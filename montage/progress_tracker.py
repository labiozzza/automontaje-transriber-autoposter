"""Centralised progress model for long-running operations.

One place builds a *plan* of stages with weights, measures each stage when a
real metric exists and keeps the aggregate honest otherwise.  Every operation
(montage, drawings, covers, publish) reports the same normalised snapshot so
the UI can render a single collapsible component.

Design rules (see task spec section 4/7):
  * A stage is ``pending``, ``running``, ``completed``, ``failed``, ``skipped``
    or ``cancelled``.
  * A stage progress is a number 0..100 or ``None`` when it cannot be measured
    (indeterminate).  We never invent 90/99 to fake movement.
  * The aggregate is a weighted mean over the stages that are actually part of
    the plan.  Disabled stages are excluded from the plan entirely.
  * The aggregate never moves backwards on its own.  A backwards move is only
    possible through an explicit ``reset_stage`` (retry) and is flagged.
  * 100 % is reserved for a genuinely finished operation.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
SKIPPED = "skipped"
CANCELLED = "cancelled"

_TERMINAL = {COMPLETED, FAILED, SKIPPED, CANCELLED}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


class Stage:
    """A single unit of work with an optional measured progress."""

    __slots__ = (
        "key", "label", "weight", "status", "progress", "detail",
        "error", "started_at", "finished_at", "parent",
    )

    def __init__(self, key: str, label: str, weight: float = 1.0,
                 parent: Optional[str] = None) -> None:
        self.key = key
        self.label = label
        self.weight = max(0.0001, float(weight))
        self.status = PENDING
        self.progress: Optional[float] = None
        self.detail = ""
        self.error = ""
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.parent = parent

    @property
    def indeterminate(self) -> bool:
        return self.status == RUNNING and self.progress is None

    def fraction(self) -> float:
        if self.status == COMPLETED:
            return 1.0
        if self.status in (SKIPPED,):
            return 1.0
        if self.status == CANCELLED:
            return 0.0
        if self.progress is None:
            return 0.0
        return _clamp(self.progress) / 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "weight": round(self.weight, 3),
            "status": self.status,
            "progress": None if self.progress is None else round(self.progress, 1),
            "indeterminate": self.indeterminate,
            "detail": self.detail,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "parent": self.parent,
        }


class ProgressTracker:
    """Holds the plan and the live state of one operation."""

    def __init__(self, operation_id: str, kind: str,
                 stages: Iterable[Stage], *,
                 label: str = "",
                 on_change: Optional[Callable[["ProgressTracker"], None]] = None) -> None:
        self.operation_id = operation_id
        self.kind = kind
        self.label = label
        self._order: List[str] = []
        self._stages: Dict[str, Stage] = {}
        for stage in stages:
            self._stages[stage.key] = stage
            self._order.append(stage.key)
        self.status = PENDING
        self.detail = ""
        self.error = ""
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.finished_at: Optional[float] = None
        self._version = 0
        self._last_overall = 0.0
        self._allow_backwards = False
        self._lock = threading.RLock()
        self._on_change = on_change

    # ------------------------------------------------------------------ plan
    def stage_keys(self) -> List[str]:
        return list(self._order)

    def stage(self, key: str) -> Stage:
        return self._stages[key]

    def _plan(self) -> List[Stage]:
        return [self._stages[k] for k in self._order
                if self._stages[k].status not in (SKIPPED, CANCELLED)]

    # -------------------------------------------------------------- mutation
    def _touch(self) -> None:
        self._version += 1
        self.updated_at = time.time()
        if self._on_change is not None:
            try:
                self._on_change(self)
            except Exception:
                pass

    def start(self, key: str, detail: str = "") -> None:
        with self._lock:
            stage = self._stages[key]
            stage.status = RUNNING
            stage.started_at = stage.started_at or time.time()
            if detail:
                stage.detail = detail
            if self.status in (PENDING,):
                self.status = RUNNING
            self._touch()

    def update(self, key: str, progress: Optional[float], detail: str = "") -> None:
        with self._lock:
            stage = self._stages[key]
            if stage.status in (PENDING,):
                stage.status = RUNNING
                stage.started_at = stage.started_at or time.time()
            stage.progress = None if progress is None else _clamp(progress)
            if detail:
                stage.detail = detail
            if self.status == PENDING:
                self.status = RUNNING
            self._touch()

    def complete(self, key: str, detail: str = "") -> None:
        with self._lock:
            stage = self._stages[key]
            stage.status = COMPLETED
            stage.progress = 100.0
            stage.finished_at = time.time()
            if detail:
                stage.detail = detail
            if self.status in (PENDING, RUNNING):
                # operation stays RUNNING until finish() is called
                self.status = RUNNING
            self._touch()

    def fail(self, key: str, error: str) -> None:
        with self._lock:
            stage = self._stages[key]
            stage.status = FAILED
            stage.error = error
            stage.finished_at = time.time()
            self.status = FAILED
            self.error = error
            self._touch()

    def skip(self, key: str, detail: str = "") -> None:
        with self._lock:
            stage = self._stages[key]
            stage.status = SKIPPED
            stage.finished_at = time.time()
            if detail:
                stage.detail = detail
            self._touch()

    def reset_stage(self, key: str, detail: str = "Повторная попытка") -> None:
        """Explicitly restart a stage (retry) — the only path allowed to move back."""
        with self._lock:
            stage = self._stages[key]
            stage.status = RUNNING
            stage.progress = None
            stage.error = ""
            stage.started_at = time.time()
            stage.finished_at = None
            stage.detail = detail
            self._allow_backwards = True
            self.status = RUNNING
            self.error = ""
            self._touch()

    def finish(self, detail: str = "") -> None:
        with self._lock:
            self.status = COMPLETED
            self.finished_at = time.time()
            if detail:
                self.detail = detail
            self._touch()

    def set_detail(self, detail: str) -> None:
        with self._lock:
            self.detail = detail
            self._touch()

    # ------------------------------------------------------------- aggregate
    def has_indeterminate(self) -> bool:
        return any(s.indeterminate for s in self._plan())

    def overall(self) -> float:
        with self._lock:
            plan = self._plan()
            if not plan:
                return 100.0 if self.status == COMPLETED else 0.0
            total_weight = sum(s.weight for s in plan)
            if total_weight <= 0:
                return 0.0
            acc = sum(s.weight * s.fraction() for s in plan)
            pct = acc / total_weight * 100.0
            all_done = all(s.status == COMPLETED for s in plan)
            if not all_done:
                pct = min(pct, 99.9)
            if self._allow_backwards:
                self._allow_backwards = False
            elif pct < self._last_overall:
                pct = self._last_overall
            self._last_overall = pct
            return round(pct, 1)

    def active_stage(self) -> Optional[Stage]:
        for key in self._order:
            stage = self._stages[key]
            if stage.status == RUNNING:
                return stage
        return None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            overall = self.overall()
            active = self.active_stage()
            return {
                "operation_id": self.operation_id,
                "kind": self.kind,
                "label": self.label,
                "status": self.status,
                "progress": overall,
                "indeterminate": self.has_indeterminate(),
                "active_stage": active.key if active else None,
                "detail": self.detail or (active.detail if active else ""),
                "error": self.error,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "finished_at": self.finished_at,
                "version": self._version,
                "stages": [self._stages[k].to_dict() for k in self._order],
            }


# ----------------------------------------------------------------- helpers --

def stage_specs_to_stages(specs: Iterable[Dict[str, Any]]) -> List[Stage]:
    return [Stage(s["key"], s.get("label", s["key"]),
                  s.get("weight", 1.0), s.get("parent")) for s in specs]


def montage_plan(options: Dict[str, Any]) -> List[Stage]:
    """Build the montage stage plan from the requested options.

    Only stages that will actually run are included; disabled features are not
    added, so their weight never drags the aggregate (spec section 4.3/5).
    """
    stages: List[Stage] = [Stage("prepare", "Проверка и подготовка исходника", 2.0)]
    rotation = bool(options.get("_rotation_applied", False))
    if rotation:
        stages.append(Stage("rotate", "Исправление ориентации", 1.0))
    if options.get("mirror_horizontal"):
        stages.append(Stage("mirror", "Отражение видео", 2.0))
    stages.append(Stage("configs", "Анализ текста и настройка эффектов", 4.0))
    do_face = bool(options.get("face_tracking")) and bool(options.get("autozoom"))
    if do_face:
        stages.append(Stage("faces", "Отслеживание лица", 15.0))
    if options.get("autozoom"):
        stages.append(Stage("zoom", "Автозум", 12.0))
    if options.get("subtitles", True):
        stages.append(Stage("subtitles", "Создание субтитров", 20.0))
    overlays = options.get("overlays") if isinstance(options.get("overlays"), list) else []
    animation = options.get("animation") or {}
    if overlays or bool(animation.get("enabled")):
        stages.append(Stage("animation", "Наложения и анимации", 8.0))
    if options.get("blur_banned_words"):
        stages.append(Stage("audio_censor", "Заглушение нежелательных слов", 3.0))
    stages.append(Stage("finalize", "Финальное кодирование", 22.0))
    stages.append(Stage("preview", "Создание версии для просмотра", 8.0))
    stages.append(Stage("verify", "Проверка и сохранение результата", 2.0))
    return stages
