from __future__ import annotations

import atexit
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional


class OpenCodeError(RuntimeError):
    """Base error for BigPickle / opencode failures."""

    reason = "cli"
    retryable = True


class NoActivityError(OpenCodeError):
    reason = "no_activity"
    retryable = True


class HardTimeoutError(OpenCodeError):
    reason = "hard_timeout"
    retryable = False


class BudgetExhaustedError(OpenCodeError):
    reason = "budget"
    retryable = False


class InvalidJSONError(OpenCodeError):
    reason = "bad_json"
    retryable = True


class CliError(OpenCodeError):
    reason = "cli"
    retryable = True


# Defaults are intentionally independent: the inactivity timeout guards against
# a stalled model, while the hard limit caps the whole attempt even if the model
# keeps emitting noise.  The total budget caps every attempt together.
DEFAULT_IDLE_TIMEOUT = 300.0
DEFAULT_HARD_TIMEOUT = 600.0
DEFAULT_TOTAL_BUDGET = 900.0

_MAX_TEXT_CHARS = 2_000_000
_MAX_RAW_CHARS = 3_000_000
_MAX_STDERR_CHARS = 8_000
_MAX_EVENTS = 500_000
_MAX_LINE_BYTES = 2_000_000
_POLL_INTERVAL = 0.2
_ACTIVITY_REPORT_INTERVAL = 1.0


def _env_float(name: str, default: float) -> float:
    try:
        value = os.environ.get(name)
        return float(value) if value not in (None, "") else float(default)
    except (TypeError, ValueError):
        return float(default)


def _assistant_text(raw: str) -> str:
    parts: list[str] = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") in {"text", "part"}:
            part = event.get("part")
            text = part.get("text") if isinstance(part, dict) else part
            if text:
                parts.append(str(text))
        elif event.get("type") == "assistant_text" and event.get("text"):
            parts.append(str(event["text"]))
    return "\n".join(parts)


def _json_object(text: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    candidates = [text]
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def _event_text(event: dict[str, Any]) -> str:
    event_type = event.get("type")
    if event_type in {"text", "part"}:
        part = event.get("part")
        if isinstance(part, dict):
            text = part.get("text")
        elif isinstance(part, str):
            text = part
        else:
            text = None
        return str(text) if text else ""
    if event_type == "assistant_text" and event.get("text"):
        return str(event["text"])
    return ""


def _event_error(event: dict[str, Any]) -> tuple[str, bool]:
    """Extracts (message, retryable) from an opencode error event."""
    error = event.get("error")
    message = ""
    retryable = True
    if isinstance(error, dict):
        data = error.get("data")
        if isinstance(data, dict):
            message = str(data.get("message") or data.get("responseBody") or "")
            if "isRetryable" in data:
                retryable = bool(data.get("isRetryable"))
        if not message:
            message = str(error.get("name") or error.get("message") or "")
    elif error:
        message = str(error)
    return message or "OpenCode вернул ошибку", retryable


def _safe_close(stream: Any) -> None:
    try:
        if stream is not None:
            stream.close()
    except Exception:
        pass


def _terminate_tree(proc: subprocess.Popen) -> None:
    """Kills the process and every child in its process group."""
    try:
        if proc.poll() is None:
            killed = False
            if hasattr(os, "killpg") and proc.pid:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    killed = True
                except (ProcessLookupError, PermissionError, OSError):
                    killed = False
            if not killed:
                try:
                    proc.kill()
                except Exception:
                    pass
    finally:
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


_ACTIVE_LOCK = threading.Lock()
_ACTIVE_PROCS: "set[subprocess.Popen]" = set()


def _register_proc(proc: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROCS.add(proc)


def _unregister_proc(proc: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROCS.discard(proc)


def terminate_active() -> None:
    """Kills every running BigPickle process (used on server shutdown/restart)."""
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE_PROCS)
    for proc in procs:
        _terminate_tree(proc)


atexit.register(terminate_active)


class _AttemptTimeout(Exception):
    def __init__(self, reason: str, event: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.event = event
        self.detail = detail


def _read_stdout(stream: Any, state: dict[str, Any], lock: threading.Lock) -> None:
    try:
        for raw in iter(stream.readline, b""):
            if not raw:
                break
            now = time.monotonic()
            line = raw[:_MAX_LINE_BYTES].decode("utf-8", errors="replace").strip()
            event: Optional[dict[str, Any]] = None
            if line:
                try:
                    loaded = json.loads(line)
                    if isinstance(loaded, dict):
                        event = loaded
                except Exception:
                    event = None
            with lock:
                state["last_activity"] = now
                if event is not None:
                    if state["events"] < _MAX_EVENTS:
                        state["events"] += 1
                    text = _event_text(event)
                    if text and state["text_len"] < _MAX_TEXT_CHARS:
                        state["text"].append(text)
                        state["text_len"] += len(text)
                    if event.get("type") == "error":
                        state["cli_error"], state["cli_retryable"] = _event_error(event)
                if state["raw_len"] < _MAX_RAW_CHARS:
                    state["raw"].append(line)
                    state["raw_len"] += len(line)
    except Exception:
        pass


def _read_stderr(stream: Any, state: dict[str, Any], lock: threading.Lock) -> None:
    try:
        for raw in iter(stream.readline, b""):
            if not raw:
                break
            chunk = raw[:_MAX_LINE_BYTES].decode("utf-8", errors="replace")
            with lock:
                state["stderr"] += chunk
                if len(state["stderr"]) > _MAX_STDERR_CHARS:
                    state["stderr"] = state["stderr"][-_MAX_STDERR_CHARS:]
    except Exception:
        pass


def _run_attempt(
    prompt: str,
    executable: str,
    model: str,
    allowed_env: dict[str, str],
    *,
    idle_timeout: float,
    hard_timeout: float,
    attempt: int,
    on_activity: Callable[[int], None],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="montage-drawing-") as temporary:
        Path(temporary, "opencode.json").write_text(json.dumps({
            "$schema": "https://opencode.ai/config.json",
            "permission": "deny",
            "share": "disabled",
        }), encoding="utf-8")
        proc = subprocess.Popen(
            [executable, "run", "--pure", "-m", model, "--format", "json", prompt],
            cwd=Path(temporary),
            env=allowed_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        _register_proc(proc)
        lock = threading.Lock()
        state: dict[str, Any] = {
            "last_activity": time.monotonic(),
            "events": 0,
            "text": [],
            "text_len": 0,
            "raw": [],
            "raw_len": 0,
            "stderr": "",
            "cli_error": "",
            "cli_retryable": True,
        }
        readers = [
            threading.Thread(target=_read_stdout, args=(proc.stdout, state, lock), daemon=True),
            threading.Thread(target=_read_stderr, args=(proc.stderr, state, lock), daemon=True),
        ]
        for reader in readers:
            reader.start()
        start = time.monotonic()
        last_report = start
        last_count = 0
        timeout: Optional[tuple[str, str, str]] = None
        try:
            while True:
                if proc.poll() is not None:
                    break
                now = time.monotonic()
                with lock:
                    since_activity = now - state["last_activity"]
                    count = state["events"]
                if since_activity > idle_timeout:
                    timeout = (
                        "no_activity",
                        "attempt_idle_timeout",
                        f"Нет активности от BigPickle более {int(idle_timeout)} с",
                    )
                    break
                if now - start > hard_timeout:
                    timeout = (
                        "hard_timeout",
                        "attempt_hard_timeout",
                        f"Попытка {attempt} превысила жёсткий лимит {int(hard_timeout)} с",
                    )
                    break
                if count > last_count and now - last_report >= _ACTIVITY_REPORT_INTERVAL:
                    on_activity(count)
                    last_report = now
                    last_count = count
                time.sleep(_POLL_INTERVAL)
        finally:
            if timeout is not None:
                _terminate_tree(proc)
            _unregister_proc(proc)
            for reader in readers:
                reader.join(timeout=5)
            _safe_close(proc.stdout)
            _safe_close(proc.stderr)
        if timeout is not None:
            raise _AttemptTimeout(*timeout)
        with lock:
            return {
                "returncode": proc.returncode,
                "text": "\n".join(state["text"]),
                "raw": "".join(state["raw"]),
                "stderr": state["stderr"].strip(),
                "cli_error": state["cli_error"],
                "cli_retryable": state["cli_retryable"],
                "events": state["events"],
            }


def _error_for(reason: str, message: str, retryable: bool = True) -> OpenCodeError:
    mapping = {
        "no_activity": NoActivityError,
        "hard_timeout": HardTimeoutError,
        "budget": BudgetExhaustedError,
        "bad_json": InvalidJSONError,
        "cli": CliError,
    }
    error = mapping.get(reason, OpenCodeError)(message)
    error.retryable = bool(retryable) and bool(getattr(error, "retryable", True))
    return error


def ask_json(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    idle_timeout: Optional[float] = None,
    hard_timeout: Optional[float] = None,
    total_budget: Optional[float] = None,
    retries: int = 1,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    if len(prompt) > 60_000:
        raise OpenCodeError("Запрос к GPT слишком большой")
    executable = os.environ.get("OPENCODE_BIN") or "opencode"
    selected_model = model or os.environ.get("DRAWING_MODEL") or "opencode/big-pickle"
    allowed_env = {
        key: value for key, value in os.environ.items()
        if key in {
            "HOME", "PATH", "SHELL", "TMPDIR", "LANG", "LC_ALL", "TERM", "USER", "LOGNAME",
            "XDG_CONFIG_HOME", "XDG_DATA_HOME", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
            "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS", "OPENCODE_CLIENT", "PWD", "OLDPWD",
            "COMMAND_MODE", "__CFBundleIdentifier",
        }
    }
    if idle_timeout is None:
        idle_timeout = _env_float("OPENCODE_IDLE_TIMEOUT", DEFAULT_IDLE_TIMEOUT)
    if hard_timeout is None:
        hard_timeout = _env_float(
            "OPENCODE_HARD_TIMEOUT", timeout if timeout is not None else DEFAULT_HARD_TIMEOUT
        )
    if total_budget is None:
        total_budget = _env_float("OPENCODE_TOTAL_BUDGET", DEFAULT_TOTAL_BUDGET)
    idle_timeout = float(idle_timeout)
    hard_timeout = float(hard_timeout)
    total_budget = float(total_budget)
    attempts_total = max(0, retries) + 1

    def report(event: str, attempt: int, detail: str) -> None:
        if not progress_callback:
            return
        try:
            progress_callback({
                "event": event,
                "attempt": attempt,
                "attempts_total": attempts_total,
                "attempt_timeout_seconds": hard_timeout,
                "idle_timeout_seconds": idle_timeout,
                "event_at": time.time(),
                "detail": detail,
            })
        except Exception:
            pass

    budget_start = time.monotonic()
    last_error: OpenCodeError = OpenCodeError("GPT не вернул результат")
    for attempt in range(1, attempts_total + 1):
        remaining = total_budget - (time.monotonic() - budget_start)
        if remaining <= 1.0:
            last_error = BudgetExhaustedError("Превышен общий лимит времени на генерацию")
            break
        report(
            "attempt_started", attempt,
            f"Запрос отправлен BigPickle, ожидаем ответ (попытка {attempt} из {attempts_total})",
        )
        try:
            result = _run_attempt(
                prompt, executable, selected_model, allowed_env,
                idle_timeout=min(idle_timeout, remaining),
                hard_timeout=min(hard_timeout, remaining),
                attempt=attempt,
                on_activity=lambda count, number=attempt: report(
                    "activity", number, f"Модель отвечает, получено событий: {count}"
                ),
            )
            if result["returncode"] != 0:
                stderr = result["stderr"][-500:]
                message = f"OpenCode завершился с кодом {result['returncode']}"
                if stderr:
                    message += f": {stderr}"
                raise CliError(message)
            if result["cli_error"]:
                raise _error_for("cli", result["cli_error"], result["cli_retryable"])
            report("response_received", attempt, "Ответ BigPickle получен, проверяем JSON")
            parsed = _json_object(result["text"])
            if parsed is None:
                raise InvalidJSONError("BigPickle вернул некорректный JSON")
            report("response_parsed", attempt, "Ответ разобран, проверяем и сохраняем схемы")
            return parsed
        except _AttemptTimeout as exc:
            last_error = _error_for(exc.reason, exc.detail)
            report(exc.event, attempt, exc.detail)
        except OpenCodeError as exc:
            last_error = exc
            report("attempt_invalid_json" if exc.reason == "bad_json" else "attempt_failed", attempt, str(exc))
        if attempt >= attempts_total or not last_error.retryable:
            break
        remaining = total_budget - (time.monotonic() - budget_start)
        if remaining <= 1.0:
            last_error = BudgetExhaustedError("Превышен общий лимит времени на генерацию")
            break
        report("retry_scheduled", attempt, f"Запускаем повторную попытку {attempt + 1} из {attempts_total}")
        time.sleep(0.5 * attempt)
    raise last_error
