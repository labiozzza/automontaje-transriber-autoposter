from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any


def stats_directory() -> Path:
    return Path(
        os.environ.get("SOCIAL_MEDIA_STATS_DIR", str(Path.home() / "Desktop" / "socialmediastats"))
    ).expanduser().resolve()


def _load_stats_module() -> ModuleType:
    script = stats_directory() / "update_stats.py"
    if not script.exists():
        raise RuntimeError(f"socialmediastats не найден: {script}")
    spec = importlib.util.spec_from_file_location("automontaje_social_media_stats", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Не удалось загрузить socialmediastats")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive_instagram_reel(
    *,
    media: dict[str, Any],
    account_id: str,
    transcript: str,
    source_video: Path,
    published_video: Path,
    post_text: str,
    job_id: str,
    variant: str,
) -> Path:
    module = _load_stats_module()
    return Path(
        module.register_published_reel(
            media=media,
            account_id=account_id,
            transcript=transcript,
            source_video=source_video,
            published_video=published_video,
            post_text=post_text,
            job_id=job_id,
            variant=variant,
        )
    )
