from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_DIR / "config"
SECRETS_FILE = CONFIG_DIR / "secrets.json"
MEDIA_REPO_DIR = Path(os.environ.get("GITHUB_REPO_DIR", str(PROJECT_DIR / "instagram-media")))

_loaded = False


def load_secrets_env() -> None:
    """Load gitignored config/secrets.json into os.environ (without overriding) and
    set defaults needed by vendored instaposter scripts."""
    global _loaded
    if _loaded:
        return
    if SECRETS_FILE.exists():
        try:
            payload = json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for key, value in payload.items():
            if value and not os.environ.get(key):
                os.environ[key] = str(value)
    os.environ.setdefault("GITHUB_REPO_DIR", str(MEDIA_REPO_DIR))
    os.environ.setdefault("INSTAGRAM_API_VERSION", "v26.0")
    _loaded = True


def reload_secrets_env() -> None:
    global _loaded
    _loaded = False
    load_secrets_env()


def get_software_defaults() -> dict:
    load_secrets_env()
    return {
        "github_repo_dir": os.environ.get("GITHUB_REPO_DIR", str(MEDIA_REPO_DIR)),
        "instagram_ready": bool(os.environ.get("INSTAGRAM_ACCESS_TOKEN") and os.environ.get("IG_USER_ID")),
        "threads_ready": bool(os.environ.get("THREADS_ACCESS_TOKEN") and os.environ.get("THREADS_USER_ID")),
        "youtube_ready": (PROJECT_DIR / "montage" / "instaposter" / "token.json").exists(),
        "telegram_ready": bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")),
    }