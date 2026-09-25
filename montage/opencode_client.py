from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


class OpenCodeError(RuntimeError):
    pass


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


def ask_json(prompt: str, *, model: str | None = None, timeout: float = 240, retries: int = 1) -> dict[str, Any]:
    if len(prompt) > 60_000:
        raise OpenCodeError("Запрос к GPT слишком большой")
    executable = os.environ.get("OPENCODE_BIN") or "opencode"
    selected_model = model or os.environ.get("DRAWING_MODEL") or "openai/gpt-5.6-sol"
    allowed_env = {
        key: value for key, value in os.environ.items()
        if key in {
            "HOME", "PATH", "SHELL", "TMPDIR", "LANG", "LC_ALL", "TERM", "USER", "LOGNAME",
            "XDG_CONFIG_HOME", "XDG_DATA_HOME", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
            "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS",
        }
    }
    error = "GPT не вернул результат"
    for attempt in range(max(0, retries) + 1):
        try:
            with tempfile.TemporaryDirectory(prefix="montage-drawing-") as temporary:
                Path(temporary, "opencode.json").write_text(json.dumps({
                    "$schema": "https://opencode.ai/config.json",
                    "permission": "deny",
                    "share": "disabled",
                }), encoding="utf-8")
                result = subprocess.run(
                    [executable, "run", "--pure", "-m", selected_model, "--format", "json", prompt],
                    cwd=Path(temporary),
                    env=allowed_env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                )
            if result.returncode != 0:
                error = f"OpenCode завершился с кодом {result.returncode}"
            else:
                raw = result.stdout[:2_000_000]
                parsed = _json_object(_assistant_text(raw)) or _json_object(raw)
                if parsed is not None:
                    return parsed
                error = "GPT вернул некорректный JSON"
        except subprocess.TimeoutExpired:
            error = "GPT не ответил вовремя"
        except OSError as exc:
            error = f"Не удалось запустить OpenCode: {exc}"
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))
    raise OpenCodeError(error)
