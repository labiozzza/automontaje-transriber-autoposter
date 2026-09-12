from __future__ import annotations

import os
import subprocess
from pathlib import Path


def github_token_configured() -> bool:
    return bool((os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip())


def git_env() -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    token = (env.get("GITHUB_TOKEN") or env.get("GH_TOKEN") or "").strip()
    if token:
        env["GITHUB_TOKEN"] = token
        env["GIT_CONFIG_COUNT"] = "2"
        env["GIT_CONFIG_KEY_0"] = "credential.helper"
        env["GIT_CONFIG_VALUE_0"] = ""
        env["GIT_CONFIG_KEY_1"] = "credential.helper"
        env["GIT_CONFIG_VALUE_1"] = (
            "!f() { echo username=x-access-token; echo password=$GITHUB_TOKEN; }; f"
        )
    return env


def ensure_push_auth(repo_dir: Path) -> None:
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=str(repo_dir), capture_output=True, text=True, env=git_env(),
    )
    if remote.returncode == 0 and remote.stdout.strip().startswith(("https://", "http://")):
        if not github_token_configured():
            raise RuntimeError(
                "Для загрузки медиа в GitHub не задан GITHUB_TOKEN. "
                "Добавьте токен с правом Contents: Read and write в окружение "
                "процесса или gitignored config/secrets.json и перезапустите сервер."
            )
