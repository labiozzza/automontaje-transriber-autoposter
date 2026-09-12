from __future__ import annotations

import argparse
import shutil
import os
import subprocess
import sys
from pathlib import Path
from PIL import Image

try:
    from montage.instaposter.git_auth import ensure_push_auth, git_env
except ModuleNotFoundError:
    from git_auth import ensure_push_auth, git_env

CURL = shutil.which("curl") or "curl"



REPO_DIR = Path(os.environ.get("GITHUB_REPO_DIR", str(Path(__file__).resolve().parent.parent / "instagram-media")))
BRANCH = "main"


def die(msg: str) -> None:
    print(f"\nОШИБКА: {msg}", file=sys.stderr)
    raise SystemExit(1)


def run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=git_env(),
    )


def convert_png_to_jpg(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, "white")
            bg.paste(rgba, mask=rgba.getchannel("A"))
            rgb = bg
        else:
            rgb = img.convert("RGB")
        rgb.save(dst, "JPEG", quality=95, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help=r"Например F:\instaposter\courusel\104")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    prefix = source_dir.name

    if not source_dir.is_dir():
        die(f"Нет папки: {source_dir}")

    txt = source_dir / f"{prefix}.txt"
    if not txt.is_file():
        die(f"Нет файла описания: {txt}")

    pngs = sorted(source_dir.glob(f"{prefix}_*.png"))
    if not pngs:
        die(f"Не найдены {prefix}_*.png в {source_dir}")

    if not (2 <= len(pngs) <= 10):
        die(f"Для Instagram карусели нужно 2–10 изображений. Найдено: {len(pngs)}")

    expected = [source_dir / f"{prefix}_{i:02d}.png" for i in range(1, len(pngs) + 1)]
    missing = [p.name for p in expected if not p.is_file()]
    if missing:
        die("Нарушена последовательность файлов:\n" + "\n".join(missing))

    if not (REPO_DIR / ".git").exists():
        die(f"Не найден git-репозиторий: {REPO_DIR}")

    print(f"Карусель: {prefix}")
    print(f"Фото: {len(expected)}")
    print(f"Описание: {txt}")

    created: list[Path] = []
    for src in expected:
        dst = REPO_DIR / f"{src.stem}.jpg"
        print(f"{src.name} -> {dst.name}")
        convert_png_to_jpg(src, dst)
        if dst.read_bytes()[:2] != b"\xff\xd8":
            die(f"Получился не JPEG: {dst}")
        created.append(dst)

    rels = [p.name for p in created]

    r = run(["git", "add", "--", *rels], cwd=REPO_DIR)
    if r.returncode != 0:
        die(r.stderr)

    status = run(["git", "status", "--porcelain", "--", *rels], cwd=REPO_DIR)
    if status.returncode != 0:
        die(status.stderr)

    if status.stdout.strip():
        commit = run(["git", "commit", "-m", f"Add carousel {prefix}", "--", *rels], cwd=REPO_DIR)
        if commit.returncode != 0:
            die(commit.stdout + "\n" + commit.stderr)
        print("git commit: OK")
    else:
        print("Изменений для commit нет.")

    try:
        ensure_push_auth(REPO_DIR)
    except RuntimeError as exc:
        die(str(exc))
    push = run(["git", "push", "origin", BRANCH], cwd=REPO_DIR)
    if push.returncode != 0:
        die(push.stdout + "\n" + push.stderr)

    print("git push: OK")
    print(f"COUNT={len(created)}")
    print(f"TEXT_FILE={txt}")


if __name__ == "__main__":
    main()
