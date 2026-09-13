from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote

try:
    from montage.instaposter.git_auth import ensure_push_auth, git_env
except ModuleNotFoundError:
    from git_auth import ensure_push_auth, git_env

CURL = shutil.which("curl") or "curl"



IG_USER_ID = os.environ.get("IG_USER_ID", "").strip()

GITHUB_REPO_DIR = Path(os.environ.get("GITHUB_REPO_DIR", str(Path(__file__).resolve().parent.parent / "instagram-media")))
GITHUB_OWNER = os.environ.get("GITHUB_OWNER", "volynecsvatoslav-png").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "instagram-media").strip()
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main").strip()
GITHUB_REELS_DIR = os.environ.get("GITHUB_REELS_DIR", "reels").strip()

GRAPH_HOST = os.environ.get("GRAPH_HOST", "graph.instagram.com").strip()
RAW_HOST = os.environ.get("RAW_HOST", "raw.githubusercontent.com").strip()

API_VERSION = os.environ.get("INSTAGRAM_API_VERSION", "v26.0").strip()
ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "").strip()

GITHUB_MAX_FILE_BYTES = 100 * 1024 * 1024
META_MAX_FILE_BYTES = 1_000_000_000

POLL_INTERVAL_SECONDS = 20
POLL_TIMEOUT_SECONDS = 10 * 60


def die(message: str) -> None:
    print(f"\nОШИБКА: {message}", file=sys.stderr)
    raise SystemExit(1)


def run_process(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 120,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=git_env(),
    )

    if check and result.returncode != 0:
        die(
            f"Команда завершилась с кодом {result.returncode}:\n"
            f"{' '.join(args)}\n\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result


def resolve_host_via_doh(host: str) -> str:
    result = run_process(
        [
            CURL,
            "-sS",
            f"https://dns.google/resolve?name={host}&type=A",
        ],
        timeout=30,
    )

    if result.returncode != 0:
        die(f"Не удалось разрешить {host} через DoH:\n{result.stderr}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        die(f"DoH вернул не JSON для {host}:\n{result.stdout}")

    for answer in data.get("Answer", []):
        if answer.get("type") == 1 and answer.get("data"):
            return str(answer["data"])

    die(f"Не найден A-record для {host}:\n{result.stdout}")
    raise AssertionError


def parse_fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None

    try:
        return float(Fraction(value))
    except Exception:
        return None


def validate_video_with_ffprobe(video_path: Path) -> None:
    ffprobe = shutil.which("ffprobe")

    if not ffprobe:
        print("   ffprobe не найден — пропускаем глубокую проверку кодеков.")
        return

    result = run_process(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration,format_name:"
                "stream=index,codec_type,codec_name,width,height,"
                "avg_frame_rate,sample_rate"
            ),
            "-of",
            "json",
            str(video_path),
        ],
        timeout=60,
    )

    if result.returncode != 0:
        die(f"ffprobe не смог прочитать видео:\n{result.stderr}")

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        die("ffprobe вернул некорректный JSON.")

    duration = None
    try:
        duration = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        pass

    streams = info.get("streams", [])
    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"),
        None,
    )
    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"),
        None,
    )

    if not video_stream:
        die("В файле не найден видеопоток.")

    codec = str(video_stream.get("codec_name", "")).lower()
    width = video_stream.get("width")
    height = video_stream.get("height")
    fps = parse_fraction(video_stream.get("avg_frame_rate"))

    if fps:
        print(
            f"   Видео: codec={codec or '?'}, "
            f"{width}x{height}, fps={fps:.2f}"
        )
    else:
        print(
            f"   Видео: codec={codec or '?'}, "
            f"{width}x{height}, fps=?"
        )

    if duration is not None:
        print(f"   Длительность: {duration:.1f} сек.")

        if duration < 3:
            die("Reel короче 3 секунд.")

        if duration > 15 * 60:
            die("Reel длиннее 15 минут.")

    if codec not in {"h264", "hevc"}:
        die(
            f"Неподходящий видеокодек: {codec}. "
            "Нужен H.264 или HEVC."
        )

    if isinstance(width, int) and width > 1920:
        die(f"Ширина видео {width}px. Максимум для Reels — 1920px.")

    if fps is not None and not (23 <= fps <= 60):
        die(f"FPS={fps:.2f}. Для Reels нужен диапазон 23–60 FPS.")

    if audio_stream:
        audio_codec = str(audio_stream.get("codec_name", "")).lower()
        sample_rate = audio_stream.get("sample_rate")

        print(
            f"   Аудио: codec={audio_codec or '?'}, "
            f"sample_rate={sample_rate or '?'}"
        )

        if audio_codec != "aac":
            die(f"Аудиокодек {audio_codec}. Для Reels нужен AAC.")

        if sample_rate and str(sample_rate) != "48000":
            print(
                f"   ПРЕДУПРЕЖДЕНИЕ: audio sample rate={sample_rate}; "
                "рекомендуется 48000 Hz."
            )

    if width and height:
        ratio = width / height
        expected = 9 / 16

        if abs(ratio - expected) > 0.03:
            print(
                f"   ПРЕДУПРЕЖДЕНИЕ: соотношение {width}:{height}; "
                "для Reels рекомендуется 9:16."
            )


def validate_local_video(video_path: Path) -> None:
    if not video_path.is_file():
        die(f"Видео не найдено:\n{video_path}")

    if video_path.suffix.lower() not in {".mp4", ".mov"}:
        die("Используй .mp4 или .mov.")

    size = video_path.stat().st_size

    print(f"   Файл: {video_path}")
    print(f"   Размер: {size / 1024 / 1024:.2f} MiB")

    if size > META_MAX_FILE_BYTES:
        die("Видео больше 1 GB — Instagram Reels API его не примет.")

    if size > GITHUB_MAX_FILE_BYTES:
        die(
            "Видео больше 100 MiB. GitHub блокирует такой файл в обычном Git.\n"
            "Для этого Reel нужен другой публичный media hosting "
            "(S3/R2/Object Storage/VPS), а не GitHub Raw."
        )

    validate_video_with_ffprobe(video_path)


def ensure_git_repo() -> None:
    if not GITHUB_REPO_DIR.is_dir():
        die(f"Нет локального репозитория:\n{GITHUB_REPO_DIR}")

    result = run_process(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=GITHUB_REPO_DIR,
    )

    if result.returncode != 0 or result.stdout.strip() != "true":
        die(f"{GITHUB_REPO_DIR} не является Git-репозиторием.")


def re_safe_filename(filename: str) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()

    cleaned = "".join(
        char if (
            char.isascii()
            and (char.isalnum() or char in {"_", "-"})
        ) else "_"
        for char in stem
    ).strip("_")

    if not cleaned:
        cleaned = f"reel_{int(time.time())}"

    return f"{cleaned}{suffix}"


def publish_video_to_github(video_path: Path) -> tuple[Path, str]:
    ensure_git_repo()

    target_dir = GITHUB_REPO_DIR / GITHUB_REELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re_safe_filename(video_path.name)
    target_path = target_dir / safe_name

    if video_path.resolve() != target_path.resolve():
        print(f"   Копируем -> {target_path}")
        shutil.copy2(video_path, target_path)
    else:
        print("   Видео уже находится в GitHub repository.")

    relative_path = target_path.relative_to(GITHUB_REPO_DIR).as_posix()

    run_process(
        ["git", "add", "--", relative_path],
        cwd=GITHUB_REPO_DIR,
        check=True,
    )

    status = run_process(
        ["git", "status", "--porcelain", "--", relative_path],
        cwd=GITHUB_REPO_DIR,
        check=True,
    )

    if status.stdout.strip():
        commit = run_process(
            [
                "git",
                "commit",
                "-m",
                f"Add Instagram Reel {safe_name}",
                "--",
                relative_path,
            ],
            cwd=GITHUB_REPO_DIR,
        )

        if commit.returncode != 0:
            die(
                f"Не удалось сделать git commit:\n"
                f"{commit.stdout}\n{commit.stderr}"
            )

        print("   Git commit: OK")
    else:
        print("   Файл в GitHub уже актуален, новый commit не нужен.")

    print("   git push...")
    try:
        ensure_push_auth(GITHUB_REPO_DIR)
    except RuntimeError as exc:
        die(str(exc))
    push = run_process(
        ["git", "push", "origin", GITHUB_BRANCH],
        cwd=GITHUB_REPO_DIR,
        timeout=300,
    )

    if push.returncode != 0:
        die(f"git push не удался:\n{push.stdout}\n{push.stderr}")

    encoded_path = quote(relative_path, safe="/")

    raw_url = (
        f"https://{RAW_HOST}/"
        f"{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{encoded_path}"
    )

    return target_path, raw_url


def check_public_video(raw_url: str, raw_ip: str) -> None:
    print(f"   RAW URL: {raw_url}")

    with tempfile.TemporaryDirectory(prefix="reel_raw_check_") as tmp:
        body_path = Path(tmp) / "video_head.bin"

        result = run_process(
            [
                CURL,
                "-sS",
                "-L",
                "--max-time",
                "60",
                "--resolve",
                f"{RAW_HOST}:443:{raw_ip}",
                "-r",
                "0-63",
                "-o",
                str(body_path),
                "-w",
                "%{http_code}|%{content_type}|%{size_download}",
                raw_url,
            ],
            timeout=90,
        )

        stats = result.stdout.strip().split("|")

        if len(stats) != 3:
            die(
                f"Не удалось проверить GitHub Raw:\n"
                f"{result.stdout}\n{result.stderr}"
            )

        status, content_type, size_download = stats

        if result.returncode != 0 or status not in {"200", "206"}:
            die(
                f"GitHub Raw недоступен:\n"
                f"HTTP={status}, type={content_type}, size={size_download}\n"
                f"{raw_url}"
            )

        head = body_path.read_bytes()[:64] if body_path.exists() else b""

        if len(head) < 12 or head[4:8] != b"ftyp":
            preview = head[:40]
            die(
                "GitHub Raw отдаёт не настоящий MP4/MOV.\n"
                f"Первые байты: {preview!r}\n"
                f"URL: {raw_url}"
            )

        print(
            f"   GitHub Raw: HTTP {status}; "
            f"Content-Type={content_type or '?'}; MP4/MOV signature OK"
        )


def meta_request(
    graph_ip: str,
    method: str,
    path: str,
    params: dict[str, str],
) -> dict:
    url = (
        f"https://{GRAPH_HOST}/"
        f"{API_VERSION}/{path.lstrip('/')}"
    )

    args = [
        CURL,
        "-sS",
        "--resolve",
        f"{GRAPH_HOST}:443:{graph_ip}",
    ]

    method = method.upper()

    if method == "GET":
        args += ["-G", url]
    elif method == "POST":
        args += ["-X", "POST", url]
    else:
        raise ValueError(method)

    for key, value in params.items():
        args += ["--data-urlencode", f"{key}={value}"]

    result = run_process(args, timeout=180)

    try:
        data = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        die(
            f"Instagram вернул не JSON.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    if result.returncode != 0 or "error" in data:
        die(
            "Ошибка Instagram API:\n"
            + json.dumps(
                data.get("error", data),
                ensure_ascii=False,
                indent=2,
            )
        )

    return data


def create_reel_container(
    graph_ip: str,
    video_url: str,
    caption: str,
    graduation_strategy: str,
    cover_url: str = "",
) -> str:
    if graduation_strategy not in {"MANUAL", "SS_PERFORMANCE"}:
        die(
            "graduation_strategy должен быть MANUAL или SS_PERFORMANCE."
        )

    params = {
        "media_type": "REELS",
        "video_url": video_url,
        "caption": caption,
        "trial_params": json.dumps({"graduation_strategy": graduation_strategy}),
        "share_to_feed": "false",
        "access_token": ACCESS_TOKEN,
    }
    if cover_url:
        params["cover_url"] = cover_url
    data = meta_request(
        graph_ip,
        "POST",
        f"{IG_USER_ID}/media",
        params,
    )

    container_id = str(data["id"])
    print(f"   CONTAINER_ID = {container_id}")

    return container_id


def wait_reel_container(graph_ip: str, container_id: str) -> None:
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    attempt = 0

    while time.time() < deadline:
        attempt += 1

        data = meta_request(
            graph_ip,
            "GET",
            container_id,
            {
                "fields": "status_code,status",
                "access_token": ACCESS_TOKEN,
            },
        )

        status_code = data.get("status_code")
        status = data.get("status", "")

        print(
            f"   [{attempt}] status_code={status_code}; "
            f"{status}"
        )

        if status_code in {"FINISHED", "PUBLISHED"}:
            return

        if status_code in {"ERROR", "EXPIRED"}:
            die(
                "Instagram не обработал Reel:\n"
                + json.dumps(data, ensure_ascii=False, indent=2)
            )

        time.sleep(POLL_INTERVAL_SECONDS)

    die(
        f"Instagram не закончил обработку Reel за "
        f"{POLL_TIMEOUT_SECONDS // 60} минут."
    )


def publish_container(graph_ip: str, container_id: str) -> str:
    data = meta_request(
        graph_ip,
        "POST",
        f"{IG_USER_ID}/media_publish",
        {
            "creation_id": container_id,
            "access_token": ACCESS_TOKEN,
        },
    )

    media_id = str(data["id"])
    print(f"   MEDIA_ID = {media_id}")

    return media_id


def get_media_info(graph_ip: str, media_id: str) -> dict:
    return meta_request(
        graph_ip,
        "GET",
        media_id,
        {
            "fields": (
                "id,media_type,media_product_type,"
                "permalink,timestamp"
            ),
            "access_token": ACCESS_TOKEN,
        },
    )


def load_caption(
    video_path: Path,
    caption: str | None,
    caption_file: str | None,
) -> str:
    if caption is not None:
        return caption

    if caption_file:
        path = Path(caption_file)

        if not path.is_file():
            die(f"Файл подписи не найден:\n{path}")

        return path.read_text(encoding="utf-8").strip()

    sidecar = video_path.with_suffix(".txt")

    if sidecar.is_file():
        print(f"   Подпись автоматически из: {sidecar}")
        return sidecar.read_text(encoding="utf-8").strip()

    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Залить локальный Reel в public GitHub Raw и опубликовать "
            "его как Instagram Trial Reel (Пробный Reels)."
        )
    )

    parser.add_argument(
        "video",
        help=r'Путь к Reel, например F:\reels\001.mp4',
    )

    parser.add_argument(
        "--caption",
        help="Подпись к Reel прямо в командной строке.",
    )

    parser.add_argument(
        "--caption-file",
        help="UTF-8 TXT-файл с подписью.",
    )

    parser.add_argument(
        "--auto-graduate",
        action="store_true",
        help=(
            "Использовать SS_PERFORMANCE: Instagram сможет автоматически "
            "показать Reel подписчикам, если пробный Reel хорошо себя покажет. "
            "Без этого флага используется MANUAL — пробный режим сохраняется, "
            "пока ты сам не выберешь публикацию для всех."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not ACCESS_TOKEN:
        die(
            "Не задан INSTAGRAM_ACCESS_TOKEN.\n\n"
            "PowerShell:\n"
            '$env:INSTAGRAM_ACCESS_TOKEN="IGAA..."'
        )

    video_path = Path(args.video).expanduser().resolve()

    print("РЕЖИМ: INSTAGRAM TRIAL REEL / ПРОБНЫЙ REELS")
    print("По умолчанию: MANUAL — подписчикам автоматически не публикуется.")

    print("1. Проверяем локальный Reel...")
    validate_local_video(video_path)

    print("\n2. Получаем DNS через DoH...")
    graph_ip = resolve_host_via_doh(GRAPH_HOST)
    raw_ip = resolve_host_via_doh(RAW_HOST)

    print(f"   {GRAPH_HOST} -> {graph_ip}")
    print(f"   {RAW_HOST} -> {raw_ip}")

    print("\n3. Кладём Reel в GitHub public repository...")
    _, raw_url = publish_video_to_github(video_path)

    time.sleep(3)

    print("\n4. Проверяем GitHub Raw...")
    check_public_video(raw_url, raw_ip)

    caption = load_caption(
        video_path,
        args.caption,
        args.caption_file,
    )

    graduation_strategy = (
        "SS_PERFORMANCE" if args.auto_graduate else "MANUAL"
    )

    print("\n5. Создаём Instagram TRIAL REELS container...")
    print(f"   graduation_strategy = {graduation_strategy}")
    container_id = create_reel_container(
        graph_ip,
        raw_url,
        caption,
        graduation_strategy=graduation_strategy,
    )

    print("\n6. Ждём обработки видео Instagram...")
    wait_reel_container(graph_ip, container_id)

    print("\n7. Публикуем Trial Reel...")
    media_id = publish_container(graph_ip, container_id)

    print("\n8. Получаем permalink...")
    media = get_media_info(graph_ip, media_id)

    print("\nГОТОВО.")
    print(json.dumps(media, ensure_ascii=False, indent=2))

    permalink = media.get("permalink")

    if permalink:
        print(f"\nTRIAL REEL: {permalink}")


if __name__ == "__main__":
    main()
