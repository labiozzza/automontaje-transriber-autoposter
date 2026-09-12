from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CURL = shutil.which("curl") or "curl"



THREADS_USER_ID = os.environ.get("THREADS_USER_ID", "").strip()
GITHUB_OWNER = os.environ.get("GITHUB_OWNER", "volynecsvatoslav-png").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "instagram-media").strip()
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main").strip()
THREADS_HOST = "graph.threads.net"
RAW_HOST = os.environ.get("RAW_HOST", "raw.githubusercontent.com").strip()
ACCESS_TOKEN = os.environ.get("THREADS_ACCESS_TOKEN", "").strip()
POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 180


def die(message: str) -> None:
    print(f"\nОШИБКА: {message}", file=sys.stderr)
    raise SystemExit(1)


def run_process(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def resolve_host_via_doh(host: str) -> str:
    result = run_process(
        [CURL, "-sS", f"https://dns.google/resolve?name={host}&type=A"],
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


def threads_request(threads_ip: str, method: str, path: str, params: dict[str, str]) -> dict:
    url = f"https://{THREADS_HOST}/{path.lstrip('/')}"
    args = [
        CURL,
        "-sS",
        "--resolve",
        f"{THREADS_HOST}:443:{threads_ip}",
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
            "Threads API вернул не JSON.\n\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if result.returncode != 0 or "error" in data:
        die(
            "Ошибка Threads API:\n"
            + json.dumps(data.get("error", data), ensure_ascii=False, indent=2)
        )
    return data


def build_raw_urls(prefix: str, count: int) -> list[str]:
    return [
        (
            f"https://{RAW_HOST}/"
            f"{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"
            f"{prefix}_{i:02d}.jpg"
        )
        for i in range(1, count + 1)
    ]


def check_raw_jpeg(url: str, raw_ip: str) -> None:
    with tempfile.TemporaryDirectory(prefix="threads_raw_check_") as tmp:
        body_path = Path(tmp) / "image.bin"
        result = run_process(
            [
                CURL,
                "-sS",
                "-L",
                "--max-time",
                "30",
                "--resolve",
                f"{RAW_HOST}:443:{raw_ip}",
                "-o",
                str(body_path),
                "-w",
                "%{http_code}|%{content_type}|%{size_download}",
                url,
            ],
            timeout=45,
        )
        parts = result.stdout.strip().split("|")
        if len(parts) != 3:
            die(
                f"Не удалось проверить GitHub Raw:\n{url}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        http_code, content_type, size_text = parts
        if result.returncode != 0 or http_code != "200":
            die(
                f"GitHub Raw не отдал файл:\n{url}\n"
                f"HTTP={http_code}, type={content_type}, size={size_text}"
            )
        if not body_path.exists() or body_path.stat().st_size < 2:
            die(f"Пустой файл:\n{url}")
        with body_path.open("rb") as f:
            magic = f.read(2)
        if magic != b"\xff\xd8":
            die(
                "По URL лежит не настоящий JPEG:\n"
                f"{url}\nПервые байты: {magic!r}"
            )
        print(
            f"      HTTP 200; {content_type or '?'}; "
            f"{body_path.stat().st_size} bytes"
        )


def wait_container(threads_ip: str, container_id: str, *, timeout_seconds: int = POLL_TIMEOUT_SECONDS) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        data = threads_request(
            threads_ip,
            "GET",
            container_id,
            {
                "fields": "id,status,error_message",
                "access_token": ACCESS_TOKEN,
            },
        )
        status = data.get("status")
        error_message = data.get("error_message")
        print(f"      status={status}" + (f"; {error_message}" if error_message else ""))
        if status in {"FINISHED", "PUBLISHED"}:
            return
        if status in {"ERROR", "EXPIRED"}:
            die(
                "Threads container завершился ошибкой:\n"
                + json.dumps(data, ensure_ascii=False, indent=2)
            )
        time.sleep(POLL_INTERVAL_SECONDS)
    die(f"Container {container_id} не стал FINISHED за {timeout_seconds} секунд.")


def create_image_child(threads_ip: str, image_url: str, alt_text: str) -> str:
    data = threads_request(
        threads_ip,
        "POST",
        "me/threads",
        {
            "media_type": "IMAGE",
            "image_url": image_url,
            "is_carousel_item": "true",
            "alt_text": alt_text,
            "access_token": ACCESS_TOKEN,
        },
    )
    container_id = str(data["id"])
    print(f"      child_id={container_id}")
    return container_id


def create_carousel(threads_ip: str, children: list[str], text: str, topic_tag: str | None) -> str:
    params = {
        "media_type": "CAROUSEL",
        "children": ",".join(children),
        "access_token": ACCESS_TOKEN,
    }
    if text:
        params["text"] = text
    if topic_tag:
        params["topic_tag"] = topic_tag
    data = threads_request(threads_ip, "POST", "me/threads", params)
    container_id = str(data["id"])
    print(f"   CAROUSEL_ID={container_id}")
    return container_id


def publish_carousel(threads_ip: str, carousel_id: str) -> str:
    data = threads_request(
        threads_ip,
        "POST",
        "me/threads_publish",
        {
            "creation_id": carousel_id,
            "access_token": ACCESS_TOKEN,
        },
    )
    thread_id = str(data["id"])
    print(f"   THREAD_ID={thread_id}")
    return thread_id


def get_thread_info(threads_ip: str, thread_id: str) -> dict:
    return threads_request(
        threads_ip,
        "GET",
        thread_id,
        {
            "fields": "id,media_product_type,media_type,permalink,username,text,timestamp",
            "access_token": ACCESS_TOKEN,
        },
    )


def load_text(prefix: str, text_arg: str | None, text_file: str | None) -> str:
    if text_arg is not None:
        return text_arg
    if text_file:
        path = Path(text_file)
        if not path.is_file():
            die(f"TXT-файл не найден:\n{path}")
        return path.read_text(encoding="utf-8").strip()
    default_path = Path(__file__).resolve().parent / "threads" / f"{prefix}.txt"
    if default_path.is_file():
        print(f"   Текст автоматически из: {default_path}")
        return default_path.read_text(encoding="utf-8").strip()
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Опубликовать карусель изображений в Threads из GitHub Raw."
    )
    parser.add_argument("prefix", help="Префикс файлов. Например 103 для 103_01.jpg ...")
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Количество изображений. Threads поддерживает 2–20. Default: 10.",
    )
    parser.add_argument("--text", help="Текст поста прямо в командной строке.")
    parser.add_argument("--text-file", help="UTF-8 TXT-файл с текстом поста.")
    parser.add_argument(
        "--topic-tag",
        help="Необязательный topic tag Threads. Передавай без #, например Финансы.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("РЕЖИМ: THREADS CAROUSEL")
    if not ACCESS_TOKEN:
        die(
            "Не задан THREADS_ACCESS_TOKEN.\n\n"
            "PowerShell:\n"
            '$env:THREADS_ACCESS_TOKEN="ТВОЙ_THREADS_TOKEN"'
        )
    if not (2 <= args.count <= 20):
        die("--count должен быть от 2 до 20.")

    print("\n1. Получаем DNS через DoH...")
    threads_ip = resolve_host_via_doh(THREADS_HOST)
    raw_ip = resolve_host_via_doh(RAW_HOST)
    print(f"   {THREADS_HOST} -> {threads_ip}")
    print(f"   {RAW_HOST} -> {raw_ip}")

    urls = build_raw_urls(args.prefix, args.count)

    print("\n2. Проверяем файлы на GitHub Raw...")
    for index, url in enumerate(urls, start=1):
        print(f"   {index:02d}/{args.count:02d} {url}")
        check_raw_jpeg(url, raw_ip)

    text = load_text(args.prefix, args.text, args.text_file)
    if text:
        print(f"\n3. Текст найден: {len(text)} символов.")
    else:
        print("\n3. Текст не задан — публикуем карусель без текста.")

    print("\n4. Создаём дочерние IMAGE containers...")
    child_ids: list[str] = []
    for index, url in enumerate(urls, start=1):
        print(f"   Изображение {index:02d}/{args.count:02d}")
        child_id = create_image_child(
            threads_ip,
            url,
            alt_text=f"Изображение {index} из {args.count}",
        )
        wait_container(threads_ip, child_id)
        child_ids.append(child_id)

    print("\n5. Создаём CAROUSEL container...")
    carousel_id = create_carousel(threads_ip, child_ids, text, args.topic_tag)

    print("\n6. Ждём готовности карусели...")
    wait_container(threads_ip, carousel_id, timeout_seconds=180)

    print("\n7. ПУБЛИКУЕМ...")
    thread_id = publish_carousel(threads_ip, carousel_id)

    print("\n8. Получаем permalink...")
    info = get_thread_info(threads_ip, thread_id)

    print("\nГОТОВО.")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    permalink = info.get("permalink")
    if permalink:
        print(f"\nTHREADS POST: {permalink}")


if __name__ == "__main__":
    main()
