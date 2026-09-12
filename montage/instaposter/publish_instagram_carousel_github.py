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



IG_USER_ID = os.environ.get("IG_USER_ID", "").strip()
GITHUB_OWNER = os.environ.get("GITHUB_OWNER", "volynecsvatoslav-png").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "instagram-media").strip()
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main").strip()
GRAPH_HOST = os.environ.get("GRAPH_HOST", "graph.instagram.com").strip()
RAW_HOST = os.environ.get("RAW_HOST", "raw.githubusercontent.com").strip()
ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "").strip()
API_VERSION = os.environ.get("INSTAGRAM_API_VERSION", "v26.0").strip()


def die(msg: str) -> None:
    print(f"\nОШИБКА: {msg}", file=sys.stderr)
    raise SystemExit(1)


def run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def resolve_doh(host: str) -> str:
    r = run([CURL, "-sS", f"https://dns.google/resolve?name={host}&type=A"], 30)
    if r.returncode != 0:
        die(r.stderr)
    data = json.loads(r.stdout)
    for item in data.get("Answer", []):
        if item.get("type") == 1:
            return str(item["data"])
    die(f"Нет A-record для {host}")
    raise AssertionError


def request(ip: str, method: str, path: str, params: dict[str, str]) -> dict:
    url = f"https://{GRAPH_HOST}/{API_VERSION}/{path.lstrip('/')}"
    args = [CURL, "-sS", "--resolve", f"{GRAPH_HOST}:443:{ip}"]
    args += ["-G", url] if method == "GET" else ["-X", "POST", url]
    for k, v in params.items():
        args += ["--data-urlencode", f"{k}={v}"]
    r = run(args, 180)
    try:
        data = json.loads(r.stdout.strip() or "{}")
    except json.JSONDecodeError:
        die(f"Meta вернула не JSON:\n{r.stdout}\n{r.stderr}")
    if r.returncode != 0 or "error" in data:
        die(json.dumps(data.get("error", data), ensure_ascii=False, indent=2))
    return data


def check_jpeg(url: str, raw_ip: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "x.jpg"
        r = run([
            CURL, "-sS", "-L", "--max-time", "30",
            "--resolve", f"{RAW_HOST}:443:{raw_ip}",
            "-o", str(out), "-w", "%{http_code}|%{content_type}", url
        ], 45)
        parts = r.stdout.strip().split("|")
        if len(parts) != 2 or parts[0] != "200":
            die(f"GitHub Raw недоступен: {url}\n{r.stdout}\n{r.stderr}")
        if not out.exists() or out.read_bytes()[:2] != b"\xff\xd8":
            die(f"По URL не настоящий JPEG: {url}")


def wait_container(graph_ip: str, cid: str) -> None:
    for _ in range(60):
        data = request(graph_ip, "GET", cid, {
            "fields": "status_code,status",
            "access_token": ACCESS_TOKEN,
        })
        code = data.get("status_code")
        print(f"      {code}")
        if code in {"FINISHED", "PUBLISHED"}:
            return
        if code in {"ERROR", "EXPIRED"}:
            die(json.dumps(data, ensure_ascii=False, indent=2))
        time.sleep(3)
    die(f"Container {cid} не готов за 180 секунд")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--text-file", required=True)
    args = parser.parse_args()

    if not ACCESS_TOKEN:
        die('Задай: $env:INSTAGRAM_ACCESS_TOKEN="..."')

    text_path = Path(args.text_file)
    if not text_path.is_file():
        die(f"Нет TXT: {text_path}")
    caption = text_path.read_text(encoding="utf-8").strip()

    if not (2 <= args.count <= 10):
        die("Instagram carousel: count должен быть 2–10")

    graph_ip = resolve_doh(GRAPH_HOST)
    raw_ip = resolve_doh(RAW_HOST)

    urls = [
        f"https://{RAW_HOST}/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{args.prefix}_{i:02d}.jpg"
        for i in range(1, args.count + 1)
    ]

    print("1. Проверяем GitHub Raw...")
    for i, url in enumerate(urls, 1):
        print(f"   {i:02d}/{args.count:02d}")
        check_jpeg(url, raw_ip)

    print("2. Создаём child containers...")
    children: list[str] = []
    for i, url in enumerate(urls, 1):
        print(f"   {i:02d}/{args.count:02d}")
        data = request(graph_ip, "POST", f"{IG_USER_ID}/media", {
            "image_url": url,
            "is_carousel_item": "true",
            "access_token": ACCESS_TOKEN,
        })
        cid = str(data["id"])
        print(f"      {cid}")
        wait_container(graph_ip, cid)
        children.append(cid)

    print("3. Создаём CAROUSEL...")
    parent = request(graph_ip, "POST", f"{IG_USER_ID}/media", {
        "media_type": "CAROUSEL",
        "children": ",".join(children),
        "caption": caption,
        "access_token": ACCESS_TOKEN,
    })
    parent_id = str(parent["id"])
    print(f"   {parent_id}")
    wait_container(graph_ip, parent_id)

    print("4. Публикуем Instagram...")
    published = request(graph_ip, "POST", f"{IG_USER_ID}/media_publish", {
        "creation_id": parent_id,
        "access_token": ACCESS_TOKEN,
    })
    media_id = str(published["id"])

    info = request(graph_ip, "GET", media_id, {
        "fields": "id,media_type,media_product_type,permalink,timestamp",
        "access_token": ACCESS_TOKEN,
    })
    print("\nINSTAGRAM ГОТОВО")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
