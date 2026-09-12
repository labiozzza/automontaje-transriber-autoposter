"""
E2E тест автомонтажа через REST API (без Playwright).

Требует запущенный сервер на 127.0.0.1:8000 и синтетическое видео /tmp/montage_test.mp4.
Запуск:
  /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 test_montage_e2e.py [skip_render]
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
VIDEO = "/tmp/montage_test.mp4"
SRT = "/tmp/montage_segments.srt"

PASS = 0
FAIL = 0
FAILURES = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail and not condition else ""))


def post_multipart(url, fields):
    boundary = "----montagee2e" + os.urandom(8).hex()
    chunks = []
    for key, value in fields:
        if isinstance(value, tuple):  # file: (filename, path)
            filename, path = value
            data = Path(path).read_bytes()
            chunks.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n".encode()
            )
            chunks.append(data + b"\r\n")
        else:
            chunks.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
            )
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    skip_render = len(sys.argv) > 1 and sys.argv[1] == "skip_render"

    check("Входное видео существует", Path(VIDEO).exists())
    check("SRT существует", Path(SRT).exists())

    # ---- Единичный рендер через transcript_job_id не проверяем (нужна транскрибация).
    if not skip_render:
        resp = post_multipart(f"{BASE}/api/montage/render", [
            ("video", ("montage_test.mp4", VIDEO)),
            ("srt", ("segments.srt", SRT)),
            ("subtitle_mode", "sentences"),
            ("face_tracking", "true"),
            ("autozoom", "true"),
            ("subtitles", "true"),
            ("bigpickle", "false"),
            ("edge_mode", "reflect"),
        ])
        job_id = resp.get("job_id")
        check("Рендер создал job_id", bool(job_id), resp)

        status = "processing"
        last = ""
        for _ in range(60):
            data = get_json(f"/api/montage/result/{job_id}")
            status = data["status"]
            last = str(data.get("detail") or "")
            if status in ("done", "error", "published"):
                break
            time.sleep(3)
        check("Рендер завершился успешно", status == "done", f"status={status} detail={last}")

        if status == "done":
            result = get_json(f"/api/montage/result/{job_id}")["result"]
            video_path = result["result_video"]
            check("result_video существует на диске", Path(video_path).exists())
            check("duration > 0", (result.get("duration") or 0) > 0)
            req = urllib.request.Request(f"{BASE}/api/montage/download/{job_id}")
            with urllib.request.urlopen(req, timeout=120) as resp_file:
                downloaded = resp_file.read()
                check("Скачивание работает", len(downloaded) > 100_000, f"bytes={len(downloaded)}")

        # request-id в пустой список целей → 400
        try:
            post_multipart(f"{BASE}/api/montage/{job_id}/publish", [("targets_json", "[]")])
            check("Пустой список целей отклонён", False, "expected HTTP error")
        except urllib.error.HTTPError as e:
            check("Пустой список целей отклонён", e.code == 400, f"code={e.code}")

        # неизвестный job → 404
        try:
            get_json("/api/montage/result/zzzz")
            check("Неизвестный job → 404", False)
        except urllib.error.HTTPError as e:
            check("Неизвестный job → 404", e.code == 404, f"code={e.code}")

    # ---- Статус / анимации
    status = get_json("/api/montage/status")
    check("Статус отдаёт publishers", isinstance(status.get("publishers"), dict))
    anims = get_json("/api/montage/animations")
    check("Список анимаций — массив", isinstance(anims.get("animations"), list))

    print("\n==================================================")
    print(f"ИТОГО: {PASS} PASS, {FAIL} FAIL")
    if FAILURES:
        for f in FAILURES:
            print("  FAIL:", f)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()