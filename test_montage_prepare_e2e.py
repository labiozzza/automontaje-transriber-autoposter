"""
E2E тест подготовки к публикации (реальный FFmpeg, без отправки в соцсети).

Проверяет, что монтаж/подготовка создают обычную и отражённую версии и
платформенные артефакты, а публикация только отправляет готовые файлы.

ВАЖНО: этот тест НИКОГДА не вызывает POST /publish и не публикует ничего в
реальные соцсети. Guard «нет подготовки → 409» проверяется отдельным мок-тестом
test_publish_endpoint_409_lists_unprepared_targets в test_publish_prepare.py.

Требует запущенный сервер на 127.0.0.1:8000 и синтетическое видео
/tmp/montage_test.mp4 + /tmp/montage_segments.srt (см. test_montage_e2e.py).

Запуск:
  python test_montage_prepare_e2e.py
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
    boundary = "----preparee2e" + os.urandom(8).hex()
    chunks = []
    for key, value in fields:
        if isinstance(value, tuple):
            filename, path = value
            chunks.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n".encode()
            )
            chunks.append(Path(path).read_bytes() + b"\r\n")
        else:
            chunks.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
            )
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_form(url, fields):
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def get_json(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def poll(job_id, predicate, timeout=180, interval=3):
    deadline = time.time() + timeout
    data = {}
    while time.time() < deadline:
        data = get_json(f"/api/montage/result/{job_id}")
        if predicate(data):
            return data
        time.sleep(interval)
    return data


def main():
    check("Входное видео существует", Path(VIDEO).exists())
    check("SRT существует", Path(SRT).exists())

    resp = post_multipart(f"{BASE}/api/montage/render", [
        ("video", ("montage_test.mp4", VIDEO)),
        ("srt", ("segments.srt", SRT)),
        ("subtitle_mode", "sentences"),
        ("face_tracking", "false"),
        ("autozoom", "false"),
        ("subtitles", "true"),
        ("bigpickle", "false"),
        ("prepare_publish", "false"),
    ])
    job_id = resp.get("job_id")
    check("Рендер создал job_id", bool(job_id), resp)

    data = poll(job_id, lambda d: d.get("status") in ("done", "error", "published"))
    check("Рендер завершился", data.get("status") == "done", data.get("status"))

    # 1. До подготовки ни одна цель не готова — публикация отклонилась бы 409.
    #    Реального запроса на /publish здесь НЕТ: этот e2e не должен публиковать.
    ready0 = data.get("publish_ready") or {}
    check("До подготовки все цели не готовы",
          bool(ready0) and not any(ready0.values()),
          str(ready0))
    # Сам guard проверяется мок-тестом test_publish_endpoint_409_lists_unprepared_targets.

    # 2. Готовим обычные артефакты (реальный FFmpeg).
    code, _ = post_form(f"{BASE}/api/montage/{job_id}/prepare-publish", {
        "targets_json": json.dumps(["instagram_feed", "youtube"]),
    })
    check("Старт подготовки обычных версий", code == 200, f"code={code}")
    data = poll(job_id, lambda d: d.get("prepare_status") in ("done", "error"), timeout=300)
    check("Подготовка обычных версий завершена", data.get("prepare_status") == "done",
          str(data.get("prepare_error")))
    ready = data.get("publish_ready") or {}
    check("instagram_feed готов", ready.get("instagram_feed"), str(ready))
    check("youtube готов", ready.get("youtube"), str(ready))

    # 3. Готовим отражённую версию, затем её артефакты.
    code, _ = post_form(f"{BASE}/api/montage/{job_id}/prepare-mirror", {})
    check("Старт подготовки отражённой версии", code == 200, f"code={code}")
    data = poll(job_id, lambda d: (d.get("mirror_variant") or {}).get("status") in ("done", "error"),
                timeout=180)
    mv = data.get("mirror_variant") or {}
    check("Отражённая версия готова", mv.get("status") == "done", str(mv.get("error")))

    code, _ = post_form(f"{BASE}/api/montage/{job_id}/prepare-publish", {
        "targets_json": json.dumps(["instagram_feed_mirrored", "youtube_mirrored"]),
    })
    check("Старт подготовки отражённых версий", code == 200, f"code={code}")
    data = poll(job_id, lambda d: d.get("prepare_status") in ("done", "error"), timeout=300)
    check("Подготовка отражённых версий завершена", data.get("prepare_status") == "done",
          str(data.get("prepare_error")))
    ready = data.get("publish_ready") or {}
    check("instagram_feed_mirrored готов", ready.get("instagram_feed_mirrored"), str(ready))
    check("youtube_mirrored готов", ready.get("youtube_mirrored"), str(ready))

    # 4. Подготовленные файлы существуют и отличаются от исходника.
    work = Path(data.get("result", {}).get("result_video") or "")
    workdir = work.parent / "publish_prep"
    files = sorted(p.name for p in workdir.glob("*.mp4")) if workdir.exists() else []
    check("Файлы подготовки созданы", len(files) >= 3, f"files={files}")
    ig_main = next((p for p in workdir.glob("*instagram_main*") if p.is_file()), None) if workdir.exists() else None
    check("Обычная Instagram-версия создана", ig_main is not None, f"files={files}")
    if ig_main is not None and work.exists():
        check("Обычная Instagram-версия отличается по размеру от исходника",
              ig_main.stat().st_size != work.stat().st_size,
              f"main={work.stat().st_size} prepared={ig_main.stat().st_size}")

    # 5. Автоподготовка: рендер с prepare_publish=true сам готовит артефакты.
    resp2 = post_multipart(f"{BASE}/api/montage/render", [
        ("video", ("montage_test.mp4", VIDEO)),
        ("srt", ("segments.srt", SRT)),
        ("subtitle_mode", "sentences"),
        ("face_tracking", "false"),
        ("autozoom", "false"),
        ("subtitles", "true"),
        ("bigpickle", "false"),
    ])
    job2 = resp2.get("job_id")
    check("Второй рендер создал job_id", bool(job2), resp2)
    data2 = poll(job2, lambda d: d.get("status") in ("done", "error"), timeout=300)
    check("Второй рендер завершился", data2.get("status") == "done", data2.get("status"))
    # автоподготовка запускается сразу после рендера
    data2 = poll(job2, lambda d: d.get("prepare_status") in ("done", "error"), timeout=300)
    check("Автоподготовка завершилась", data2.get("prepare_status") == "done",
          str(data2.get("prepare_error")))
    ready2 = data2.get("publish_ready") or {}
    check("Автоподготовка: instagram_feed готов", ready2.get("instagram_feed"), str(ready2))

    print("\n==================================================")
    print(f"ИТОГО: {PASS} PASS, {FAIL} FAIL")
    if FAILURES:
        print("FAILURES:")
        for f in FAILURES:
            print(" -", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    _code = main()
    sys.stdout.flush()
    os._exit(_code)
