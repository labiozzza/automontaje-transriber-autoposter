"""
E2E тест единого автоматического процесса монтажа (реальный FFmpeg, без отправки в соцсети).

Проверяет, что один запуск монтажа с включённым отражением сам создаёт обычную и
отражённую версии и все платформенные артефакты, без ручных кнопок подготовки.
Публикация (сетевые операции) здесь не запускается вообще.

ВАЖНО: этот тест НИКОГДА не вызывает POST /publish и не публикует ничего в
реальные соцсети. Авто-подготовка при публикации и guard-поведение проверяются
отдельными мок-тестами (test_publish_prepare.py).

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

    # 1. Монтаж без подготовки: платформенные артефакты не создаются, ни одна
    #    цель публикации и отражённая версия не готовы. Запроса на /publish НЕТ.
    ready0 = data.get("publish_ready") or {}
    check("Без подготовки все цели не готовы",
          bool(ready0) and not any(ready0.values()),
          str(ready0))
    check("Без подготовки отражённая версия не готова",
          not (data.get("variants") or {}).get("mirrored_ready"),
          str(data.get("variants")))

    # 2. Единый автоматический процесс: один запуск монтажа с prepare_publish и
    #    prepare_mirrored сам готовит обычную + отражённую версии и все
    #    платформенные артефакты. Ручных вызовов подготовки больше нет.
    resp2 = post_multipart(f"{BASE}/api/montage/render", [
        ("video", ("montage_test.mp4", VIDEO)),
        ("srt", ("segments.srt", SRT)),
        ("subtitle_mode", "sentences"),
        ("face_tracking", "false"),
        ("autozoom", "false"),
        ("subtitles", "true"),
        ("bigpickle", "false"),
        ("prepare_publish", "true"),
        ("prepare_mirrored", "true"),
    ])
    job2 = resp2.get("job_id")
    check("Второй рендер создал job_id", bool(job2), resp2)

    data2 = poll(job2, lambda d: d.get("status") in ("done", "error"), timeout=300)
    check("Единый процесс завершился", data2.get("status") == "done", data2.get("status"))
    check("Фаза ready (без предупреждений)", data2.get("phase") == "ready",
          f"phase={data2.get('phase')} warnings={data2.get('warnings')}")
    check("Предупреждений нет", not (data2.get("warnings") or []), str(data2.get("warnings")))
    check("Подготовка к публикации завершена", data2.get("prepare_status") == "done",
          str(data2.get("prepare_error")))

    mv = data2.get("mirror_variant") or {}
    check("Отражённая версия готова (авто)",
          mv.get("status") == "done" and (data2.get("variants") or {}).get("mirrored_ready"),
          str(mv.get("error")))

    ready = data2.get("publish_ready") or {}
    check("instagram_feed готов (авто)", ready.get("instagram_feed"), str(ready))
    check("youtube готов (авто)", ready.get("youtube"), str(ready))
    check("instagram_feed_mirrored готов (авто)", ready.get("instagram_feed_mirrored"), str(ready))
    check("youtube_mirrored готов (авто)", ready.get("youtube_mirrored"), str(ready))

    # 3. Подготовленные файлы существуют и отличаются от исходника.
    work = Path(data2.get("result", {}).get("result_video") or "")
    workdir = work.parent / "publish_prep"
    files = sorted(p.name for p in workdir.glob("*.mp4")) if workdir.exists() else []
    check("Файлы подготовки созданы", len(files) >= 3, f"files={files}")
    ig_main = next((p for p in workdir.glob("*instagram_main*") if p.is_file()), None) if workdir.exists() else None
    check("Обычная Instagram-версия создана", ig_main is not None, f"files={files}")
    ig_mirror = next((p for p in workdir.glob("*instagram_mirror*") if p.is_file()), None) if workdir.exists() else None
    check("Отражённая Instagram-версия создана", ig_mirror is not None, f"files={files}")
    if ig_main is not None and work.exists():
        check("Обычная Instagram-версия отличается по размеру от исходника",
              ig_main.stat().st_size != work.stat().st_size,
              f"main={work.stat().st_size} prepared={ig_main.stat().st_size}")

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
