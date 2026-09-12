import os, json, uuid, asyncio, subprocess, time, re, tempfile, hashlib, urllib.request, importlib, threading
import shutil
import traceback
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import whisper
from pyannote.audio import Pipeline as DiarizationPipeline
import torch
import soundfile as sf
import numpy as np
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cpu"

app = FastAPI(title="Transcriber API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# In-memory job storage
jobs = {}
job_queues = {}
whisper_models = {}
WHISPER_PROGRESS_LOCK = threading.Lock()
diarization_pipeline = None
PROJECTS = {}
PROJECTS_LOCK = asyncio.Lock()
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".3gp", ".ts"}


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_whisper_model(root: str, name: str, progress_cb=None) -> Optional[str]:
    """Докачиваемое (resume) скачивание модели whisper с прогрессом.
    Возвращает путь к локальному файлу модели либо None."""
    url = whisper._MODELS.get(name)
    if url is None:
        return None
    expected = url.split("/")[-2]
    target = os.path.join(root, name + ".pt")
    part = target + ".part"

    if os.path.isfile(target):
        if _file_sha256(target) == expected:
            return target
        os.replace(target, part)

    start = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "Mozilla/5.0 (whisper-mirror-downloader)"}
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab") as out:
        total = start + int(resp.headers.get("Content-Length") or 0)
        downloaded = start
        last_pct = -1
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = int(downloaded * 100 / total)
            else:
                pct = int(downloaded * 100 / (start or 1))
            if pct != last_pct and progress_cb:
                last_pct = pct
                progress_cb(pct, downloaded, total)
        if progress_cb:
            progress_cb(100, downloaded, total)
    if _file_sha256(part) != expected:
        raise RuntimeError(f"Скачивание модели {name}: контрольная сумма не совпала")
    os.replace(part, target)
    return target


def load_whisper_model(name: str, progress_cb=None, load_cb=None):
    if name not in whisper_models:
        print(f"[transcriber] Checking whisper model '{name}'...")
        root = os.path.expanduser("~/.cache/whisper")
        path = ensure_whisper_model(root, name, progress_cb)
        if load_cb:
            load_cb()
        print(f"[transcriber] Loading whisper model '{name}'...")
        whisper_models[name] = whisper.load_model(path or name, device=DEVICE)
        print(f"[transcriber] Whisper '{name}' loaded on {DEVICE}")
    return whisper_models[name]


def transcribe_with_progress(model, audio_path: str, progress_cb, **kwargs):
    """Expose Whisper's internal audio-frame progress without changing Whisper itself."""
    transcribe_module = importlib.import_module("whisper.transcribe")
    original_tqdm = transcribe_module.tqdm.tqdm

    class ProgressTqdm(original_tqdm):
        def update(self, amount=1):
            result = super().update(amount)
            total = float(self.total or 0)
            if total:
                progress_cb(min(1.0, float(self.n) / total))
            return result

    with WHISPER_PROGRESS_LOCK:
        transcribe_module.tqdm.tqdm = ProgressTqdm
        try:
            return model.transcribe(audio_path, verbose=False, **kwargs)
        finally:
            transcribe_module.tqdm.tqdm = original_tqdm


def load_diarization_pipeline():
    global diarization_pipeline
    if diarization_pipeline is None:
        load_secrets_env()
        print("[transcriber] Loading pyannote/speaker-diarization-3.1...")
        diarization_pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=os.environ.get("HF_TOKEN")
        )
        diarization_pipeline.to(torch.device(DEVICE))
        print(f"[transcriber] Diarization pipeline loaded on {DEVICE}")
    return diarization_pipeline


def convert_to_wav(input_path: str, output_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0


def fmt_srt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_vtt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


async def broadcast_progress(job_id: str, progress: float, stage: str, detail: str = ""):
    job = jobs.get(job_id)
    if job:
        job["progress"] = progress
        job["stage"] = stage
        job["detail"] = detail
        q = job_queues.get(job_id)
        if q:
            try:
                await q.put({
                    "type": "progress",
                    "progress": progress,
                    "stage": stage,
                    "detail": detail,
                })
            except Exception:
                pass


async def process_job(job_id: str, params: dict):
    job = jobs[job_id]
    try:
        job["status"] = "processing"
        wav_path = job["wav_path"]
        original_name = job["original_name"]

        # Stage 1: Conversion
        await broadcast_progress(job_id, 5, "conversion", "Конвертация в WAV...")
        wav_output = str(UPLOAD_DIR / f"{job_id}.wav")
        is_wav = wav_path.lower().endswith(".wav")
        if is_wav:
            wav_output = wav_path
            await broadcast_progress(job_id, 15, "conversion", "Файл уже в формате WAV")
        else:
            success = await asyncio.to_thread(convert_to_wav, wav_path, wav_output)
            if not success:
                job["status"] = "error"
                job["error"] = "Ошибка конвертации файла"
                await broadcast_progress(job_id, 0, "error", "Ошибка конвертации")
                return

        await broadcast_progress(job_id, 15, "conversion", "Конвертация завершена")

        # Stage 2: Transcription
        model_name = params.get("model", "small")
        language = params.get("language", "ru")
        mode = params.get("mode", "sentences")

        loop = asyncio.get_running_loop()

        def model_progress(pct, downloaded, total):
            mb = downloaded / (1 << 20)
            total_mb = total / (1 << 20)
            asyncio.run_coroutine_threadsafe(
                broadcast_progress(
                    job_id, 20 + pct * 0.28, "model",
                    f"Скачивание модели {model_name}: {mb:.0f} / {total_mb:.0f} MB ({pct}%)"
                ),
                loop
            )

        def model_load_start():
            asyncio.run_coroutine_threadsafe(
                broadcast_progress(
                    job_id, 48, "model",
                    f"Загрузка модели {model_name} в память..."
                ),
                loop
            )

        await broadcast_progress(job_id, 20, "model", f"Проверка модели {model_name}...")
        wm = await asyncio.to_thread(load_whisper_model, model_name, model_progress, model_load_start)

        await broadcast_progress(job_id, 50, "transcription", "Транскрибация аудио...")

        transcription_span = 12 if params.get("diarization", False) else 38

        def transcription_progress(ratio: float):
            pct = 50 + ratio * transcription_span
            asyncio.run_coroutine_threadsafe(
                broadcast_progress(job_id, pct, "transcription", f"Транскрибация: {ratio * 100:.1f}%"),
                loop,
            )

        result = await asyncio.to_thread(
            transcribe_with_progress, wm, wav_output, transcription_progress,
            language=language, word_timestamps=(mode == "words")
        )
        segments = result["segments"]

        transcription_done = 63 if params.get("diarization", False) else 89
        await broadcast_progress(job_id, transcription_done, "transcription", f"Транскрибация: {len(segments)} сегментов")

        # Stage 3: Diarization
        do_diarization = params.get("diarization", False)
        num_speakers = params.get("num_speakers", None)
        diarization = None

        if do_diarization:
            await broadcast_progress(job_id, 65, "diarization", "Загрузка модели диаризации...")
            dp = await asyncio.to_thread(load_diarization_pipeline)

            await broadcast_progress(job_id, 70, "diarization", "Диаризация...")
            diarization_kwargs = {}
            if num_speakers:
                diarization_kwargs["num_speakers"] = int(num_speakers)

            diarization = await asyncio.to_thread(dp, wav_output, **diarization_kwargs)
            await broadcast_progress(job_id, 85, "diarization", "Диаризация завершена")

        # Stage 4: Merge
        await broadcast_progress(job_id, 90, "merge", "Объединение результатов...")

        merged = await asyncio.to_thread(merge_results, segments, diarization, mode)
        job["result"] = merged
        job["status"] = "done"
        job["progress"] = 100

        await broadcast_progress(job_id, 100, "done", "Готово!")
        if await register_video_project(job_id):
            await broadcast_progress(job_id, 100, "done",
                                     "Готово! Ролик добавлен в проекты (Автомонтаж).")

    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()
        await broadcast_progress(job_id, 0, "error", str(e))


def merge_results(segments, diarization, mode):
    result_segments = []

    if diarization is not None:
        for seg in segments:
            start, end = seg["start"], seg["end"]

            if mode == "words" and "words" in seg:
                word_groups = []
                current_group = {"words": [], "speaker": None, "start": start, "end": start}

                for word_info in seg["words"]:
                    w_start = word_info["start"]
                    w_end = word_info["end"]
                    w_text = word_info["word"]

                    speaker = find_speaker(w_start, w_end, diarization)

                    if speaker != current_group["speaker"] and current_group["words"]:
                        current_group["end"] = w_start
                        current_group["text"] = " ".join(current_group["words"])
                        word_groups.append(current_group)
                        current_group = {"words": [], "speaker": speaker, "start": w_start, "end": w_start}

                    current_group["words"].append(w_text.strip())
                    current_group["end"] = w_end

                if current_group["words"]:
                    current_group["text"] = " ".join(current_group["words"])
                    word_groups.append(current_group)

                result_segments.extend(word_groups)
            else:
                speaker = find_speaker(start, end, diarization)
                result_segments.append({
                    "start": start,
                    "end": end,
                    "text": seg["text"].strip(),
                    "speaker": speaker,
                })
    else:
        for seg in segments:
            if mode == "words" and "words" in seg:
                for wi in seg["words"]:
                    result_segments.append({
                        "start": wi["start"],
                        "end": wi["end"],
                        "text": wi["word"].strip(),
                        "speaker": None,
                    })
            else:
                result_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    "speaker": None,
                })

    return result_segments


def find_speaker(start, end, diarization):
    best_speaker = "SPEAKER_00"
    best_overlap = 0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap_start = max(start, turn.start)
        overlap_end = min(end, turn.end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


def export_srt(segments, show_speakers=True, show_timecodes=True):
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        if show_timecodes:
            lines.append(f"{fmt_srt_time(seg['start'])} --> {fmt_srt_time(seg['end'])}")
        speaker = f"[{seg['speaker']}] " if show_speakers and seg.get('speaker') else ""
        lines.append(f"{speaker}{seg['text']}")
        lines.append("")
    return "\n".join(lines)


def export_vtt(segments, show_speakers=True, show_timecodes=True):
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        if show_timecodes:
            lines.append(f"{fmt_vtt_time(seg['start'])} --> {fmt_vtt_time(seg['end'])}")
        speaker = f"[{seg['speaker']}] " if show_speakers and seg.get('speaker') else ""
        lines.append(f"{speaker}{seg['text']}")
        lines.append("")
    return "\n".join(lines)


def export_txt(segments, show_speakers=True):
    lines = []
    for seg in segments:
        speaker = f"[{seg['speaker']}] " if show_speakers and seg.get('speaker') else ""
        lines.append(f"{speaker}{seg['text']}")
    return "\n".join(lines)


def export_dialogue(segments, show_timecodes=True):
    lines = []
    current_speaker = None
    for seg in segments:
        sp = seg.get('speaker')
        if sp and sp != current_speaker:
            current_speaker = sp
            lines.append("")
            lines.append(f"═══ {sp} ═══")
        tc = f"[{fmt_srt_time(seg['start'])}] " if show_timecodes else ""
        lines.append(f"{tc}{seg['text']}")
    return "\n".join(lines).strip()


def export_json(segments, show_timecodes=True):
    data = []
    for seg in segments:
        entry = {"text": seg["text"]}
        if seg.get("speaker"):
            entry["speaker"] = seg["speaker"]
        if show_timecodes:
            entry["start"] = seg["start"]
            entry["end"] = seg["end"]
        data.append(entry)
    return json.dumps(data, ensure_ascii=False, indent=2)


def export_tsv(segments, show_speakers=True, show_timecodes=True):
    lines = []
    headers = []
    if show_timecodes:
        headers.extend(["start", "end"])
    if show_speakers:
        headers.append("speaker")
    headers.append("text")
    lines.append("\t".join(headers))

    for seg in segments:
        row = []
        if show_timecodes:
            row.extend([str(seg['start']), str(seg['end'])])
        if show_speakers:
            row.append(seg.get('speaker') or '')
        row.append(seg['text'])
        lines.append("\t".join(row))
    return "\n".join(lines)


# ========== ROUTES ==========

@app.get("/", response_class=HTMLResponse)
async def index():
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/manifest.json")
async def manifest():
    return FileResponse(path=str(BASE_DIR / "static" / "manifest.json"), media_type="application/manifest+json")


@app.get("/service-worker.js")
async def service_worker():
    return FileResponse(path=str(BASE_DIR / "static" / "sw.js"), media_type="application/javascript")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename or "file").suffix or ".wav"
    file_path = UPLOAD_DIR / f"{job_id}{ext}"

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    jobs[job_id] = {
        "id": job_id,
        "original_name": file.filename,
        "wav_path": str(file_path),
        "upload_path": str(file_path),
        "status": "uploaded",
        "progress": 0,
        "stage": "",
        "detail": "",
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
    }

    return {"job_id": job_id, "filename": file.filename}


async def register_video_project(job_id: str) -> Optional[str]:
    job = jobs.get(job_id)
    if not job:
        return None
    name = str(job.get("original_name") or "")
    ext = Path(name).suffix.lower()
    src = job.get("upload_path") or job.get("wav_path") or ""
    if ext not in VIDEO_EXTS or not src or not Path(src).exists():
        return None
    if job.get("status") != "done" or not job.get("result"):
        return None
    pid = "p_" + job_id
    async with PROJECTS_LOCK:
        PROJECTS[pid] = {
            "id": pid,
            "name": name,
            "video_path": str(Path(src).resolve()),
            "transcript_job_id": job_id,
            "status": "ready",
            "created_at": job.get("created_at"),
        }
    return pid


# ---------------- Thumbs & Video preview ----------------

def _thumb_path(job_id: str) -> Path:
    return UPLOAD_DIR / f"{job_id}.thumb.jpg"


def _generate_thumb(video_path: Path, out_path: Path) -> bool:
    try:
        if not video_path.exists() or out_path.exists():
            return out_path.exists()
        subprocess.run(
            [shutil.which("ffmpeg") or "ffmpeg", "-y", "-i", str(video_path),
             "-vframes", "1", "-vf", "scale=405:-1", "-q:v", "3",
             str(out_path)],
            capture_output=True, timeout=30,
        )
        return out_path.exists() and out_path.stat().st_size > 100
    except Exception:
        return False


@app.get("/api/thumb/{job_id}")
async def serve_thumb(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    video_src = job.get("upload_path") or job.get("video_path") or ""
    if not video_src or not Path(video_src).exists():
        raise HTTPException(404, "Video not available")
    thumb = _thumb_path(job_id)
    if not thumb.exists():
        ok = await asyncio.to_thread(_generate_thumb, Path(video_src), thumb)
        if not ok:
            raise HTTPException(500, "Thumb generation failed")
    return FileResponse(str(thumb), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})


@app.get("/api/video/{job_id}")
async def serve_video(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    video_src = job.get("upload_path") or job.get("video_path") or ""
    video = Path(video_src)
    if not video.exists():
        raise HTTPException(404, "Video not available")
    media = {
        ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
        ".webm": "video/webm", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
    }.get(video.suffix.lower(), "video/mp4")
    return FileResponse(str(video), media_type=media, filename=video.name,
                        headers={"Accept-Ranges": "bytes"})


@app.post("/api/projects/link")
async def link_transcript_to_project(
    transcript_job_id: str = Form(...),
):
    job = jobs.get(transcript_job_id)
    if not job:
        raise HTTPException(404, "Задание транскрибации не найдено")
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(400, "Транскрибация ещё не завершена")
    async with PROJECTS_LOCK:
        existing = next(
            (p for p in PROJECTS.values()
             if p.get("transcript_job_id") == transcript_job_id), None)
    if existing:
        return {"project_id": existing["id"], "status": "linked"}
    pid = await register_video_project(transcript_job_id)
    if not pid:
        raise HTTPException(400, "Видео недоступно — проект не создан")
    return {"project_id": pid, "status": "created"}


@app.get("/api/projects")
async def list_projects():
    projects = []
    for pid, pr in PROJECTS.items():
        entry = dict(pr)
        trj = jobs.get(pr.get("transcript_job_id")) if pr.get("transcript_job_id") else None
        entry["transcript_status"] = trj.get("status") if trj else "missing"
        result = trj.get("result") if trj else None
        entry["segments"] = len(result) if isinstance(result, list) else 0
        entry["video_ok"] = bool(pr.get("video_path")) and Path(pr["video_path"]).exists()
        projects.append(entry)
    projects.sort(key=lambda p: p.get("created_at") or "", reverse=True)
    return {"projects": projects}


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    async with PROJECTS_LOCK:
        if project_id in PROJECTS:
            del PROJECTS[project_id]
    return {"ok": True}


@app.post("/api/transcribe/{job_id}")
async def start_transcription(
    job_id: str,
    model: str = Form("small"),
    language: str = Form("ru"),
    mode: str = Form("sentences"),
    diarization: bool = Form(False),
    num_speakers: Optional[int] = Form(None),
):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    params = {
        "model": model,
        "language": language,
        "mode": mode,
        "diarization": diarization,
        "num_speakers": num_speakers,
    }
    jobs[job_id]["params"] = params

    asyncio.create_task(process_job(job_id, params))
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "detail": job["detail"],
        "error": job.get("error"),
    }


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Job not completed")
    return {"segments": job["result"], "original_name": job["original_name"]}


@app.get("/api/export/{job_id}/{format_name}")
async def export_file(
    job_id: str,
    format_name: str,
    speakers: bool = True,
    timecodes: bool = True,
):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Job not completed")

    segments = job["result"]
    original = Path(job["original_name"]).stem

    exporters = {
        "srt": lambda: export_srt(segments, speakers, timecodes),
        "vtt": lambda: export_vtt(segments, speakers, timecodes),
        "txt": lambda: export_txt(segments, speakers),
        "json": lambda: export_json(segments, timecodes),
        "tsv": lambda: export_tsv(segments, speakers, timecodes),
        "dialogue": lambda: export_dialogue(segments, timecodes),
    }

    if format_name not in exporters:
        raise HTTPException(400, f"Unknown format: {format_name}")

    content = exporters[format_name]()
    ext = format_name if format_name != "dialogue" else "txt"
    media_type = "text/plain"
    if format_name == "json":
        media_type = "application/json"

    tmp_path = OUTPUT_DIR / f"{job_id}.{ext}"
    tmp_path.write_text(content, encoding="utf-8")

    return FileResponse(
        path=str(tmp_path),
        filename=f"{original}.{ext}",
        media_type=media_type,
    )


# ========== MONTAGE (Автомонтаж) ==========

MONTAGE_WORK = BASE_DIR / "montage_work"
MONTAGE_WORK.mkdir(exist_ok=True)

from montage import engine as montage_engine
from montage.secrets_env import load_secrets_env

app.mount("/animation-assets", StaticFiles(directory=str(montage_engine.ANIMATIONS_DIR)), name="animation-assets")
app.mount("/font-assets", StaticFiles(directory=str(montage_engine.FONTS_DIR)), name="font-assets")


def _threaded_progress(loop, job_id):
    def progress(pct, stage, detail=""):
        asyncio.run_coroutine_threadsafe(
            broadcast_progress(job_id, pct, stage, detail), loop
        )
    return progress


async def _montage_announce(job_id: str, text: str):
    q = job_queues.get(job_id)
    if q:
        try:
            await q.put({"type": "log", "message": text})
        except Exception:
            pass


async def process_montage(job_id: str, params: dict):
    job = jobs[job_id]
    loop = asyncio.get_running_loop()
    try:
        job["status"] = "processing"
        progress = _threaded_progress(loop, job_id)

        await _montage_announce(job_id, "Проверка установленных компонентов...")
        status = montage_engine.montage_status()

        video = Path(job["video_path"])
        if not video.exists():
            raise RuntimeError("Видеофайл не найден")

        srt_text: str = job["srt_text"] or ""
        transcript_job = job.get("transcript_job_id")
        if not srt_text and transcript_job and transcript_job in jobs:
            source = jobs[transcript_job]
            if source.get("status") == "done" and source.get("result"):
                srt_text = montage_engine.build_words_srt(source["result"])

        if not srt_text.strip():
            raise RuntimeError("Нет текста: загрузите SRT или укажите готовую транскрибацию")

        if params.get("face_tracking", True) and not status["face_model"]:
            await broadcast_progress(job_id, 3, "model", "Загрузка модели face_landmarker.task...")
            download_face_model(job_id, progress)
            status = montage_engine.montage_status()

        workdir = MONTAGE_WORK / job_id
        workdir.mkdir(parents=True, exist_ok=True)

        await _montage_announce(job_id, "Запуск рендера...")
        meta = await asyncio.to_thread(
            montage_engine.render_montage,
            job_id=job_id,
            workdir=workdir,
            source_video=video,
            srt_text=srt_text,
            options=params,
            progress_cb=progress,
            log_cb=lambda message: asyncio.run_coroutine_threadsafe(
                _montage_announce(job_id, message), loop
            ),
        )

        job["status"] = "done"
        job["progress"] = 100
        job["stage"] = "done"
        job["detail"] = "Готово"
        job["result"] = meta
        await broadcast_progress(job_id, 100, "done", "Рендер завершён")
        await _montage_announce(job_id, f"Готово: {Path(meta['result_video']).name} ({(os.path.getsize(meta['result_video']) >> 20)} МБ)")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[montage] job {job_id} failed: {traceback.format_exc()}", file=sys.stderr)
        await broadcast_progress(job_id, 0, "error", str(e))


async def process_publish(job_id: str, targets: list):
    job = jobs[job_id]
    loop = asyncio.get_running_loop()
    try:
        job["status"] = "publishing"
        progress = _threaded_progress(loop, job_id)
        workdir = MONTAGE_WORK / job_id
        source_video = Path(job["result"]["result_video"])

        await _montage_announce(job_id, "Публикация...")
        result = await asyncio.to_thread(
            montage_engine.publish_job,
            job_id=job_id,
            workdir=workdir,
            source_video=source_video,
            targets=targets,
            progress_cb=progress,
            log_cb=lambda message: asyncio.run_coroutine_threadsafe(
                _montage_announce(job_id, message), loop
            ),
        )
        job["status"] = "done"
        job["progress"] = 100
        job["stage"] = "published"
        job["detail"] = "Опубликовано"
        job["publish"] = result["results"]
        await broadcast_progress(job_id, 100, "published", "Публикация завершена")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        await broadcast_progress(job_id, 0, "error", str(e))


def download_face_model(job_id: str, progress):
    model = montage_engine.MODELS_DIR / "face_landmarker.task"
    if model.exists():
        return
    montage_engine.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    part = str(model) + ".part"
    start = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if start:
        req.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab") as out:
        total = start + int(resp.headers.get("Content-Length") or 0)
        downloaded = start
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = int(downloaded * 100 / total)
                progress(pct * 0.03, "model", f"Скачивание модели: {pct}%")
    os.replace(part, str(model))


@app.get("/api/montage/status")
async def montage_status():
    load_secrets_env()
    status = montage_engine.montage_status()
    entries = [
        {"id": j["id"], "name": j["original_name"], "status": j["status"]}
        for j in jobs.values() if j.get("kind") == "montage"
    ]
    return {**status, "jobs": entries}


@app.post("/api/montage/fonts/download")
async def download_comfortaa_font():
    target = montage_engine.FONTS_DIR / "Comfortaa.ttf"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return {"font": {"id": target.name, "name": "Comfortaa", "path": str(target)}}
    try:
        await asyncio.to_thread(
            urllib.request.urlretrieve,
            "https://github.com/google/fonts/raw/main/ofl/comfortaa/Comfortaa%5Bwght%5D.ttf",
            target,
        )
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"Не удалось скачать Comfortaa: {exc}")
    return {"font": {"id": target.name, "name": "Comfortaa", "path": str(target)}}


@app.post("/api/montage/render")
async def start_montage(
    video: Optional[UploadFile] = File(None),
    srt: Optional[UploadFile] = File(None),
    transcript_job_id: Optional[str] = Form(None),
    project_id: Optional[str] = Form(None),
    subtitle_mode: str = Form("words"),
    subtitle_position: str = Form("custom"),
    subtitle_x: Optional[float] = Form(None),
    subtitle_y: Optional[float] = Form(None),
    subtitle_scale: Optional[float] = Form(None),
    subtitle_box: Optional[bool] = Form(False),
    subtitle_font: Optional[str] = Form(None),
    face_tracking: bool = Form(False),
    autozoom: bool = Form(False),
    subtitles: bool = Form(True),
    bigpickle: bool = Form(True),
    edge_mode: str = Form("scale"),
    animation_id: str = Form("Running_Cat_f1718808"),
    animation_start: float = Form(0.0),
    animation_offset_x: int = Form(0),
    animation_item_size: int = Form(72),
    animation_bar: bool = Form(True),
    zoom_timeline_json: Optional[str] = Form(None),
):
    job_id = str(uuid.uuid4())[:8]
    work_dir = MONTAGE_WORK / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path: Optional[Path] = None
    srt_text = ""
    src_job_id: Optional[str] = None

    if project_id:
        async with PROJECTS_LOCK:
            project = PROJECTS.get(project_id)
        if not project:
            raise HTTPException(404, "Проект не найден")
        if not project.get("video_path") or not Path(project["video_path"]).exists():
            raise HTTPException(400, "Видео проекта недоступно")
        video_path = work_dir / f"project_video{Path(project['video_path']).suffix or '.mp4'}"
        shutil.copy2(project["video_path"], video_path)
        src_job_id = project.get("transcript_job_id")
        if src_job_id and jobs.get(src_job_id) and jobs[src_job_id].get("status") == "done" \
                and isinstance(jobs[src_job_id].get("result"), list):
            srt_text = montage_engine.build_words_srt(jobs[src_job_id]["result"])
    else:
        if video is None:
            raise HTTPException(400, "Загрузите видео или выберите проект")
        ext = Path(video.filename or "video.mp4").suffix or ".mp4"
        video_path = work_dir / f"source{ext}"
        content = await video.read()
        video_path.write_bytes(content)

    if video is not None and video.filename:
        original_name = video.filename
    elif project_id:
        original_name = PROJECTS.get(project_id, {}).get("name", "Проект")
    else:
        original_name = "video.mp4"

    if srt is not None and srt.filename:
        srt_text = (await srt.read()).decode("utf-8", errors="replace")

    if not srt_text:
        raise HTTPException(400, "Нет текста для субтитров (загрузите SRT или выберите проект с транскрибацией)")

    zoom_override = None
    if zoom_timeline_json and zoom_timeline_json.strip():
        try:
            zoom_override = json.loads(zoom_timeline_json)
        except json.JSONDecodeError:
            raise HTTPException(400, "zoom_timeline_json должен быть валидным JSON")

    jobs[job_id] = {
        "id": job_id,
        "original_name": original_name,
        "video_path": str(video_path),
        "srt_text": srt_text,
        "transcript_job_id": src_job_id,
        "project_id": project_id,
        "kind": "montage",
        "status": "uploaded",
        "progress": 0,
        "stage": "",
        "detail": "",
        "result": None,
        "publish": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
    }
    params = {
        "subtitle_mode": subtitle_mode,
        "subtitle_position": subtitle_position,
        "subtitle_x": subtitle_x,
        "subtitle_y": subtitle_y,
        "subtitle_scale": subtitle_scale,
        "subtitle_box": subtitle_box,
        "subtitle_font": subtitle_font,
        "face_tracking": face_tracking,
        "autozoom": autozoom,
        "subtitles": subtitles,
        "bigpickle": bigpickle,
        "edge_mode": edge_mode,
        "animation": {
            "enabled": bool(animation_id),
            "id": animation_id,
            "start": animation_start,
            "offset_x": animation_offset_x,
            "item_size": max(20, animation_item_size),
            "bar": animation_bar,
        },
        "zoom_timeline_override": zoom_override,
    }
    jobs[job_id]["params"] = params
    if not project_id and video and video.filename:
        name = video.filename
        ext = Path(name).suffix.lower()
        if ext in VIDEO_EXTS and srt_text:
            async with PROJECTS_LOCK:
                PROJECTS[f"p_{job_id}"] = {
                    "id": f"p_{job_id}",
                    "name": name,
                    "video_path": str(video_path),
                    "transcript_job_id": src_job_id,
                    "status": "ready",
                    "created_at": datetime.now().isoformat(),
                }
    asyncio.create_task(process_montage(job_id, params))
    return {"job_id": job_id, "filename": original_name}


@app.post("/api/montage/{job_id}/publish")
async def publish_montage(
    job_id: str,
    targets_json: str = Form(...),
):
    if job_id not in jobs or jobs[job_id].get("kind") != "montage":
        raise HTTPException(404, "Montage job not found")
    job = jobs[job_id]
    if job["status"] != "done" or not job.get("result"):
        raise HTTPException(400, "Сначала завершите рендер")
    try:
        targets = json.loads(targets_json)
    except json.JSONDecodeError:
        raise HTTPException(400, "targets_json must be valid JSON")
    if not isinstance(targets, list) or not targets:
        raise HTTPException(400, "Список целей пуст")
    asyncio.create_task(process_publish(job_id, targets))
    return {"job_id": job_id, "status": "publishing", "targets": [t.get("kind") for t in targets]}


@app.get("/api/montage/result/{job_id}")
async def montage_result(job_id: str):
    if job_id not in jobs or jobs[job_id].get("kind") != "montage":
        raise HTTPException(404, "Montage job not found")
    job = jobs[job_id]
    return {
        "id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "detail": job["detail"],
        "error": job.get("error"),
        "result": job.get("result"),
        "publish": job.get("publish"),
    }


@app.get("/api/montage/download/{job_id}")
async def montage_download(job_id: str):
    if job_id not in jobs or jobs[job_id].get("kind") != "montage":
        raise HTTPException(404, "Montage job not found")
    result = jobs[job_id].get("result")
    if not result:
        raise HTTPException(400, "Рендер не завершён")
    video = Path(result["result_video"])
    if not video.exists():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(path=str(video), media_type="video/mp4", filename=video.name, content_disposition_type="inline")


@app.get("/api/montage/preview/{job_id}")
async def montage_preview(job_id: str):
    if job_id not in jobs or jobs[job_id].get("kind") != "montage":
        raise HTTPException(404, "Montage job not found")
    result = jobs[job_id].get("result")
    if not result:
        raise HTTPException(400, "Рендер не завершён")
    video = Path(result.get("preview_video") or result["result_video"])
    if not video.exists():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(path=str(video), media_type="video/mp4", filename=video.name, content_disposition_type="inline")


@app.get("/api/montage/animations")
async def montage_animations():
    return {"animations": montage_engine.animation_ids()}


@app.post("/api/montage/animations")
async def upload_animation(zipfile: UploadFile = File(...)):
    import zipfile as _zipfile
    if not zipfile.filename or not zipfile.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Ожидается ZIP-архив")
    tmp = MONTAGE_WORK / f"anim_{uuid.uuid4().hex[:8]}.zip"
    tmp.write_bytes(await zipfile.read())
    target = montage_engine.ANIMATIONS_DIR / (zipfile.filename.rsplit(".", 1)[0])
    target.mkdir(parents=True, exist_ok=True)
    with _zipfile.ZipFile(tmp) as z:
        for name in z.namelist():
            safe = Path(name).name
            if safe:
                z.extract(name, str(target))
    tmp.unlink(missing_ok=True)
    return {"animations": montage_engine.animation_ids()}


@app.post("/api/montage/animations/create")
async def create_animation(
    name: str = Form(...),
    fps: float = Form(10.0),
    frames: list[UploadFile] = File(...),
):
    from PIL import Image

    clean_name = re.sub(r"[^\w\- ]+", "", name, flags=re.UNICODE).strip() or "Animation"
    slug = re.sub(r"\s+", "_", clean_name)
    animation_id = f"{slug}_{uuid.uuid4().hex[:8]}"
    target = montage_engine.ANIMATIONS_DIR / animation_id
    target.mkdir(parents=True, exist_ok=False)
    manifest_frames = []
    try:
        for index, upload in enumerate(frames):
            raw = await upload.read()
            if not raw:
                continue
            source = target / f"upload_{index:04d}"
            source.write_bytes(raw)
            output = target / f"frame_{len(manifest_frames):04d}.png"
            try:
                with Image.open(source) as image:
                    image.convert("RGBA").save(output, "PNG")
            finally:
                source.unlink(missing_ok=True)
            manifest_frames.append({"file": output.name})
        if not manifest_frames:
            raise HTTPException(400, "Добавьте хотя бы один корректный кадр")
        manifest = {
            "id": animation_id,
            "name": clean_name,
            "fps": max(1.0, min(60.0, float(fps))),
            "frames": manifest_frames,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return {"animation": {"id": animation_id, "name": clean_name}}


@app.get("/api/jobs")
async def list_jobs():
    return [{"id": j["id"], "name": j["original_name"], "status": j["status"]} for j in jobs.values()]


@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    q = asyncio.Queue()
    job_queues[job_id] = q
    try:
        job = jobs.get(job_id)
        if job:
            await websocket.send_json({
                "type": "progress",
                "progress": job["progress"],
                "stage": job["stage"],
                "detail": job["detail"],
            })
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=10)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        job_queues.pop(job_id, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
