"""Part 2 (spec sections 8-11) mock tests.

Covers, without any real render/publish:

* 9.3/9.5 — the mirrored version is prepared by the montage pipeline and reused
  by publish (never rendered inline), with stale-signature detection;
* 10 — per-platform publish steps are recorded;
* 8.2 — drawing generation exposes 8 explicit stages and never shows 100% on
  error;
* 8/9 — the result serializer exposes the mirror state.

Run:
  python -m unittest test_part2_progress
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

import app as app_module


def make_job(job_id, workdir, **extra):
    source = workdir / "source.mp4"
    source.write_bytes(b"source-video")
    result_video = workdir / "final.mp4"
    preview_video = workdir / "preview.mp4"
    result_video.write_bytes(b"final")
    preview_video.write_bytes(b"preview")
    job = {
        "id": job_id,
        "kind": "montage",
        "status": "done",
        "stage": "done",
        "detail": "",
        "progress": 100,
        "error": None,
        "video_path": str(source),
        "srt_text": "1\n00:00:00,000 --> 00:00:01,000\ntext\n",
        "result": {"job_id": job_id, "result_video": str(result_video), "preview_video": str(preview_video), "timeline": {}},
        "params": {"mirror_horizontal": False, "overlays": []},
        "publish": None,
        "publish_errors": None,
        "publish_progress": None,
        "covers": [],
        "cover_status": "idle",
        "variants": {},
        "mirror_variant": None,
    }
    job.update(extra)
    app_module.jobs[job_id] = job
    return job


class Part2Base(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.old_work = app_module.MONTAGE_WORK
        self.old_jobs = app_module.jobs
        app_module.MONTAGE_WORK = self.root / "montage_work"
        app_module.MONTAGE_WORK.mkdir()
        app_module.jobs = {}
        app_module.publish_locks.clear()

    def tearDown(self):
        app_module.MONTAGE_WORK = self.old_work
        app_module.jobs = self.old_jobs
        app_module.publish_locks.clear()
        self.tempdir.cleanup()

    def make(self, job_id="p2job", **extra):
        workdir = app_module.MONTAGE_WORK / job_id
        workdir.mkdir(parents=True)
        return make_job(job_id, workdir, **extra), workdir

    @staticmethod
    def _client():
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app), base_url="http://t")


class MirrorVariantTests(Part2Base):
    def test_signature_toggles_mirror_flag(self):
        job, _ = self.make()
        normal = app_module._montage_signature(job["params"])
        mirrored = app_module._mirrored_variant_signature(job)
        toggled = dict(job["params"])
        toggled["mirror_horizontal"] = True
        self.assertEqual(mirrored, app_module._montage_signature(toggled))
        self.assertNotEqual(normal, mirrored)

    def test_ready_requires_existing_file_and_fresh_signature(self):
        job, workdir = self.make()
        self.assertFalse(app_module.mirror_variant_ready(job))
        mirror = workdir / "variants_mirrored" / "final.mp4"
        mirror.parent.mkdir(parents=True)
        mirror.write_bytes(b"m")
        job["variants"] = {"mirrored": {
            "path": str(mirror), "signature": app_module._mirrored_variant_signature(job),
            "ready": True, "mirror_horizontal": True, "created_at": 0,
        }}
        self.assertTrue(app_module.mirror_variant_ready(job))
        # stale signature (params changed) invalidates the cached file
        job["params"]["blur_banned_words"] = True
        self.assertFalse(app_module.mirror_variant_ready(job))

    def test_process_mirror_variant_builds_and_persists(self):
        job, workdir = self.make()
        produced = workdir / "variants_mirrored" / "final.mp4"

        def fake_render(**kwargs):
            self.assertTrue(kwargs["options"]["mirror_horizontal"])
            produced.parent.mkdir(parents=True, exist_ok=True)
            produced.write_bytes(b"mirror")
            return {"result_video": str(produced), "preview_video": str(produced), "timeline": {}}

        with mock.patch.object(app_module.montage_engine, "render_montage", side_effect=fake_render), \
                mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
            asyncio.run(app_module.process_mirror_variant(job["id"]))

        self.assertTrue(app_module.mirror_variant_ready(job))
        self.assertEqual(job["mirror_variant"]["status"], "done")
        self.assertEqual(job["variants"]["mirrored"]["path"], str(produced))

    def test_process_mirror_variant_reuses_ready_without_render(self):
        job, workdir = self.make()
        mirror = workdir / "variants_mirrored" / "final.mp4"
        mirror.parent.mkdir(parents=True)
        mirror.write_bytes(b"m")
        job["variants"] = {"mirrored": {
            "path": str(mirror), "signature": app_module._mirrored_variant_signature(job),
            "ready": True, "mirror_horizontal": True, "created_at": 0,
        }}
        with mock.patch.object(app_module.montage_engine, "render_montage") as render_mock, \
                mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
            asyncio.run(app_module.process_mirror_variant(job["id"]))
        render_mock.assert_not_called()
        self.assertEqual(job["mirror_variant"]["status"], "done")

    def test_process_mirror_variant_records_error(self):
        job, _ = self.make()

        def boom(**kwargs):
            raise RuntimeError("ffmpeg died")

        with mock.patch.object(app_module.montage_engine, "render_montage", side_effect=boom), \
                mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
            asyncio.run(app_module.process_mirror_variant(job["id"]))

        self.assertEqual(job["mirror_variant"]["status"], "error")
        self.assertIn("ffmpeg died", job["mirror_variant"]["error"])
        self.assertFalse(app_module.mirror_variant_ready(job))

    def test_publish_endpoint_rejects_unprepared_mirror(self):
        async def scenario():
            job_id = "p2pub_reject"
            workdir = app_module.MONTAGE_WORK / job_id
            workdir.mkdir(parents=True)
            make_job(job_id, workdir)
            with mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
                async with self._client() as client:
                    resp = await client.post(
                        f"/api/montage/{job_id}/publish",
                        data={"targets_json": json.dumps(["youtube_mirrored"]), "title": "T"},
                    )
            self.assertEqual(resp.status_code, 409)
        asyncio.run(scenario())

    def test_publish_endpoint_accepts_prepared_mirror(self):
        async def scenario():
            job_id = "p2pub_ok"
            workdir = app_module.MONTAGE_WORK / job_id
            workdir.mkdir(parents=True)
            job = make_job(job_id, workdir)
            mirror = workdir / "variants_mirrored" / "final.mp4"
            mirror.parent.mkdir(parents=True)
            mirror.write_bytes(b"m")
            job["variants"] = {"mirrored": {
                "path": str(mirror), "signature": app_module._mirrored_variant_signature(job),
                "ready": True, "mirror_horizontal": True, "created_at": 0,
            }}
            job["publish_prep"] = {
                "youtube:mirror": {
                    "path": str(mirror), "purpose": "youtube", "kind": "video",
                    "mirrored": True, "ready": True,
                    "signature": app_module._publish_prep_signature(job, "youtube:mirror", mirror),
                }
            }
            started = []

            async def fake_publish(job_id_, targets, publish_lock=None):
                started.append(job_id_)
                if publish_lock and publish_lock.locked():
                    publish_lock.release()

            with mock.patch.object(app_module, "process_publish", side_effect=fake_publish), \
                    mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
                async with self._client() as client:
                    resp = await client.post(
                        f"/api/montage/{job_id}/publish",
                        data={"targets_json": json.dumps(["youtube_mirrored"]), "title": "T"},
                    )
            self.assertEqual(resp.status_code, 200, resp.text)
        asyncio.run(scenario())

    def test_result_serializer_exposes_mirror_state(self):
        async def scenario():
            job_id = "p2result"
            workdir = app_module.MONTAGE_WORK / job_id
            workdir.mkdir(parents=True)
            job = make_job(job_id, workdir)
            mirror = workdir / "variants_mirrored" / "final.mp4"
            mirror.parent.mkdir(parents=True)
            mirror.write_bytes(b"m")
            job["variants"] = {"mirrored": {
                "path": str(mirror), "signature": app_module._mirrored_variant_signature(job),
                "ready": True, "mirror_horizontal": True, "created_at": 0,
            }}
            job["mirror_variant"] = {"status": "done", "progress": 100, "detail": "ok", "error": ""}
            async with self._client() as client:
                resp = await client.get(f"/api/montage/result/{job_id}")
            data = resp.json()
            self.assertTrue(data["variants"]["mirrored_ready"])
            self.assertEqual(data["mirror_variant"]["status"], "done")
        asyncio.run(scenario())


class PublishStepTests(Part2Base):
    def test_publish_records_per_target_steps(self):
        async def scenario():
            job, _ = self.make("p2steps")
            job["result"] = {"result_video": str((app_module.MONTAGE_WORK / "p2steps" / "final.mp4"))}

            def fake_publish_job(**kwargs):
                cb = kwargs.get("target_step_cb")
                if cb:
                    cb("youtube", "auth", "Проверка авторизации", "completed", "")
                    cb("youtube", "upload", "Загрузка видео", "running", "25%")
                    cb("youtube", "upload", "Загрузка видео", "completed", "")
                return {"results": {"youtube": {"watch_url": "u"}}, "errors": {}}

            with mock.patch.object(app_module.montage_engine, "publish_job", side_effect=fake_publish_job), \
                    mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None), \
                    mock.patch.object(app_module, "_montage_announce", mock.AsyncMock()):
                await app_module.process_publish(job["id"], [
                    {"id": "youtube", "kind": "youtube", "mirrored": False, "label": "YouTube", "title": "T"}
                ])

            steps = {s["key"]: s for s in job["publish_progress"]["youtube"]["steps"]}
            self.assertEqual(set(steps), {"auth", "upload"})
            self.assertEqual(steps["upload"]["status"], "completed")
            self.assertEqual(steps["auth"]["label"], "Проверка авторизации")
        asyncio.run(scenario())


class DrawingStageTests(Part2Base):
    def setUp(self):
        super().setUp()
        app_module._drawing_generation_jobs.clear()

    async def _generate(self, request):
        app_module._drawing_generation_lock = asyncio.Lock()
        return await app_module.generate_montage_drawings(request)

    def _request(self):
        payload = {"job_id": "dj-part2", "mode": "auto", "srt_text": "1\n00:00:00,000 --> 00:00:01,000\nПривет мир\n"}

        class FakeRequest:
            headers = {}
            client = type("C", (), {"host": "127.0.0.1"})()

            async def json(self):
                return payload

        return FakeRequest()

    def test_drawing_success_marks_all_stages_done(self):
        def fake_generate(transcript, segments, count, progress_cb):
            progress_cb({"event": "attempt_started", "attempt": 1, "attempts_total": 2, "detail": "Запрос"})
            progress_cb({"event": "response_received", "attempt": 1, "detail": "Ответ получен"})
            progress_cb({"event": "response_parsed", "attempt": 1, "detail": "Разбор"})
            return [{"id": "d1", "points": [[0, 0], [1, 1]]}]

        with mock.patch.object(app_module.drawing_generator, "generate_smart_drawings", side_effect=fake_generate), \
                mock.patch.dict(app_module._drawing_generation_jobs, {}, clear=True):
            request = self._request()
            result = asyncio.run(self._generate(request))
            job = app_module._drawing_generation_jobs[result["job_id"]]
            stages = {s["key"]: s for s in job["stages"]}
            self.assertEqual(job["status"], "done")
            self.assertTrue(all(s["status"] == "completed" for s in stages.values()))
            self.assertEqual(job["progress"], 100)

    def test_drawing_error_never_reports_full_progress(self):
        from fastapi import HTTPException

        def fake_generate(transcript, segments, count, progress_cb):
            progress_cb({"event": "attempt_started", "attempt": 1, "attempts_total": 2, "detail": "Запрос"})
            progress_cb({"event": "response_received", "attempt": 1, "detail": "Ответ"})
            raise RuntimeError("no schemas")

        with mock.patch.object(app_module.drawing_generator, "generate_smart_drawings", side_effect=fake_generate), \
                mock.patch.dict(app_module._drawing_generation_jobs, {}, clear=True):
            request = self._request()
            with self.assertRaises(HTTPException):
                asyncio.run(self._generate(request))
            # job is removed from the live registry on error; read the persisted snapshot
            persisted = app_module._load_drawing_job("dj-part2")
            self.assertIsNotNone(persisted)
            self.assertLessEqual(persisted["progress"], 95)
            running = [s for s in persisted["stages"] if s["status"] == "error"]
            self.assertTrue(running)

    def test_drawing_recovery_marks_running_stage_error_and_caps_progress(self):
        workdir = app_module.MONTAGE_WORK / "dj-recover"
        workdir.mkdir(parents=True)
        state = {
            "id": "dj-recover", "status": "running", "phase": "x", "progress": 88,
            "stages": [
                {"key": "prepare", "label": "Подготовка", "status": "completed", "progress": None, "detail": "", "error": ""},
                {"key": "request", "label": "Запрос", "status": "running", "progress": None, "detail": "", "error": ""},
            ],
        }
        (workdir / "drawing_generation.json").write_text(json.dumps(state), encoding="utf-8")
        with mock.patch.dict(app_module._drawing_generation_jobs, {}, clear=True):
            app_module._recover_drawing_jobs()
            job = app_module._drawing_generation_jobs["dj-recover"]
            self.assertEqual(job["status"], "error")
            self.assertLessEqual(job["progress"], 95)
            stages = {s["key"]: s for s in job["stages"]}
            self.assertEqual(stages["prepare"]["status"], "completed")
            self.assertEqual(stages["request"]["status"], "error")


if __name__ == "__main__":
    unittest.main()
