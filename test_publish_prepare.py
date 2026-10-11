"""Section 4 mock tests: publishing never transcodes; preparation does it.

Verifies, without any real FFmpeg/PIL/network work:

* Instagram/YouTube publish consume already-prepared files (no ``prepare_variant``
  / ``prepare_video`` call inside the publish path);
* preparation renders each needed variant once and reuses ready artefacts;
* a stale prepared file (source changed) is invalidated and re-prepared;
* the prepared cover is converted before publishing;
* the publish endpoint returns ``409`` with ``prepare_targets`` when a requested
  target has no ready artefact;
* the preparation operation records nested stages for the three-level progress bar.

Run:
  python -m unittest test_publish_prepare
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
    result_video = workdir / "final.mp4"
    result_video.write_bytes(b"final-video")
    job = {
        "id": job_id,
        "kind": "montage",
        "status": "done",
        "stage": "done",
        "detail": "",
        "progress": 100,
        "error": None,
        "result": {"job_id": job_id, "result_video": str(result_video), "timeline": {}},
        "video_path": str(result_video),
        "srt_text": "",
        "params": {},
        "publish": None,
        "publish_errors": None,
        "publish_progress": {},
        "covers": [],
        "cover_status": "idle",
        "variants": {},
        "mirror_variant": None,
        "publish_prep": {},
        "prepare_operation": None,
        "prepare_status": "idle",
        "prepare_error": None,
    }
    job.update(extra)
    app_module.jobs[job_id] = job
    return job


class PrepareBase(unittest.TestCase):
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

    def make(self, job_id="prepjob", **extra):
        workdir = app_module.MONTAGE_WORK / job_id
        workdir.mkdir(parents=True)
        return make_job(job_id, workdir, **extra), workdir

    @staticmethod
    def _client():
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app), base_url="http://t")

    def _targets(self):
        return [
            {"id": "instagram_feed", "kind": "instagram", "mirrored": False, "label": "Instagram"},
            {"id": "youtube", "kind": "youtube", "mirrored": False, "label": "YouTube"},
        ]

    def _run_prepare(self, job_id, targets, cover_path=None, cover_id=""):
        async def scenario():
            with mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None), \
                 mock.patch.object(app_module, "_montage_announce", _noop):
                await app_module.process_prepare_publish(job_id, targets, cover_path, cover_id)
        asyncio.run(scenario())


async def _noop(*args, **kwargs):
    return None


def _fake_render_factory(calls):
    def fake_render(purpose, source, target_dir, stem, mirrored, progress):
        calls.append((purpose, mirrored))
        if callable(progress):
            progress(100, "готово")
        output = Path(target_dir) / f"{stem}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"prepared-" + purpose.encode())
        return output
    return fake_render


class PreparePipelineTests(PrepareBase):
    def test_prepares_each_variant_once_and_reuses(self):
        job, workdir = self.make("prep_reuse")
        calls = []
        with mock.patch.object(app_module, "_render_prepared_video", _fake_render_factory(calls)), \
             mock.patch.object(app_module, "_validate_prepared_video",
                               lambda path: {"size": Path(path).stat().st_size, "duration": "30"}):
            self._run_prepare("prep_reuse", self._targets())
            self.assertEqual(sorted(calls), [("instagram", False), ("youtube", False)])
            prepared = app_module.jobs["prep_reuse"]["publish_prep"]
            self.assertIn("instagram:main", prepared)
            self.assertIn("youtube:main", prepared)
            self.assertTrue(prepared["instagram:main"]["ready"])
            self.assertEqual(app_module.jobs["prep_reuse"]["prepare_status"], "done")
            # second run must reuse, not re-render
            calls.clear()
            self._run_prepare("prep_reuse", self._targets())
        self.assertEqual(calls, [])

    def test_stale_prepared_video_is_invalidated(self):
        job, workdir = self.make("prep_stale")
        calls = []
        with mock.patch.object(app_module, "_render_prepared_video", _fake_render_factory(calls)), \
             mock.patch.object(app_module, "_validate_prepared_video",
                               lambda path: {"size": Path(path).stat().st_size, "duration": "30"}):
            self._run_prepare("prep_stale", [{"id": "instagram_feed", "kind": "instagram", "mirrored": False}])
        entry = app_module.jobs["prep_stale"]["publish_prep"]["instagram:main"]
        self.assertIsNotNone(app_module._prepared_video_entry(app_module.jobs["prep_stale"], "instagram", False))
        # mutate the source -> signature no longer matches
        source = Path(app_module.jobs["prep_stale"]["result"]["result_video"])
        source.write_bytes(b"a-completely-different-payload")
        self.assertIsNone(app_module._prepared_video_entry(app_module.jobs["prep_stale"], "instagram", False))
        self.assertTrue(entry["ready"])

    def test_publish_readiness_marks_missing_and_prepared(self):
        job, workdir = self.make("prep_ready")
        target = {"id": "instagram_feed", "kind": "instagram", "mirrored": False}
        prepared, missing = app_module._publish_readiness(app_module.jobs["prep_ready"], [target])
        self.assertEqual(prepared, {})
        self.assertEqual(len(missing), 1)
        source = Path(job["result"]["result_video"])
        out = workdir / "publish_prep" / "ig.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"ig")
        job["publish_prep"]["instagram:main"] = {
            "path": str(out), "purpose": "instagram", "kind": "video", "mirrored": False,
            "ready": True, "signature": app_module._publish_prep_signature(job, "instagram:main", source),
        }
        prepared, missing = app_module._publish_readiness(job, [target])
        self.assertEqual(missing, [])
        self.assertEqual(prepared["instagram_feed"]["video_path"], str(out))

    def test_prepared_cover_converted_and_required(self):
        job, workdir = self.make("prep_cover")
        cover = workdir / "cover.jpg"
        cover.write_bytes(b"cover-bytes")
        target = {"id": "instagram_feed", "kind": "instagram", "mirrored": False,
                  "cover_path": str(cover)}
        calls = []
        with mock.patch.object(app_module, "_render_prepared_video", _fake_render_factory(calls)), \
             mock.patch.object(app_module, "_render_prepared_cover",
                               lambda purpose, cover_path, output: _write(Path(output), b"conv")), \
             mock.patch.object(app_module, "_validate_prepared_video",
                               lambda path: {"size": Path(path).stat().st_size, "duration": "30"}):
            self._run_prepare("prep_cover", [target], cover_path=str(cover), cover_id="c1")
        prepared = app_module.jobs["prep_cover"]["publish_prep"]
        self.assertIn("cover:instagram", prepared)
        self.assertTrue(prepared["cover:instagram"]["ready"])
        ready, missing = app_module._publish_readiness(app_module.jobs["prep_cover"], [target])
        self.assertEqual(missing, [])
        self.assertEqual(ready["instagram_feed"]["cover_path"], prepared["cover:instagram"]["path"])

    def test_prepare_records_nested_stages(self):
        job, workdir = self.make("prep_stages")
        calls = []
        with mock.patch.object(app_module, "_render_prepared_video", _fake_render_factory(calls)), \
             mock.patch.object(app_module, "_validate_prepared_video",
                               lambda path: {"size": Path(path).stat().st_size, "duration": "30"}):
            self._run_prepare("prep_stages", self._targets())
        operation = app_module.jobs["prep_stages"]["prepare_operation"]
        self.assertIsNotNone(operation)
        keys = {stage["key"] for stage in operation["stages"]}
        self.assertEqual(keys, {"instagram:main", "youtube:main"})
        self.assertTrue(all(stage["status"] == "completed" for stage in operation["stages"]))


class PublishWithoutTranscodeTests(PrepareBase):
    def test_instagram_publish_uses_prepared_file_without_prepare_variant(self):
        job, workdir = self.make("pub_ig")
        prepared = workdir / "publish_prep" / "ig_ready.mp4"
        prepared.parent.mkdir(parents=True, exist_ok=True)
        prepared.write_bytes(b"already-prepared")
        engine = app_module.montage_engine
        ig_module = __import__("montage.instagram_publisher", fromlist=["InstagramPublisher"])

        class FakeIG:
            IG_USER_ID = "1"
            prepared_variant_called = False

            def __init__(self, *a, **k):
                pass

            def resolve_hosts(self):
                return ("1.1.1.1", "2.2.2.2")

            def upload_to_github(self, staged):
                FakeIG.uploaded = str(staged)
                return ("graph", "raw")

            def verify_public_video(self, url, ip, cb):
                cb(100, "ok")

            def upload_cover_file(self, path):
                return ("graph", "raw")

            def create_container(self, *a, **k):
                return "container"

            def wait_container(self, *a, **k):
                cb = a[3] if len(a) > 3 else (k.get("progress") or (lambda p, d: None))
                cb(100, "ok")

            def publish_container(self, *a, **k):
                return "media"

            def media_info(self, *a, **k):
                return {"id": "media", "permalink": "https://x/ig"}

            def prepare_variant(self, *a, **k):
                FakeIG.prepared_variant_called = True
                raise AssertionError("prepare_variant must not run during publish")

        with mock.patch.object(ig_module, "InstagramPublisher", FakeIG), \
             mock.patch.object(engine, "load_secrets_env", lambda: None), \
             mock.patch.object(engine, "_load_module", lambda *a, **k: mock.MagicMock()), \
             mock.patch.object(engine, "get_software_defaults",
                               lambda: {"instagram_ready": True, "github_ready": True}), \
             mock.patch.object(engine, "time", mock.Mock()), \
             mock.patch.dict("sys.modules", {"montage.social_stats": mock.MagicMock()}):
            result = engine._publish_targets(
                job_id="pub_ig",
                workdir=workdir,
                source_video=Path(job["result"]["result_video"]),
                mirrored_source_video=None,
                original_video=Path(job["result"]["result_video"]),
                transcript_text="text",
                targets=[{"id": "instagram_feed", "kind": "instagram", "mirrored": False, "label": "Instagram"}],
                progress_cb=lambda *a: None,
                log_cb=lambda *a: None,
                prepared_targets={"instagram_feed": {"video_path": str(prepared)}},
            )
        self.assertFalse(FakeIG.prepared_variant_called)
        self.assertEqual(FakeIG.uploaded, str(prepared))
        self.assertIn("instagram_feed", result["results"])

    def test_prepared_targets_threaded_into_publish_job(self):
        job, workdir = self.make("pub_thread")
        source = Path(job["result"]["result_video"])
        out = workdir / "publish_prep" / "ig.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"ig")
        job["publish_prep"]["instagram:main"] = {
            "path": str(out), "purpose": "instagram", "kind": "video", "mirrored": False,
            "ready": True, "signature": app_module._publish_prep_signature(job, "instagram:main", source),
        }
        captured = {}

        async def fake_announce(*a, **k):
            return None

        def fake_publish_job(**kwargs):
            captured.update(kwargs)
            return {"results": {"instagram_feed": {"kind": "instagram", "id": "instagram_feed"}}, "errors": {}}

        async def scenario():
            with mock.patch.object(app_module.montage_engine, "publish_job", side_effect=fake_publish_job), \
                 mock.patch.object(app_module, "_montage_announce", side_effect=fake_announce), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None), \
                 mock.patch.object(app_module, "_cleanup_publish_artifacts", lambda *a, **k: {}):
                await app_module.process_publish(
                    "pub_thread",
                    [{"id": "instagram_feed", "kind": "instagram", "mirrored": False, "label": "Instagram"}],
                )
        asyncio.run(scenario())
        self.assertIn("prepared_targets", captured)
        self.assertEqual(captured["prepared_targets"]["instagram_feed"]["video_path"], str(out))


class PublishGuardTests(PrepareBase):
    def test_publish_endpoint_409_lists_unprepared_targets(self):
        async def scenario():
            job_id = "guard_409"
            workdir = app_module.MONTAGE_WORK / job_id
            workdir.mkdir(parents=True)
            make_job(job_id, workdir)
            with mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
                async with self._client() as client:
                    resp = await client.post(
                        f"/api/montage/{job_id}/publish",
                        data={"targets_json": json.dumps(["instagram_feed", "youtube"]), "title": "T"},
                    )
            self.assertEqual(resp.status_code, 409)
            data = resp.json()
            self.assertIn("prepare_targets", data)
            self.assertIn("instagram_feed", data["prepare_targets"])
            self.assertIn("youtube", data["prepare_targets"])
        asyncio.run(scenario())

    def test_publish_endpoint_allows_prepared_target(self):
        async def scenario():
            job_id = "guard_ok"
            workdir = app_module.MONTAGE_WORK / job_id
            workdir.mkdir(parents=True)
            job = make_job(job_id, workdir)
            source = Path(job["result"]["result_video"])
            out = workdir / "publish_prep" / "ig.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"ig")
            job["publish_prep"]["instagram:main"] = {
                "path": str(out), "purpose": "instagram", "kind": "video", "mirrored": False,
                "ready": True, "signature": app_module._publish_prep_signature(job, "instagram:main", source),
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
                        data={"targets_json": json.dumps(["instagram_feed"]), "title": "T"},
                    )
            self.assertEqual(resp.status_code, 200, resp.text)
        asyncio.run(scenario())


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def sys_executable():
    import sys
    return sys.executable


if __name__ == "__main__":
    unittest.main()
