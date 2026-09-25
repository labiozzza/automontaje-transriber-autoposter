import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from starlette.requests import Request

import app


class MontageCleanupTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.old_work = app.MONTAGE_WORK
        self.old_jobs = app.jobs
        self.old_queues = app.job_queues
        app.MONTAGE_WORK = self.root / "montage_work"
        app.MONTAGE_WORK.mkdir()
        app.jobs = {}
        app.job_queues = {}

    def tearDown(self):
        app.MONTAGE_WORK = self.old_work
        app.jobs = self.old_jobs
        app.job_queues = self.old_queues
        self.tempdir.cleanup()

    def make_job(self, job_id="cleanup01", status="done"):
        workdir = app.MONTAGE_WORK / job_id
        workdir.mkdir(parents=True)
        source = workdir / "source.mp4"
        source.write_bytes(b"source-video")
        words = workdir / f"{job_id}_words.srt"
        words.write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")
        result_video = workdir / "final.mp4"
        preview_video = workdir / "preview.mp4"
        result_video.write_bytes(b"final-video")
        preview_video.write_bytes(b"preview-video")
        job = {
            "id": job_id,
            "kind": "montage",
            "status": status,
            "progress": 100 if status == "done" else 10,
            "stage": "done" if status == "done" else status,
            "detail": "",
            "error": None,
            "original_name": source.name,
            "video_path": str(source),
            "srt_text": words.read_text(encoding="utf-8"),
            "transcript_job_id": None,
            "project_id": None,
            "result": {
                "job_id": job_id,
                "result_video": str(result_video),
                "preview_video": str(preview_video),
                "timeline": {},
            },
            "params": {"face_tracking": False, "mirror_horizontal": False, "overlays": []},
            "publish": None,
            "publish_errors": None,
            "publish_progress": None,
            "covers": [],
            "cover_status": "idle",
        }
        app.jobs[job_id] = job
        return job, workdir

    @staticmethod
    async def response_body(response):
        return b"".join([chunk async for chunk in response.body_iterator])

    def test_render_cleanup_is_allowlisted_idempotent_and_protects_result_files(self):
        job, workdir = self.make_job()
        for name in app._RENDER_CLEANUP_FILES:
            (workdir / name).write_bytes(b"temporary")
        for name in (
            "subtitle_timeline_config.json",
            "subtitle_timeline_config_words.json",
            "subtitle_timeline_config_phrases.json",
        ):
            (workdir / name).write_text("{}", encoding="utf-8")
        for name in ("overlays", "covers"):
            directory = workdir / name
            directory.mkdir()
            (directory / "keep.txt").write_text("keep", encoding="utf-8")
        (workdir / "project_video.mov").write_bytes(b"project")
        (workdir / "unlisted.mp4").write_bytes(b"keep")
        (workdir / "subtitle_timeline_config_backup.json").write_text("keep", encoding="utf-8")
        (workdir / "final.mp4").write_bytes(b"final")
        (workdir / "preview.mp4").write_bytes(b"preview")
        job["result"]["result_video"] = str(workdir / "sub_words.mp4")
        job["result"]["preview_video"] = str(workdir / "zoom.mp4")
        app._save_montage_job(job["id"])

        first = app._cleanup_render_artifacts(job["id"], job)
        second = app._cleanup_render_artifacts(job["id"], job)

        self.assertFalse(first["errors"])
        self.assertFalse(second["errors"])
        self.assertEqual(second["removed"], [])
        self.assertTrue((workdir / "sub_words.mp4").is_file())
        self.assertTrue((workdir / "zoom.mp4").is_file())
        self.assertTrue((workdir / "final.mp4").is_file())
        self.assertTrue((workdir / "preview.mp4").is_file())
        self.assertTrue((workdir / "job_state.json").is_file())
        self.assertTrue((workdir / "source.mp4").is_file())
        self.assertTrue((workdir / "project_video.mov").is_file())
        self.assertTrue((workdir / f"{job['id']}_words.srt").is_file())
        self.assertTrue((workdir / "overlays" / "keep.txt").is_file())
        self.assertTrue((workdir / "covers" / "keep.txt").is_file())
        self.assertTrue((workdir / "unlisted.mp4").is_file())
        self.assertTrue((workdir / "subtitle_timeline_config_backup.json").is_file())
        self.assertFalse((workdir / "normalized_source.mp4").exists())
        self.assertFalse((workdir / "subtitle_timeline_config_words.json").exists())
        self.assertEqual(len((workdir / "cleanup.log").read_text(encoding="utf-8").splitlines()), 2)

        download = asyncio.run(app.montage_download(job["id"]))
        self.assertEqual(Path(download.path), workdir / "sub_words.mp4")
        request = Request({
            "type": "http",
            "method": "GET",
            "path": f"/api/montage/preview/{job['id']}",
            "headers": [(b"range", b"bytes=1-3")],
        })
        preview = asyncio.run(app.montage_preview(job["id"], request))
        self.assertEqual(preview.status_code, 206)
        self.assertEqual(asyncio.run(self.response_body(preview)), b"emp")

    def test_cleanup_errors_do_not_change_successful_status(self):
        job, workdir = self.make_job()
        disposable = workdir / "normalized_source.mp4"
        disposable.write_bytes(b"busy")
        original_remove = app._remove_cleanup_path

        def fail_one(path, recursive):
            if path == disposable:
                raise OSError("file is busy")
            return original_remove(path, recursive)

        with mock.patch.object(app, "_remove_cleanup_path", side_effect=fail_one):
            summary = app._cleanup_render_artifacts(job["id"], job)

        self.assertEqual(job["status"], "done")
        self.assertTrue(disposable.exists())
        self.assertTrue(any("file is busy" in error for error in summary["errors"]))
        self.assertIn("file is busy", (workdir / "cleanup.log").read_text(encoding="utf-8"))

    def test_cleanup_does_not_follow_links_or_accept_paths_outside_job(self):
        job, workdir = self.make_job()
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"outside")
        os.symlink(outside, workdir / "zoom.mp4")

        summary = app._cleanup_render_artifacts(job["id"], job)
        rejected = app._cleanup_allowed_paths(
            job["id"], job, "render", [(outside, False)]
        )
        invalid = app._cleanup_render_artifacts("../outside", job)

        external_job = self.root / "external-job"
        external_job.mkdir()
        external_state = external_job / "job_state.json"
        external_state.write_text('{"id":"linked"}', encoding="utf-8")
        os.symlink(external_job, app.MONTAGE_WORK / "linked")
        app._recover_montage_jobs()

        self.assertFalse(summary["errors"])
        self.assertFalse((workdir / "zoom.mp4").exists())
        self.assertEqual(outside.read_bytes(), b"outside")
        self.assertTrue(rejected["errors"])
        self.assertTrue(invalid["errors"])
        self.assertNotIn("linked", app.jobs)
        self.assertEqual(external_state.read_text(encoding="utf-8"), '{"id":"linked"}')
        self.assertEqual(outside.read_bytes(), b"outside")

    def test_process_montage_cleans_only_after_successful_state_save(self):
        job, workdir = self.make_job(status="uploaded")
        job["result"] = None
        params = job["params"]

        def render(**kwargs):
            (workdir / "normalized_source.mp4").write_bytes(b"temporary")
            (workdir / "final.mp4").write_bytes(b"final")
            (workdir / "preview.mp4").write_bytes(b"preview")
            return {
                "job_id": job["id"],
                "result_video": str(workdir / "final.mp4"),
                "preview_video": str(workdir / "preview.mp4"),
                "timeline": {},
            }

        with mock.patch.object(app.montage_engine, "montage_status", return_value={"face_model": True}), \
                mock.patch.object(app.montage_engine, "render_montage", side_effect=render):
            asyncio.run(app.process_montage(job["id"], params))

        self.assertEqual(job["status"], "done")
        self.assertFalse((workdir / "normalized_source.mp4").exists())
        saved = json.loads((workdir / "job_state.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "done")
        self.assertEqual(saved["result"]["result_video"], str(workdir / "final.mp4"))

    def test_failed_render_does_not_cleanup(self):
        job, workdir = self.make_job(status="uploaded")
        job["result"] = None
        disposable = workdir / "normalized_source.mp4"
        disposable.write_bytes(b"diagnostic")

        with mock.patch.object(app.montage_engine, "montage_status", return_value={"face_model": True}), \
                mock.patch.object(app.montage_engine, "render_montage", side_effect=RuntimeError("render failed")):
            asyncio.run(app.process_montage(job["id"], job["params"]))

        self.assertEqual(job["status"], "error")
        self.assertTrue(disposable.is_file())
        self.assertFalse((workdir / "cleanup.log").exists())

    def test_unexpected_cleanup_exception_does_not_change_render_success(self):
        job, workdir = self.make_job(status="uploaded")
        job["result"] = None

        def render(**kwargs):
            return {
                "job_id": job["id"],
                "result_video": str(workdir / "final.mp4"),
                "preview_video": str(workdir / "preview.mp4"),
                "timeline": {},
            }

        with mock.patch.object(app.montage_engine, "montage_status", return_value={"face_model": True}), \
                mock.patch.object(app.montage_engine, "render_montage", side_effect=render), \
                mock.patch.object(app, "_cleanup_render_artifacts", side_effect=RuntimeError("cleanup crashed")):
            asyncio.run(app.process_montage(job["id"], job["params"]))

        self.assertEqual(job["status"], "done")
        self.assertIsNone(job["error"])
        self.assertIn("cleanup crashed", (workdir / "cleanup.log").read_text(encoding="utf-8"))

    def test_recovery_retries_cleanup_and_preserves_download_and_preview(self):
        job, workdir = self.make_job(job_id="recover01", status="processing")
        protected_result = workdir / "sub_phrases.mp4"
        protected_result.write_bytes(b"result-after-restart")
        (workdir / "normalized_source.mp4").write_bytes(b"temporary")
        job["result"]["result_video"] = str(protected_result)
        job["last_publish_targets"] = []
        app._save_montage_job(job["id"])
        app.jobs.clear()

        app._recover_montage_jobs()

        recovered = app.jobs[job["id"]]
        self.assertEqual(recovered["status"], "done")
        self.assertTrue(protected_result.is_file())
        self.assertFalse((workdir / "normalized_source.mp4").exists())
        persisted = json.loads((workdir / "job_state.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "done")
        download = asyncio.run(app.montage_download(job["id"]))
        self.assertEqual(Path(download.path), protected_result)
        request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
        preview = asyncio.run(app.montage_preview(job["id"], request))
        self.assertEqual(Path(preview.path), workdir / "preview.mp4")

    def test_successful_mirrored_publish_cleans_only_after_publish_returns(self):
        job, workdir = self.make_job(job_id="publish01")
        (workdir / "normalized_source.mp4").write_bytes(b"temporary")
        app._cleanup_render_artifacts(job["id"], job)
        targets = [{"id": "instagram_mirror", "kind": "instagram", "mirrored": True, "label": "Mirror"}]

        def render_mirror(**kwargs):
            self.assertTrue(Path(kwargs["source_video"]).is_file())
            self.assertTrue((workdir / f"{job['id']}_words.srt").is_file())
            mirror_dir = Path(kwargs["workdir"])
            mirror_dir.mkdir(parents=True, exist_ok=True)
            mirror = mirror_dir / "final.mp4"
            mirror.write_bytes(b"mirror")
            return {"result_video": str(mirror), "preview_video": str(mirror), "timeline": {}}

        def publish(**kwargs):
            self.assertTrue(Path(kwargs["mirrored_source_video"]).is_file())
            for name in ("staging", "youtube_staging"):
                directory = workdir / name
                directory.mkdir()
                (directory / "active.mp4").write_bytes(b"active")
            (workdir / f"{job['id']}_tg.mp4").write_bytes(b"telegram")
            self.assertTrue((workdir / "publish_mirrored").is_dir())
            return {"results": {"instagram_mirror": {"ok": True}}, "errors": {}}

        with mock.patch.object(app.montage_engine, "render_montage", side_effect=render_mirror), \
                mock.patch.object(app.montage_engine, "publish_job", side_effect=publish):
            asyncio.run(app.process_publish(job["id"], targets))

        self.assertEqual(job["status"], "done")
        self.assertEqual(job["stage"], "published")
        self.assertFalse((workdir / "publish_mirrored").exists())
        self.assertFalse((workdir / "staging").exists())
        self.assertFalse((workdir / "youtube_staging").exists())
        self.assertFalse((workdir / f"{job['id']}_tg.mp4").exists())
        self.assertTrue((workdir / "source.mp4").is_file())
        self.assertTrue((workdir / f"{job['id']}_words.srt").is_file())

    def test_partial_publish_error_keeps_staging_for_diagnostics(self):
        job, workdir = self.make_job(job_id="publish02")
        for name in ("publish_mirrored", "staging", "youtube_staging"):
            directory = workdir / name
            directory.mkdir()
            (directory / "diagnostic.txt").write_text("keep", encoding="utf-8")
        telegram = workdir / f"{job['id']}_tg.mp4"
        telegram.write_bytes(b"keep")

        with mock.patch.object(
            app.montage_engine,
            "publish_job",
            return_value={"results": {}, "errors": {"youtube": "upload failed"}},
        ):
            asyncio.run(app.process_publish(
                job["id"],
                [{"id": "youtube", "kind": "youtube", "mirrored": False, "label": "YouTube"}],
            ))

        self.assertEqual(job["status"], "done")
        self.assertTrue(job["publish_errors"])
        self.assertTrue((workdir / "publish_mirrored").is_dir())
        self.assertTrue((workdir / "staging").is_dir())
        self.assertTrue((workdir / "youtube_staging").is_dir())
        self.assertTrue(telegram.is_file())

    def test_publish_cleanup_protects_result_inside_staging(self):
        job, workdir = self.make_job(job_id="publish03")
        staging = workdir / "staging"
        staging.mkdir()
        protected = staging / "fallback.mp4"
        protected.write_bytes(b"fallback")
        job["result"]["result_video"] = str(protected)
        job["publish_errors"] = {}
        job["stage"] = "published"

        summary = app._cleanup_publish_artifacts(job["id"], job)

        self.assertIn("staging", summary["protected"])
        self.assertTrue(protected.is_file())


if __name__ == "__main__":
    unittest.main()
