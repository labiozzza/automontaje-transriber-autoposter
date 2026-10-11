"""Server-side publish progress tests (spec 1.3 / 4: successful and partial
publish, retry only failed targets, errors are not masked as success).

Run:
  python -m unittest test_publish_progress
"""
import asyncio
import unittest
from pathlib import Path
from unittest import mock

import montage.engine as engine


class PublishJobAggregationTest(unittest.TestCase):
    def _target(self, target_id, kind):
        return {"id": target_id, "kind": kind, "label": target_id}

    def test_all_success_no_errors(self):
        targets = [self._target("ig", "instagram"), self._target("yt", "youtube")]
        seen = {}

        def fake_publish_targets(**kwargs):
            target = kwargs["targets"][0]
            seen[target["id"]] = True
            kwargs["progress_cb"](50.0, "upload", "uploading")
            return {"results": {target["id"]: {"kind": target["kind"], "id": target["id"]}}}

        with mock.patch.object(engine, "_publish_targets", side_effect=fake_publish_targets):
            result = engine.publish_job(
                job_id="j", workdir=Path("/tmp"), source_video=Path("/tmp/x.mp4"),
                targets=targets, progress_cb=lambda *a: None, log_cb=lambda *a: None,
            )
        self.assertEqual(set(result["results"]), {"ig", "yt"})
        self.assertEqual(result["errors"], {})
        self.assertEqual(set(seen), {"ig", "yt"})

    def test_partial_failure_is_not_masked(self):
        targets = [self._target("ig", "instagram"), self._target("yt", "youtube")]
        statuses = {}

        def fake_publish_targets(**kwargs):
            target = kwargs["targets"][0]
            if target["kind"] == "youtube":
                raise RuntimeError("YouTube quota exceeded")
            return {"results": {target["id"]: {"kind": target["kind"], "id": target["id"]}}}

        def target_cb(target_id, percent, detail, status):
            statuses[target_id] = status

        with mock.patch.object(engine, "_publish_targets", side_effect=fake_publish_targets):
            result = engine.publish_job(
                job_id="j", workdir=Path("/tmp"), source_video=Path("/tmp/x.mp4"),
                targets=targets, progress_cb=lambda *a: None, log_cb=lambda *a: None,
                target_progress_cb=target_cb,
            )
        self.assertEqual(set(result["results"]), {"ig"})
        self.assertEqual(set(result["errors"]), {"yt"})
        self.assertIn("quota", result["errors"]["yt"])
        self.assertEqual(statuses["ig"], "done")
        self.assertEqual(statuses["yt"], "error")

    def test_all_failure_returns_errors(self):
        targets = [self._target("ig", "instagram")]

        def fake_publish_targets(**kwargs):
            raise RuntimeError("boom")

        with mock.patch.object(engine, "_publish_targets", side_effect=fake_publish_targets):
            result = engine.publish_job(
                job_id="j", workdir=Path("/tmp"), source_video=Path("/tmp/x.mp4"),
                targets=targets, progress_cb=lambda *a: None, log_cb=lambda *a: None,
            )
        self.assertEqual(result["results"], {})
        self.assertEqual(set(result["errors"]), {"ig"})

    def test_retry_only_failed_targets(self):
        targets = [self._target("ig", "instagram"), self._target("yt", "youtube")]
        calls = []
        yt_failures = {"n": 0}

        def fake_publish_targets(**kwargs):
            target = kwargs["targets"][0]
            calls.append(target["id"])
            if target["id"] == "yt" and yt_failures["n"] < 1:
                yt_failures["n"] += 1
                raise RuntimeError("temp")
            return {"results": {target["id"]: {"kind": target["kind"], "id": target["id"]}}}

        with mock.patch.object(engine, "_publish_targets", side_effect=fake_publish_targets):
            first = engine.publish_job(
                job_id="j", workdir=Path("/tmp"), source_video=Path("/tmp/x.mp4"),
                targets=targets, progress_cb=lambda *a: None, log_cb=lambda *a: None,
            )
        self.assertEqual(set(first["errors"]), {"yt"})

        failed_ids = set(first["errors"])
        retry_targets = [t for t in targets if t["id"] in failed_ids]
        calls.clear()
        with mock.patch.object(engine, "_publish_targets", side_effect=fake_publish_targets):
            second = engine.publish_job(
                job_id="j", workdir=Path("/tmp"), source_video=Path("/tmp/x.mp4"),
                targets=retry_targets, progress_cb=lambda *a: None, log_cb=lambda *a: None,
            )
        self.assertEqual(set(second["errors"]), set())
        self.assertEqual(set(second["results"]), {"yt"})
        self.assertEqual(calls, ["yt"])


class ProcessPublishStatusTest(unittest.TestCase):
    def test_partial_publish_job_status_not_masked(self):
        import app as app_module

        job_id = "test_publish_partial"
        app_module.jobs[job_id] = {
            "kind": "montage",
            "status": "done",
            "result": {"result_video": "/tmp/does-not-matter.mp4"},
            "video_path": "/tmp/does-not-matter.mp4",
            "params": {},
            "publish_progress": {},
        }
        targets = [
            {"id": "ig", "kind": "instagram", "label": "Instagram"},
            {"id": "yt", "kind": "youtube", "label": "YouTube"},
        ]

        def fake_publish_job(**kwargs):
            return {
                "results": {"ig": {"kind": "instagram", "id": "ig", "permalink": "https://x/ig"}},
                "errors": {"yt": "quota exceeded"},
            }

        async def fake_announce(*a, **k):
            return None

        try:
            with mock.patch.object(app_module.montage_engine, "publish_job", side_effect=fake_publish_job), \
                 mock.patch.object(app_module, "_montage_announce", side_effect=fake_announce), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None), \
                 mock.patch.object(app_module, "_cleanup_publish_artifacts", lambda *a, **k: {}):
                asyncio.run(app_module.process_publish(job_id, targets))
        finally:
            job = app_module.jobs.pop(job_id, None)

        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["stage"], "published")
        self.assertEqual(set(job["publish_errors"]), {"yt"})
        self.assertIn("ошибк", job["detail"].lower())
        self.assertIn("ig", job["publish"])

    def test_full_publish_has_no_errors(self):
        import app as app_module

        job_id = "test_publish_full"
        app_module.jobs[job_id] = {
            "kind": "montage",
            "status": "done",
            "result": {"result_video": "/tmp/does-not-matter.mp4"},
            "video_path": "/tmp/does-not-matter.mp4",
            "params": {},
            "publish_progress": {},
        }
        targets = [{"id": "ig", "kind": "instagram", "label": "Instagram"}]

        def fake_publish_job(**kwargs):
            return {"results": {"ig": {"kind": "instagram", "id": "ig"}}, "errors": {}}

        async def fake_announce(*a, **k):
            return None

        try:
            with mock.patch.object(app_module.montage_engine, "publish_job", side_effect=fake_publish_job), \
                 mock.patch.object(app_module, "_montage_announce", side_effect=fake_announce), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None), \
                 mock.patch.object(app_module, "_cleanup_publish_artifacts", lambda *a, **k: {}):
                asyncio.run(app_module.process_publish(job_id, targets))
        finally:
            job = app_module.jobs.pop(job_id, None)

        self.assertEqual(job["publish_errors"], {})
        self.assertEqual(job["status"], "done")


if __name__ == "__main__":
    unittest.main()
