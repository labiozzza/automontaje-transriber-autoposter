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

    def test_confirmed_callback_receives_result_before_done(self):
        targets = [self._target("ig", "instagram")]
        order = []
        confirmed = {}

        def fake_publish_targets(**kwargs):
            return {"results": {"ig": {"media_id": "m1", "permalink": "https://x/ig"}}}

        def target_cb(target_id, percent, detail, status):
            order.append(("progress", status))

        def confirmed_cb(target_id, data):
            confirmed[target_id] = data
            order.append(("confirmed",))

        with mock.patch.object(engine, "_publish_targets", side_effect=fake_publish_targets):
            result = engine.publish_job(
                job_id="j", workdir=Path("/tmp"), source_video=Path("/tmp/x.mp4"),
                targets=targets, progress_cb=lambda *a: None, log_cb=lambda *a: None,
                target_progress_cb=target_cb, confirmed_cb=confirmed_cb,
            )
        self.assertEqual(set(result["results"]), {"ig"})
        self.assertEqual(confirmed["ig"]["media_id"], "m1")
        self.assertEqual(order[0], ("confirmed",))
        self.assertEqual(order[1], ("progress", "done"))

    def test_ambiguous_publish_reported_as_unknown(self):
        targets = [self._target("ig", "instagram")]
        statuses = {}

        def fake_publish_targets(**kwargs):
            raise RuntimeError(
                "AMBIGUOUS_MEDIA_PUBLISH: Meta могла опубликовать Reel, но media_id "
                "не подтверждён; автоматический повтор запрещён"
            )

        def target_cb(target_id, percent, detail, status):
            statuses[target_id] = status

        with mock.patch.object(engine, "_publish_targets", side_effect=fake_publish_targets):
            result = engine.publish_job(
                job_id="j", workdir=Path("/tmp"), source_video=Path("/tmp/x.mp4"),
                targets=targets, progress_cb=lambda *a: None, log_cb=lambda *a: None,
                target_progress_cb=target_cb,
            )
        self.assertEqual(statuses["ig"], "unknown")
        self.assertEqual(set(result["errors"]), {"ig"})
        self.assertIn("AMBIGUOUS_MEDIA_PUBLISH", result["errors"]["ig"])


class ProcessPublishStatusTest(unittest.TestCase):
    def _job(self, job_id, **extra):
        job = {
            "kind": "montage",
            "status": "done",
            "result": {"result_video": "/tmp/does-not-matter.mp4"},
            "video_path": "/tmp/does-not-matter.mp4",
            "params": {},
            "publish_progress": {},
        }
        job.update(extra)
        return job

    def _run(self, job_id, targets, fake_publish_job):
        import app as app_module

        async def fake_announce(*a, **k):
            return None

        try:
            with mock.patch.object(app_module.montage_engine, "publish_job", side_effect=fake_publish_job), \
                 mock.patch.object(app_module, "_montage_announce", side_effect=fake_announce), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None), \
                 mock.patch.object(app_module, "_cleanup_publish_artifacts", lambda *a, **k: {}):
                asyncio.run(app_module.process_publish(job_id, targets))
        finally:
            return app_module.jobs.pop(job_id, None)

    def test_partial_publish_job_status_not_masked(self):
        import app as app_module

        job_id = "test_publish_partial"
        app_module.jobs[job_id] = self._job(job_id)
        targets = [
            {"id": "ig", "kind": "instagram", "label": "Instagram"},
            {"id": "yt", "kind": "youtube", "label": "YouTube"},
        ]

        def fake_publish_job(**kwargs):
            return {
                "results": {"ig": {"kind": "instagram", "id": "ig", "permalink": "https://x/ig"}},
                "errors": {"yt": "quota exceeded"},
            }

        job = self._run(job_id, targets, fake_publish_job)

        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["stage"], "published")
        self.assertEqual(set(job["publish_errors"]), {"yt"})
        self.assertEqual(job["publish_status"]["ig"], "published")
        self.assertEqual(job["publish_status"]["yt"], "failed")
        self.assertIn("ошибк", job["detail"].lower())
        self.assertIn("ig", job["publish"])

    def test_full_publish_has_no_errors(self):
        import app as app_module

        job_id = "test_publish_full"
        app_module.jobs[job_id] = self._job(job_id)
        targets = [{"id": "ig", "kind": "instagram", "label": "Instagram"}]

        def fake_publish_job(**kwargs):
            return {"results": {"ig": {"kind": "instagram", "id": "ig"}}, "errors": {}}

        job = self._run(job_id, targets, fake_publish_job)

        self.assertEqual(job["publish_errors"], {})
        self.assertEqual(job["status"], "done")

    def test_unknown_after_restart_not_auto_retried(self):
        import app as app_module

        job_id = "test_publish_unknown"
        app_module.jobs[job_id] = self._job(job_id)
        targets = [{"id": "ig", "kind": "instagram", "label": "Instagram"}]

        def fake_publish_job(**kwargs):
            return {"results": {}, "errors": {"ig": "AMBIGUOUS_MEDIA_PUBLISH: not confirmed"}}

        job = self._run(job_id, targets, fake_publish_job)

        self.assertEqual(job["publish_status"]["ig"], "unknown")
        self.assertEqual(set(job["publish_unknown"]), {"ig"})
        self.assertEqual(job["publish_errors"], {})
        self.assertIn("проверк", job["detail"].lower())

    def test_refresh_retains_outcomes_and_retries_only_failed(self):
        import app as app_module

        job_id = "test_publish_refresh"
        app_module.jobs[job_id] = self._job(job_id)
        targets = [
            {"id": "ig", "kind": "instagram", "label": "Instagram"},
            {"id": "yt", "kind": "youtube", "label": "YouTube"},
        ]

        def fake_publish_job(**kwargs):
            return {"results": {}, "errors": {"ig": "quota", "yt": "quota"}}

        job = self._run(job_id, targets, fake_publish_job)
        self.assertEqual(set(job["publish_errors"]), {"ig", "yt"})
        self.assertEqual(set(job["publish_status"]), {"ig", "yt"})
        self.assertEqual(set(job["publish_status"].values()), {"failed"})
        self.assertNotIn("ig", job.get("publish_unknown") or {})

        retry_targets = [t for t in targets if t["id"] in job["publish_errors"]]
        app_module.jobs[job_id] = dict(job)
        app_module.jobs[job_id]["id"] = job_id

        def fake_publish_job_success(**kwargs):
            return {
                "results": {t["id"]: {"kind": t["kind"], "id": t["id"], "permalink": "https://x/" + t["id"]} for t in kwargs["targets"]},
                "errors": {},
            }

        def fake_announce(*a, **k):
            return None

        try:
            with mock.patch.object(app_module.montage_engine, "publish_job", side_effect=fake_publish_job_success), \
                 mock.patch.object(app_module, "_montage_announce", side_effect=fake_announce), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None), \
                 mock.patch.object(app_module, "_cleanup_publish_artifacts", lambda *a, **k: {}):
                asyncio.run(app_module.process_publish(job_id, retry_targets))
        finally:
            job = app_module.jobs.pop(job_id, None)

        self.assertEqual(set(job["publish"]), {"ig", "yt"})
        self.assertEqual(job["publish_errors"], {})


if __name__ == "__main__":
    unittest.main()
