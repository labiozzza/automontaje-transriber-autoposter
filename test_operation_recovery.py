"""Server-side operation state recovery tests (spec 1.2 / 4: interrupted jobs
are not advertised as running after a restart, errors are coherent and not
masked).

Run:
  python -m unittest test_operation_recovery
"""
import json
import shutil
import unittest

import app as app_module


class OperationRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.job_id = "recover_test_job"
        self.workdir = app_module.MONTAGE_WORK / self.job_id
        self.workdir.mkdir(parents=True, exist_ok=True)
        app_module.jobs.pop(self.job_id, None)

    def tearDown(self):
        app_module.jobs.pop(self.job_id, None)
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _write_state(self, **overrides):
        job = {
            "id": self.job_id,
            "kind": "montage",
            "status": "processing",
            "stage": "render",
            "detail": "Рендер",
            "result": {"result_video": str(self.workdir / "final.mp4")},
            "operation": {
                "version": 3,
                "status": "running",
                "active_stage": "render",
                "progress": 40,
                "stages": [{"key": "render", "label": "Рендер", "status": "running", "progress": 40}],
            },
            "last_publish_targets": [{"id": "ig"}, {"id": "yt"}],
        }
        job.update(overrides)
        (self.workdir / "job_state.json").write_text(json.dumps(job), encoding="utf-8")

    def test_mark_operation_failed_is_coherent(self):
        app_module.jobs[self.job_id] = {
            "kind": "montage",
            "operation": {"version": 1, "status": "running", "progress": 55,
                          "stages": [{"key": "render", "label": "Рендер", "status": "running", "progress": 55}]},
        }
        snap = app_module.mark_operation_failed(self.job_id, "boom")
        self.assertIsNotNone(snap)
        self.assertEqual(snap["status"], "failed")
        self.assertEqual(snap["error"], "boom")
        running = [s for s in snap["stages"] if s["status"] == "running"]
        self.assertEqual(running, [])
        self.assertEqual(app_module.jobs[self.job_id]["operation"]["status"], "failed")

    def test_interrupted_job_recovered_as_failed_not_running(self):
        self._write_state()
        app_module._recover_montage_jobs()
        job = app_module.jobs.get(self.job_id)
        self.assertIsNotNone(job, "job was not recovered")
        self.assertNotIn(job["status"], {"processing", "publishing"})
        self.assertEqual(job["status"], "done")
        op = job.get("operation") or {}
        self.assertEqual(op.get("status"), "failed")
        self.assertIn("прерван", op.get("error", "").lower())
        self.assertEqual(set(job.get("publish_errors") or {}), {"ig", "yt"})

    def test_unfinished_job_is_persisted_after_recovery(self):
        self._write_state()
        app_module._recover_montage_jobs()
        saved = json.loads((self.workdir / "job_state.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "done")
        self.assertEqual((saved.get("operation") or {}).get("status"), "failed")

    def test_completed_job_kept_intact(self):
        self._write_state(status="done", stage="published", operation={"version": 4, "status": "completed", "progress": 100, "stages": []})
        app_module._recover_montage_jobs()
        job = app_module.jobs.get(self.job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["operation"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
