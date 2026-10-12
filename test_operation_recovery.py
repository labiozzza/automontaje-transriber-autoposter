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
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["stage"], "interrupted")
        self.assertIsNone(job.get("result"))
        self.assertLess(job["progress"], 100)
        op = job.get("operation") or {}
        self.assertEqual(op.get("status"), "failed")
        self.assertIn("прерван", op.get("error", "").lower())
        self.assertEqual(job.get("detail"), "Монтаж прерван перезапуском сервера")

    def test_interrupted_keeps_completed_stages(self):
        self._write_state(
            operation={
                "version": 3,
                "status": "running",
                "active_stage": "render",
                "progress": 57,
                "stages": [
                    {"key": "transcript", "label": "Стенограмма", "status": "completed", "progress": 100},
                    {"key": "render", "label": "Рендер", "status": "running", "progress": 57},
                    {"key": "audio", "label": "Аудио", "status": "pending", "progress": 0},
                ],
            }
        )
        app_module._recover_montage_jobs()
        job = app_module.jobs.get(self.job_id)
        stages = {s["key"]: s for s in (job.get("operation") or {}).get("stages") or []}
        self.assertEqual(stages["transcript"]["status"], "completed")
        self.assertEqual(stages["render"]["status"], "failed")
        self.assertEqual(stages["audio"]["status"], "skipped")

    def test_interrupted_with_result_file_recovered_as_done(self):
        (self.workdir / "final.mp4").write_bytes(b"fake-video")
        self._write_state(
            status="publishing",
            stage="published",
            detail="Публикация",
            operation={
                "version": 3,
                "status": "running",
                "active_stage": "publish",
                "progress": 88,
                "stages": [{"key": "render", "label": "Рендер", "status": "completed", "progress": 100}],
            },
            publish_tracking=True,
            publish_started={"ig": True},
            publish={"yt": {"kind": "youtube", "video_id": "abc"}},
            publish_progress={
                "ig": {"status": "publishing", "progress": 60, "detail": "Отправка"},
                "yt": {"status": "publishing", "progress": 90, "detail": "Отправка"},
            },
        )
        app_module._recover_montage_jobs()
        job = app_module.jobs.get(self.job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["detail"], "Публикация прервана перезапуском сервера")
        self.assertEqual((job.get("operation") or {}).get("status"), "failed")
        statuses = job.get("publish_status") or {}
        self.assertEqual(statuses.get("yt"), "published")
        self.assertEqual(statuses.get("ig"), "unknown")
        self.assertIn("ig", job.get("publish_unknown") or {})
        self.assertNotIn("ig", job.get("publish_errors") or {})

    def test_unfinished_job_is_persisted_after_recovery(self):
        self._write_state()
        app_module._recover_montage_jobs()
        saved = json.loads((self.workdir / "job_state.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "failed")
        self.assertEqual((saved.get("operation") or {}).get("status"), "failed")

    def test_recover_publish_outcomes_classification(self):
        job = {
            "last_publish_targets": [{"id": "ig"}, {"id": "fb"}, {"id": "yt"}, {"id": "tg"}],
            "publish_tracking": True,
            "publish_started": {"ig": True},
            "publish": {"yt": {"kind": "youtube", "video_id": "abc"}},
            "publish_status": {"fb": "failed"},
            "publish_progress": {
                "ig": {"status": "publishing", "progress": 40, "detail": "Отправка"},
                "fb": {"status": "error", "progress": 100, "detail": "quota"},
                "yt": {"status": "publishing", "progress": 70, "detail": "Отправка"},
            },
        }
        app_module._recover_publish_outcomes(job)
        statuses = job["publish_status"]
        self.assertEqual(statuses["ig"], "unknown")
        self.assertEqual(statuses["fb"], "failed")
        self.assertEqual(statuses["yt"], "published")
        self.assertEqual(statuses["tg"], "failed")
        self.assertIn("ig", job["publish_unknown"])
        self.assertIn("fb", job["publish_errors"])
        self.assertEqual(job["publish_progress"]["ig"]["status"], "unknown")
        self.assertIsNone(job["publish_progress"]["ig"]["progress"])
        self.assertEqual(job["publish_progress"]["yt"]["status"], "done")
        self.assertEqual(job["publish_progress"]["yt"]["progress"], 100)

    def test_recover_publish_outcomes_legacy_is_unknown(self):
        job = {
            "last_publish_targets": [{"id": "ig"}, {"id": "yt"}],
            "publish_progress": {"ig": {"status": "publishing", "progress": 30}},
        }
        app_module._recover_publish_outcomes(job)
        self.assertEqual(job["publish_status"]["ig"], "unknown")
        self.assertEqual(job["publish_status"]["yt"], "unknown")
        self.assertEqual(set(job["publish_unknown"]), {"ig", "yt"})

    def test_completed_job_kept_intact(self):
        self._write_state(status="done", stage="published", operation={"version": 4, "status": "completed", "progress": 100, "stages": []})
        app_module._recover_montage_jobs()
        job = app_module.jobs.get(self.job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["operation"]["status"], "completed")

    def test_restart_at_main_ready_reports_warnings_not_full_readiness(self):
        (self.workdir / "final.mp4").write_bytes(b"v")
        self._write_state(
            status="processing",
            phase="main_ready",
            stage="main_ready",
            detail="Основной ролик готов",
            params={"prepare_publish": True, "prepare_mirrored": True},
            prepare_status="idle",
            mirror_variant=None,
            last_publish_targets=[],
        )
        app_module._recover_montage_jobs()
        job = app_module.jobs.get(self.job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["phase"], "ready_with_warnings")
        joined = " ".join(job["warnings"])
        self.assertIn("Отражённая версия", joined)
        self.assertIn("не завершена", joined)
        self.assertEqual(job["progress"], 100)
        self.assertNotIn((job.get("operation") or {}).get("status"), {"running", "pending"})

    def test_restart_during_mirror_marks_variant_interrupted(self):
        (self.workdir / "final.mp4").write_bytes(b"v")
        self._write_state(
            status="processing",
            phase="rendering_mirror",
            params={"prepare_publish": True, "prepare_mirrored": True},
            prepare_status="idle",
            mirror_variant={
                "status": "generating",
                "operation": {"status": "running", "progress": 30, "stages": []},
            },
            last_publish_targets=[],
        )
        app_module._recover_montage_jobs()
        job = app_module.jobs.get(self.job_id)
        self.assertEqual(job["mirror_variant"]["status"], "error")
        self.assertIn("прерван", job["mirror_variant"]["error"].lower())
        self.assertEqual(job["mirror_variant"]["operation"]["status"], "failed")
        self.assertEqual(job["phase"], "ready_with_warnings")
        self.assertEqual(job["status"], "done")

    def test_restart_during_preparation_marks_prepare_interrupted(self):
        (self.workdir / "final.mp4").write_bytes(b"v")
        self._write_state(
            status="processing",
            phase="preparing_platform_variants",
            params={"prepare_publish": True, "prepare_mirrored": False},
            prepare_status="running",
            prepare_operation={
                "version": 4,
                "status": "running",
                "progress": 50,
                "stages": [
                    {"key": "video:instagram:main", "label": "Видео", "status": "running", "progress": 50},
                ],
            },
            last_publish_targets=[],
        )
        app_module._recover_montage_jobs()
        job = app_module.jobs.get(self.job_id)
        self.assertEqual(job["prepare_status"], "error")
        self.assertIn("прерван", job["prepare_error"].lower())
        self.assertEqual(job["prepare_operation"]["status"], "failed")
        self.assertEqual(job["phase"], "ready_with_warnings")
        self.assertEqual(job["status"], "done")
        running = [s for s in job["prepare_operation"]["stages"] if s["status"] == "running"]
        self.assertEqual(running, [])

    def test_restart_after_completed_preparation_is_clean(self):
        (self.workdir / "final.mp4").write_bytes(b"v")
        self._write_state(
            status="processing",
            phase="verifying_outputs",
            params={"prepare_publish": True, "prepare_mirrored": False},
            prepare_status="done",
            mirror_variant=None,
            last_publish_targets=[],
        )
        app_module._recover_montage_jobs()
        job = app_module.jobs.get(self.job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["phase"], "ready")
        self.assertEqual(job["warnings"], [])
        self.assertEqual(job["progress"], 100)


if __name__ == "__main__":
    unittest.main()
