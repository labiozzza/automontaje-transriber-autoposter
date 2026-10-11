import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from montage.progress_tracker import (
    CANCELLED,
    COMPLETED,
    PENDING,
    RUNNING,
    SKIPPED,
    ProgressTracker,
    Stage,
    montage_plan,
)


def make_tracker(*weighted):
    stages = [Stage(f"s{i}", f"Стадия {i}", weight) for i, weight in enumerate(weighted)]
    return ProgressTracker("op-1", "test", stages, label="Тест")


class ProgressTrackerTest(unittest.TestCase):
    def test_plan_excludes_disabled_stages(self):
        plan = montage_plan({
            "face_tracking": False,
            "autozoom": False,
            "blur_banned_words": False,
            "mirror_horizontal": False,
            "subtitles": True,
        })
        keys = [s.key for s in plan]
        self.assertIn("subtitles", keys)
        self.assertIn("finalize", keys)
        self.assertNotIn("faces", keys)
        self.assertNotIn("zoom", keys)
        self.assertNotIn("audio_censor", keys)
        self.assertNotIn("mirror", keys)

    def test_plan_includes_enabled_stages(self):
        plan = montage_plan({
            "face_tracking": True,
            "autozoom": True,
            "blur_banned_words": True,
            "mirror_horizontal": True,
            "subtitles": True,
        })
        keys = {s.key for s in plan}
        for expected in ("prepare", "mirror", "configs", "faces", "zoom",
                         "subtitles", "audio_censor", "finalize", "preview", "verify"):
            self.assertIn(expected, keys)

    def test_aggregate_is_weighted(self):
        tracker = make_tracker(50, 50)
        tracker.complete("s0")
        self.assertEqual(tracker.overall(), 50.0)
        tracker.complete("s1")
        self.assertEqual(tracker.overall(), 100.0)

    def test_aggregate_never_goes_back_when_unfinished(self):
        tracker = make_tracker(50, 50)
        tracker.update("s0", 80)
        first = tracker.overall()
        tracker.update("s0", 60)
        self.assertEqual(tracker.overall(), first)

    def test_reset_allows_backward_move(self):
        tracker = make_tracker(50, 50)
        tracker.update("s0", 80)
        before = tracker.overall()
        tracker.reset_stage("s0")
        after = tracker.overall()
        self.assertLess(after, before)

    def test_100_reserved_until_finished(self):
        tracker = make_tracker(50, 50)
        tracker.complete("s0")
        tracker.update("s1", 100)
        self.assertLess(tracker.overall(), 100.0)
        tracker.complete("s1")
        tracker.finish()
        self.assertEqual(tracker.overall(), 100.0)

    def test_indeterminate_stage_flagged(self):
        tracker = make_tracker(100)
        tracker.start("s0")
        snap = tracker.snapshot()
        self.assertTrue(snap["indeterminate"])
        tracker.update("s0", 50)
        self.assertFalse(tracker.snapshot()["indeterminate"])

    def test_skipped_stage_excluded_from_denominator(self):
        tracker = make_tracker(80, 20)
        tracker.skip("s0")
        tracker.complete("s1")
        self.assertEqual(tracker.overall(), 100.0)

    def test_failed_stage_keeps_partial_progress(self):
        tracker = make_tracker(80, 20)
        tracker.update("s0", 45)
        tracker.fail("s0", "boom")
        self.assertEqual(tracker.status, "failed")
        snap = tracker.snapshot()
        self.assertEqual(snap["error"], "boom")
        self.assertAlmostEqual(snap["progress"], 0.8 * 45)

    def test_active_stage_reported(self):
        tracker = make_tracker(50, 50)
        tracker.start("s1", "работаем")
        snap = tracker.snapshot()
        self.assertEqual(snap["active_stage"], "s1")
        self.assertEqual(snap["detail"], "работаем")

    def test_snapshot_shape(self):
        tracker = make_tracker(100)
        tracker.update("s0", 33, "деталь")
        snap = tracker.snapshot()
        self.assertEqual(snap["operation_id"], "op-1")
        self.assertEqual(snap["kind"], "test")
        stage = snap["stages"][0]
        for field in ("key", "label", "weight", "status", "progress",
                      "indeterminate", "detail", "error", "parent"):
            self.assertIn(field, stage)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
class FfmpegProgressTest(unittest.TestCase):
    def test_run_ffmpeg_progress_reports_real_percent(self):
        from montage.engine import run_ffmpeg_progress

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "src.mp4"
            result = subprocess.run(
                [shutil.which("ffmpeg"), "-y", "-f", "lavfi", "-i",
                 "color=c=red:s=320x240:d=2",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
                capture_output=True)
            self.assertEqual(result.returncode, 0)
            duration = 2.0
            out = tmp / "out.mp4"
            values = []

            def on_progress(pct):
                values.append(pct)

            code = run_ffmpeg_progress(
                [shutil.which("ffmpeg"), "-y", "-i", str(source),
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", str(out)],
                duration=duration, on_progress=on_progress,
                log_path=tmp / "enc.log", timeout=120)
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())
            self.assertTrue(values)
            self.assertGreaterEqual(values[-1], 99.0)


class MontagePlanSerializationTest(unittest.TestCase):
    def test_snapshot_json_serializable(self):
        tracker = make_tracker(10)
        tracker.update("s0", 50)
        json.dumps(tracker.snapshot())


if __name__ == "__main__":
    unittest.main()