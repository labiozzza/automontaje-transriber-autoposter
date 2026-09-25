import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from montage import engine


PROJECT_DIR = Path(__file__).resolve().parent
GENERATOR = PROJECT_DIR / "montage" / "generate_configs_from_transcript_srt.py"


class SubtitleHighlightingTests(unittest.TestCase):
    def test_build_words_srt_keeps_edited_segment_text(self):
        srt = engine.build_words_srt([{
            "start": 0.0,
            "end": 2.0,
            "text": "исправленный пользователем текст",
            "words": [{"start": 0.2, "end": 0.7, "word": "старый"}],
        }])
        self.assertIn("исправленный пользователем текст", srt)
        self.assertNotIn("\nстарый\n", srt)

    def test_word_timing_srt_prefers_nested_word_timestamps(self):
        srt = engine.build_word_timing_srt([{
            "start": 0.0,
            "end": 2.0,
            "text": "деньги совершил",
            "words": [
                {"start": 0.2, "end": 0.7, "word": "деньги"},
                {"start": 0.8, "end": 1.5, "word": "совершил"},
            ],
        }])
        self.assertIn("00:00:00,200 --> 00:00:00,700\nденьги", srt)
        self.assertIn("00:00:00,800 --> 00:00:01,500\nсовершил", srt)

    def test_run_cmd_passes_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log_path = directory / "env.log"
            env = dict(os.environ)
            env["MONTAGE_TEST_OPTION"] = "disabled"
            code = engine.run_cmd(
                [sys.executable, "-c", "import os; print(os.environ['MONTAGE_TEST_OPTION'])"],
                cwd=directory,
                log_path=log_path,
                env=env,
            )
            self.assertEqual(code, 0)
            self.assertEqual(log_path.read_text(encoding="utf-8").strip(), "disabled")

    def test_disabled_bigpickle_produces_only_default_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            srt = directory / "words.srt"
            srt.write_text(
                """1
00:00:00,000 --> 00:00:01,000
убийство

2
00:00:01,000 --> 00:00:02,000
деньги

3
00:00:02,000 --> 00:00:03,000
заработок
""",
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["USE_BIGPICKLE"] = "0"
            result = subprocess.run(
                [sys.executable, str(GENERATOR), str(srt)],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in (
                "subtitle_timeline_config.json",
                "subtitle_timeline_config_words.json",
                "subtitle_timeline_config_phrases.json",
            ):
                payload = json.loads((directory / name).read_text(encoding="utf-8"))
                colors = [
                    token["color"]
                    for subtitle in payload["subtitles"]
                    for token in subtitle["tokens"]
                ]
                self.assertTrue(colors)
                self.assertEqual(set(colors), {"default"})
                self.assertEqual(payload["highlight_rules"]["green_words"], [])
                self.assertEqual(payload["highlight_rules"]["yellow_words"], [])
                self.assertEqual(payload["highlight_rules"]["red_words"], [])


if __name__ == "__main__":
    unittest.main()
