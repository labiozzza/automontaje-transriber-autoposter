import json
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from montage import opencode_client as oc


FAKE_CLI = r'''
import json, sys, subprocess, time

def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def text(value):
    emit({"type": "text", "part": {"type": "text", "text": value}})

RESULT = {"suggestions": [{"text": "тезис", "start": 0.0, "end": 1.0}]}

args = sys.argv[1:]
model = "fake:success"
for i, a in enumerate(args):
    if a == "-m" and i + 1 < len(args):
        model = args[i + 1]
spec = model.split(":", 1)[1] if ":" in model else model
mode, _, extra = spec.partition(":")

if mode == "success":
    emit({"type": "step_start"})
    text(json.dumps(RESULT, ensure_ascii=False))
    emit({"type": "step_finish", "part": {"reason": "stop"}})
elif mode == "fenced":
    text("```json\n" + json.dumps(RESULT, ensure_ascii=False) + "\n```")
elif mode == "active":
    emit({"type": "step_start"})
    deadline = time.time() + 2.0
    while time.time() < deadline:
        emit({"type": "step"})
        time.sleep(0.15)
    text(json.dumps(RESULT, ensure_ascii=False))
elif mode == "silent":
    time.sleep(30)
elif mode == "forever":
    emit({"type": "step_start"})
    deadline = time.time() + 30
    while time.time() < deadline:
        emit({"type": "step"})
        time.sleep(0.1)
elif mode == "badjson":
    text("это совсем не JSON")
elif mode == "nonzero":
    sys.stderr.write("boom\n")
    sys.exit(3)
elif mode == "error_event":
    emit({"type": "error", "error": {"name": "FreeTierError", "data": {
        "message": "free tier disabled", "isRetryable": False}}})
    time.sleep(0.4)
elif mode == "huge":
    emit({"type": "step_start"})
    chunk = "".join(json.dumps({"type": "tick", "i": i}) + "\n" for i in range(2000))
    for _ in range(80):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    text(json.dumps(RESULT, ensure_ascii=False))
elif mode == "child":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with open(extra, "w") as fh:
            fh.write(str(child.pid))
    except Exception:
        pass
    time.sleep(30)
else:
    text(json.dumps(RESULT, ensure_ascii=False))

sys.exit(0)
'''


class FakeOpenCodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.bin = Path(cls._tmp.name) / "fake_opencode"
        cls.bin.write_text("#!" + sys.executable + "\n" + FAKE_CLI, encoding="utf-8")
        cls.bin.chmod(cls.bin.stat().st_mode | stat.S_IEXEC)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _ask(self, mode, **kwargs):
        with mock.patch.dict(os.environ, {"OPENCODE_BIN": str(self.bin)}):
            return oc.ask_json("prompt", model="fake:" + mode, **kwargs)

    def test_fast_success_emits_ordered_progress(self):
        events = []
        result = self._ask("success", progress_callback=events.append)
        self.assertEqual(result, {"suggestions": [{"text": "тезис", "start": 0.0, "end": 1.0}]})
        self.assertEqual(
            [event["event"] for event in events],
            ["attempt_started", "response_received", "response_parsed"],
        )

    def test_markdown_fenced_json_is_parsed(self):
        result = self._ask("fenced")
        self.assertEqual(result["suggestions"][0]["text"], "тезис")

    def test_long_active_run_survives_past_idle_limit(self):
        events = []
        result = self._ask(
            "active", idle_timeout=0.6, hard_timeout=8, total_budget=20, progress_callback=events.append,
        )
        self.assertIn("suggestions", result)
        names = [event["event"] for event in events]
        self.assertIn("activity", names)
        self.assertNotIn("attempt_idle_timeout", names)

    def test_silent_process_hits_idle_timeout_without_retry(self):
        events = []
        with self.assertRaises(oc.NoActivityError):
            self._ask("silent", idle_timeout=0.4, hard_timeout=5, retries=0, progress_callback=events.append)
        self.assertEqual([event["event"] for event in events], ["attempt_started", "attempt_idle_timeout"])

    def test_constant_activity_hits_hard_timeout(self):
        with self.assertRaises(oc.HardTimeoutError):
            self._ask("forever", idle_timeout=10, hard_timeout=0.8, retries=0)

    def test_invalid_json_retried_once_then_fails(self):
        events = []
        with self.assertRaises(oc.InvalidJSONError):
            self._ask("badjson", idle_timeout=2, hard_timeout=5, retries=1, progress_callback=events.append)
        names = [event["event"] for event in events]
        self.assertEqual(names.count("attempt_started"), 2)
        self.assertIn("attempt_invalid_json", names)
        self.assertIn("retry_scheduled", names)

    def test_nonzero_exit_raises_cli_error(self):
        with self.assertRaises(oc.CliError):
            self._ask("nonzero", retries=0)

    def test_non_retryable_error_event_is_not_retried(self):
        events = []
        with self.assertRaises(oc.CliError) as ctx:
            self._ask("error_event", retries=1, progress_callback=events.append)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual([event["event"] for event in events].count("attempt_started"), 1)

    def test_huge_event_stream_is_capped_and_parsed(self):
        result = self._ask("huge", idle_timeout=5, hard_timeout=30)
        self.assertIn("suggestions", result)

    def test_timeout_kills_child_process_group(self):
        pidfile = Path(self._tmp.name) / "child.pid"
        with self.assertRaises(oc.NoActivityError):
            self._ask("child:" + str(pidfile), idle_timeout=0.5, hard_timeout=5, retries=0)
        self.assertTrue(pidfile.exists(), "child pid file was not written")
        pid = int(pidfile.read_text().strip())
        alive = True
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                alive = False
                break
            time.sleep(0.1)
        self.assertFalse(alive, "child process survived the timeout kill")


if __name__ == "__main__":
    unittest.main()
