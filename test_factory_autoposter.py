import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import factory_autoposter as autoposter
from montage.instagram_publisher import InstagramPublisher


class FactoryAutoposterTests(unittest.TestCase):
    def test_recovers_media_after_ambiguous_publish_response(self):
        caption = "same caption"
        timestamp = datetime.now(timezone.utc).replace(microsecond=0)

        class FakeModule:
            IG_USER_ID = "account"
            ACCESS_TOKEN = "secret"

            @staticmethod
            def meta_request(graph_ip, method, path, params):
                return {
                    "data": [{
                        "id": "recovered-media-id",
                        "caption": caption,
                        "permalink": "https://example.invalid/recovered",
                        "timestamp": timestamp.isoformat(),
                    }]
                }

        publisher = InstagramPublisher.__new__(InstagramPublisher)
        publisher.main = FakeModule
        publisher.trial = FakeModule
        publisher._tokens = []
        publisher.on_log = lambda message: None
        recovered = publisher.find_recent_publication(
            "trial",
            "127.0.0.1",
            caption,
            timestamp - timedelta(minutes=1),
            timeout=1,
        )
        self.assertEqual(recovered["id"], "recovered-media-id")

    def test_creates_all_combinations_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            connection = autoposter.connect_database(state)
            autoposter.connect_database(state).close()
            count = connection.execute("SELECT COUNT(*) FROM combinations").fetchone()[0]
            connection.close()
            self.assertEqual(count, 125)

    def test_random_queue_is_persistent(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            with patch.object(autoposter.random, "SystemRandom") as randomizer:
                randomizer.return_value.shuffle.side_effect = lambda items: items.reverse()
                connection = autoposter.connect_database(state)
            first = autoposter.next_combination(connection)
            orders = [
                row[0]
                for row in connection.execute(
                    "SELECT queue_order FROM combinations WHERE published = 0"
                ).fetchall()
            ]
            connection.close()
            reopened = autoposter.connect_database(state)
            second = autoposter.next_combination(reopened)
            reopened.close()
            self.assertEqual(first, (5, 5, 5))
            self.assertEqual(second, first)
            self.assertEqual(len(orders), len(set(orders)))

    def test_caption_keeps_category_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            rendered = Path(temporary)
            for category, value in zip(autoposter.CATEGORIES, ("HOOK", "MAIN", "FINAL")):
                directory = rendered / category
                directory.mkdir()
                (directory / "1.txt").write_text(value, encoding="utf-8")
            self.assertEqual(
                autoposter.combination_caption(rendered, (1, 1, 1)),
                "HOOK\n\nMAIN\n\nFINAL",
            )

    def test_marks_published_only_with_media_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            rendered = root / "rendered"
            for category in autoposter.CATEGORIES:
                directory = rendered / category
                directory.mkdir(parents=True)
                (directory / "1.txt").write_text(category, encoding="utf-8")
            connection = autoposter.connect_database(state)
            def fake_concat(_rendered, _combination, destination, progress=None):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"fake")

            fake_result = {
                "results": {
                    "instagram_trial": {
                        "media_id": "123",
                        "permalink": "https://example.invalid/reel/123",
                        "archive_dir": "/archive/123",
                    }
                },
                "errors": {},
            }
            with patch.object(autoposter, "concatenate_videos", fake_concat), patch.object(
                autoposter, "start_progress_overlay"
            ), patch.object(autoposter, "archived_publication", return_value=None):
                result = autoposter.publish_combination(
                    connection,
                    rendered,
                    state,
                    (1, 1, 1),
                    publish_fn=lambda **kwargs: fake_result,
                )
            row = connection.execute(
                "SELECT * FROM combinations WHERE hook=1 AND main=1 AND final=1"
            ).fetchone()
            connection.close()
            self.assertTrue(result["published"])
            self.assertEqual(row["published"], 1)
            self.assertEqual(row["media_id"], "123")
            self.assertEqual(row["archive_result"], "/archive/123")

    def test_missing_media_id_stays_unpublished(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            rendered = root / "rendered"
            for category in autoposter.CATEGORIES:
                directory = rendered / category
                directory.mkdir(parents=True)
                (directory / "1.txt").write_text(category, encoding="utf-8")
            connection = autoposter.connect_database(state)

            def fake_concat(_rendered, _combination, destination, progress=None):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"fake")

            with patch.object(autoposter, "concatenate_videos", fake_concat), patch(
                "factory_autoposter.subprocess.run"
            ), patch.object(autoposter, "start_progress_overlay"), patch.object(
                autoposter, "archived_publication", return_value=None
            ):
                result = autoposter.publish_combination(
                    connection,
                    rendered,
                    state,
                    (1, 1, 1),
                    publish_fn=lambda **kwargs: {"results": {}, "errors": {}},
                )
            published = connection.execute(
                "SELECT published FROM combinations WHERE hook=1 AND main=1 AND final=1"
            ).fetchone()[0]
            checkpoint = autoposter.active_job(connection)
            connection.close()
            self.assertFalse(result["published"])
            self.assertEqual(published, 0)
            self.assertIn("instagram_trial", result["errors"])
            self.assertEqual(checkpoint["stage"], "failed")
            self.assertEqual(checkpoint["retry_delay_minutes"], 1)
            self.assertFalse(autoposter.retry_due(checkpoint))

    def test_retry_delay_can_be_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            connection = autoposter.connect_database(state)
            now = autoposter.iso_now()
            connection.execute(
                """
                INSERT INTO active_job(
                    id, hook, main, final, job_id, workdir, stitched_path,
                    stage, started_at, updated_at
                ) VALUES (1, 1, 1, 1, 'job', '/work', '/work/video.mp4',
                          'failed', ?, ?)
                """,
                (now, now),
            )
            connection.commit()
            delay, retry_at = autoposter.schedule_retry(connection, 17)
            checkpoint = autoposter.active_job(connection)
            connection.close()
            self.assertEqual(delay, 17)
            self.assertEqual(checkpoint["retry_delay_minutes"], 17)
            self.assertEqual(checkpoint["retry_at"], retry_at)

    def test_waiting_progress_contains_next_run_timer(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            connection = autoposter.connect_database(state)
            autoposter.set_setting(connection, "last_publish_attempt_at", autoposter.iso_now())
            connection.close()
            autoposter.update_waiting_progress(state)
            payload = json.loads(
                (state / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "waiting")
            self.assertTrue(payload["next_run_at"])
            self.assertIn("Следующий Trial Reel", payload["title"])

    def test_publish_now_request_bypasses_only_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            rendered = root / "rendered"
            connection = autoposter.connect_database(state)
            autoposter.set_setting(connection, "last_publish_attempt_at", autoposter.iso_now())
            connection.close()
            request = autoposter.request_publish_now(state)

            with patch.object(autoposter, "validate_sources"), patch.object(
                autoposter,
                "publish_combination",
                return_value={"published": True, "media_id": "now"},
            ) as publish:
                result = autoposter.run_once(rendered, state, dry_run=False)

            self.assertTrue(request["requested"])
            self.assertTrue(result["published"])
            publish.assert_called_once()
            self.assertFalse((state / autoposter.PUBLISH_NOW_REQUEST).exists())

    def test_publish_now_rejected_when_job_is_active(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            connection = autoposter.connect_database(state)
            now = autoposter.iso_now()
            connection.execute(
                """
                INSERT INTO active_job(
                    id, hook, main, final, job_id, workdir, stitched_path,
                    stage, started_at, updated_at
                ) VALUES (1, 1, 1, 1, 'job', '/work', '/work/video.mp4',
                          'publishing', ?, ?)
                """,
                (now, now),
            )
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(RuntimeError, "уже выполняется"):
                autoposter.request_publish_now(state)
            self.assertFalse((state / autoposter.PUBLISH_NOW_REQUEST).exists())

    def test_posting_can_be_disabled_and_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            rendered = root / "rendered"
            autoposter.connect_database(state).close()
            self.assertTrue(autoposter.posting_enabled(state))

            disabled = autoposter.set_posting_enabled(state, False)
            with patch.object(autoposter, "validate_sources"):
                paused = autoposter.run_once(rendered, state, dry_run=False)
            with self.assertRaisesRegex(RuntimeError, "выключен"):
                autoposter.request_publish_now(state)

            self.assertFalse(disabled["posting_enabled"])
            self.assertFalse(autoposter.posting_enabled(state))
            self.assertTrue(paused["paused"])

            enabled = autoposter.set_posting_enabled(state, True)
            self.assertTrue(enabled["posting_enabled"])
            self.assertTrue(autoposter.posting_enabled(state))

    def test_interrupted_job_reuses_stitched_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            rendered = root / "rendered"
            for category in autoposter.CATEGORIES:
                directory = rendered / category
                directory.mkdir(parents=True)
                (directory / "1.txt").write_text(category, encoding="utf-8")
            connection = autoposter.connect_database(state)
            concat_calls = []

            def fake_concat(_rendered, _combination, destination, progress=None):
                concat_calls.append(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"fake")

            completed = {
                "results": {
                    "instagram_trial": {
                        "media_id": "resumed-123",
                        "permalink": "https://example.invalid/resumed-123",
                        "archive_dir": "/archive/resumed-123",
                    }
                },
                "errors": {},
            }
            with patch.object(autoposter, "concatenate_videos", fake_concat), patch.object(
                autoposter, "start_progress_overlay"
            ), patch.object(autoposter, "archived_publication", return_value=None), patch.object(
                autoposter.engine, "ffprobe", return_value={"format": {"duration": "1.0"}}
            ):
                with self.assertRaises(KeyboardInterrupt):
                    autoposter.publish_combination(
                        connection,
                        rendered,
                        state,
                        (1, 1, 1),
                        publish_fn=lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
                    )
                checkpoint = autoposter.active_job(connection)
                self.assertEqual(checkpoint["stage"], "publishing")
                self.assertTrue(Path(checkpoint["stitched_path"]).is_file())

                result = autoposter.publish_combination(
                    connection,
                    rendered,
                    state,
                    (1, 1, 1),
                    publish_fn=lambda **kwargs: completed,
                )

            connection.close()
            self.assertTrue(result["published"])
            self.assertEqual(len(concat_calls), 1)

    def test_ambiguous_media_publish_disables_automatic_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            rendered = root / "rendered"
            for category in autoposter.CATEGORIES:
                directory = rendered / category
                directory.mkdir(parents=True)
                (directory / "1.txt").write_text(category, encoding="utf-8")
            connection = autoposter.connect_database(state)

            def fake_concat(_rendered, _combination, destination, progress=None):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"fake")

            ambiguous = {
                "results": {},
                "errors": {
                    "instagram_trial": "AMBIGUOUS_MEDIA_PUBLISH: media_id не подтверждён"
                },
            }
            with patch.object(autoposter, "concatenate_videos", fake_concat), patch.object(
                autoposter, "start_progress_overlay"
            ), patch.object(autoposter, "archived_publication", return_value=None), patch(
                "factory_autoposter.subprocess.run"
            ):
                result = autoposter.publish_combination(
                    connection,
                    rendered,
                    state,
                    (1, 1, 1),
                    publish_fn=lambda **kwargs: ambiguous,
                )
            checkpoint = autoposter.active_job(connection)
            connection.close()
            self.assertFalse(result["published"])
            self.assertEqual(checkpoint["stage"], "ambiguous")
            self.assertIsNone(checkpoint["retry_at"])


if __name__ == "__main__":
    unittest.main()
