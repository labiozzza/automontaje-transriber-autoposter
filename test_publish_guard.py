"""Server-side guards against duplicate publications (task 5): the publish
endpoint must reject re-publication of platforms with an unknown outcome
unless confirm_unknown=true (and never leak one confirmation to later
requests), and per-job atomic publishing with no two background publishes for
the same job_id.

Run:
  python -m unittest test_publish_guard
"""
import asyncio
import json
import unittest
from unittest import mock

import httpx

import app as app_module

TARGET = "instagram_trial"


def make_job(job_id, **extra):
    job = {
        "kind": "montage",
        "status": "done",
        "stage": "done",
        "progress": 100,
        "result": {"result_video": "/tmp/does-not-matter.mp4"},
        "video_path": "/tmp/does-not-matter.mp4",
        "params": {},
        "publish_progress": {},
        "publish_tracking": True,
    }
    job.update(extra)
    return job


class PublishEndpointGuardTest(unittest.TestCase):
    def setUp(self):
        app_module.jobs.clear()
        app_module.publish_locks.clear()

    @staticmethod
    def _client():
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_module.app), base_url="http://t"
        )

    def _run(self, coro):
        return asyncio.run(coro)

    def test_unknown_without_confirm_is_409_and_no_task(self):
        async def scenario():
            job_id = "guard_unknown_no_confirm"
            app_module.jobs[job_id] = make_job(
                job_id,
                publish_status={TARGET: "unknown"},
                publish_unknown={TARGET: "Результат неизвестен"},
                last_publish_targets=[{"id": TARGET, "kind": "trial", "label": "ig"}],
            )
            state_before = json.loads(json.dumps(app_module.jobs[job_id]))
            calls = []

            async def fake(job_id_, targets, publish_lock=None):
                calls.append(job_id_)

            with mock.patch.object(app_module, "process_publish", side_effect=fake), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
                async with self._client() as client:
                    resp = await client.post(
                        f"/api/montage/{job_id}/publish",
                        data={"targets_json": json.dumps([TARGET])},
                    )
            self.assertEqual(resp.status_code, 409)
            self.assertEqual(resp.json()["confirm_targets"], [TARGET])
            self.assertEqual(calls, [])
            self.assertEqual(app_module.jobs[job_id], state_before)

        self._run(scenario())

    def test_unknown_with_confirm_allows_one_run(self):
        async def scenario():
            job_id = "guard_unknown_confirm"
            app_module.jobs[job_id] = make_job(
                job_id,
                publish_status={TARGET: "unknown"},
                publish_unknown={TARGET: "Результат неизвестен"},
            )
            calls = []

            async def fake(job_id_, targets, publish_lock=None):
                calls.append((job_id_, [t["id"] for t in targets]))

            with mock.patch.object(app_module, "process_publish", side_effect=fake), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
                async with self._client() as client:
                    resp = await client.post(
                        f"/api/montage/{job_id}/publish",
                        data={"targets_json": json.dumps([TARGET]), "confirm_unknown": "true"},
                    )
                    self.assertEqual(resp.status_code, 200)
                    await asyncio.sleep(0)
                    self.assertEqual(calls, [(job_id, [TARGET])])
                    # the confirmation is not sticky: the very same request
                    # again must still demand a fresh confirmation
                    resp2 = await client.post(
                        f"/api/montage/{job_id}/publish",
                        data={"targets_json": json.dumps([TARGET])},
                    )
            self.assertEqual(resp2.status_code, 409)
            self.assertEqual(calls, [(job_id, [TARGET])])

        self._run(scenario())

    def test_new_publication_does_not_bypass_unknown_confirm(self):
        async def scenario():
            job_id = "guard_unknown_new_pub"
            app_module.jobs[job_id] = make_job(
                job_id,
                publish_status={TARGET: "unknown"},
                publish_unknown={TARGET: "Результат неизвестен"},
            )
            calls = []

            async def fake(job_id_, targets, publish_lock=None):
                calls.append(job_id_)

            with mock.patch.object(app_module, "process_publish", side_effect=fake), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
                async with self._client() as client:
                    resp = await client.post(
                        f"/api/montage/{job_id}/publish",
                        data={"targets_json": json.dumps([TARGET]), "new_publication": "true"},
                    )
            self.assertEqual(resp.status_code, 409)
            self.assertEqual(resp.json()["confirm_targets"], [TARGET])
            self.assertEqual(calls, [])

        self._run(scenario())

    def test_confirmed_published_target_not_repeated_in_normal_mode(self):
        async def scenario():
            job_id = "guard_published_no_repeat"
            app_module.jobs[job_id] = make_job(
                job_id,
                publish_status={TARGET: "published"},
                publish={TARGET: {"permalink": "https://x/1"}},
            )
            calls = []

            async def fake(job_id_, targets, publish_lock=None):
                calls.append(job_id_)

            with mock.patch.object(app_module, "process_publish", side_effect=fake), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
                async with self._client() as client:
                    normal = await client.post(
                        f"/api/montage/{job_id}/publish",
                        data={"targets_json": json.dumps([TARGET])},
                    )
                    self.assertEqual(normal.status_code, 400)
                    self.assertEqual(calls, [])
                    published_again = await client.post(
                        f"/api/montage/{job_id}/publish",
                        data={"targets_json": json.dumps([TARGET]), "new_publication": "true"},
                    )
                    self.assertEqual(published_again.status_code, 200)
                    await asyncio.sleep(0)
                    self.assertEqual(calls, ["guard_published_no_repeat"])

        self._run(scenario())

    def test_retry_after_error_is_allowed(self):
        async def scenario():
            job_id = "guard_retry_error"
            app_module.jobs[job_id] = make_job(
                job_id,
                publish_status={TARGET: "failed"},
                publish_errors={TARGET: "quota exceeded"},
            )
            calls = []

            async def fake(job_id_, targets, publish_lock=None):
                calls.append([t["id"] for t in targets])

            with mock.patch.object(app_module, "process_publish", side_effect=fake), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
                async with self._client() as client:
                    resp = await client.post(
                        f"/api/montage/{job_id}/publish",
                        data={"targets_json": json.dumps([TARGET])},
                    )
            self.assertEqual(resp.status_code, 200)
            await asyncio.sleep(0)
            self.assertEqual(calls, [[TARGET]])

        self._run(scenario())

    def test_two_concurrent_requests_start_exactly_one_publish(self):
        async def scenario():
            job_id = "guard_concurrent"
            app_module.jobs[job_id] = make_job(job_id)
            calls = []
            release = asyncio.Event()
            started = asyncio.Event()

            async def fake(job_id_, targets, publish_lock=None):
                calls.append(job_id_)
                started.set()
                await release.wait()

            with mock.patch.object(app_module, "process_publish", side_effect=fake), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
                async with self._client() as client:
                    results = await asyncio.gather(
                        client.post(f"/api/montage/{job_id}/publish",
                                    data={"targets_json": json.dumps([TARGET])}),
                        client.post(f"/api/montage/{job_id}/publish",
                                    data={"targets_json": json.dumps([TARGET])}),
                    )
            self.assertEqual(sorted((r.status_code for r in results)), [200, 409])
            ok = next(r for r in results if r.status_code == 200)
            conflict = next(r for r in results if r.status_code == 409)
            self.assertEqual(conflict.json()["detail"], "Публикация уже выполняется для этого ролика")
            # the winning request reserved the job and saw its background task start
            self.assertEqual(calls, [job_id])
            self.assertIn("last_publish_targets", app_module.jobs[job_id])
            state_after = json.loads(json.dumps(app_module.jobs[job_id]))
            # a fresh 409 (lock held) must not mutate the job again
            async with self._client() as client:
                third = await client.post(f"/api/montage/{job_id}/publish",
                                          data={"targets_json": json.dumps([TARGET])})
            self.assertEqual(third.status_code, 409)
            self.assertEqual(app_module.jobs[job_id], state_after)
            self.assertEqual(calls, [job_id])
            await started.wait()
            release.set()
            await asyncio.sleep(0)

        self._run(scenario())

    def test_two_different_jobs_publish_independently(self):
        async def scenario():
            app_module.jobs["job_a"] = make_job("job_a")
            app_module.jobs["job_b"] = make_job("job_b")
            calls = []

            async def fake(job_id_, targets, publish_lock=None):
                calls.append(job_id_)
                return None

            with mock.patch.object(app_module, "process_publish", side_effect=fake), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None):
                async with self._client() as client:
                    results = await asyncio.gather(
                        client.post("/api/montage/job_a/publish",
                                    data={"targets_json": json.dumps([TARGET])}),
                        client.post("/api/montage/job_b/publish",
                                    data={"targets_json": json.dumps([TARGET])}),
                    )
            self.assertTrue(all(r.status_code == 200 for r in results))
            await asyncio.sleep(0)
            self.assertEqual(sorted(calls), ["job_a", "job_b"])

        self._run(scenario())

    def test_exception_releases_the_lock(self):
        async def scenario():
            job_id = "guard_exception"
            app_module.jobs[job_id] = make_job(job_id)

            def fake_announce(*a, **k):
                return None

            async def fake_broadcast(*a, **k):
                return None

            def fake_publish_job(**kwargs):
                raise RuntimeError("mock publish boom")

            with mock.patch.object(app_module.montage_engine, "publish_job", side_effect=fake_publish_job), \
                 mock.patch.object(app_module, "_montage_announce", side_effect=fake_announce), \
                 mock.patch.object(app_module, "broadcast_progress", side_effect=fake_broadcast), \
                 mock.patch.object(app_module, "_save_montage_job", lambda *a, **k: None), \
                 mock.patch.object(app_module, "_cleanup_publish_artifacts", lambda *a, **k: {}):
                async with self._client() as client:
                    resp = await client.post(
                        f"/api/montage/{job_id}/publish",
                        data={"targets_json": json.dumps([TARGET])},
                    )
                    self.assertEqual(resp.status_code, 200)
                    for _ in range(200):
                        await asyncio.sleep(0)
                        if not app_module._publish_lock(job_id).locked():
                            break
            self.assertFalse(app_module._publish_lock(job_id).locked())
            self.assertEqual(app_module.jobs[job_id]["status"], "error")

        self._run(scenario())


if __name__ == "__main__":
    unittest.main()