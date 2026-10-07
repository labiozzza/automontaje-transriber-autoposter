import json
import shutil
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from montage.drawing_generator import (
    classify_request,
    generate_drawings,
    generate_smart_drawings,
    parse_srt_segments,
    revise_drawing,
    validate_drawing_overlay,
)
from montage import drawing_generator as app_chart
from montage.apply_timeline_animation import render_drawing
import app


class DrawingGeneratorTests(unittest.TestCase):
    def test_validates_freeform_drawing(self):
        drawing = validate_drawing_overlay({
            "name": "Собака у миски",
            "start": 2,
            "end": 6,
            "draw_speed": 420,
            "drawing": {
                "width": 1000,
                "height": 800,
                "paths": [{
                    "points": [[10, 20], [200, 120], [1100, -5]],
                    "stroke": "#aabbcc",
                    "stroke_width": 9,
                }],
            },
        })
        self.assertEqual(drawing["kind"], "drawing")
        self.assertEqual(drawing["drawing"]["paths"][0]["points"][-1], [1000, 0.0])
        self.assertEqual(drawing["drawing"]["paths"][0]["stroke"], "#AABBCC")

    def test_rejects_markup_instead_of_paths(self):
        with self.assertRaises(ValueError):
            validate_drawing_overlay({"drawing": {"html": "<script>alert(1)</script>"}})

    def test_renderer_reveals_lines_progressively_on_transparent_background(self):
        drawing = {
            "width": 100,
            "height": 100,
            "paths": [{"points": [[10, 50], [90, 50]], "stroke": "#FFFFFF", "stroke_width": 6}],
        }
        empty = render_drawing(drawing, 100, 0, 100)
        partial = render_drawing(drawing, 100, 0.3, 100)
        complete = render_drawing(drawing, 100, 1.0, 100)
        self.assertIsNone(empty.getchannel("A").getbbox())
        self.assertLess(partial.getchannel("A").getbbox()[2], complete.getchannel("A").getbbox()[2])
        self.assertEqual(complete.getpixel((50, 50))[:3], (255, 255, 255))

    def test_generation_endpoint_is_local_and_returns_validated_proposals(self):
        class Request:
            client = SimpleNamespace(host="127.0.0.1")
            headers = {"origin": "http://127.0.0.1:8000"}

            async def json(self):
                return {
                    "srt_text": "1\n00:00:00,000 --> 00:00:01,000\nСобака ест.\n",
                    "selected_fragments": [{"segment_index": 0, "start": 0, "end": 1, "text": "Собака ест."}],
                }

        proposal = validate_drawing_overlay({
            "drawing": {"paths": [{"points": [[0, 0], [10, 10]]}]},
        })
        with mock.patch.object(app.drawing_generator, "generate_drawings", return_value=[proposal]):
            response = __import__("asyncio").run(app.generate_montage_drawings(Request()))
        self.assertEqual(response["drawings"][0]["kind"], "drawing")

        Request.client = SimpleNamespace(host="192.168.1.10")
        with self.assertRaises(HTTPException) as raised:
            __import__("asyncio").run(app.generate_montage_drawings(Request()))
        self.assertEqual(raised.exception.status_code, 403)

    def test_bigpickle_draws_only_selected_targets(self):
        response = {"drawings": [{
            "target_id": "4",
            "name": "Стикмен с монетой",
            "start": 99,
            "end": 100,
            "drawing": {"paths": [{"points": [[10, 10], [20, 20]]}]},
        }, {
            "target_id": "unselected",
            "drawing": {"paths": [{"points": [[0, 0], [10, 10]]}]},
        }]}
        with mock.patch("montage.drawing_generator.ask_json", return_value=response) as ask:
            drawings = generate_drawings(
                "Полный контекст транскрипции",
                [{"target_id": "4", "start": 2, "end": 3, "text": "за деньги"}],
            )
        self.assertEqual(len(drawings), 1)
        self.assertEqual((drawings[0]["start"], drawings[0]["end"]), (2.0, 3.0))
        self.assertEqual(drawings[0]["trigger_text"], "за деньги")
        self.assertEqual(ask.call_args.kwargs["model"], "opencode/big-pickle")
        self.assertIn("Полный контекст транскрипции", ask.call_args.args[0])


    def test_endpoint_sends_edited_and_manual_replicas(self):
            app.jobs["job-edits"] = {"status": "done", "result": [
                {"start": 0.0, "end": 8.0, "text": "Оригинальная реплика"},
            ]}

            class Request:
                client = SimpleNamespace(host="127.0.0.1")
                headers = {"origin": "http://127.0.0.1:8000"}

                async def json(self):
                    return {
                        "transcript_job_id": "job-edits",
                        "selected_fragments": [
                            {"segment_index": 0, "start": 0.5, "end": 1.5, "text": "Исправленная реплика"},
                            {"start": 5, "end": 7, "text": "Новая добавленная реплика"},
                        ],
                    }

            proposal = validate_drawing_overlay({"drawing": {"paths": [{"points": [[0, 0], [10, 10]]}]}})
            with mock.patch.object(app.drawing_generator, "generate_drawings", return_value=[proposal, proposal]) as generate:
                response = __import__("asyncio").run(app.generate_montage_drawings(Request()))
            fragments = generate.call_args.args[1]
            self.assertEqual(fragments, [
                {"target_id": "0", "start": 0.5, "end": 1.5, "text": "Исправленная реплика"},
                {"target_id": "manual-1", "start": 5.0, "end": 7.0, "text": "Новая добавленная реплика"},
            ])
            self.assertEqual(len(response["drawings"]), 2)

    def test_endpoint_rejects_empty_replica_text(self):
        class Request:
            client = SimpleNamespace(host="127.0.0.1")
            headers = {"origin": "http://127.0.0.1:8000"}

            async def json(self):
                return {
                    "srt_text": "1\n00:00:00,000 --> 00:00:01,000\nКу-ку.\n",
                    "selected_fragments": [{"start": 0, "end": 1, "text": "   "}],
                }

        with self.assertRaises(HTTPException) as raised:
            __import__("asyncio").run(app.generate_montage_drawings(Request()))
        self.assertEqual(raised.exception.status_code, 400)

    def test_endpoint_rejects_invalid_replica_time(self):
        class Request:
            client = SimpleNamespace(host="127.0.0.1")
            headers = {"origin": "http://127.0.0.1:8000"}

            async def json(self):
                return {
                    "srt_text": "1\n00:00:00,000 --> 00:00:01,000\nКу-ку.\n",
                    "selected_fragments": [{"start": 5, "end": 2, "text": "Реплика"}],
                }

        with self.assertRaises(HTTPException) as raised:
            __import__("asyncio").run(app.generate_montage_drawings(Request()))
        self.assertEqual(raised.exception.status_code, 400)

    def test_endpoint_rejects_more_than_twelve_replicas(self):
        class Request:
            client = SimpleNamespace(host="127.0.0.1")
            headers = {"origin": "http://127.0.0.1:8000"}

            async def json(self):
                return {
                    "srt_text": "1\n00:00:00,000 --> 00:00:01,000\nКу-ку.\n",
                    "selected_fragments": [{"start": 0, "end": 1, "text": f"Реплика {index}"} for index in range(13)],
                }

        with self.assertRaises(HTTPException) as raised:
            __import__("asyncio").run(app.generate_montage_drawings(Request()))
        self.assertEqual(raised.exception.status_code, 400)


class ChartIntentTests(unittest.TestCase):
    def test_classify_request(self):
        self.assertEqual(classify_request("Возрастающий график вертикали клиента горизонталь контент"), "chart")
        self.assertEqual(classify_request("Покажи процесс: шаг 1 и шаг 2"), "process")
        self.assertEqual(classify_request("Сравнение до и после"), "comparison")
        self.assertEqual(classify_request("Карточка с числом подписчиков"), "card")
        self.assertEqual(classify_request("Схема: этапы воронки"), "abstract")
        self.assertEqual(classify_request("Человек идёт за деньгами"), "scene")

    def test_parse_chart_spec_exact_query(self):
        spec = app_chart.parse_chart_spec("Возрастающий график вертикали клиента горизонталь контент")
        self.assertEqual(spec["axes"]["y"], "клиента")
        self.assertEqual(spec["axes"]["x"], "контент")
        self.assertEqual(spec["trend"], "up")
        self.assertEqual(spec["kind"], "line")

    def test_parse_chart_spec_down_and_bar(self):
        spec = app_chart.parse_chart_spec("Убывающая гистограмма по вертикали расходов и по горизонтали месяцам")
        self.assertEqual(spec["trend"], "down")
        self.assertEqual(spec["kind"], "bar")

    def test_chart_viz_for_request(self):
        viz = app_chart.chart_viz_for_request("Возрастающий график вертикали подписчиков горизонталь время")
        self.assertEqual(viz["type"], "chart")
        self.assertEqual(viz["trend"], "up")
        self.assertEqual(viz["axes"]["y"], "подписчиков")
        self.assertIsNone(app_chart.chart_viz_for_request("Просто карточка с числом"))

    def test_validate_drawing_overlay_viz_sanitized(self):
        drawing = validate_drawing_overlay({
            "start": 1, "end": 3, "trigger_text": "Текст",
            "viz": {"type": "chart", "axes": {"x": "КОНТЕНТ", "y": "КЛИЕНТЫ"},
                    "kind": "bar", "trend": "flat"},
            "drawing": {"width": 1000, "height": 1000,
                        "paths": [{"points": [[140, 880], [500, 520], [860, 200]],
                                   "stroke": "#55D66B", "stroke_width": 14}]},
        })
        self.assertEqual(drawing["viz"]["type"], "chart")
        self.assertEqual(drawing["viz"]["axes"]["y"], "КЛИЕНТЫ")
        self.assertEqual(drawing["viz"]["kind"], "bar")
        self.assertEqual(drawing["viz"]["trend"], "flat")

    def test_validate_drawing_overlay_rejects_bad_viz(self):
        drawing = validate_drawing_overlay({
            "start": 1, "end": 3, "trigger_text": "Текст",
            "viz": {"type": "video"}, "drawing": {"width": 1000, "height": 1000,
                                                  "paths": [{"points": [[0, 0], [1, 1]]}]},
        })
        self.assertIsNone(drawing["viz"])

    def test_render_drawing_draws_chart_axes(self):
        drawing = validate_drawing_overlay({
            "start": 1, "end": 3, "trigger_text": "Текст",
            "viz": {"type": "chart", "axes": {"x": "Контент", "y": "Клиенты"},
                    "kind": "line", "trend": "up"},
            "drawing": {"width": 1000, "height": 1000,
                        "paths": [{"points": [[140, 880], [500, 520], [860, 200]],
                                   "stroke": "#55D66B", "stroke_width": 14}]},
        })
        image = render_drawing(drawing, 300, 9999.0)
        self.assertEqual(image.size, (300, 300))
        axis_pixel = image.getpixel((int(round(120 * 300 / 1000)), int(round(920 * 300 / 1000))))
        self.assertGreater(axis_pixel[3], 0, "ось X должна рисоваться белым маркером")

    def test_viz_embedded_inside_drawing_payload(self):
        drawing = validate_drawing_overlay({
            "start": 1, "end": 3, "trigger_text": "Текст",
            "viz": {"type": "chart", "axes": {"x": "Контент", "y": "Клиенты"},
                    "kind": "bar", "trend": "up"},
            "drawing": {"width": 1000, "height": 1000,
                        "paths": [{"points": [[140, 880], [500, 520], [860, 200]]}]},
        })
        self.assertEqual(drawing["drawing"]["viz"], drawing["viz"])
        self.assertEqual(drawing["drawing"]["viz"]["type"], "chart")


class DrawingRevisionTests(unittest.TestCase):
    def _base_overlay(self):
        return validate_drawing_overlay({
            "start": 2, "end": 5, "x": 0.4, "y": 0.3, "size": 500, "draw_speed": 900,
            "trigger_text": "Исходная реплика", "name": "График клиентов",
            "viz": {"type": "chart", "axes": {"x": "Контент", "y": "Клиенты"},
                    "kind": "line", "trend": "up"},
            "drawing": {"width": 1000, "height": 1000,
                        "paths": [{"points": [[140, 880], [500, 520], [860, 200]],
                                   "stroke": "#55D66B", "stroke_width": 14}]},
        })

    def test_revise_drawing_replaces_content_preserving_meta(self):
        overlay = self._base_overlay()
        with mock.patch("montage.drawing_generator.ask_json", return_value={
            "drawing": {"width": 1000, "height": 1000,
                        "paths": [{"points": [[120, 900], [300, 700], [500, 300], [700, 120]],
                                   "stroke": "#FFFFFF", "stroke_width": 12}]},
        }) as ask:
            revised = revise_drawing(overlay, "Сделай рост круче и добавь стрелку", index=0)
        self.assertEqual((revised["start"], revised["end"]), (2.0, 5.0))
        self.assertEqual((revised["x"], revised["y"]), (0.4, 0.3))
        self.assertEqual(revised["size"], 500)
        self.assertEqual(revised["draw_speed"], 900)
        self.assertEqual(revised["name"], "График клиентов")
        self.assertEqual(revised["trigger_text"], "Исходная реплика")
        self.assertEqual(revised["drawing"]["paths"][0]["points"][-1], [700, 120])
        self.assertEqual(revised["drawing"]["viz"]["trend"], "up", "подписи осей сохраняются")
        self.assertIn("Сделай рост круче", ask.call_args.args[0])

    def test_revise_drawing_forces_timeline_fields_even_if_model_changes_them(self):
        overlay = self._base_overlay()
        with mock.patch("montage.drawing_generator.ask_json", return_value={
            "start": 99, "end": 100, "size": 999,
            "drawing": {"width": 1000, "height": 1000,
                        "paths": [{"points": [[0, 900], [300, 500], [900, 100]]}]},
        }):
            revised = revise_drawing(overlay, "измени")
        self.assertEqual((revised["start"], revised["end"], revised["size"]), (2.0, 5.0, 500))

    def test_revise_drawing_rejects_invalid_model_output(self):
        overlay = self._base_overlay()
        with mock.patch("montage.drawing_generator.ask_json", return_value={"drawing": {"paths": []}}):
            with self.assertRaises(ValueError):
                revise_drawing(overlay, "сделай пустым")

    def test_revise_endpoint_validates_instruction_and_result(self):
        class Request:
            client = SimpleNamespace(host="127.0.0.1")
            headers = {"origin": "http://127.0.0.1:8000"}

            async def json(self):
                return {"instruction": "", "drawing": self._overlay}

            def __init__(self):
                self._overlay = validate_drawing_overlay({
                    "drawing": {"paths": [{"points": [[0, 0], [10, 10]]},
                                          {"points": [[20, 20], [30, 30]]}]},
                })

        with self.assertRaises(HTTPException) as raised:
            __import__("asyncio").run(app.revise_montage_drawing(Request()))
        self.assertEqual(raised.exception.status_code, 400)

    def test_revise_endpoint_calls_model_and_returns_proposal(self):
        class Request:
            client = SimpleNamespace(host="127.0.0.1")
            headers = {"origin": "http://127.0.0.1:8000"}
            payload = {
                "instruction": "Добавь стрелку",
                "drawing": validate_drawing_overlay({
                    "drawing": {"paths": [{"points": [[0, 0], [10, 10]]},
                                          {"points": [[20, 20], [30, 30]]}]},
                }),
            }

            async def json(self):
                return self.payload

        revised = validate_drawing_overlay({
            "drawing": {"paths": [{"points": [[5, 5], [60, 60]]}]},
        })
        with mock.patch.object(app.drawing_generator, "revise_drawing", return_value=revised) as called:
            response = __import__("asyncio").run(app.revise_montage_drawing(Request()))
        self.assertEqual(response["drawing"]["drawing"]["paths"][0]["points"][-1], [60, 60])
        self.assertEqual(called.call_args.args[1], "Добавь стрелку")

        Request.client = SimpleNamespace(host="192.168.1.10")
        with self.assertRaises(HTTPException) as raised:
            __import__("asyncio").run(app.revise_montage_drawing(Request()))
        self.assertEqual(raised.exception.status_code, 403)


class DrawingGenerationJobTests(unittest.TestCase):
    """Режим авто-подбора: без выбранных фрагментов, честный статус, восстановление."""

    JOB_IDS = ("dj-test-auto-mode", "dj-test-smart-err", "dj-test-status-job", "dj-test-recover")

    def tearDown(self):
        for job_id in self.JOB_IDS:
            app._drawing_generation_jobs.pop(job_id, None)
            shutil.rmtree(app.MONTAGE_WORK / job_id, ignore_errors=True)

    @staticmethod
    def _request(payload, host="127.0.0.1"):
        class Request:
            client = SimpleNamespace(host=host)
            headers = {"origin": "http://127.0.0.1:8000"}

            async def json(self):
                return payload
        return Request()

    @staticmethod
    def _proposal(start=1.0, end=3.0):
        return validate_drawing_overlay({
            "start": start, "end": end,
            "drawing": {"paths": [{"points": [[0, 0], [10, 10]]}]},
        })

    def test_auto_mode_starts_without_selected_fragments(self):
        payload = {
            "mode": "auto",
            "job_id": "dj-test-auto-mode",
            "srt_text": "1\n00:00:00,000 --> 00:00:04,000\nПроцесс обучения нейросети.\n",
            "selected_fragments": [],
        }
        with mock.patch.object(app.drawing_generator, "generate_smart_drawings", return_value=[self._proposal()]) as smart:
            response = __import__("asyncio").run(app.generate_montage_drawings(self._request(payload)))
        self.assertEqual(len(response["drawings"]), 1)
        self.assertIn("Процесс обучения нейросети", smart.call_args.args[0])
        job = app._drawing_generation_jobs["dj-test-auto-mode"]
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["progress"], 100)
        self.assertEqual(job["drawings"], response["drawings"])
        stored = app._load_drawing_job("dj-test-auto-mode")
        self.assertEqual(stored["status"], "done")

    def test_manual_mode_still_requires_fragments(self):
        payload = {"mode": "manual", "job_id": "dj-test-auto-mode", "srt_text": "Ку-ку.\n", "selected_fragments": []}
        with self.assertRaises(HTTPException) as raised:
            __import__("asyncio").run(app.generate_montage_drawings(self._request(payload)))
        self.assertEqual(raised.exception.status_code, 400)

    def test_status_endpoint_returns_saved_job(self):
        app._drawing_generation_jobs["dj-test-status-job"] = {
            "id": "dj-test-status-job", "status": "done", "progress": 100, "drawings": [],
        }
        request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
        job = __import__("asyncio").run(
            app.montage_drawing_job_status(request, job_id="dj-test-status-job")
        )
        self.assertEqual(job["status"], "done")

    def test_generation_error_is_recorded_in_job(self):
        payload = {
            "mode": "auto",
            "job_id": "dj-test-smart-err",
            "srt_text": "1\n00:00:00,000 --> 00:00:04,000\nТекст.\n",
            "selected_fragments": [],
        }
        with mock.patch.object(app.drawing_generator, "generate_smart_drawings", side_effect=RuntimeError("модель упала")):
            with self.assertRaises(HTTPException) as raised:
                __import__("asyncio").run(app.generate_montage_drawings(self._request(payload)))
        self.assertEqual(raised.exception.status_code, 502)
        stored = app._load_drawing_job("dj-test-smart-err")
        self.assertEqual(stored["status"], "error")
        self.assertIn("модель упала", stored["error"])

    def test_recover_marks_interrupted_job_as_error(self):
        path = app.MONTAGE_WORK / "dj-test-recover" / "drawing_generation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": "dj-test-recover", "status": "running", "phase": "Анализ"}), encoding="utf-8")
        app._recover_drawing_jobs()
        job = app._drawing_generation_jobs["dj-test-recover"]
        self.assertEqual(job["status"], "error")
        self.assertIn("перезапуском", job["error"])


class SmartDrawingSelectionTests(unittest.TestCase):
    def test_segments_to_phrases_merges_word_segments(self):
        segments = [
            {"start": index * 0.4, "end": index * 0.4 + 0.4, "text": word}
            for index, word in enumerate(["Привет", "мир", "это", "тест"])
        ]
        phrases = app_chart._segments_to_phrases(segments)
        self.assertEqual(len(phrases), 1)
        self.assertEqual(phrases[0]["text"], "Привет мир это тест")
        self.assertEqual((phrases[0]["start"], phrases[0]["end"]), (0.0, 1.6))

    def test_segments_to_phrases_starts_new_phrase_after_long_pause(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Первая мысль"},
            {"start": 6.0, "end": 7.0, "text": "Вторая мысль"},
        ]
        phrases = app_chart._segments_to_phrases(segments)
        self.assertEqual([p["text"] for p in phrases], ["Первая мысль", "Вторая мысль"])

    def test_parse_srt_segments(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:03,500\nПервая фраза\n\n"
            "2\n00:00:04,000 --> 00:00:06,250\nВторая фраза\n"
        )
        segments = parse_srt_segments(srt)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0], {"start": 1.0, "end": 3.5, "text": "Первая фраза"})
        self.assertEqual(segments[1], {"start": 4.0, "end": 6.25, "text": "Вторая фраза"})

    @staticmethod
    def _suggestion(text, start, end):
        return {
            "text": text, "start": start, "end": end, "name": text, "rationale": "почему-то",
            "drawing": {"width": 1000, "height": 1000, "paths": [{"points": [[100, 100], [900, 400]]}]},
        }

    def test_smart_drawings_skip_overlap_and_tail_outside_video(self):
        suggestions = [
            self._suggestion("Первый момент", 0.5, 3.0),
            self._suggestion("Наложение", 1.0, 2.0),
            self._suggestion("После конца ролика", 40.0, 44.0),
        ]
        segments = [{"start": 0.0, "end": 10.0, "text": "Смысловая фраза."}]
        with mock.patch("montage.drawing_generator.ask_json", return_value={"suggestions": suggestions}):
            drawings = generate_smart_drawings("Транскрипция целиком", segments, 3)
        self.assertEqual(len(drawings), 1)
        self.assertEqual(drawings[0]["trigger_text"], "Первый момент")
        self.assertEqual(drawings[0]["rationale"], "почему-то")
        self.assertLessEqual(drawings[0]["end"], 10.0)

    def test_smart_drawings_without_segments_parse_srt(self):
        suggestions = [self._suggestion("Момент", 0.0, 2.0)]
        srt = "1\n00:00:00,000 --> 00:00:05,000\nФраза.\n"
        with mock.patch("montage.drawing_generator.ask_json", return_value={"suggestions": suggestions}) as ask:
            drawings = generate_smart_drawings(srt, [], 3)
        self.assertEqual(len(drawings), 1)
        self.assertIn("ФРАЗЫ", ask.call_args.args[0])
        self.assertIn("Фраза.", ask.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
