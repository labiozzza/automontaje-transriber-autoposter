import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from montage.drawing_generator import validate_drawing_overlay
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
        empty = render_drawing(drawing, 100, 0)
        partial = render_drawing(drawing, 100, 30)
        complete = render_drawing(drawing, 100, 100)
        self.assertIsNone(empty.getchannel("A").getbbox())
        self.assertLess(partial.getchannel("A").getbbox()[2], complete.getchannel("A").getbbox()[2])
        self.assertEqual(complete.getpixel((50, 50))[:3], (255, 255, 255))

    def test_generation_endpoint_is_local_and_returns_validated_proposals(self):
        class Request:
            client = SimpleNamespace(host="127.0.0.1")
            headers = {"origin": "http://127.0.0.1:8000"}

            async def json(self):
                return {"srt_text": "1\n00:00:00,000 --> 00:00:01,000\nСобака ест.\n"}

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


if __name__ == "__main__":
    unittest.main()
