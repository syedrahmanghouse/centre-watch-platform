"""Evidence must use the sampled pixels and coordinates without changing counts."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import ImageDraw

from server import vision


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_snapshot_retains_background_and_scales_boxes_with_exact_sample(self):
        source = np.full((1080, 1920, 3), (170, 60, 40), dtype=np.uint8)
        observation = {"width": 1920, "height": 1080, "time": 3.25, "people": 1, "detections": [
            {"label": "person", "xyxy": [300, 150, 900, 600], "includedInCount": True},
            {"label": "person", "xyxy": [1500, 150, 1800, 600], "includedInCount": False},
        ]}
        with patch.object(ImageDraw.ImageDraw, "text", autospec=True, side_effect=ImageDraw.ImageDraw.text) as text:
            evidence = vision._save_evidence(observation, self.root, 7, vision._snapshot_bytes(source))
        rendered = cv2.imread(evidence["path"])
        self.assertEqual(rendered.shape[1], 1280)
        np.testing.assert_allclose(rendered[600, 1100], source[900, 1650], atol=5)
        # x=300 in the 1920px source maps to x=200 in the retained 1280px image.
        self.assertGreater(int(rendered[260, 200, 1]), 200)
        self.assertLess(int(rendered[260, 205, 1]), 90)
        self.assertEqual(evidence["privacy"], "source_pixels_retained")
        self.assertEqual(evidence["type"], "annotated_snapshot")
        self.assertEqual((evidence["time"], evidence["frameIndex"], evidence["people"]), (3.25, 7, 1))
        labels = [call.args[2] for call in text.call_args_list]
        self.assertIn("Person 1", labels)
        self.assertIn("Outside count area", labels)
        self.assertNotIn("Person 2", labels)

    def test_default_evidence_is_still_an_abstract_diagram(self):
        observation = {"width": 640, "height": 352, "time": 1, "people": 0, "detections": []}
        evidence = vision._save_evidence(observation, self.root, 0)
        rendered = cv2.imread(evidence["path"])
        np.testing.assert_allclose(rendered[100, 100], (26, 35, 46), atol=4)
        self.assertEqual(evidence["privacy"], "all_source_pixels_removed")
        self.assertEqual(evidence["type"], "abstract_diagram")

    def analyze_fixture(self, **options):
        samples = [np.full((352, 640, 3), color, dtype=np.uint8)
                   for color in ((40, 90, 160), (130, 60, 90), (170, 140, 60))]
        def detections(frame, **kwargs):
            count = 2 if frame[0, 0, 0] == 130 else 1
            return [{"label": "person", "confidence": .8, "xyxy": [30 + i * 100, 90, 90 + i * 100, 220], "detector": "people"}
                    for i in range(count)], {"qualityFlags": [], "model": "fixture", "sha256": "0" * 64}
        def quality(frame):
            return {"height": frame.shape[0], "width": frame.shape[1], "qualityFlags": []}
        frames = [(index + .5, sample, {"captureFlags": [], "isImage": False, "sourceType": "rtsp"})
                  for index, sample in enumerate(samples)]
        with patch.object(vision, "_load_model", return_value=SimpleNamespace(names={0: "person"})), \
             patch.object(vision, "detect_people", side_effect=detections), \
             patch.object(vision, "frame_quality", side_effect=quality), \
             patch.object(vision, "_read_frames", return_value=iter(frames)) as reader:
            result = vision.analyze_source("rtsp://fixture", [], options)
        reader.assert_called_once()
        return result, samples

    def test_pipeline_preserves_each_selected_sample_without_reopening_stream(self):
        result, samples = self.analyze_fixture(output_dir=str(self.root), evidence_mode="annotated")
        self.assertEqual(result["count"], 1)
        self.assertEqual([frame["people"] for frame in result["frames"]], [1, 2, 1])
        self.assertEqual(len(result["evidence"]), 3)
        for item in result["evidence"]:
            index = item["frameIndex"]
            image = cv2.imread(item["path"])
            scale = image.shape[1] / samples[index].shape[1]
            np.testing.assert_allclose(image[round(300 * scale), round(500 * scale)], samples[index][300, 500], atol=5)
            self.assertEqual(item["time"], result["frames"][index]["time"])
            self.assertEqual(item["people"], result["frames"][index]["people"])
        # Transient source images/encoded bytes are never exposed in results.
        json.dumps(result)
        self.assertNotIn("snapshot", result["frames"][0])

    def test_pixels_are_never_captured_for_edge_or_disabled_evidence(self):
        for options in ({"output_dir": str(self.root)},
                        {"evidence_mode": "annotated"},
                        {"output_dir": str(self.root), "evidence_mode": "annotated", "save_evidence": False}):
            with self.subTest(options=options), patch.object(vision, "_snapshot_bytes") as capture:
                self.analyze_fixture(**options)
                capture.assert_not_called()

    def test_evidence_mode_typo_cannot_silently_retain_source_pixels(self):
        with self.assertRaisesRegex(ValueError, "evidence_mode"):
            vision._options({"evidence_mode": "raw"})


if __name__ == "__main__":
    unittest.main()
