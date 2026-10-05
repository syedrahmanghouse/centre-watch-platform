import unittest
from server.vision import _options, canonical_name, sample_times, summarize_observations, representative_frame_indices


def frame(people=12, chairs=10, flags=None):
    return {"people": people, "equipment": {"chair": chairs}, "qualityFlags": flags or [], "detections": [], "usable": True}


class VisionLogicTests(unittest.TestCase):
    def test_median_rejects_single_count_outlier(self):
        result = summarize_observations([frame(12), frame(13), frame(60)], [], ["person"])
        self.assertIsNone(result["count"])
        self.assertEqual(result["candidateCount"], 13)
        self.assertEqual(result["confidence"], "low")
        self.assertIn("unstable_headcount", result["qualityFlags"])

    def test_evidence_first_frame_matches_aggregate_not_first_timestamp(self):
        frames = [frame(11), frame(14), frame(17), frame(14), frame(16)]
        chosen = representative_frame_indices(frames, 14)
        self.assertEqual(frames[chosen[0]]["people"], 14)
        self.assertEqual(sorted(frames[i]["people"] for i in chosen), [11, 14, 17])

    def test_no_detections_does_not_prove_empty(self):
        result = summarize_observations([frame(0)] * 3, [], ["person"])
        self.assertIsNone(result["count"])
        self.assertEqual(result["confidence"], "low")

    def test_dark_frames_do_not_override_visible_count(self):
        result = summarize_observations([frame(12), frame(0, flags=["dark_frame"]), frame(12), frame(12)], [], ["person"])
        self.assertEqual(result["count"], 12)
        self.assertEqual(result["usableFrames"], 3)

    def test_inventory_requires_confirmed_visibility(self):
        result = summarize_observations([frame()] * 3, [{"name": "chairs", "quantity": 15}], ["person", "chair"])
        self.assertEqual(result["equipment"][0]["status"], "unknown")

    def test_gap_is_sustained_and_operability_unverified(self):
        result = summarize_observations([frame()] * 3, [{"name": "chairs", "quantity": 15, "observable": True}], ["person", "chair"])
        item = result["equipment"][0]
        self.assertEqual(item["status"], "gap")
        self.assertEqual(item["observed"], 10)
        self.assertEqual(item["operability"], "unverified")

    def test_single_photo_never_proves_missing_item(self):
        result = summarize_observations([frame()], [{"name": "chairs", "quantity": 15, "observable": True}], ["person", "chair"], is_image=True)
        self.assertEqual(result["equipment"][0]["status"], "unknown")

    def test_transient_missing_detection_is_unknown(self):
        result = summarize_observations([frame(chairs=10), frame(chairs=15), frame(chairs=10)], [{"name": "chair", "quantity": 15, "observable": True}], ["person", "chair"])
        self.assertEqual(result["equipment"][0]["status"], "unknown")

    def test_frozen_stream_cannot_verify_inventory(self):
        result = summarize_observations([frame(flags=["frozen_frames"])] * 3, [{"name": "chair", "quantity": 10, "observable": True}], ["person", "chair"])
        self.assertEqual(result["confidence"], "low")
        self.assertEqual(result["equipment"][0]["status"], "unknown")

    def test_unsupported_inventory_is_explicit(self):
        result = summarize_observations([frame()] * 3, [{"name": "special hydraulic press", "quantity": 1, "observable": True}], ["person", "chair"])
        self.assertIsNone(result["equipment"][0]["observed"])
        self.assertIn("vocabulary", result["equipment"][0]["reason"])

    def test_timestamps_do_not_sample_eof(self):
        values = sample_times(1.0, 12)
        self.assertTrue(all(0 <= value < 1.0 for value in values))
        self.assertEqual(sample_times(10, 3, [0, 2.5, 9]), [0, 2.5, 9])
        with self.assertRaises(ValueError):
            sample_times(10, 3, [10])

    def test_aliases_do_not_invent_machine_types(self):
        self.assertEqual(canonical_name("Work Benches"), "workbench")
        self.assertEqual(canonical_name("machines"), "machines")

    def test_partial_stream_cannot_verify_attendance_or_inventory(self):
        observations = [frame(flags=["incomplete_observation"])] * 3
        result = summarize_observations(observations, [{"name": "chair", "quantity": 10, "observable": True}], ["person", "chair"])
        self.assertEqual(result["confidence"], "low")
        self.assertEqual(result["equipment"][0]["status"], "unknown")

    def test_mostly_obstructed_session_cannot_prove_inventory_gap(self):
        observations = [frame()] * 3 + [frame(flags=["dark_frame"])] * 9
        result = summarize_observations(observations, [{"name": "chair", "quantity": 20, "observable": True}], ["person", "chair"])
        self.assertEqual(result["equipment"][0]["status"], "unknown")

    def test_invalid_numerical_options_fail_before_inference(self):
        for invalid in ({"detector_confidence": float("nan")}, {"sample_seconds": float("inf")},
                        {"max_frames": 2.5}, {"image_size": 999999}, {"trainee_zone": []}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _options(invalid)


if __name__ == "__main__":
    unittest.main()
