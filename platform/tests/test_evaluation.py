"""Synthetic unit fixtures test arithmetic; these are not model-accuracy results."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from server.evaluation import EvaluationError, run_evaluation, validate_manifest


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "clip.mp4").write_bytes(b"synthetic unit test placeholder; injected analyzer only")
        self.manifest = {"version": 1, "settings": {"attendance_tolerance": 0}, "videos": [{
            "file": "clip.mp4", "room": "Room A", "split": "test", "scene_id": "camera-A", "reported_count": 5,
            "inventory": [{"name": "chair", "quantity": 4}], "observations": [
                {"time_seconds": 1, "people": 3, "equipment": {"chair": 3}, "visibility": "clear",
                 "boxes": [{"class": "person", "bbox": [0, 0, 10, 10]}]},
                {"time_seconds": 2, "people": 5, "equipment": {"chair": 4}, "visibility": "occluded"},
                {"time_seconds": 3, "people": 2, "equipment": {"chair": 3}, "visibility": "clear"},
            ]}]}

    def tearDown(self):
        self.temp.cleanup()

    def fake_analyzer(self, source, inventory, options=None):
        self.assertEqual(options["sample_times"], [1, 2, 3])
        return {"modelVersion": "unit-fixture-only", "frames": [
            {"time": 1, "people": 5, "equipment": {"chair": 4}, "usable": True,
             "detections": [{"label": "person", "box": [0, 0, 10, 10], "confidence": .9},
                            {"label": "person", "box": [50, 50, 10, 10], "confidence": .8}]},
            {"time": 2, "people": 4, "equipment": {"chair": 3}, "usable": True, "detections": []},
            {"time": 3, "people": 1, "equipment": {"chair": 0}, "usable": False, "qualityFlags": ["dark"]},
        ]}

    def test_count_alert_and_object_metrics_are_distinct(self):
        report = run_evaluation(self.manifest, self.root, self.root / "result", self.fake_analyzer)
        metrics = report["metrics"]
        self.assertEqual(metrics["count"]["countMAE"], 1.5)
        self.assertAlmostEqual(metrics["count"]["countRMSE"], math.sqrt(2.5), places=5)
        self.assertEqual(metrics["count"]["overcount_total"], 2)
        self.assertEqual(metrics["count"]["undercount_total"], 1)
        self.assertEqual(metrics["attendance_alerts"]["fp"], 1)
        self.assertEqual(metrics["attendance_alerts"]["fn"], 1)
        self.assertEqual(metrics["attendance_alerts"]["unverified_positive"], 1)
        self.assertEqual(metrics["attendance_alerts"]["total"], 3)
        self.assertEqual(metrics["infrastructure_alerts"]["chair"]["fn"], 1)
        self.assertEqual(metrics["object_detection"]["classes"]["person"]["tp"], 1)
        self.assertEqual(metrics["object_detection"]["classes"]["person"]["fp"], 1)
        self.assertEqual(report["coverage"]["abstained_or_missing"], 1)
        for name in report["files"]:
            self.assertTrue((self.root / "result" / name).is_file())

    def test_unverified_positive_lowers_operational_recall(self):
        def analyzer(*args, **kwargs):
            return {"frames": [{"time": 1, "people": 3, "usable": True}, {"time": 2, "people": 5, "usable": True}]}
        result = run_evaluation(self.manifest, self.root, self.root / "result", analyzer)
        alerts = result["metrics"]["attendance_alerts"]
        self.assertEqual(alerts["recall_on_assessed"], 1)
        self.assertEqual(alerts["recall_including_unverified"], .5)
        self.assertEqual(alerts["tn"], 1)

    def test_no_boxes_reports_unavailable(self):
        del self.manifest["videos"][0]["observations"][0]["boxes"]
        report = run_evaluation(self.manifest, self.root, self.root / "result", self.fake_analyzer)
        self.assertFalse(report["metrics"]["object_detection"]["available"])
        self.assertEqual(report["metrics"]["object_detection"]["classes"], {})

    def test_unknown_ground_truth_and_absent_equipment_not_zero(self):
        self.manifest["videos"][0]["observations"][0]["visibility"] = "unknown"
        del self.manifest["videos"][0]["observations"][1]["equipment"]["chair"]
        report = run_evaluation(self.manifest, self.root, self.root / "result", self.fake_analyzer)
        self.assertEqual(report["metrics"]["attendance_alerts"]["unknown_ground_truth"], 1)
        self.assertEqual(report["metrics"]["infrastructure_alerts"]["chair"]["unknown_ground_truth"], 2)

    def test_reject_path_traversal_and_absolute_path(self):
        for name in ("../clip.mp4", str(self.root / "clip.mp4"), "x\\clip.mp4"):
            self.manifest["videos"][0]["file"] = name
            with self.assertRaises(EvaluationError):
                validate_manifest(self.manifest, self.root)

    def test_reject_symlink_outside_incoming(self):
        child = self.root / "incoming"
        child.mkdir()
        (child / "clip.mp4").symlink_to(self.root / "clip.mp4")
        with self.assertRaises(EvaluationError):
            validate_manifest(self.manifest, child)

    def test_reject_scene_leakage(self):
        second = copy.deepcopy(self.manifest["videos"][0])
        second.update(file="clip2.mp4", split="development")
        (self.root / "clip2.mp4").write_bytes(b"placeholder")
        self.manifest["videos"].append(second)
        with self.assertRaisesRegex(EvaluationError, "Scene leakage"):
            validate_manifest(self.manifest, self.root)

    def test_reject_invalid_counts_timestamps_and_boxes(self):
        for field, value in (("people", -1), ("people", True), ("time_seconds", float("nan"))):
            manifest = copy.deepcopy(self.manifest)
            manifest["videos"][0]["observations"][0][field] = value
            with self.assertRaises(EvaluationError):
                validate_manifest(manifest, self.root)
        self.manifest["videos"][0]["observations"][0]["boxes"][0]["bbox"][2] = 0
        with self.assertRaises(EvaluationError):
            validate_manifest(self.manifest, self.root)

    def test_mismatched_timestamp_is_abstained(self):
        report = run_evaluation(self.manifest, self.root, self.root / "result", lambda *a, **k: {"frames": [{"time": 100, "people": 5, "usable": True}]})
        self.assertIsNone(report["metrics"]["count"]["countMAE"])
        self.assertEqual(report["coverage"]["abstained_or_missing"], 3)

    def test_csv_formula_escaping(self):
        self.manifest["videos"][0]["room"] = "=EXAMPLE()"
        run_evaluation(self.manifest, self.root, self.root / "result", self.fake_analyzer)
        self.assertIn("'=EXAMPLE()", (self.root / "result" / "predictions.csv").read_text())


class SeatedWindowEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'clip.mp4').write_bytes(b'injected analyzer fixture only')
        self.manifest={'version':1,'settings':{'countingMode':'seated_only','observationSeconds':300},
            'videos':[{'file':'clip.mp4','room':'A','scene_id':'A','split':'development',
                'window_truth':{'people':13,'visibility':'occluded','provenance':'user_reported'},'observations':[]}]}

    def tearDown(self):
        self.temp.cleanup()

    def result(self,**overrides):
        return {'countingMode':'seated_only','count':12,'candidateCount':12,'confidence':'medium',
                'frames':[{'time':i,'people':11,'temporalCount':12,'usable':True} for i in range(301)],**overrides}

    def test_one_dense_call_keeps_expected_and_reported_counts_out_of_inference(self):
        self.manifest['videos'][0]['reported_count']=27
        self.manifest['videos'][0]['observations']=[{'time_seconds':150,'people':13,'visibility':'clear'}]
        calls=[]
        def analyzer(source,inventory,options):
            calls.append(options)
            self.assertEqual(options,{'counting_mode':'seated_only','observation_seconds':300,
                'sample_interval_seconds':1,'max_frames':301,'save_evidence':False})
            return self.result()
        report=run_evaluation(self.manifest,self.root,self.root/'report',analyzer)
        self.assertEqual(len(calls),1)
        self.assertEqual(report['window_metrics']['count']['countMAE'],1)
        self.assertEqual(report['metrics']['count']['countMAE'],1)
        self.assertFalse(report['window_metrics']['object_detection']['available'])

    def test_uncertain_candidate_is_diagnostic_not_a_scored_count(self):
        report=run_evaluation(self.manifest,self.root,self.root/'report',lambda *a,**k:self.result(count=None,candidateCount=8,confidence='low'))
        self.assertIsNone(report['window_metrics']['count']['countMAE'])
        self.assertEqual(report['window_metrics']['count']['scored_observations'],0)
        self.assertEqual(report['window_predictions'][0]['candidate_absolute_error_diagnostic_only'],5)
        self.assertEqual(report['window_coverage']['abstained_or_missing'],1)
        self.assertFalse(report['metrics']['object_detection']['available'])

    def test_sparse_frames_cannot_claim_window_accuracy(self):
        report=run_evaluation(self.manifest,self.root,self.root/'report',lambda *a,**k:self.result(frames=[{'time':0},{'time':150},{'time':300}]))
        self.assertFalse(report['window_predictions'][0]['dense_window_complete'])
        self.assertIsNone(report['window_metrics']['count']['countMAE'])

    def test_unknown_ground_truth_is_not_forced_from_user_number(self):
        self.manifest['videos'][0]['window_truth']['visibility']='unknown'
        report=run_evaluation(self.manifest,self.root,self.root/'report',lambda *a,**k:self.result())
        self.assertIsNone(report['window_metrics']['count']['countMAE'])
        self.assertIsNone(report['window_predictions'][0]['candidate_absolute_error_diagnostic_only'])

    def test_seated_object_matching_excludes_annotated_standing_people(self):
        self.manifest['settings']['observationSeconds']=2
        self.manifest['videos'][0]['observations']=[{'time_seconds':2,'people':1,'visibility':'clear','boxes':[
            {'class':'person','posture':'seated','bbox':[0,0,10,10]},
            {'class':'person','posture':'standing','bbox':[30,0,10,10]}]}]
        detections=[{'label':'person','box':[0,0,10,10],'confidence':.9,'includedInCount':True},
                    {'label':'person','box':[30,0,10,10],'confidence':.9,'includedInCount':False}]
        frames=[{'time':i,'people':1,'temporalCount':1,'usable':True,'detections':detections} for i in range(3)]
        report=run_evaluation(self.manifest,self.root,self.root/'report',lambda *a,**k:self.result(count=1,frames=frames))
        person=report['metrics']['object_detection']['classes']['person']
        self.assertEqual((person['tp'],person['fp'],person['fn']),(1,0,0))

    def test_seated_person_boxes_require_posture_labels(self):
        self.manifest['videos'][0]['observations']=[{'time_seconds':2,'people':1,'visibility':'clear','boxes':[{'class':'person','bbox':[0,0,10,10]}]}]
        with self.assertRaisesRegex(EvaluationError,'require posture'):
            validate_manifest(self.manifest,self.root)

    def test_reject_bad_mode_duration_and_annotation_outside_window(self):
        for settings in ({'countingMode':'trainer'}, {'countingMode':'seated_only','observationSeconds':901},
                         {'countingMode':'seated_only','observationSeconds':float('nan')}):
            manifest=copy.deepcopy(self.manifest);manifest['settings']=settings
            with self.assertRaises(EvaluationError):validate_manifest(manifest,self.root)
        self.manifest['videos'][0]['observations']=[{'time_seconds':301,'people':13,'visibility':'clear'}]
        with self.assertRaisesRegex(EvaluationError,'contiguous observation window'):
            validate_manifest(self.manifest,self.root)


if __name__ == "__main__":
    unittest.main()
