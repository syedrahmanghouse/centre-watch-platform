import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from edge_worker import EdgeWorker, parsed_time, validate_config
from offline_sync import Outbox


class EdgeWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.db = self.root / "outbox.sqlite3"
        self.outbox = Outbox(self.db)
        self.config = {"server": "https://centre.example", "deviceId": "device-123", "classes": [{
            "classId": "class-123", "source": "rtsp://user:private-secret@192.168.1.2/stream",
            "inventory": [{"name": "chair", "quantity": 4}], "submittedCount": 3,
            "startsAt": "2026-10-04T10:00:00+05:30", "endsAt": "2026-10-04T12:00:00+05:30",
            "intervalMinutes": 30, "coverageConfirmed": True,
        }]}
        self.calls = []
    def tearDown(self):
        self.outbox.close()
        self.temporary.cleanup()
    def analyzer(self, source, inventory, options):
        self.calls.append((source, inventory, options))
        return {"count": 3, "confidence": "high", "qualityFlags": [], "equipment": [{"name": "chair", "expected": 4, "observed": 3, "status": "gap"}], "source": source}
    def test_active_schedule_and_persisted_next_due(self):
        worker = EdgeWorker(self.outbox, self.config, analyzer=self.analyzer)
        active = parsed_time("2026-10-04T10:05:00+05:30")
        self.assertEqual(worker.capture_due(active - 3600), [])
        captured = worker.capture_due(active)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["count"], 3)
        self.assertEqual(worker.capture_due(active + 60), [])
        self.outbox.close()
        self.outbox = Outbox(self.db)
        restarted = EdgeWorker(self.outbox, analyzer=self.analyzer)
        self.assertEqual(restarted.capture_due(active + 120), [])
        self.assertEqual(len(restarted.capture_due(active + 1800)), 1)
        self.assertEqual(restarted.capture_due(active + 7200), [])
        self.assertEqual(len(self.calls), 2)
        body = self.outbox.connection.execute("SELECT body FROM outbox LIMIT 1").fetchone()["body"]
        self.assertNotIn("private-secret", body)
        self.assertEqual(json.loads(body)["payload"]["analysis"]["equipment"][0]["status"], "gap")
    def test_inference_failure_is_explicit_unverified_event(self):
        def failing(*a, **k):
            raise RuntimeError("rtsp://user:private-secret@camera was not reachable")
        worker = EdgeWorker(self.outbox, self.config, analyzer=failing)
        captured = worker.capture_due(parsed_time("2026-10-04T10:05:00+05:30"))
        self.assertEqual(captured[0]["status"], "unverified")
        self.assertIsNone(captured[0]["count"])
        record = self.outbox.connection.execute("SELECT body FROM outbox").fetchone()["body"]
        self.assertNotIn("private-secret", record)
        self.assertIn("analysis_failed", record)
    def test_continuous_mode_refuses_recorded_video(self):
        source = self.root / "test.mp4"
        source.write_bytes(b"synthetic scheduler test only")
        self.config["classes"][0]["source"] = str(source)
        with self.assertRaisesRegex(ValueError, "live RTSP"):
            validate_config(self.config, self.root)
        checked = validate_config(self.config, self.root, once=True)
        self.assertEqual(checked["classes"][0]["source"], str(source.resolve()))
    def test_reject_device_key_in_config_and_unbounded_schedule(self):
        self.config["deviceKey"] = "must-stay-in-environment"
        with self.assertRaises(ValueError):
            validate_config(self.config, self.root)
        del self.config["deviceKey"]
        del self.config["classes"][0]["startsAt"]
        del self.config["classes"][0]["endsAt"]
        with self.assertRaises(ValueError):
            validate_config(self.config, self.root)


if __name__ == "__main__":
    unittest.main()
