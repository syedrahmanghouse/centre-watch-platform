import json
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from offline_sync import Outbox, compact_analysis, endpoint_url, make_event


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "outbox.sqlite3"
        self.outbox = Outbox(self.path)
        self.event = make_event("device-test", "class-test", {"count": 23, "confidence": "medium", "qualityFlags": [], "equipment": [{"name": "chair", "expected": 25, "observed": 25, "status": "present"}]}, "2026-10-04T00:00:00+00:00")

    def tearDown(self):
        self.outbox.close()
        self.temp.cleanup()

    def test_restart_and_deduplication(self):
        self.outbox.enqueue(self.event)
        self.outbox.close()
        self.outbox = Outbox(self.path)
        self.assertEqual(self.outbox.status()["states"], {"pending": 1})
        self.assertTrue(self.outbox.enqueue(self.event)["already_queued"])
        self.event["payload"]["analysis"]["count"] = 20
        with self.assertRaises(ValueError):
            self.outbox.enqueue(self.event)

    def test_media_and_private_paths_are_not_serialized(self):
        analysis = {"count": 1, "frames": [{"image": "data:image/jpg;base64,aaaa"}], "path": "/private/raw.mp4", "evidence": "face.jpg", "source": "rtsp://user:password@camera"}
        safe = compact_analysis(analysis)
        encoded = json.dumps(safe)
        for forbidden in ("base64", "private", "password", "face.jpg"):
            self.assertNotIn(forbidden, encoded)

    def test_https_required_and_local_override_explicit(self):
        for value in ("http://example.org", "https://user:secret@example.org", "https://example.org?key=x"):
            with self.assertRaises(ValueError):
                endpoint_url(value, True)
        with self.assertRaises(ValueError):
            endpoint_url("http://127.0.0.1:8000")
        self.assertEqual(endpoint_url("http://127.0.0.1:8000", True), "http://127.0.0.1:8000/api/edge/events")

    def test_http_200_without_matching_acknowledgment_stays_queued(self):
        self.outbox.enqueue(self.event)
        with patch("offline_sync.urllib.request.build_opener") as build:
            response = build.return_value.open.return_value.__enter__.return_value
            response.status = 200
            response.read.return_value = b'{"accepted":true,"eventId":"different-event"}'
            result = self.outbox.sync("https://central.example", "test-key")
        self.assertEqual(result["acked"], 0)
        self.assertEqual(result["deferred"], 1)
        self.assertEqual(self.outbox.status()["states"], {"pending": 1})

    def test_http_retry_keeps_uuid_and_measures_actual_bodies(self):
        received, ids = [], set()
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received.append(body)
                event = json.loads(body)
                if self.headers.get("X-Device-Key") != "test-key":
                    self.send_response(401)
                elif len(received) == 1:
                    self.send_response(503)
                else:
                    ids.add(event["event_id"])
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"accepted": True, "eventId": event["event_id"]}, separators=(",", ":")).encode())
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            queued = self.outbox.enqueue(self.event)
            url = f"http://127.0.0.1:{server.server_port}"
            first = self.outbox.sync(url, "test-key", allow_local_http=True, now=time.time())
            self.assertEqual(first["deferred"], 1)
            self.outbox.close()
            self.outbox = Outbox(self.path)
            second = self.outbox.sync(url, "test-key", allow_local_http=True, now=time.time() + 100)
            self.assertEqual(second["acked"], 1)
            self.assertEqual(len(ids), 1)
            self.assertEqual(received[0], received[1])
            self.assertEqual(len(received[0]), queued["bytes"])
            counters = self.outbox.status()["counters"]
            self.assertEqual(counters["attempted_request_body_bytes"], sum(map(len, received)))
            self.assertEqual(counters["acknowledged_event_bytes"], len(received[1]))
            self.assertEqual(self.outbox.status()["states"], {"acked": 1})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
