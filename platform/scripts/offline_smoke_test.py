#!/usr/bin/env python3
"""Measure actual localhost metadata bodies and durable retry; no CCTV is fabricated."""
import argparse
import json
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from offline_sync import Outbox, make_event


def run_smoke_test():
    received, accepted = [], set()
    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            received.append(data)
            event = json.loads(data)
            authorized = self.headers.get("X-Device-Key") == "smoke-test-key"
            success = authorized and len(received) > 1
            if success:
                accepted.add(event["event_id"])
            self.send_response(200 if success else 503 if authorized else 401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"accepted": True, "eventId": event["event_id"]}, separators=(",", ":")).encode() if success else b'{"retry":true}')
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.sqlite3"
            box = Outbox(database)
            event = make_event("smoke-device", "smoke-class", {
                "count": 23, "confidence": "medium", "qualityFlags": [],
                "equipment": [{"name": "chair", "expected": 25, "observed": 25, "status": "present"}],
            }, "2026-10-04T00:00:00+00:00")
            queued = box.enqueue(event)
            endpoint = f"http://127.0.0.1:{server.server_port}"
            failed = box.sync(endpoint, "smoke-test-key", allow_local_http=True)
            box.close()
            box = Outbox(database)
            restarted_pending = box.status()["states"].get("pending", 0)
            synced = box.sync(endpoint, "smoke-test-key", allow_local_http=True, now=time.time() + 100)
            stats = box.status()
            box.close()
            assert failed["deferred"] == restarted_pending == synced["acked"] == len(accepted) == 1
            assert received[0] == received[1]
            assert queued["bytes"] == len(received[0])
            return {"created_at": datetime.now(timezone.utc).isoformat(), "kind": "transport_smoke_test_only",
                    "result": "pass", "initial_status": 503, "retry_status": 200,
                    "restart_preserved_queue": True, "retry_preserved_event_uuid": True,
                    "unique_events_received": len(accepted), "http_requests": len(received),
                    "event_body_bytes": len(received[0]), "total_received_request_body_bytes": sum(map(len, received)),
                    "total_response_body_bytes": stats["counters"]["response_body_bytes"],
                    "raw_video_bytes_sent": 0, "raw_image_bytes_sent": 0,
                    "scope": "Measured HTTP request and response bodies over localhost; excludes HTTP headers/TLS/transport overhead. Counts are synthetic metadata for a transport test, not model accuracy."}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_smoke_test()
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
