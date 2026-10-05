#!/usr/bin/env python3
"""Run the real --once detector-to-outbox-to-HTTP path on an explicit local source."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def run(source: Path):
    received = []
    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("X-Device-Key") != "test-only-device-key":
                self.send_response(401)
                self.end_headers()
                return
            received.append(raw)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"accepted": True, "eventId": json.loads(raw)["event_id"]}, separators=(",", ":")).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"server": f"http://127.0.0.1:{server.server_port}", "deviceId": "pipeline-test-device", "classes": [{
                "classId": "pipeline-test-class", "source": str(source.resolve()), "inventory": [], "maxFrames": 6,
            }]}
            file = root / "edge.json"
            file.write_text(json.dumps(config))
            completed = subprocess.run([sys.executable, str(Path(__file__).with_name("edge_worker.py")), "--config", str(file),
                                        "--db", str(root / "outbox.sqlite3"), "--once", "--allow-local-http"],
                                       env={**os.environ, "CENTREWATCH_DEVICE_KEY": "test-only-device-key"},
                                       capture_output=True, text=True, timeout=180)
            if completed.returncode:
                raise RuntimeError("Worker process failed; inspect the local model environment")
            assert len(received) == 1, "Expected one received metadata event"
            event = json.loads(received[0])
            analysis = event["payload"]["analysis"]
            assert "analysis_failed" not in analysis["qualityFlags"], "Local inference failed"
            assert set(analysis) == {"count", "confidence", "qualityFlags", "equipment"}
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            assert result["sync"]["acked"] == 1
            assert result["queue"]["states"] == {"acked": 1}
            return {"created_at": datetime.now(timezone.utc).isoformat(), "kind": "actual_inference_pipeline_smoke_only",
                    "result": "pass", "worker_once": True, "real_model_inference": True,
                    "outbox_acknowledged": True, "received_events": len(received), "event_body_bytes": len(received[0]),
                    "raw_media_bytes_transferred": 0, "analysis_quality_flags": analysis["qualityFlags"],
                    "scope": "Actual local model, SQLite outbox and localhost HTTP receiver. No labelled ground truth; not a model-accuracy assessment. Source may be a synthetic demonstration fixture."}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error("source must be an existing local photo/video")
    report = run(args.source)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
