#!/usr/bin/env python3
"""Continuously analyze scheduled local cameras and forward only durable metadata."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from offline_sync import Outbox, endpoint_url, make_event

PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM))


@contextlib.contextmanager
def private_decoder_errors():
    """Native decoder stderr can include a credential-bearing camera URL."""
    original = os.dup(2)
    sink = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink, 2)
        yield
    finally:
        os.dup2(original, 2)
        os.close(original)
        os.close(sink)


def parsed_time(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError()
        return result.timestamp()
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Schedule timestamps must include a timezone") from None


def validate_config(config, base: Path, once=False, allow_local_http=False):
    if not isinstance(config, dict) or set(config) - {"server", "deviceId", "classes"}:
        raise ValueError("Config accepts server, deviceId and classes; keep the device key in the environment")
    endpoint_url(config.get("server", ""), allow_local_http)
    device = config.get("deviceId", os.environ.get("CENTREWATCH_DEVICE_ID", ""))
    if not isinstance(device, str) or not 1 <= len(device) <= 100:
        raise ValueError("A registered deviceId is required")
    classes = config.get("classes")
    if not isinstance(classes, list) or not 1 <= len(classes) <= 50:
        raise ValueError("Provide 1–50 local classes")
    seen, normalized = set(), []
    for item in classes:
        if not isinstance(item, dict) or set(item) - {"classId", "source", "inventory", "submittedCount", "startsAt", "endsAt", "intervalMinutes", "coverageConfirmed", "traineeZone", "maxFrames", "sampleSeconds", "observationSeconds"}:
            raise ValueError("Unsupported class configuration fields")
        identity = item.get("classId")
        if not isinstance(identity, str) or not 1 <= len(identity) <= 100 or identity in seen:
            raise ValueError("Class IDs must be unique nonempty strings")
        seen.add(identity)
        source = item.get("source")
        if not isinstance(source, str) or not source or len(source) > 4096:
            raise ValueError("Every class needs a local RTSP source or demo file")
        live = source.lower().startswith(("rtsp://", "rtsps://"))
        if not live:
            if not once:
                raise ValueError("Continuous monitoring requires live RTSP; use --once for recorded footage")
            candidate = (base / source).resolve()
            if not candidate.is_file():
                raise ValueError("A configured local demo source does not exist")
            source = str(candidate)
        elif not urlsplit(source).hostname:
            raise ValueError("Invalid RTSP source")
        starts, ends = item.get("startsAt"), item.get("endsAt")
        if starts is not None or ends is not None:
            if parsed_time(starts) >= parsed_time(ends):
                raise ValueError("Class end must follow class start")
        elif not once:
            raise ValueError("Continuous monitoring requires startsAt and endsAt for each class")
        interval = item.get("intervalMinutes", 30)
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or not 1 <= interval <= 120:
            raise ValueError("intervalMinutes must be 1–120")
        observation = item.get("observationSeconds", 300)
        if isinstance(observation,bool) or not isinstance(observation,int) or not 300 <= observation <= 900:
            raise ValueError("observationSeconds must be 300–900")
        frames, seconds = item.get("maxFrames", 12), item.get("sampleSeconds", 30)
        if isinstance(frames, bool) or not isinstance(frames, int) or not 1 <= frames <= 1801:
            raise ValueError("maxFrames must be 1–1801")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 1 <= seconds <= 900:
            raise ValueError("sampleSeconds must be 1–900")
        inventory = item.get("inventory", [])
        if not isinstance(inventory, list) or len(inventory) > 100:
            raise ValueError("inventory must be a list with at most 100 items")
        for equipment in inventory:
            if not isinstance(equipment, dict) or not isinstance(equipment.get("name"), str) or not 1 <= len(equipment["name"]) <= 100:
                raise ValueError("Equipment requires a short class name")
            count = equipment.get("quantity")
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 10000:
                raise ValueError("Equipment quantity must be 0–10000")
            if "observable" in equipment and not isinstance(equipment["observable"], bool):
                raise ValueError("observable must be a boolean")
        submitted = item.get("submittedCount")
        if submitted is not None and (isinstance(submitted, bool) or not isinstance(submitted, int) or not 0 <= submitted <= 1000):
            raise ValueError("submittedCount must be 0–1000 or null")
        coverage = item.get("coverageConfirmed", False)
        if not isinstance(coverage, bool):
            raise ValueError("coverageConfirmed must be boolean")
        zone = item.get("traineeZone")
        if zone is not None and (not isinstance(zone,list) or len(zone)!=4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in zone) or not 0<=zone[0]<zone[2]<=1 or not 0<=zone[1]<zone[3]<=1):
            raise ValueError("traineeZone requires normalized [left,top,right,bottom]")
        normalized.append({**item, "source": source, "observationSeconds":observation, "intervalMinutes": interval, "maxFrames": frames, "sampleSeconds": seconds, "inventory": inventory, "coverageConfirmed": coverage})
    return {"server": config["server"], "deviceId": device, "classes": normalized}


class EdgeWorker:
    def __init__(self, outbox, config=None, *, config_base=None, once=False, allow_local_http=False, analyzer=None):
        self.outbox = outbox
        self.once = once
        self.allow_local_http = allow_local_http
        self.analyzer = analyzer
        connection = outbox.connection
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS worker_config(id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS worker_schedule(class_id TEXT PRIMARY KEY,next_due REAL NOT NULL,last_captured REAL,last_error TEXT);
        """)
        if config is None:
            cached = connection.execute("SELECT body FROM worker_config WHERE id=1").fetchone()
            if not cached:
                raise ValueError("No cached configuration; provide --config on first start")
            config = json.loads(cached["body"])
        self.config = validate_config(config, config_base or Path.cwd(), once, allow_local_http)
        with connection:
            connection.execute("INSERT INTO worker_config(id,body) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body", (json.dumps(self.config),))

    def capture_due(self, now=None):
        current = time.time() if now is None else now
        records = []
        for cls in self.config["classes"]:
            if now is None:
                current = time.time()
            if cls.get("startsAt") and not parsed_time(cls["startsAt"]) <= current < parsed_time(cls["endsAt"]):
                continue
            row = self.outbox.connection.execute("SELECT * FROM worker_schedule WHERE class_id=?", (cls["classId"],)).fetchone()
            if row and row["next_due"] > current and not self.once:
                continue
            captured = datetime.fromtimestamp(current, timezone.utc).isoformat()
            failure = None
            try:
                if self.analyzer is None:
                    from server.vision import analyze_source
                    self.analyzer = analyze_source
                with private_decoder_errors():
                    analysis = self.analyzer(cls["source"], cls["inventory"], options={
                        "max_frames": cls["maxFrames"], "sample_seconds": cls["sampleSeconds"],
                        "save_evidence": False, "coverage_confirmed": cls["coverageConfirmed"],
                        "observation_seconds":cls["observationSeconds"], "sample_interval_seconds":1,
                        "trainee_zone": cls.get("traineeZone"), "submitted_count": cls.get("submittedCount"),
                    })
                if not isinstance(analysis, dict) or analysis.get("error"):
                    raise RuntimeError("Analysis did not complete")
            except Exception as exc:
                # Never return exception strings: decoder errors may contain camera credentials.
                failure = type(exc).__name__
                analysis = {"count": None, "confidence": "low", "qualityFlags": ["analysis_failed", "camera_or_model_unavailable"],
                            "equipment": [{"name": i["name"], "expected": i["quantity"], "observed": None, "status": "unknown"} for i in cls["inventory"]]}
            event = make_event(self.config["deviceId"], cls["classId"], analysis, captured)
            # Enqueue first: a crash here can repeat a capture, but cannot lose a completed event.
            queued = self.outbox.enqueue(event)
            delay = min(60, cls["intervalMinutes"] * 60) if failure else cls["intervalMinutes"] * 60 * random.uniform(.8, 1.0)
            with self.outbox.connection:
                self.outbox.connection.execute("INSERT INTO worker_schedule(class_id,next_due,last_captured,last_error) VALUES(?,?,?,?) ON CONFLICT(class_id) DO UPDATE SET next_due=excluded.next_due,last_captured=excluded.last_captured,last_error=excluded.last_error", (cls["classId"], current + delay, current, failure))
            records.append({"classId": cls["classId"], "eventId": queued["event_id"], "bodyBytes": queued["bytes"], "capturedAt": captured,
                            "count": analysis.get("count"), "status": "unverified" if failure else "observed"})
        return records

    def sync(self, device_key):
        # Limit each poll to three requests × three seconds so outages cannot stall capture indefinitely.
        return self.outbox.sync(self.config["server"], device_key, limit=3, timeout=3, allow_local_http=self.allow_local_http)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Protected local configuration; cached in SQLite for restart")
    parser.add_argument("--db", type=Path, default=Path(".edge/outbox.sqlite3"))
    parser.add_argument("--once", action="store_true", help="Capture active classes once, attempt sync, then exit; allows local demonstration files")
    parser.add_argument("--allow-local-http", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be 1–60")
    key = os.environ.get("CENTREWATCH_DEVICE_KEY", "")
    if not key:
        parser.error("Set CENTREWATCH_DEVICE_KEY in the protected environment")
    outbox = Outbox(args.db)
    running = True
    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        config = json.loads(args.config.read_text()) if args.config else None
        worker = EdgeWorker(outbox, config, config_base=args.config.parent if args.config else None,
                            once=args.once, allow_local_http=args.allow_local_http)
        while running:
            captures = worker.capture_due()
            synced = worker.sync(key)
            print(json.dumps({"captured": captures, "sync": synced, "queue": outbox.status()}), flush=True)
            if args.once:
                break
            deadline = time.monotonic() + args.poll_seconds
            while running and time.monotonic() < deadline:
                time.sleep(max(0, min(1, deadline - time.monotonic())))
    except (ValueError, OSError) as exc:
        # Validation messages never contain the configured source or raw exception details.
        parser.exit(2, f"Edge worker stopped: {type(exc).__name__}; check the local configuration and device environment.\n")
    finally:
        outbox.close()


if __name__ == "__main__":
    main()
