#!/usr/bin/env python3
"""Durable centre-device metadata outbox. Video never leaves through this client."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

MAX_EVENT_BYTES = 64 * 1024
MAX_QUEUE_EVENTS = 100_000
MAX_ATTEMPTS = 8


def compact_analysis(analysis):
    """Positive field selection: no video, frame, boxes, faces, credentials or paths."""
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be an object")
    count = analysis.get("count")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 1000):
        raise ValueError("count must be an integer from 0 to 1000, or null")
    flags = analysis.get("qualityFlags", [])
    if not isinstance(flags, list) or len(flags) > 30 or any(not isinstance(v, str) or len(v) > 200 for v in flags):
        raise ValueError("qualityFlags must contain at most 30 short strings")
    inventory = analysis.get("equipment", [])
    if not isinstance(inventory, list) or len(inventory) > 100:
        raise ValueError("equipment must contain at most 100 items")
    equipment = []
    for item in inventory:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not 1 <= len(item["name"]) <= 100:
            raise ValueError("Equipment requires a short name")
        converted = {"name": item["name"]}
        for field in ("expected", "observed"):
            value = item.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10000):
                raise ValueError(f"equipment {field} must be a count or null")
            converted[field] = value
        status = item.get("status", "unknown")
        if status not in ("present", "gap", "missing", "unknown", "compliant", "review", "insufficient", "verified", "insufficient_evidence"):
            status = "unknown"
        converted["status"] = status
        equipment.append(converted)
    confidence = analysis.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    return {"count": count, "confidence": confidence, "qualityFlags": flags, "equipment": equipment}


def make_event(device_id, class_id, analysis, captured_at=None):
    if not all(isinstance(v, str) and 1 <= len(v) <= 100 for v in (device_id, class_id)):
        raise ValueError("Provide the registered device ID and assigned class ID")
    timestamp = captured_at or datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("captured_at must include a timezone")
    return {"event_id": str(uuid.uuid4()), "device_id": device_id, "occurred_at": timestamp,
            "event_type": "analysis", "payload": {"classId": class_id, "analysis": compact_analysis(analysis)}}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def endpoint_url(base_url, allow_local_http=False):
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Server URL must not contain credentials, query parameters or a fragment")
    if not parsed.hostname:
        raise ValueError("Invalid server URL")
    local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if parsed.scheme != "https" and not (allow_local_http and local and parsed.scheme == "http"):
        raise ValueError("HTTPS is required; --allow-local-http permits only loopback development servers")
    return base_url.rstrip("/") + "/api/edge/events"


class Outbox:
    def __init__(self, path: Path | str):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not target.exists():
            target.touch(mode=0o600)
        self.connection = sqlite3.connect(target, timeout=15)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS outbox (
              event_id TEXT PRIMARY KEY, body TEXT NOT NULL, body_bytes INTEGER NOT NULL,
              state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
              next_attempt REAL NOT NULL DEFAULT 0, created REAL NOT NULL,
              acknowledged REAL, last_error TEXT);
            CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
        """)
        self.connection.commit()
        try:
            target.chmod(0o600)
        except OSError:
            pass

    def close(self):
        self.connection.close()

    def enqueue(self, event):
        uuid.UUID(event["event_id"])
        if set(event) != {"event_id", "device_id", "occurred_at", "event_type", "payload"}:
            raise ValueError("Unsupported event fields")
        if event["event_type"] != "analysis" or set(event["payload"]) != {"classId", "analysis"}:
            raise ValueError("Only analysis metadata events are supported")
        # Reconstruct to enforce field selection even for caller-built dictionaries.
        safe = make_event(event["device_id"], event["payload"]["classId"], event["payload"]["analysis"], event["occurred_at"])
        safe["event_id"] = event["event_id"]
        body = json.dumps(safe, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        size = len(body.encode("utf-8"))
        if size > MAX_EVENT_BYTES:
            raise ValueError("Metadata event exceeds 64 KiB")
        with self.connection:
            existing = self.connection.execute("SELECT body FROM outbox WHERE event_id=?", (event["event_id"],)).fetchone()
            if existing:
                if existing["body"] != body:
                    raise ValueError("An event ID cannot be reused with a different payload")
                return {"event_id": event["event_id"], "bytes": size, "already_queued": True}
            count = self.connection.execute("SELECT COUNT(*) FROM outbox WHERE state != 'acked'").fetchone()[0]
            if count >= MAX_QUEUE_EVENTS:
                raise ValueError("Outbox is full; sync or inspect failed events before capturing more")
            self.connection.execute("INSERT INTO outbox(event_id,body,body_bytes,created) VALUES(?,?,?,?)", (event["event_id"], body, size, time.time()))
        return {"event_id": event["event_id"], "bytes": size, "already_queued": False}

    def _increment(self, name, amount):
        self.connection.execute("INSERT INTO counters(name,value) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value", (name, amount))

    def sync(self, base_url, device_key, *, limit=100, timeout=10, allow_local_http=False, now=None):
        endpoint = endpoint_url(base_url, allow_local_http)
        if not device_key or len(device_key) > 500 or "\n" in device_key or "\r" in device_key:
            raise ValueError("A valid X-Device-Key is required")
        if not 1 <= limit <= 1000 or not .1 <= timeout <= 60:
            raise ValueError("limit must be 1–1000 and timeout 0.1–60 seconds")
        current = time.time() if now is None else now
        rows = self.connection.execute("SELECT * FROM outbox WHERE state='pending' AND next_attempt<=? ORDER BY created LIMIT ?", (current, limit)).fetchall()
        opener = urllib.request.build_opener(NoRedirect())
        result = {"attempted": 0, "acked": 0, "deferred": 0, "failed": 0, "request_body_bytes": 0, "response_body_bytes": 0}
        for row in rows:
            payload = row["body"].encode("utf-8")
            request = urllib.request.Request(endpoint, data=payload, method="POST", headers={
                "Content-Type": "application/json", "X-Device-Key": device_key, "User-Agent": "CentreWatch-Edge/1.0"})
            error, permanent, success, response_size = None, False, False, 0
            try:
                with opener.open(request, timeout=timeout) as response:
                    # Require the API's explicit acknowledgment for this event.
                    data = response.read(MAX_EVENT_BYTES)
                    response_size = len(data)
                    try:
                        acknowledgment = json.loads(data)
                    except (ValueError, UnicodeDecodeError):
                        acknowledgment = None
                    success = (200 <= response.status < 300 and isinstance(acknowledgment, dict)
                               and acknowledgment.get("accepted") is True and acknowledgment.get("eventId") == row["event_id"])
                    if not success:
                        error = f"HTTP {response.status}: missing matching event acknowledgment"
            except urllib.error.HTTPError as exc:
                response_size = len(exc.read(MAX_EVENT_BYTES))
                error = f"HTTP {exc.code}"
                permanent = 400 <= exc.code < 500 and exc.code not in (408, 425, 429)
                exc.close()
            except (urllib.error.URLError, TimeoutError, OSError):
                error = "Connection failed or timed out"
            attempts = row["attempts"] + 1
            with self.connection:
                self._increment("attempted_request_body_bytes", len(payload))
                self._increment("response_body_bytes", response_size)
                self._increment("requests_attempted", 1)
                if success:
                    self.connection.execute("UPDATE outbox SET state='acked',attempts=?,acknowledged=?,last_error=NULL WHERE event_id=?", (attempts, current, row["event_id"]))
                    self._increment("acknowledged_event_bytes", len(payload))
                    result["acked"] += 1
                else:
                    dead = permanent or attempts >= MAX_ATTEMPTS
                    delay = min(3600, 5 * 2 ** (attempts - 1)) + random.uniform(0, 5)
                    self.connection.execute("UPDATE outbox SET state=?,attempts=?,next_attempt=?,last_error=? WHERE event_id=?", ("failed" if dead else "pending", attempts, current + delay, error, row["event_id"]))
                    result["failed" if dead else "deferred"] += 1
            result["attempted"] += 1
            result["request_body_bytes"] += len(payload)
            result["response_body_bytes"] += response_size
        return result

    def status(self):
        states = {row["state"]: row["count"] for row in self.connection.execute("SELECT state,COUNT(*) AS count FROM outbox GROUP BY state")}
        stats = {row["name"]: row["value"] for row in self.connection.execute("SELECT * FROM counters")}
        pending_bytes = self.connection.execute("SELECT COALESCE(SUM(body_bytes),0) FROM outbox WHERE state!='acked'").fetchone()[0]
        return {"states": states, "queued_bytes": pending_bytes, "counters": stats,
                "byte_scope": "Serialized UTF-8 HTTP bodies only; attempted bytes may include requests that failed before transmission. TLS, HTTP headers and transport overhead excluded."}

    def retry_failed(self):
        with self.connection:
            cursor = self.connection.execute("UPDATE outbox SET state='pending',attempts=0,next_attempt=0 WHERE state='failed'")
        return cursor.rowcount

    def prune(self, older_than_days=30):
        if not 1 <= older_than_days <= 3650:
            raise ValueError("Retention must be 1–3650 days")
        with self.connection:
            cursor = self.connection.execute("DELETE FROM outbox WHERE state='acked' AND acknowledged<?", (time.time() - older_than_days * 86400,))
        return cursor.rowcount


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(".edge/outbox.sqlite3"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("enqueue", "capture"):
        command = commands.add_parser(name)
        command.add_argument("--device-id", default=os.getenv("CENTREWATCH_DEVICE_ID"), required=not bool(os.getenv("CENTREWATCH_DEVICE_ID")))
        command.add_argument("--class-id", required=True)
        if name == "enqueue":
            command.add_argument("--analysis", type=Path, required=True, help="Local analyzer result JSON")
        else:
            command.add_argument("--source", required=True, help="Local video/image path or local RTSP URL")
            command.add_argument("--inventory", type=Path, help="JSON list of {name,quantity} items")
            command.add_argument("--max-frames", type=int, default=12)
    command = commands.add_parser("sync")
    command.add_argument("--server", required=True)
    command.add_argument("--limit", type=int, default=100)
    command.add_argument("--timeout", type=float, default=10)
    command.add_argument("--allow-local-http", action="store_true")
    commands.add_parser("status")
    commands.add_parser("retry-failed")
    command = commands.add_parser("prune")
    command.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    outbox = Outbox(args.db)
    try:
        if args.command in ("enqueue", "capture"):
            captured_at = datetime.now(timezone.utc).isoformat()
            if args.command == "enqueue":
                analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
            else:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from server.vision import analyze_source
                inventory = json.loads(args.inventory.read_text(encoding="utf-8")) if args.inventory else []
                if not 1 <= args.max_frames <= 120:
                    raise ValueError("max-frames must be 1–120")
                analysis = analyze_source(args.source, inventory, options={"max_frames": args.max_frames, "save_evidence": False})
                if analysis.get("error"):
                    raise ValueError(str(analysis["error"]))
            result = outbox.enqueue(make_event(args.device_id, args.class_id, analysis, captured_at))
        elif args.command == "sync":
            result = outbox.sync(args.server, os.getenv("CENTREWATCH_DEVICE_KEY", ""), limit=args.limit, timeout=args.timeout, allow_local_http=args.allow_local_http)
        elif args.command == "status":
            result = outbox.status()
        elif args.command == "retry-failed":
            result = {"retried": outbox.retry_failed()}
        else:
            result = {"deleted_acknowledged": outbox.prune(args.days)}
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        parser.exit(2, f"Edge client error: {exc}\n")
    finally:
        outbox.close()


if __name__ == "__main__":
    main()
