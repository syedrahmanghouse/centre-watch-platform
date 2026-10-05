# Local inference and low-bandwidth operation

## Implemented mode

The centre service reads a local file or local camera stream, samples frames and runs a prepared detector at the centre. `scripts/edge_worker.py` continuously connects scheduled local inference to the durable queue and central synchronization. It uses `scripts/offline_sync.py`, a SQLite outbox that sends compact JSON through HTTPS. No video or image data is sent by that client. The central service validates a registered device key, restricts the class to that device's institution and deduplicates the device/event UUID.

Pending events survive client restart. An event is acknowledged only when the API returns `accepted: true` with its matching event ID; an unrelated HTTP 200 page cannot clear the queue. Transient errors retry with exponential backoff and jitter, capped at one hour; a sync invocation only processes events currently due. After eight attempts, or a permanent 4xx error, the event is retained as failed for review. Fix the cause and explicitly retry it. Authentication errors never loop indefinitely. Redirects are refused so device keys are not forwarded to another URL.

Keep inference on the centre device. Uploading a large video through a browser connected directly to a remote server still consumes video-upload bandwidth. Use the local service or the edge capture command for limited-connectivity deployment.

## Prepare once while connected

Use the same Python environment as the platform service. From `centre-watch/platform/`:

```sh
.venv/bin/python -m pip install -r requirements-vision.txt
.venv/bin/python scripts/prepare_models.py
.venv/bin/python scripts/prepare_people_model.py
.venv/bin/python scripts/prepare_posture_model.py
.venv/bin/python scripts/prepare_models.py --verify-only
.venv/bin/python scripts/prepare_people_model.py --verify-only
.venv/bin/python scripts/prepare_posture_model.py --verify-only
```

Preparation downloads the separate person, inventory and posture model assets, writes the prepared manifests and records checksums. Posture uses pretrained pose and CLIP evidence; preparation does not train a classroom-specific model. The verify-only commands load the models and run inference in child processes that block outbound socket connections. Retain the full environment and the prepared `models/` directory on the centre device. Do not delete the cache and expect a disconnected device to recreate it.

The checked-in default vocabulary covers people, chairs, workbenches, tables and selected workshop equipment. Prepare an explicit custom vocabulary while online for additional items. A prepared class name does not establish detection accuracy; validate it on the centre's footage. Unsupported items remain unknown. Do not claim every type of workshop machinery can be recognised reliably by the default model.

## Register and capture

An institution head or administrator registers a device through `POST /api/devices` with a name (and institution ID for an administrator). Save the returned ID and one-time key securely; the key is scoped to that institution. `GET /api/edge/config` accepts `X-Device-Key` and returns that institution's classes and schedules.

Set `CENTREWATCH_DEVICE_ID` and `CENTREWATCH_DEVICE_KEY` in the centre machine's protected environment. Do not include the key in source code, a public `.env` file or screenshots. The sync command reads it from the environment, keeping it out of process arguments.

## Continuous scheduled edge worker

Create a private `.edge/centre.json` on the centre machine. Use the class IDs registered on the central server; keep attendance submissions in the central class record. All camera sources stay local. Example:

```json
{
  "server": "https://YOUR_CENTRAL_SERVER",
  "deviceId": "REGISTERED_DEVICE_ID",
  "classes": [{
    "classId": "CENTRAL_CLASS_ID",
    "source": "rtsp://CAMERA_HOST/STREAM_PATH",
    "startsAt": "2026-10-05T09:00:00+05:30",
    "endsAt": "2026-10-05T11:00:00+05:30",
    "intervalMinutes": 30,
    "submittedCount": 25,
    "inventory": [{"name": "chair", "quantity": 25, "observable": true}],
    "coverageConfirmed": true,
    "traineeZone": [0.05, 0.15, 0.95, 0.95],
    "countingMode": "seated_only",
    "observationSeconds": 300,
    "maxFrames": 301,
    "sampleSeconds": 300
  }]
}
```

The schedule is an actual class date and includes its time zone; update it for new sessions. Relative demo file paths resolve from the configuration file's directory. Optional `traineeZone` is a normalized rectangle `[left, top, right, bottom]`, with increasing coordinates between 0 and 1; polygons are not accepted. Only confirm inventory coverage after checking the view. `submittedCount` provides local session context; synchronization does not modify the trainer's central attendance submission.

The example requests a five-minute seated observation, targeting one sample per second with a budget of 301 frames. `observationSeconds` must be an integer from 300 to 900; seated mode requires `maxFrames` of at least `observationSeconds + 1`. Set `sampleSeconds` to the same requested duration. `countingMode` can explicitly be `all_people` for a check that does not filter posture.

Live capture drains the decoder continuously and gives inference the latest available frame rather than an old frame buffered while inference was busy. Five-minute live capture requires actual elapsed time. If the device cannot process samples fast enough, gaps are recorded and can make the result unverified; this does not create continuous coverage. Measure throughput on the intended edge hardware before claiming a reliable sample rate.

Use a fixed camera and calibrated region. Seated mode excludes standing detections immediately, can support a missing detection for at most three seconds, and can reuse seated posture for up to 30 seconds only while the person remains visible and spatially stable. Unknown posture, ambiguous position matching, sparse samples and camera movement require review. A seated trainer can still count and a standing student is excluded; posture does not identify roles. No face matching or cross-camera identities are created.

```sh
chmod 600 .edge/centre.json
.venv/bin/python scripts/edge_worker.py --config .edge/centre.json \
  --db .edge/outbox.sqlite3 --poll-seconds 15
```

The worker captures active, due classes, queues the actual result and attempts bounded synchronization (at most three requests per poll, each with a three-second timeout). A 30-minute configured interval uses a random 24–30 minute delay between captures to reduce predictability. Failed inference produces an explicit unverified event with `count: null` and retries analysis after at most one minute. Native decoder error output is suppressed because it may contain camera credentials; the metadata records generic camera/model failure flags.

The SQLite database caches class configuration and next due times so a restart can operate without fetching remote configuration. Restart with the same `--db` and omit `--config` to use that cache. Local configuration may contain camera credentials; its cached copy stays in the protected centre database and is not included in events or console status. The device key is never written into the configuration cache and must be restored through the protected process environment. Stop with Ctrl+C or the service manager. The same command can be managed as a restartable operating-system service.

Continuous mode requires live RTSP and explicit start/end times; it will not repeatedly pass a recorded file off as fresh attendance. For a local footage demonstration, replace `source` with the footage path, omit the schedule or set an active one, and run once:

```sh
.venv/bin/python scripts/edge_worker.py --config .edge/demo.json \
  --db .edge/demo-outbox.sqlite3 --once
```

This invokes the real local model, writes the real metadata result and attempts synchronization before exiting. If disconnected, the queued event remains available to the next worker or sync run. `--once` honours a supplied schedule and processes a local file only when that session is active; omitting the schedule is useful for a deliberate file-based demonstration.

Keep the same seated-mode and 300-second settings for an offline demonstration. A shorter file is analysed only for its available duration, and its observation window is labelled incomplete. A ten-second clip is not treated as five minutes of attendance. Requested/observed duration, counting mode, count range and quality flags travel as metadata; raw frames, posture crops and appearance features do not.

An earlier `--once` model → SQLite queue → localhost HTTP receiver test passed and is recorded in `data/results/edge-worker-pipeline-smoke.json`: one acknowledged event, 279 body bytes, no raw media transferred. Its source was an explicitly synthetic demonstration video, so it is historical transport/pipeline evidence, not validation of the five-minute seated workflow or an accuracy score. Reproduce using your footage:

```sh
.venv/bin/python scripts/edge_worker_smoke_test.py \
  --source ../data/incoming/room01.mp4 \
  --output ../data/results/edge-worker-pipeline-smoke.json
```

## Individual capture and sync commands

The low-level `offline_sync.py capture` command performs a bounded all-visible-people check; use `edge_worker.py --once` with the configuration above for the five-minute seated workflow. Inventory JSON is a list of `{ "name": "chair", "quantity": 25, "observable": true }` items. Mark an item observable only after checking that its entire relevant area is covered.

```sh
.venv/bin/python scripts/offline_sync.py --db .edge/outbox.sqlite3 capture \
  --class-id CLASS_ID --source ../data/incoming/room01.mp4 \
  --inventory ../data/annotations/room01-inventory.json --max-frames 12

.venv/bin/python scripts/offline_sync.py --db .edge/outbox.sqlite3 status

.venv/bin/python scripts/offline_sync.py --db .edge/outbox.sqlite3 sync \
  --server https://YOUR_CENTRAL_SERVER --limit 100
```

For cameras, pass the locally reachable RTSP source instead of a file, using the centre's protected camera configuration. Avoid exposing credentials in saved shell history. The source itself is stripped from the uploaded event. `enqueue --class-id CLASS_ID --analysis result.json` queues an existing local analyzer result instead of running inference again.

The low-level sync client is deliberately finite: it handles currently due events then exits. Use the continuous worker above for automatic capture and synchronization. Scheduled checks in the full local application continue using its own persistent jobs; the separate edge worker is for metadata forwarding to a central instance. Do not run duplicate capture schedulers for the same class.

For a localhost development service only, add `--allow-local-http --server http://127.0.0.1:8000`. Remote HTTP is rejected even with this switch.

## Reconnect and manage the queue

```sh
.venv/bin/python scripts/offline_sync.py --db .edge/outbox.sqlite3 status
.venv/bin/python scripts/offline_sync.py --db .edge/outbox.sqlite3 retry-failed
.venv/bin/python scripts/offline_sync.py --db .edge/outbox.sqlite3 sync \
  --server https://YOUR_CENTRAL_SERVER
.venv/bin/python scripts/offline_sync.py --db .edge/outbox.sqlite3 prune --days 30
```

`retry-failed` preserves event UUIDs while resetting retry counters. `prune` deletes only acknowledged events older than the selected period; it never deletes pending data. A full queue stops new enqueue operations with an error rather than silently losing data. The current limit is 100,000 unacknowledged events, and each event is limited to 64 KiB. SQLite holds only metadata; protect the device and its database with normal operating-system permissions.

## Measured transport test

`data/results/offline-transport-smoke.json` records an actual loopback HTTP test. A receiver initially returned HTTP 503, the client restarted with its SQLite queue intact, and a second request succeeded using the same UUID and exact payload.

| Measurement | Observed |
|---|---:|
| Unique event delivered | 1 |
| HTTP requests including retry | 2 |
| JSON body for the test event | 322 bytes |
| Request bodies actually received, including retry | 644 bytes |
| Response bodies actually received | 80 bytes |
| Original image/video bytes sent | 0 |

The counts inside this test are synthetic transport metadata, not an accuracy dataset. These measurements exclude HTTP headers, TLS setup and network transport overhead. They do not establish a fixed bandwidth budget per camera per day. Actual consumption depends on item count, event frequency, reconnect attempts and the deployment's protocol overhead.

Reproduce and replace the report on the target device:

```sh
.venv/bin/python scripts/offline_smoke_test.py \
  --output ../data/results/offline-transport-smoke.json
```

The status counters distinguish attempted request-body bytes from acknowledged-event bytes. A request failing before transmission can still increase the attempted counter; the smoke receiver's byte count measures what actually arrived. For a field network budget, record interface/router counters alongside these application counters.

## Demonstrate a real disconnection

1. Prepare and verify the model while connected; run one known footage sample successfully.
2. Disconnect the internet while keeping the local CCTV network available. Stop and restart the service/client to demonstrate that the model was truly cached.
3. Run `edge_worker.py --once` with a five-minute live view or independently labelled recording and inspect the queue. Check its actual observed duration and quality flags. A short recording must remain an incomplete window; failed or uncertain evidence must not become an invented count.
4. Attempt sync while the central host is unavailable. Confirm the event remains pending and a future retry time is recorded.
5. Reconnect, run sync after its retry deadline and verify that the central class gets one event with the original capture time.
6. Inspect the event and interface counters to confirm no raw footage left through the edge client.

The repository's automated test verifies a server-failure/restart/retry path. A physical rural-link test and throughput measurements on the intended edge hardware remain deployment checks; neither is claimed to have been performed on user footage.
