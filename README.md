# CentreWatch

A working local monitoring platform for SIH problem statement 26245: attendance verification and infrastructure observations at training centres.

**Use `platform/` for the complete application.** The older `dist/` website is a separate browser-only demonstration. Its private hosted URL does not run the Python service described here.

## Start on this machine

```sh
cd centre-watch/platform
./run.sh
```

Open **http://127.0.0.1:8787**. Create the first administrator account in the browser. There are no default credentials or automatic sample records. Dependencies and the prepared offline model have already been installed in this workspace.

On a fresh machine, install Python 3.12, then run `./install.sh` once with internet access. It creates an isolated virtual environment, installs the dependencies, prepares the separate person and inventory detectors, and tests them without network access. Subsequent application and inference runs use local assets. For an existing installation, run `.venv/bin/python scripts/prepare_people_model.py` from `platform/` to prepare the dedicated person detector.

## Your CCTV videos go here

```text
data/incoming/
```

Absolute path on this machine:

```text
/Users/syedrahman/Desktop/SIH-Project/PostAI/centre-watch/data/incoming/
```

MP4 is recommended. MOV, AVI, MKV, WebM, JPG and PNG are also supported, subject to the file’s installed decoder. Use a separate file for each room/camera. Do not put credentials or trainee names in filenames.

After copying files:

1. **Admin:** create an institution and its head account. In **System & cameras → Assign footage**, assign the incoming file to that institution.
2. **Institution head:** create trainers, then create classes. Choose an assigned incoming file, upload a photo/video, or provide a live RTSP camera URL. Enter sanctioned equipment quantities and the class schedule.
3. **Institution head:** choose the counting mode (default: **Seated people only**) and observation length (default: **300 seconds**, configurable to 900). Confirm inventory coverage only after checking the camera view. Optionally specify the trainee seating region to exclude the trainer’s area.
4. **Trainer:** sign in and submit attendance for an assigned class. The server queues and runs actual local inference.
5. **Institution head:** inspect mismatches, actual observation duration, seated/standing/posture uncertainty, infrastructure gaps and history; record an explanation or request follow-up. Live checks need actual capture time. Short uploads cover only their available duration and are labelled incomplete for a five-minute request.
6. **Admin:** review daily institution/class reports and export CSV. Submitted and observed counts remain separate after review.

Uploaded sources expire after 24 hours by default; annotated camera snapshots and legacy diagrams after 30 days. Snapshot retention is independent of source expiry. The administrator can change those periods. Incoming originals are never automatically deleted. Real footage, runtime databases, keys, model weights, annotations, and test environments are excluded from Git and container build contexts.

## What is implemented

| Outcome | Working implementation | Validation still needed |
| --- | --- | --- |
| Video analytics | OpenCV decoding; offline YOLO11m person detection with overlapping crops and duplicate removal; pretrained posture classification for seated-only counting; temporary spatial presence support through brief occlusions; separate YOLO-World inventory checks; requested/actual observation duration | Independent labelled centre footage, seated/standing posture labels, full five-minute recordings and a real camera field test |
| Monitoring dashboards | Actual admin/head/trainer accounts, tenant-scoped access, classes, attendance, durable jobs, discrepancies, infrastructure observations, review history and CSV reports | User workflow review with actual institutions |
| Privacy design | No facial identification; local inference; restricted camera snapshots with frame-local detection numbers; authenticated source/evidence access; retention and audit records; metadata-only edge sync | Site-specific operating policy and human consent/notice procedures |
| Accuracy assessment | Ground-truth manifest runner, exact timestamp comparisons, count MAE/RMSE, alert TP/FP/FN/TN, abstention coverage, optional bounding-box detection metrics and exports | User clips + independent labels; no classroom accuracy percentage is claimed yet |
| Low bandwidth | Offline model assets and UI, local SQLite, durable metadata outbox, reconnect retry and duplicate-safe receiver; continuous edge worker | Real rural connectivity, device throughput and camera coverage measurements |

### Technical boundaries

- Seated mode counts confirmed **seated people**, excludes standing people and requires review for uncertain posture. Posture does not identify students or trainers: a seated trainer can count and a standing student is excluded. All-visible-people mode remains available explicitly.
- Each check requests 5–15 minutes, default 5. Uploaded video is analysed only for the available duration; a short recording or photo cannot demonstrate a completed five-minute observation. Live RTSP capture requires the requested wall-clock time.
- A missing detection may be supported by recent seated evidence for at most **3 seconds**. A person who remains **visible and spatially stable** may separately use seated posture confirmed within **30 seconds** if the current posture is unclear; standing detections are excluded immediately. Posture-history use is shown in the frame table. The count is simultaneously supported presence, not the maximum count or every unique person seen across a clip. No face identification or cross-session person matching is performed.
- The detectors and posture classifier use pretrained models with inference logic; they have not been newly trained to achieve perfect classroom accuracy. A matching aggregate count cannot establish who attended.
- Supported equipment labels are recorded in `platform/models/vocabulary.json`. Unknown items stay unverified; extend and evaluate the vocabulary before claiming support.
- An equipment detection does not establish operability. Idle machines are never labelled broken. Functional assessment needs an appropriate observable test or manual inspection.
- Visibility confirmation is a human calibration step. Occlusion and out-of-frame areas can still hide people or equipment.
- RTSP works with cameras/NVRs that expose compatible streams. Universal camera compatibility and automatic PTZ scanning are not claimed.
- Periodic checks run on **live RTSP sources during scheduled class hours**, with jitter around the configured interval. Saved clips run on submission or manual verification.
- The current deployment is a single application instance with a persistent SQLite database and a single durable inference worker. It is suitable for a working demonstration and an initial measured pilot; national-scale performance has not been measured.
- Public hosting of the backend requires a host with Python/ML compute, persistent disk, TLS, and access to cameras or centre edge workers. The earlier static Site cannot execute this service.

## Stack and source map

| Layer | Implementation |
| --- | --- |
| Dashboard | Native ES modules, HTML/CSS, accessible forms, responsive layout; no CDN dependencies |
| API / access | FastAPI, Pydantic, scrypt passwords, hashed server sessions, HttpOnly cookies, CSRF and origin checks |
| Records / jobs | SQLite WAL, persistent work queue, versioned submissions, audit records |
| Vision | Python, OpenCV, PyTorch, YOLO11m for people, pretrained posture classification and temporary spatial presence support, prepared YOLO-World vocabulary for inventory; local model assets |
| Edge | Local detector + durable SQLite outbox, compact allowlisted metadata, backoff and event UUID deduplication |
| Deployment | Isolated Python environment; optional Docker Compose; local files and mounted volumes |

- `platform/server/`: API, access control, jobs, inference, evaluation.
- `platform/web/`: authenticated dashboard.
- `platform/scripts/`: model preparation, dataset evaluation, edge worker/sync and integration checks.
- `platform/tests/`: meaningful access, data integrity, vision aggregation, evaluation and offline retry tests.
- `platform/requirements.lock.txt`: versions installed for the verified macOS Python 3.12 environment. Fresh installs use the pinned top-level requirement files; other OS/GPU wheels may differ.
- `docs/PRIVACY_DESIGN.md`: data identified, privacy boundaries, retention and access.
- `docs/ACCURACY_PROTOCOL.md`: annotation format and metrics.
- `docs/LOW_BANDWIDTH_DEPLOYMENT.md`: offline inference, continuous worker, metadata sync and measured transport smoke test.
- `docs/INTEGRATION_READINESS.md`: verified backend flow and current boundaries.
- `docs/CLASSROOM_COUNT_FIX.md`: classroom overcount diagnosis, detector changes, measured counts and validation limits.

## Dataset evaluation

Make a copy of `data/annotations/manifest.example.json`, replace the filenames and independently label the headcounts/equipment at its timestamps. Do not copy model predictions into ground truth. Keep development and test cameras/rooms separate.

As administrator, open **Accuracy lab → Run evaluation** and upload that JSON manifest. Results and raw prediction exports appear when the job finishes. The video files remain in the intake directory; the manifest does not contain footage.

CLI details and edge worker instructions are in their dedicated documents. Only use a reported score for the exact labelled dataset it describes.

## Verification

```sh
cd centre-watch/platform
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
node --check web/app.js
```

The HTTP integration script uses an isolated temporary database and explicit source path. It never creates accounts in your application’s real database. Public-photo/video fixtures prove the pipeline works; they are not representative classroom accuracy evidence.

## Container deployment

Prepare model assets locally with `platform/install.sh`, then from `centre-watch/` run:

```sh
docker compose up --build
```

The container binds to loopback, persists records in a named volume, mounts incoming footage read-only, and mounts prepared models read-only. Docker execution has not been verified in this workspace. For a remote pilot, terminate HTTPS at a reverse proxy, set `CW_PUBLIC_ORIGIN` and `CW_SECURE_COOKIES=1`, and protect the persistent data and camera key. Complete first-admin setup before making the service reachable to others.

## Model attribution

The model/runtime use Ultralytics YOLO11 for people, YOLO-World for inventory, and OpenAI CLIP for one-time vocabulary preparation. Check their published licensing before distributing a commercial or closed-source deployment. See the model manifests and vision readiness notes for the actual artifact identities.
