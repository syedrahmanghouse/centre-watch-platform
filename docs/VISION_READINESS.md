# Vision readiness — 4 October 2026

The local detectors have run successfully on real images and decoded video. These checks establish working software; they are **not a general accuracy assessment on training-centre footage**.

## Classroom counting update

Attendance now uses a separate local YOLO11m person detector, overlapping crops, and duplicate removal. YOLO-World serves inventory observations only. The user's classroom clip now produces an aggregate count of 14, with sampled counts from 13–15; low-quality unstable results abstain from a reliable attendance count. See [the classroom diagnosis and measured results](CLASSROOM_COUNT_FIX.md) for the selected settings, development/reserved frame split and remaining limitations. No classroom-specific training or general accuracy percentage is claimed.

The checks below document the original YOLO-World pipeline before this change. They remain historical smoke tests rather than validation of the new person detector. Both prepared model assets are required at runtime.

## Prepared artifact

- Runtime: Python 3.12.14, Ultralytics 8.4.172, PyTorch 2.14.1, OpenCV 4.14.0.94; exact versions are pinned in `platform/requirements-vision.txt`.
- Checkpoint: `platform/models/centrewatch-world.pt`.
- SHA-256: `ca16b1da973937a043dd92039f3c8cfd9867753dc9e4ec80b161da412c9e9553`.
- Original weights SHA-256: `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792`.
- Embedded vocabulary: person, chair, workbench, table, lathe machine, drill press, sewing machine, computer monitor, laptop, fire extinguisher, whiteboard.
- Source: [official Ultralytics YOLO-World v2 weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt); [offline vocabulary documentation](https://docs.ultralytics.com/models/yolo-world/).

Preparation caches the CLIP text encoder once, saves object-name embeddings into the detector, then removes the encoder from the saved checkpoint. These embeddings describe object categories, not people. Runtime inference neither changes vocabulary nor downloads a missing model. The manifest contains hashes, versions, sources and offline verification results.

Licenses: [Ultralytics AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), [original YOLO-World GPL-3.0](https://github.com/AILab-CVC/YOLO-World/blob/master/LICENSE), and [CLIP MIT](https://github.com/openai/CLIP/blob/main/LICENSE).

## Checks actually executed

| Check | Observed result |
| --- | --- |
| Fresh checkpoint load and prediction with DNS and outbound sockets blocked | Passed; zero network attempts recorded |
| Public two-person JPEG, SHA-256 `16d73869e3267a7d4ed00de8e860833bd1657c1b252e94c0c348277adc7b6edb` | Two person detections; aggregate count 2, evidence quality medium because it is a still image |
| Three-second MP4 derived from that image with a changing timestamp overlay | Six explicit samples at 0, 0.5, 1, 1.5, 2 and 2.5 seconds each returned 2; complete local analysis took 1.057 seconds in this single development-machine run |
| Equipment image from [COCO128](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip), image ID `000000000564` | Detector emitted two person boxes and 25 chair boxes; this is an inference smoke test, not a verified room-inventory total |
| Small portrait fixture, COCO image ID `000000000283` | Chair detection ran, but aggregate assessment abstained because the frame failed the configured resolution check |
| Saved evidence inspected visually | Blank-background diagrams with labelled boxes; every original image pixel removed |
| Vision regression suite | 14 tests passed, including interrupted observations, frozen video, numerical validation, unsupported inventory and uncertain visibility |

The generated MP4 is deliberately synthetic pipeline evidence. Reusing a still image does not establish real-video accuracy, tracking quality or fraud resistance. Timings above are observations from individual runs, not a performance guarantee.

## Interpretation and remaining validation

Counts estimate visible people; a configured trainee zone can exclude the trainer's area. They do not identify trainees. Low image quality, frozen or interrupted observations, and unstable counts require review. Zero detections do not establish an empty room.

An inventory gap requires an explicitly confirmed item view, at least three usable samples, and at least 75% usable coverage of the sampled session. An item below its approved quantity in all clear samples becomes a **possible gap for review**. The model cannot prove that an unseen item is absent or determine which objects are hidden behind people. A coverage confirmation is an operator statement, not automatic proof of visibility. Operability remains unverified; idle machines are never classified as broken.

The RTSP adapter uses bounded capture and decoder timeouts. It has not yet been tested against the user's physical CCTV or DVR/NVR. Camera compatibility, centre-specific counting accuracy, false positives/negatives and workshop-equipment performance require the user's footage and independent labels. Put that footage in `data/incoming/`, then run the supplied evaluation workflow.

Reproduce the offline checkpoint check from `platform/`:

```sh
.venv/bin/python scripts/prepare_models.py --verify-only
.venv/bin/python scripts/prepare_people_model.py --verify-only
.venv/bin/python -m unittest tests.test_vision_logic -v
```
