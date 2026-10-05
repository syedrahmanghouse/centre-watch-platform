# Labelled-footage evaluation protocol

## Current evidence status

No accuracy percentage is claimed before the user's centre footage and checked ground-truth labels are supplied. The annotation example is a format illustration, not measured data. Unit tests exercise scoring arithmetic and failure handling with injected predictions; their scores are not detector performance. The offline transport smoke report is likewise unrelated to model accuracy.

The evaluation page and CLI run real inference on the supplied files and export every sampled prediction with the report. They never fill missing results with plausible-looking values.

## Place footage and prepare labels

Put original files in `centre-watch/data/incoming/`. Create a manifest under `centre-watch/data/annotations/`, starting from `manifest.example.json`. Use relative file names; parent-directory paths, absolute paths and symlinks leading outside the incoming folder are rejected.

For ordinary class creation, the administrator first assigns each incoming file to the correct institution in the Intake dashboard (`POST /api/incoming/assign` with `filename` and `institutionId`). Heads see only their assigned files. Administrator-only dataset evaluation can read its labelled files directly from the incoming folder.

```json
{
  "version": 1,
  "settings": {"attendance_tolerance": 2, "iou_threshold": 0.5},
  "videos": [{
    "file": "institutionA_room01_session01.mp4",
    "room": "Institution A / Room 01",
    "scene_id": "institutionA-camera01",
    "split": "test",
    "reported_count": 25,
    "inventory": [{"name": "chair", "quantity": 25}],
    "observations": [{
      "time_seconds": 10,
      "people": 23,
      "equipment": {"chair": 25},
      "visibility": "clear",
      "boxes": [
        {"class": "person", "bbox": [100, 80, 40, 110]}
      ]
    }]
  }]
}
```

The example box is a syntax illustration. A real `boxes` array must contain **every visible object of each evaluated class** in that frame, including all people and the inventory classes. A partial set of boxes produces invalid precision/recall. Omit `boxes` when only count labels are available; the report explicitly marks object-level false-positive/false-negative metrics unavailable. Boxes use original-image pixel `[x, y, width, height]` coordinates, never normalized or resized coordinates.

Label the actual visible people at each exact timestamp, not the number expected to be present. Include the trainer/visitors if evaluation covers the whole frame. A missing equipment key or a null count means unknown ground truth, not zero objects. Use `visibility: "occluded"` when the scene is challenging but the ground-truth count can still be independently established; `"unknown"` excludes unverifiable truth from scoring and keeps it in coverage totals. Keep timestamps unique within a file.

For inventory, choose approved quantities for the area actually covered by the camera and verify that the labelled view covers those items. Snapshot inventory metrics test count discrepancy decisions on those annotated visible areas. They do not claim that the full operational multi-frame review rule flagged the entire room.

## Dataset design and split

Include empty rooms, normal attendance, inflated and understated submissions, crowded seating, partial occlusion, poor lighting, blur, low resolution, equipment removed or obscured, a blocked camera and frozen footage. Record whether each condition was deliberately staged. Use consented demonstration footage.

Keep camera/room scenes together. Every video has a `scene_id`; absent one, its `room` is the scene group. Assign each scene to `development` or `test`. The evaluator rejects a scene appearing across both splits and duplicate source files. Use development footage to tune the detector threshold, count tolerance and camera view. Freeze those settings before evaluating test scenes. A missing split is reported as `unassigned` and is not represented as a held-out test.

The software enforces the declared split, but cannot prove that two differently named scenes are different, that labels are correct, or that pretrained-model data never included a similar scene. Have a second person review labels and split assignments. Report the small dataset's limits rather than extrapolating nationwide accuracy.

## Run

After installing the vision dependencies and preparing the local model:

```sh
cd centre-watch/platform
.venv/bin/python scripts/evaluate_dataset.py ../data/annotations/manifest.json \
  --incoming ../data/incoming --output ../data/results/evaluation-run01
```

Alternatively, an administrator can submit the manifest on the Evaluation dashboard. Jobs run in the background and publish the same downloadable artifacts: `report.json`, `predictions.json`, `predictions.csv` and the submitted `manifest.json`. Source footage is not duplicated into the output folder.

For the default `all_people` mode, the evaluator requests labelled timestamps, chunks each inference call to at most 120 frames and aligns decoded observations within 0.5 seconds. Missing frames remain unverified. A model/decode error fails the run with its reason; it is not replaced with synthetic performance. Very short or truncated files must have their labels corrected.

## Metrics and interpretation

| Metric | Calculation / meaning |
|---|---|
| Count MAE | Mean absolute difference between usable predicted and known true counts. |
| Count RMSE | Square root of mean squared count error. |
| Overcount / undercount totals | Sum of positive / negative count errors; these are **not object FP/FN**. |
| Count coverage | Scored count observations divided by every submitted observation, including unknown/abstained cases. |
| Attendance alert TP / FP / FN / TN | A mismatch is `abs(count - reported_count) > attendance_tolerance`; compare this decision using predicted versus true count. The default tolerance is 2 people. |
| Inventory alert TP / FP / FN / TN | Per class, compare `observed quantity < sanctioned quantity` against the same comparison using ground truth. |
| Unverified positives / negatives | Known true discrepancy/no discrepancy where prediction abstained or was unavailable. Kept separate from scored confusion counts. |
| Recall on assessed observations | TP / (TP + FN). Always read together with coverage. |
| Recall including unverified positives | TP / (TP + FN + unverified positives). Prevents abstention from hiding the system's inability to detect a known actionable case. |
| Object precision / recall | Confidence-ordered, one-to-one same-class IoU matching against exhaustive boxes; default IoU threshold 0.5. Unmatched predictions are FP, unmatched labels FN. |

No denominator means `null` rather than 100%. Reports contain overall and split-specific metrics. Unknown-ground-truth observations, failed view checks and unavailable detections remain visible in totals and raw predictions; they are not silently discarded. Heuristic high/medium/low evidence quality and raw detector confidence are not measured accuracy probabilities.

Snapshot alert metrics do **not** measure detection delay, false alerts per camera-hour, continued attendance, cross-view tracking or the operational rule requiring repeated clear views. For a pilot, annotate discrepancy intervals and review end-to-end scheduled alerts separately. Present this boundary clearly to judges alongside the real live workflow demonstration.


## Seated-only temporal evaluation

Set `countingMode` and `observationSeconds` on a video, or in `settings` for the whole manifest. The default remains `all_people`, preserving existing snapshot/object evaluations. A seated-only run processes every second from zero through the requested window in **one inference call**, preserving temporal history. Sparse annotations select scoring points afterward; they never select the frames fed into temporal tracking. Supported observation windows are 1–900 seconds.

```json
{
  "version": 1,
  "settings": {"countingMode": "seated_only", "observationSeconds": 300},
  "videos": [{
    "file": "classroom.mp4",
    "room": "Room A",
    "scene_id": "camera-A",
    "split": "development",
    "window_truth": {
      "people": 13,
      "visibility": "occluded",
      "provenance": "User-confirmed window occupancy; not exhaustive person annotations"
    },
    "observations": []
  }]
}
```

The `13` above illustrates a supplied label; it is never passed to inference or used to select the output. Omit `reported_count` when the submitted attendance is unknown; attendance-alert metrics then remain unassessed. `window_truth` is optional when timestamp annotations exist. Its label must describe the occupancy statistic being assessed, rather than the number enrolled or the union of everyone who entered over five minutes.

The report separates `window_metrics`, `window_predictions`, and `window_coverage` from existing snapshot `metrics`. A low-confidence aggregate is an abstention, even if its diagnostic candidate equals the label. `candidate_absolute_error_diagnostic_only` describes a candidate's difference from known truth; it is not a scored verified prediction. An incomplete, sparse, or too-short capture cannot earn window accuracy. The report retains such windows in coverage.

For seated-only frame annotations, `people` means seated occupants, including independently established short occlusions where labelled. Person boxes must each declare `posture`: `seated`, `standing`, or `unknown`. Object matching compares visible, included seated detections against seated boxes. Any unknown person-posture box leaves that frame's object scores unassessed. Exhaustive boxes remain mandatory for object FP/FN; a window count alone supplies no such ground truth.

Standing posture does not prove someone is a trainer, and seated posture does not prove someone is a trainee. A standing person leaning over a desk can resemble a seated person when legs are hidden. Posture uncertainty, temporal exclusions and brief occlusion support must be reported with count coverage. This workflow is pretrained inference plus temporal processing, not evidence of classroom-specific training.
