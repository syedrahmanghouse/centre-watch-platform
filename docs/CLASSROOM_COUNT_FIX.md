# Classroom count investigation

This report evaluates the dedicated **all-visible-people** detector correction.
The later seated-only posture and temporal observation workflow requires its own
labelled evaluation; the figures below are not seated-posture accuracy scores or
proof of five-minute attendance.

## What caused the reported 16

The supplied 10.95-second, 640 × 352 classroom clip contains 14 students according
to the user. The original YOLO-World small detector returned frame counts
`13, 17, 12, 11, 15, 16, 20, 17, 16, 20, 17`; the displayed median was 16.
Reviewing the detection overlays found duplicate torso/full-body boxes, occasional
chair/backpack false detections, and missed students bending over their desks.

Reducing overlap tolerance alone produced a median of 14, but visible misses and
false positives sometimes cancelled each other. A correct total alone is not
evidence that every person was detected correctly.

## Selected correction

Use a separate pretrained **YOLO11m person detector** for attendance; keep
YOLO-World for infrastructure. This is an inference correction, **not newly
trained model weights**.

The selected configuration was fixed before evaluating the five reserved
timestamps:

- 960-pixel inference size, confidence 0.25, predictor NMS IoU 0.70.
- Whole frame plus four corner crops, each 65% of the image width and height.
- Reject boxes within 5 pixels of an artificial crop edge to avoid counting cut
  bodies as additional people; original image edges remain valid.
- Merge all detections in original image coordinates, highest score first;
  suppress duplicates with IoU greater than 0.50 or intersection over the smaller
  box area greater than 0.90.
- Aggregate sampled frame counts; retain the count range and review uncertainty.
  Submitted attendance never determines detector output.

Official weights: [Ultralytics YOLO11m release](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt).
SHA256: `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`.
[Model documentation](https://docs.ultralytics.com/models/yolo11/).

## Measured comparison

Eleven uniformly spaced timestamps were sampled. The first six were used to
choose the configuration; the remaining five were reserved. The constant count
of 14 is the user's clip-level label, assumed valid throughout this clip.
MAE below means **mean absolute count error**, not detection precision or recall.

| Configuration | Development counts (first 6) | Development count MAE |
| --- | --- | ---: |
| Original YOLO-World, 640 | 13, 17, 12, 11, 15, 16 | 2.00 |
| YOLO11m, whole frame, 1280, IoU 0.45 | 13, 13, 13, 14, 13, 13 | 0.83 |
| YOLO11l, whole frame, 1280, IoU 0.45 | 14, 13, 13, 14, 15, 14 | 0.50 |
| YOLO11x, whole frame, 1280, IoU 0.45 | 15, 14, 14, 13, 14, 15 | 0.50 |
| Selected tiled YOLO11m | 14, 13, 13, 14, 14, 14 | 0.33 |

Selected configuration on the reserved five timestamps: **15, 14, 14, 14, 14**;
median **14**, count MAE **0.20**. One reserved frame retained a duplicate box.
No parameters were changed after those results. Across all eleven timestamps,
the median is **14**, range **13–15**, and count MAE **0.27**; the original count
MAE was **2.91**. A separate public two-person photograph returned two people
with the selected configuration.

The integrated `analyze_source` pipeline reproduced all eleven selected counts
and returned 14 with `medium` evidence quality and a 13–15 range. This local run
took 8.015 seconds for the eleven samples; that is a measurement on the development
computer, not a deployment latency guarantee. The evidence diagrams show the
representative 14-person sample followed by the 13- and 15-person extremes, with
all source pixels removed.

Subsequent snapshot-display update: dashboard analysis now retains those sampled
camera frames and overlays numbered person boxes, under authenticated class access.
The detector and counting configuration are unchanged. Person numbers apply only
within each snapshot. The abstract-diagram mode remains available to the inference
library; the original run described above used that mode.

## Limits and reproducibility

These are **reserved timestamps from the same clip**, not independent classrooms.
There are no manually annotated person boxes, so object-level false-positive and
false-negative rates are unavailable. Overlay inspection was qualitative. Two
development frames still missed a bent-over student; one reserved frame retained
a duplicate. This does not establish perfect accuracy or generalization to other
cameras, lighting, crowding, or poses. Five inference passes per frame also require
latency measurement on the deployment hardware.

`platform/scripts/benchmark_people.py` records whole-frame alternatives, settings,
hashes, scores, boxes, and timing. Private diagnostic JSON and overlays stay under
the ignored `platform/runtime/diagnostics/` directory; no source footage or
unredacted classroom images are included in this document. Dataset evaluation on
separately labelled rooms is the next step before claiming a general accuracy
figure or training a classroom-specific model.
