#!/usr/bin/env python3
"""Local person-detector diagnostic. Never uploads source video or invents labels.

Use --split dev to choose settings; only then run those frozen settings with
--split holdout. A constant expected count is an assumption, not box-level truth.
All raw frames/results belong in the ignored runtime directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.45)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.model.is_file() or not args.source.is_file():
        parser.error("Source and local model files must exist; downloads are disabled.")
    if args.samples < 2:
        parser.error("--samples must be at least two for a dev/holdout split")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(args.output / "ultralytics")
    import cv2
    from ultralytics import YOLO, settings
    from ultralytics.utils.events import events
    settings.update({"sync": False})
    events.enabled = False
    model = YOLO(str(args.model.resolve()))
    if model.names.get(0) != "person":
        parser.error("Expected a COCO model with class 0 = person")
    is_image = args.source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    frames = []
    if is_image:
        frame = cv2.imread(str(args.source))
        if frame is None:
            parser.error("Cannot decode source image")
        frames = [(0., frame)]
        duration = None
    else:
        cap = cv2.VideoCapture(str(args.source))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or fps <= 0 or n <= 0:
            parser.error("Cannot decode source video metadata")
        duration = n / fps
        times = [duration * (index + .5) / args.samples for index in range(args.samples)]
        cut = math.floor(args.samples * .6)
        selected = times[:cut] if args.split == "dev" else times[cut:] if args.split == "holdout" else times
        for timestamp in selected:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode timestamp {timestamp:.4f}")
            frames.append((timestamp, frame))
        cap.release()
    observations = []
    for index, (timestamp, frame) in enumerate(frames):
        started = time.perf_counter()
        prediction = model.predict(frame, classes=[0], conf=args.confidence, iou=args.iou,
                                   imgsz=args.size, device=args.device, verbose=False)[0]
        elapsed = time.perf_counter() - started
        boxes = []
        for box in prediction.boxes:
            boxes.append({"box": [round(float(x), 2) for x in box.xyxy[0]],
                          "confidence": round(float(box.conf[0]), 5)})
        observation = {"time": round(timestamp, 4), "people": len(boxes), "boxes": boxes,
                       "width": frame.shape[1], "height": frame.shape[0],
                       "elapsedSeconds": round(elapsed, 4)}
        observations.append(observation)
        annotated = frame.copy()
        for number, box in enumerate(boxes, 1):
            x1, y1, x2, y2 = map(round, box["box"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (30, 210, 40), 1)
            cv2.putText(annotated, f"{number}: {box['confidence']:.2f}", (x1, max(12, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 0, 255), 1)
        cv2.imwrite(str(args.output / f"frame-{index:02d}-{timestamp:.3f}.jpg"), annotated)
        print(json.dumps({key: value for key, value in observation.items() if key != "boxes"}), flush=True)
    counts = [frame["people"] for frame in observations]
    report = {
        "source": args.source.name, "duration": duration,
        "sourceSha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "model": args.model.name, "modelSha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "configuration": {"imageSize": args.size, "confidence": args.confidence, "iou": args.iou,
                          "device": args.device},
        "split": args.split, "splitRule": "First floor(0.6*N) uniformly spaced timestamps are dev; remainder holdout.",
        "expectedCountAssumption": args.expected,
        "limitations": ["Same-clip holdout is correlated and does not establish accuracy on new rooms.",
                        "No box labels supplied; object precision/recall and false positives/negatives are unavailable.",
                        "Constant expected count requires manual confirmation for every timestamp."],
        "counts": counts, "median": statistics.median(counts), "min": min(counts), "max": max(counts),
        "meanAbsoluteCountError": (sum(abs(count - args.expected) for count in counts) / len(counts)
                                   if args.expected is not None else None),
        "observations": observations,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "observations"}), flush=True)


if __name__ == "__main__":
    main()
