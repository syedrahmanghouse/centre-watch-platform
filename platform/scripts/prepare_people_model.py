#!/usr/bin/env python3
"""Prepare an official COCO detector for offline person-only inference.

Run once with internet: python scripts/prepare_people_model.py --model yolo11m.pt
Check without downloads: python scripts/prepare_people_model.py --verify-only
The weights are pretrained; this command does not train on or upload CCTV media.
It leaves the separate YOLO-World vocabulary and weights unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request


PLATFORM = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("CENTREWATCH_MODELS_DIR", PLATFORM / "models"))
ALLOWED_MODELS = ("yolo11m.pt", "yolo11l.pt", "yolo11x.pt")
RELEASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
MAX_WEIGHT_BYTES = 600 * 1024 * 1024
TILING_CONFIG = {"tiling": True, "tileFraction": 0.65, "tileEdgeMargin": 5,
                 "containmentThreshold": 0.90, "predictorIou": 0.70}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_probability(value: str, minimum: float, maximum: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise argparse.ArgumentTypeError(f"must be a finite number from {minimum} through {maximum}")
    return number


def confidence_threshold(value: str) -> float:
    return bounded_probability(value, 0.05, 0.90)


def iou_threshold(value: str) -> float:
    return bounded_probability(value, 0.10, 0.95)


def image_size(value: str) -> int:
    number = int(value)
    if not 320 <= number <= 1280 or number % 32:
        raise argparse.ArgumentTypeError("must be a multiple of 32 between 320 and 1280")
    return number


def offline_verify(model_path: Path, config: dict) -> dict:
    """Load and predict in a new process that rejects sockets and DNS lookups."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing person detector: {model_path}. Run model preparation first.")
    script = r'''
import os, socket, json, sys, time
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["DO_NOT_TRACK"] = "1"
attempts = []
def denied(*args, **kwargs):
    attempts.append("network_attempt")
    raise RuntimeError("Person-detector offline check blocked an attempted network connection")
socket.getaddrinfo = denied
socket.create_connection = denied
socket.socket.connect = denied
socket.socket.connect_ex = denied
import numpy as np
from ultralytics import YOLO, settings
settings.update({"sync": False})
model = YOLO(sys.argv[1])
assert model.task == "detect", "Only object detection weights are supported"
names = model.names
assert len(names) == 80 and names[0] == "person", "Expected COCO's 80 classes with person at class 0"
config = json.loads(sys.argv[2])
prediction = model.predict(np.zeros((480, 640, 3), dtype=np.uint8), classes=[0],
    imgsz=config["imageSize"], conf=config["confidence"], iou=config["predictorIou"],
    device="cpu", verbose=False, save=False)[0]
assert all(int(value) == 0 for value in prediction.boxes.cls.tolist()), "Unexpected non-person detections"
time.sleep(.1)
assert not attempts, "An inference callback attempted network access"
print(json.dumps({"offlineLoadAndInference": True, "networkAttempts": len(attempts),
    "personClassId": 0, "modelClassCount": len(names), "task": model.task}))
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(model_path.resolve()), json.dumps(config)],
            capture_output=True, text=True, timeout=300, check=True,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "Unknown validation failure").strip()
        raise RuntimeError(f"Person-detector offline validation failed: {detail[-1800:]}") from error
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("offlineLoadAndInference"):
            return value
    raise RuntimeError("Offline verification did not produce its confirmation")


def verify_prepared() -> dict:
    manifest_path = MODEL_DIR / "people.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Person detector is not prepared: people.json is missing. Run scripts/prepare_people_model.py first.")
    manifest = json.loads(manifest_path.read_text())
    filename = manifest.get("modelFile")
    if filename not in ALLOWED_MODELS:
        raise ValueError("The person-detector manifest does not reference an approved official model")
    model_path = MODEL_DIR / filename
    if not model_path.is_file():
        raise FileNotFoundError(f"Person-detector weights are missing: {filename}")
    if manifest.get("sha256") != sha256(model_path):
        raise ValueError("Person-detector checksum differs from its prepared manifest")
    if manifest.get("classes") != ["person"] or manifest.get("personClassId") != 0:
        raise ValueError("Expected a person-only inference manifest with class ID 0")
    if any(manifest.get(key) != value for key, value in TILING_CONFIG.items()):
        raise ValueError("Person-detector tiling metadata differs from the supported configuration")
    config = {
        "imageSize": image_size(str(manifest["imageSize"])),
        "confidence": confidence_threshold(str(manifest["confidence"])),
        "iou": iou_threshold(str(manifest["iou"])),
        **TILING_CONFIG,
    }
    return {"ready": True, "model": str(model_path), "sha256Verified": True,
            "offlineVerification": offline_verify(model_path, config)}


def download_and_verify(name: str, config: dict) -> tuple[Path, dict]:
    """Validate a temporary download before atomically installing its weights."""
    destination = MODEL_DIR / name
    if destination.is_file():
        return destination, offline_verify(destination, config)
    request = urllib.request.Request(f"{RELEASE_URL}/{name}", headers={"User-Agent": "CentreWatch-Model-Setup/1.0"})
    descriptor, temporary_name = tempfile.mkstemp(prefix=".people-download-", suffix=".pt", dir=MODEL_DIR)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            with urllib.request.urlopen(request, timeout=120) as response:
                downloaded = 0
                while block := response.read(1024 * 1024):
                    downloaded += len(block)
                    if downloaded > MAX_WEIGHT_BYTES:
                        raise ValueError("Official person-detector download exceeds the expected size limit")
                    target.write(block)
            target.flush()
            os.fsync(target.fileno())
        verification = offline_verify(temporary, config)
        temporary.replace(destination)
        return destination, verification
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=ALLOWED_MODELS, default="yolo11m.pt")
    parser.add_argument("--image-size", type=image_size, default=960, help="Multiple of 32 from 320 to 1280 (default: 960)")
    parser.add_argument("--confidence", type=confidence_threshold, default=0.25, help="Detection threshold from 0.05 to 0.90 (default: 0.25)")
    parser.add_argument("--iou", type=iou_threshold, default=0.5, help="Global merge overlap threshold from 0.10 to 0.95 (default: 0.5)")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    os.environ["YOLO_CONFIG_DIR"] = str(MODEL_DIR / ".ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["DO_NOT_TRACK"] = "1"
    if args.verify_only:
        print(json.dumps(verify_prepared(), indent=2))
        return 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / ".ultralytics").mkdir(exist_ok=True)
    config = {"imageSize": args.image_size, "confidence": args.confidence, "iou": args.iou,
              **TILING_CONFIG}
    print(f"Preparing official {args.model}; all validation stays on this computer...", flush=True)
    model_path, verification = download_and_verify(args.model, config)
    manifest = {
        "modelFile": model_path.name, "baseModel": model_path.stem,
        "sha256": sha256(model_path), "source": f"{RELEASE_URL}/{args.model}",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "classes": ["person"], "personClassId": 0, **config,
        "ultralyticsVersion": importlib.metadata.version("ultralytics"),
        "torchVersion": importlib.metadata.version("torch"),
        "offlineVerification": verification,
        "documentation": "https://docs.ultralytics.com/models/yolo11/",
        "license": "AGPL-3.0 or Ultralytics commercial license",
        "hashPurpose": "Local file integrity after preparation; not an upstream published checksum.",
        "limitation": "Pretrained person detection is not classroom-specific training or proof of exact attendance.",
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=".people-manifest-", suffix=".json", dir=MODEL_DIR)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as target:
            target.write(json.dumps(manifest, indent=2) + "\n")
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(MODEL_DIR / "people.json")
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"ready": True, "model": str(model_path), "offlineVerified": True, **config}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, argparse.ArgumentTypeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"ready": False, "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
