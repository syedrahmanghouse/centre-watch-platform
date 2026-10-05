"""Local, offline image/video observations; no face recognition or identity tracking.

Only prepare_models.py may fetch a model or text encoder. This module requires the
prepared checkpoint and manifest. Detection confidence is not claimed accuracy.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
from typing import Callable
from urllib.parse import urlsplit
from .people import detect_people, model_status as person_model_status

# Native decoder logs can contain credential-bearing RTSP URLs. The service
# emits its own sanitized failures instead of forwarding FFmpeg diagnostics.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

PLATFORM = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("CENTREWATCH_MODELS_DIR", PLATFORM / "models"))
MODEL_PATH = MODEL_DIR / "centrewatch-world.pt"
MANIFEST_PATH = MODEL_DIR / "vocabulary.json"
_MODEL = None
_LOCK = threading.RLock()
SEVERE_FLAGS = {"low_resolution", "dark_frame", "overexposed_frame", "blurred_frame", "detector_limit", "read_failure", "frozen_frames", "invalid_model_output"}
ALIASES = {
    "people": "person", "trainee": "person", "trainees": "person", "student": "person",
    "chairs": "chair", "seating": "chair", "seats": "chair", "seat": "chair",
    "workbenches": "workbench", "work bench": "workbench", "work benches": "workbench",
    "tables": "table", "desks": "table", "desk": "table",
    "lathes": "lathe machine", "lathe": "lathe machine", "lathe machines": "lathe machine",
    "drill presses": "drill press", "drilling machine": "drill press",
    "sewing machines": "sewing machine", "computers": "computer monitor", "computer": "computer monitor",
    "monitors": "computer monitor", "monitor": "computer monitor", "laptops": "laptop",
    "fire extinguishers": "fire extinguisher", "whiteboards": "whiteboard",
}


class VisionUnavailable(RuntimeError):
    """An actionable missing model/dependency error, never simulated output."""


def canonical_name(name: str) -> str:
    normalized = " ".join(str(name).lower().replace("_", " ").strip().split())
    return ALIASES.get(normalized, normalized)


def model_status() -> dict:
    manifest = {}
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except (OSError, ValueError):
        pass
    if not isinstance(manifest, dict):
        manifest = {}
    vocabulary = manifest.get("classes")
    valid_vocabulary = (isinstance(vocabulary, list) and bool(vocabulary)
                        and all(isinstance(value, str) and value.strip() for value in vocabulary)
                        and len(set(vocabulary)) == len(vocabulary) and "person" in vocabulary)
    checksum = manifest.get("sha256", "")
    valid_checksum = isinstance(checksum, str) and len(checksum) == 64 and all(c in "0123456789abcdef" for c in checksum)
    dependencies = all(importlib.util.find_spec(module) for module in ("ultralytics", "torch", "cv2"))
    inventory_ready = MODEL_PATH.is_file() and valid_vocabulary and valid_checksum and dependencies
    people_status = person_model_status()
    ready = inventory_ready and people_status["ready"]
    return {
        "ready": ready, "modelPath": str(MODEL_PATH), "model": f"{people_status['model']} people + YOLO-World inventory",
        "people": people_status, "inventoryReady": inventory_ready,
        "vocabulary": vocabulary if valid_vocabulary else [], "manifest": manifest,
        "detail": "Local model and vocabulary available" if ready else "Install requirements-vision.txt, then run scripts/prepare_models.py and scripts/prepare_people_model.py once while online.",
    }


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    state = model_status()
    if not state["ready"]:
        raise VisionUnavailable(state["detail"])
    # Environment is configured before importing Ultralytics. No model fallback or
    # set_classes call is allowed here: both could initiate a network download.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    (MODEL_DIR / ".ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(MODEL_DIR / ".ultralytics")
    expected_hash = state["manifest"].get("sha256")
    if expected_hash and hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != expected_hash:
        raise VisionUnavailable("Prepared model checksum differs from vocabulary manifest. Run prepare_models.py again.")
    from ultralytics import YOLOWorld
    from ultralytics import settings
    settings.update({"sync": False})
    # Also disable an already-instantiated analytics object when a host process
    # imported Ultralytics before this service configured its offline settings.
    from ultralytics.utils.events import events
    events.enabled = False
    model = YOLOWorld(str(MODEL_PATH))
    actual_names = [model.names[key] for key in sorted(model.names)]
    if actual_names != state["vocabulary"]:
        raise VisionUnavailable("Model vocabulary does not match manifest; prepare the model again.")
    _MODEL = model
    return _MODEL


def _number(value, name: str, low: float, high: float, *, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or not low <= number <= high or (integer and not number.is_integer()):
        raise ValueError(f"{name} must be {'an integer' if integer else 'finite'} between {low} and {high}")
    return int(number) if integer else number


def _options(options: dict) -> dict:
    validated = dict(options)
    for name, default, low, high, integer in (
        ("max_frames", 12, 1, 120, True), ("detector_confidence", .25, .05, .9, False),
        ("max_detections", 300, 20, 1000, True), ("image_size", 640, 320, 1280, True),
        ("read_timeout_ms", 5000, 1000, 15000, True), ("sample_seconds", 15, 1, 120, False),
    ):
        validated[name] = _number(validated.get(name, default), name, low, high, integer=integer)
    # Validate a zone even on an empty frame; detection absence must not conceal
    # malformed configuration that would be applied on the next observation.
    if validated.get("trainee_zone") is not None:
        _in_region(0, 0, validated["trainee_zone"], 1, 1)
    if validated.get("evidence_mode", "abstract") not in {"abstract", "annotated"}:
        raise ValueError("evidence_mode must be abstract or annotated")
    return validated


def sample_times(duration: float, max_frames: int = 12, explicit: list | None = None) -> list[float]:
    duration = _number(duration, "duration", 0, 60 * 60 * 24 * 366)
    max_frames = _number(max_frames, "max_frames", 1, 120, integer=True)
    if explicit is not None:
        values = [float(value) for value in explicit]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("sample_times must contain finite, non-negative seconds")
        if len(values) > max_frames:
            raise ValueError("sample_times exceeds max_frames; increase the explicit frame budget")
        if duration > 0 and any(value >= duration for value in values):
            raise ValueError("A requested sample timestamp is outside the video duration")
        return values
    if duration <= 0:
        return [0.0]
    # Centre each sample inside its interval; never seek to the final EOF frame.
    count = min(max_frames, max(1, math.ceil(duration)))
    return [round(duration * (index + 0.5) / count, 4) for index in range(count)]


def frame_quality(frame) -> dict:
    import cv2
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    flags = []
    if width < 480 or height < 270:
        flags.append("low_resolution")
    if brightness < 25:
        flags.append("dark_frame")
    if brightness > 245:
        flags.append("overexposed_frame")
    if sharpness < 18:
        flags.append("blurred_frame")
    return {"width": width, "height": height, "brightness": round(brightness, 2), "sharpness": round(sharpness, 2), "qualityFlags": flags}


def _in_region(x: float, y: float, region, width: int, height: int) -> bool:
    """Optional normalized [left,top,right,bottom] trainee zone; not identity."""
    if region is None:
        return True
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        raise ValueError("trainee_zone must be normalized [left, top, right, bottom]")
    left, top, right, bottom = [float(value) for value in region]
    if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
        raise ValueError("trainee_zone coordinates must be increasing within 0–1")
    return left <= x / width <= right and top <= y / height <= bottom


def summarize_observations(frames: list[dict], inventory: list[dict], vocabulary: list[str], *, is_image=False, coverage_confirmed=False) -> dict:
    """Pure aggregation, shared by inference and meaningful regression tests."""
    flags = sorted({flag for frame in frames for flag in frame.get("qualityFlags", [])})
    if is_image:
        flags.append("single_image_only")
    usable = [frame for frame in frames if frame.get("usable", True) and not SEVERE_FLAGS.intersection(frame.get("qualityFlags", []))]
    counts = [int(frame["people"]) for frame in usable]
    if not counts:
        count = None
        flags.append("insufficient_usable_frames")
    elif max(counts) == 0:
        count = None
        flags.append("no_people_detected_not_proof_of_empty_room")
    else:
        count = int(math.floor(statistics.median(counts) + 0.5))
    variation = (max(counts) - min(counts)) if counts else None
    if variation is not None and variation > max(2, (count or 0) * 0.2):
        flags.append("unstable_headcount")
    if "frozen_frames" in flags or "unstable_headcount" in flags or "incomplete_observation" in flags or not counts or count is None:
        confidence = "low"
    elif is_image or len(usable) < 3 or len(usable) / max(len(frames), 1) < 0.75:
        confidence = "medium"
    else:
        confidence = "high"
    if variation and confidence == "high":
        confidence = "medium"
    if variation and "unstable_headcount" not in flags:
        flags.append("count_variation")
    equipment = []
    known = set(vocabulary)
    for item in inventory:
        name = str(item.get("name", "Unnamed item"))
        label = canonical_name(item.get("detectorLabel") or name)
        expected = max(0, int(item.get("quantity", item.get("expected", 0))))
        observable = item.get("observable", coverage_confirmed) is True
        supported = label in known and label != "person"
        readings = [int(frame.get("equipment", {}).get(label, 0)) for frame in usable] if supported else []
        observed = int(math.floor(statistics.median(readings) + 0.5)) if readings else None
        reason = ""
        status = "unknown"
        if not supported:
            reason = "Item is not in the prepared detector vocabulary; add an explicit visual label and prepare the model."
        elif not usable:
            reason = "No clear, usable view of this inventory item."
        elif not observable:
            reason = "Full visibility of the sanctioned item area has not been confirmed."
        elif "frozen_frames" in flags:
            reason = "Repeated identical video frames cannot verify current inventory."
        elif "incomplete_observation" in flags:
            reason = "The observation was interrupted; complete a fresh check before assessing inventory."
        elif observed is not None and min(readings) >= expected:
            status = "present"
            reason = "Expected quantity detected in the configured view."
        elif is_image or len(usable) < 3 or len(usable) / max(len(frames), 1) < .75:
            reason = "A sustained clear view is needed before flagging a possible inventory gap."
        elif max(readings) < expected:
            status = "gap"
            reason = "Below the sanctioned quantity across all clear samples; human review required."
        else:
            reason = "Counts change between views; an obstruction or detection miss is possible."
        object_scores = [d["confidence"] for frame in usable for d in frame.get("detections", []) if d["label"] == label]
        equipment.append({
            "name": name, "detectorLabel": label, "expected": expected, "observed": observed,
            "status": status, "operability": "unverified", "confidence": "medium" if status != "unknown" else "low",
            "detectorScoreMean": round(statistics.mean(object_scores), 3) if object_scores else None,
            "observable": observable, "reason": reason,
            "operabilityNote": "Video presence does not establish working condition; idle equipment is never labelled broken.",
        })
    candidate_count = count
    # A wildly varying estimate must not be presented as a verified observation.
    # Preserve it for review, without comparing a fabricated exact count to attendance.
    if confidence == "low":
        count = None
    return {"count": count, "candidateCount": candidate_count, "confidence": confidence, "qualityFlags": sorted(set(flags)), "equipment": equipment,
            "countRange": [min(counts), max(counts)] if counts else None, "usableFrames": len(usable),
            "confidenceMeaning": "Heuristic evidence quality, not a calibrated accuracy probability"}


def _read_frames(source: str, options: dict):
    """Bounded file sampling or short RTSP observations; never infinite capture."""
    import cv2
    scheme = urlsplit(source).scheme.lower()
    is_rtsp = scheme in {"rtsp", "rtsps"}
    if scheme and not is_rtsp:
        raise ValueError("Use a local uploaded file or RTSP stream; remote HTTP video is not downloaded during analysis")
    path = Path(source)
    if not is_rtsp and not path.is_file():
        raise ValueError("Media file was not found")
    max_frames = max(1, min(120, int(options.get("max_frames", 12))))
    # Shared by yielded observations so a later read failure can invalidate the
    # complete session without losing already gathered diagnostic evidence.
    info = {"captureFlags": []}
    is_image = options.get("media_type", "").startswith("image") or path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if is_image and not is_rtsp:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError("Image could not be decoded")
        yield 0.0, frame, {**info, "isImage": True, "duration": 0, "sourceType": "image"}
        return
    timeout_ms = max(1000, min(15000, int(options.get("read_timeout_ms", 5000))))
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms]
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    if not capture.isOpened():
        capture.release()
        raise ValueError("Camera or video could not be opened. Check codec, connectivity and credentials.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 and total_frames > 0 and math.isfinite(total_frames / fps) else 0
        if not is_rtsp and duration <= 0:
            raise ValueError("Cannot determine video duration; convert the upload to MP4/H.264.")
        if is_rtsp:
            seconds = max(1.0, min(120.0, float(options.get("sample_seconds", 15))))
            info.update(isImage=False, duration=seconds, sourceType="rtsp")
            # Use wall-clock timestamps; cap reads by both deadline and max_frames.
            deadline = time.monotonic() + seconds
            start = time.monotonic()
            next_sample = start
            captured = 0
            interval = seconds / max_frames
            while time.monotonic() < deadline and captured < max_frames:
                okay, frame = capture.read()
                if not okay:
                    info["captureFlags"].append("incomplete_observation")
                    break
                now = time.monotonic()
                if now >= next_sample:
                    captured += 1
                    next_sample = now + interval
                    yield round(now - start, 3), frame, info
            if captured == 0:
                raise ValueError("No camera frames received within the bounded observation window")
        else:
            info.update(isImage=False, duration=duration, sourceType="video")
            for timestamp in sample_times(duration, max_frames, options.get("sample_times")):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                okay, frame = capture.read()
                if not okay:
                    info["captureFlags"].append("incomplete_observation")
                    if options.get("sample_times") is not None:
                        raise ValueError(f"Could not decode evaluation timestamp {timestamp:.3f}s")
                    continue
                yield timestamp, frame, info
    finally:
        capture.release()


def representative_frame_indices(frames: list[dict], candidate_count: int | None) -> list[int]:
    """Show a representative sample first, then distinct count extremes.

    Frames are selected only from observed evidence, never the submitted count.
    Selection is independent of whether evidence is a diagram or a snapshot.
    """
    indices = [i for i, frame in enumerate(frames) if frame.get("usable", True)] or list(range(len(frames)))
    if not indices:
        return []
    target = candidate_count if candidate_count is not None else statistics.median(frames[i]["people"] for i in indices)
    centre = (len(frames) - 1) / 2
    representative = min(indices, key=lambda i: (abs(frames[i]["people"] - target), abs(i - centre)))
    low = min(indices, key=lambda i: (frames[i]["people"], abs(i - centre)))
    high = max(indices, key=lambda i: (frames[i]["people"], -abs(i - centre)))
    chosen = list(dict.fromkeys([representative, low, high]))
    for i in (indices[0], indices[-1]):
        if len(chosen) < 3 and i not in chosen:
            chosen.append(i)
    return chosen


def _snapshot_bytes(frame) -> bytes:
    """Keep the exact sampled image, bounded in size, separate from result JSON."""
    import cv2
    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / max(height, width))
    if scale < 1:
        frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    okay, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not okay:
        raise RuntimeError("Could not capture evidence snapshot")
    return encoded.tobytes()


def _save_evidence(observation: dict, directory: Path, index: int, snapshot: bytes | None = None) -> dict:
    """Render frame-local numbered detections over an optional exact sample.

    A snapshot is provided only by the explicitly requested annotated mode.
    The default keeps source pixels removed. Numbers never identify trainees.
    """
    import io
    from PIL import Image
    from .evidence_renderer import render_observation
    annotated = snapshot is not None
    source = Image.open(io.BytesIO(snapshot)).convert("RGB") if annotated else None
    canvas = render_observation(observation, source)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"observation-{index:03d}.jpg"
    destination = directory / filename
    canvas.save(destination, format="JPEG", quality=92, subsampling=0)
    return {"path": str(destination), "filename": filename, "time": observation["time"],
            "people": observation["people"], "frameIndex": index,
            "privacy": "source_pixels_retained" if annotated else "all_source_pixels_removed",
            "type": "annotated_snapshot" if annotated else "abstract_diagram",
            "bytes": destination.stat().st_size}


def analyze_source(source: str, inventory: list[dict], options: dict | None = None, progress: Callable | None = None) -> dict:
    options = _options(options or {})
    for item in inventory:
        _number(item.get("quantity", item.get("expected", 0)), "inventory quantity", 0, 100000, integer=True)
    started = time.monotonic()
    frames, evidence, hashes = [], [], []
    # At most max_frames JPEGs, each capped at a 1280px long edge. These are
    # transient and never added to observations, API JSON, or edge payloads.
    snapshots = []
    annotated_evidence = (options.get("evidence_mode", "abstract") == "annotated"
                          and options.get("save_evidence", True) and bool(options.get("output_dir")))
    media_info = {}
    with _LOCK:
        model = _load_model()
        vocabulary = [model.names[index] for index in sorted(model.names)]
        threshold = max(0.05, min(0.9, float(options.get("detector_confidence", 0.25))))
        device = options.get("device", os.environ.get("CENTREWATCH_VISION_DEVICE", "cpu"))
        max_detections = max(20, min(1000, int(options.get("max_detections", 300))))
        for index, (timestamp, frame, media_info) in enumerate(_read_frames(source, options)):
            if progress:
                progress({"phase": "analysing", "framesProcessed": index, "time": timestamp})
            quality = frame_quality(frame)
            raw_detections, person_metadata = detect_people(frame, device=device, max_detections=max_detections)
            quality["qualityFlags"].extend(person_metadata["qualityFlags"])
            # Open-vocabulary detections serve infrastructure only. Their person
            # boxes must never be mixed into the dedicated attendance count.
            if inventory:
                result = model.predict(source=frame, conf=threshold, imgsz=int(options.get("image_size", 640)),
                                       device=device, max_det=max_detections, verbose=False, save=False)[0]
                if len(result.boxes) >= max_detections:
                    quality["qualityFlags"].append("detector_limit")
                for box in result.boxes:
                    class_value = float(box.cls.item())
                    score = float(box.conf.item())
                    raw_coords = [float(value) for value in box.xyxy[0].tolist()]
                    if (not math.isfinite(class_value) or not class_value.is_integer() or not 0 <= class_value < len(vocabulary)
                            or not math.isfinite(score) or not 0 <= score <= 1 or len(raw_coords) != 4
                            or not all(math.isfinite(value) for value in raw_coords)
                            or raw_coords[2] <= raw_coords[0] or raw_coords[3] <= raw_coords[1]):
                        quality["qualityFlags"].append("invalid_model_output")
                        continue
                    label = vocabulary[int(class_value)]
                    if label != "person":
                        raw_detections.append({"label":label,"confidence":score,"xyxy":raw_coords,"detector":"inventory"})
            detections = []
            equipment_counts = {label: 0 for label in vocabulary if label != "person"}
            people = 0
            for detection in raw_detections:
                label, score = detection["label"], detection["confidence"]
                coords = [round(value, 2) for value in detection["xyxy"]]
                x1, y1, x2, y2 = coords
                in_zone = _in_region((x1 + x2) / 2, (y1 + y2) / 2, options.get("trainee_zone"), quality["width"], quality["height"])
                if label == "person" and in_zone:
                    people += 1
                elif label != "person":
                    equipment_counts[label] += 1
                detections.append({"label": label, "confidence": round(score, 4), "xyxy": coords,
                                   "box": [x1, y1, round(x2 - x1, 2), round(y2 - y1, 2)],
                                   "includedInCount": label == "person" and in_zone,
                                   "detector": detection["detector"]})
            if len(detections) >= max_detections:
                quality["qualityFlags"].append("detector_limit")
            person_scores = [row["confidence"] for row in detections if row["includedInCount"]]
            observation = {"time": timestamp, "people": people, "equipment": equipment_counts, "detections": detections,
                           "personDetector": person_metadata,
                           **quality, "usable": not bool(SEVERE_FLAGS.intersection(quality["qualityFlags"])),
                           "detectorScoreMean": round(statistics.mean(person_scores), 3) if person_scores else None}
            frames.append(observation)
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            if annotated_evidence:
                snapshots.append(_snapshot_bytes(frame))
    if not frames:
        raise ValueError("No decodable frames were sampled")
    for frame in frames:
        frame["qualityFlags"].extend(media_info.get("captureFlags", []))
    if len(hashes) >= 3 and len(set(hashes)) == 1:
        for frame in frames:
            frame["qualityFlags"].append("frozen_frames")
            frame["usable"] = False
    summary = summarize_observations(frames, inventory, vocabulary, is_image=media_info.get("isImage", False),
                                     coverage_confirmed=bool(options.get("coverage_confirmed", False)))
    if options.get("save_evidence", True) and options.get("output_dir"):
        evidence = [_save_evidence(frames[index], Path(options["output_dir"]), index,
                                   snapshots[index] if annotated_evidence else None)
                    for index in representative_frame_indices(frames, summary["candidateCount"])]
    result = {**summary, "frames": frames, "source": "edge-model", "model": f"{person_metadata['model']} (tiled people) + YOLO-World (inventory)",
              "modelVersion": person_metadata["sha256"], "personDetector": person_metadata,
              "method": "Median dedicated person detections across clear frames; overlapping views merged before counting; inventory checked separately",
              "elapsedSeconds": round(time.monotonic() - started, 3), "sampleTimes": [frame["time"] for frame in frames],
              "evidence": evidence, "media": media_info, "detectorThreshold": threshold,
              "observationScope": "configured trainee zone" if options.get("trainee_zone") else "whole room; visible trainers and visitors are included",
              "identityRecognition": False, "audioProcessed": False, "rawMediaUploaded": False,
              "reviewRequired": summary["confidence"] == "low" or any(row["status"] != "present" for row in summary["equipment"])}
    # Estimate payload bytes without footage or the optional evidence files.
    result["bytesSentEstimate"] = len(json.dumps(result, separators=(",", ":")).encode())
    if progress:
        progress({"phase": "complete", "framesProcessed": len(frames)})
    return result
