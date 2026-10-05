"""Reproducible, labelled-footage evaluation. Never substitutes synthetic scores."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from .vision import canonical_name


class EvaluationError(ValueError):
    pass


def _number(value, label, minimum=0, maximum=100_000, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EvaluationError(f"{label} must be a finite number")
    if not minimum <= value <= maximum or (integer and int(value) != value):
        raise EvaluationError(f"{label} is outside its allowed bounds")
    return value


def _name(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 150:
        raise EvaluationError(f"{label} must be a non-empty string of at most 150 characters")
    return value.strip().lower()


def _counting_options(value, defaults=None):
    defaults = defaults or {}
    mode = value.get('countingMode', defaults.get('countingMode', 'all_people'))
    if mode not in ('all_people', 'seated_only'):
        raise EvaluationError('countingMode must be all_people or seated_only')
    seconds = _number(value.get('observationSeconds', defaults.get('observationSeconds', 300)),
                      'observationSeconds', minimum=1, maximum=900, integer=True)
    return mode, int(seconds)


def validate_manifest(manifest: dict, incoming_dir: Path) -> list[dict]:
    """Validate before invoking inference, including symlinks and scene leakage."""
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise EvaluationError("Manifest version must be 1")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or not 1 <= len(videos) <= 200:
        raise EvaluationError("Provide 1–200 labelled videos")
    settings = manifest.get("settings", {})
    if not isinstance(settings, dict):
        raise EvaluationError("settings must be an object")
    _number(settings.get("attendance_tolerance", 2), "attendance_tolerance", maximum=1000, integer=True)
    _number(settings.get("iou_threshold", .5), "iou_threshold", minimum=.01, maximum=1)
    _counting_options(settings)
    root = Path(incoming_dir).resolve()
    scenes, files, normalized, total = {}, set(), [], 0
    for video in videos:
        if not isinstance(video, dict):
            raise EvaluationError("Each video must be an object")
        filename = video.get("file")
        if not isinstance(filename, str) or not filename or len(filename) > 500:
            raise EvaluationError("Every video needs a relative file path")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or "\\" in filename:
            raise EvaluationError("Video paths must stay inside data/incoming")
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise EvaluationError(f"Video is missing or outside data/incoming: {filename}")
        if source in files:
            raise EvaluationError("A file may occur only once in a manifest")
        files.add(source)
        room = _name(video.get("room"), "room")
        scene = _name(video.get("scene_id", room), "scene_id")
        split = video.get("split", "unassigned")
        if split not in ("development", "test", "unassigned"):
            raise EvaluationError("split must be development or test when supplied")
        if scene in scenes and scenes[scene] != split:
            raise EvaluationError(f"Scene leakage: {scene} occurs in multiple splits")
        scenes[scene] = split
        mode, seconds = _counting_options(video, settings)
        if video.get('reported_count') is not None or mode == 'all_people':
            _number(video.get("reported_count"), "reported_count", maximum=1000, integer=True)
        window_truth = video.get('window_truth')
        if window_truth is not None:
            if mode != 'seated_only' or not isinstance(window_truth, dict):
                raise EvaluationError('window_truth is an object supported only for seated_only evaluation')
            if window_truth.get('people') is not None:
                _number(window_truth['people'], 'window_truth.people', maximum=1000, integer=True)
            if window_truth.get('visibility') not in ('clear', 'occluded', 'unknown'):
                raise EvaluationError('window_truth.visibility must be clear, occluded or unknown')
            _name(window_truth.get('provenance'), 'window_truth.provenance')
        inventory = video.get("inventory", [])
        if not isinstance(inventory, list) or len(inventory) > 100:
            raise EvaluationError("inventory must contain at most 100 items")
        names = set()
        for item in inventory:
            if not isinstance(item, dict):
                raise EvaluationError("Each inventory item must be an object")
            name = canonical_name(_name(item.get("name"), "inventory name"))
            if name in names:
                raise EvaluationError("Inventory names must be unique")
            names.add(name)
            _number(item.get("quantity"), "inventory quantity", maximum=10000, integer=True)
        observations = video.get("observations", [])
        if not isinstance(observations, list) or (not observations and window_truth is None):
            raise EvaluationError("Every video needs labelled observations")
        total += len(observations)
        if total > 10000:
            raise EvaluationError("A run supports at most 10000 observations")
        times = set()
        for observation in observations:
            if not isinstance(observation, dict):
                raise EvaluationError("Every observation must be an object")
            t = _number(observation.get("time_seconds"), "time_seconds", maximum=86400)
            if mode == 'seated_only' and t > seconds:
                raise EvaluationError('Seated annotations must fall inside the contiguous observation window starting at zero')
            if t in times:
                raise EvaluationError("Observation timestamps must be unique within a file")
            times.add(t)
            if observation.get("people") is not None:
                _number(observation["people"], "people", maximum=1000, integer=True)
            if observation.get("visibility") not in ("clear", "occluded", "unknown"):
                raise EvaluationError("visibility must be clear, occluded or unknown")
            equipment = observation.get("equipment", {})
            if not isinstance(equipment, dict) or len(equipment) > 100:
                raise EvaluationError("equipment must be an object with at most 100 classes")
            for name, count in equipment.items():
                _name(name, "equipment name")
                if count is not None:
                    _number(count, "equipment count", maximum=10000, integer=True)
            if "boxes" in observation:
                boxes = observation["boxes"]
                if not isinstance(boxes, list) or len(boxes) > 3000:
                    raise EvaluationError("boxes must be a list of at most 3000 annotations")
                for box in boxes:
                    if not isinstance(box, dict):
                        raise EvaluationError("Each box must be an object")
                    _name(box.get("class"), "box class")
                    if mode == 'seated_only' and canonical_name(box['class']) == 'person' and box.get('posture') not in ('seated', 'standing', 'unknown'):
                        raise EvaluationError('Seated-only person boxes require posture seated, standing or unknown')
                    coords = box.get("bbox")
                    if not isinstance(coords, list) or len(coords) != 4:
                        raise EvaluationError("bbox must be pixel [x,y,width,height]")
                    for n in coords:
                        _number(n, "bbox coordinate")
                    if coords[2] <= 0 or coords[3] <= 0:
                        raise EvaluationError("Box width and height must be positive")
        normalized.append({**video, "source": source, "split": split, "scene_id": scene,
                           'countingMode':mode, 'observationSeconds':seconds, 'observations':observations})
    return normalized


def _ratio(a, b):
    return round(a / b, 6) if b else None


def _confusion(rows):
    result = dict(tp=0, fp=0, fn=0, tn=0, unverified_positive=0, unverified_negative=0, unknown_ground_truth=0)
    for truth, prediction in rows:
        if truth is None:
            result["unknown_ground_truth"] += 1
        elif prediction is None:
            result["unverified_positive" if truth else "unverified_negative"] += 1
        else:
            result["tp" if truth and prediction else "fn" if truth else "fp" if prediction else "tn"] += 1
    result.update(precision=_ratio(result["tp"], result["tp"] + result["fp"]),
                  recall_on_assessed=_ratio(result["tp"], result["tp"] + result["fn"]),
                  recall_including_unverified=_ratio(result["tp"], result["tp"] + result["fn"] + result["unverified_positive"]),
                  assessed=result["tp"] + result["fp"] + result["fn"] + result["tn"], total=len(rows))
    result["assessment_coverage"] = _ratio(result["assessed"], len(rows))
    return result


def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0


def _match_boxes(truth, detections, threshold):
    """Confidence-ordered one-to-one IoU matching, separately for each class."""
    unmatched = set(range(len(truth)))
    tp = 0
    for prediction in sorted(detections, key=lambda b: b.get("confidence", 0), reverse=True):
        candidates = [(i, _iou(prediction["box"], truth[i]["bbox"])) for i in unmatched]
        index, overlap = max(candidates, key=lambda v: v[1], default=(None, 0))
        if index is not None and overlap >= threshold:
            unmatched.remove(index)
            tp += 1
    return {"tp": tp, "fp": len(detections) - tp, "fn": len(unmatched)}


def _summarize(rows, tolerance, iou_threshold):
    errors, attendance, inventory, object_counts = [], [], defaultdict(list), defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    box_frames = 0
    labelled_box_frames = 0
    for row in rows:
        ground = row["true_people"]
        predicted = row["predicted_people"]
        if ground is not None and predicted is not None:
            errors.append(predicted - ground)
        reported = row['reported_count']
        attendance.append((abs(ground - reported) > tolerance if ground is not None and reported is not None else None,
                           abs(predicted - reported) > tolerance if predicted is not None and reported is not None else None))
        for item in row["inventory"]:
            name, expected = canonical_name(item["name"]), item["quantity"]
            actual = row["true_equipment"].get(name)
            detected = row["predicted_equipment"].get(name)
            inventory[name].append((actual < expected if actual is not None else None,
                                    detected < expected if detected is not None else None))
        if row["boxes"] is not None:
            labelled_box_frames += 1
            if row["usable"] and row["visibility"] != "unknown" and row.get('box_score_eligible', True):
                box_frames += 1
                labels = {"person"} | {canonical_name(item["name"]) for item in row["inventory"]}
                labels |= {canonical_name(b["class"]) for b in row["boxes"]}
                for label in labels:
                    gt = [b for b in row["boxes"] if canonical_name(b["class"]) == label]
                    det = [b for b in row["detections"] if canonical_name(b["label"]) == label]
                    if row.get('countingMode') == 'seated_only' and label == 'person':
                        gt = [box for box in gt if box.get('posture') == 'seated']
                        det = [box for box in det if box.get('includedInCount') is True]
                    matched = _match_boxes(gt, det, iou_threshold)
                    for key in ("tp", "fp", "fn"):
                        object_counts[label][key] += matched[key]
    object_report = {}
    for label, counts in object_counts.items():
        object_report[label] = {**counts, "precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]), "recall": _ratio(counts["tp"], counts["tp"] + counts["fn"])}
    count_metrics = {
        "countMAE": _ratio(sum(abs(e) for e in errors), len(errors)),
        "countRMSE": round(math.sqrt(sum(e * e for e in errors) / len(errors)), 6) if errors else None,
        "overcount_total": sum(max(0, e) for e in errors),
        "undercount_total": sum(max(0, -e) for e in errors),
        "overcount_observations": sum(e > 0 for e in errors),
        "undercount_observations": sum(e < 0 for e in errors),
        "scored_observations": len(errors),
        "total_observations": len(rows),
        "coverage": _ratio(len(errors), len(rows)),
    }
    return {"count": count_metrics, "attendance_alerts": _confusion(attendance),
            "infrastructure_alerts": {name: _confusion(items) for name, items in inventory.items()},
            "object_detection": {"available": bool(box_frames), "iou_threshold": iou_threshold,
                                 "labelled_frames": labelled_box_frames, "scored_frames": box_frames,
                                 "unscored_frames": labelled_box_frames - box_frames,
                                 "coverage": _ratio(box_frames, labelled_box_frames), "classes": object_report,
                                 "note": "Object FP/FN requires exhaustive boxes; count under/overestimates are not object FP/FN."}}


def _valid_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _window_prediction(video, result):
    """Score verified window output separately from diagnostic candidate counts."""
    truth = video.get('window_truth') or {}
    frames = result.get('frames', [])
    times = [frame.get('time') for frame in frames]
    valid_times = bool(times) and all(isinstance(t, (int,float)) and not isinstance(t,bool) and math.isfinite(t) for t in times)
    dense = bool(valid_times and len(times) >= 2 and times[0] <= .5 and times[-1] >= video['observationSeconds'] - 1.1
                 and all(0 < later-earlier <= 1.1 for earlier,later in zip(times,times[1:])))
    quality = list(result.get('qualityFlags', []))
    if not dense:
        quality.append('incomplete_or_sparse_evaluation_window')
    candidate = result.get('candidateCount', result.get('count'))
    candidate = candidate if _valid_count(candidate) else None
    count = result.get('count')
    usable = bool(dense and result.get('countingMode') == 'seated_only' and _valid_count(count)
                  and result.get('confidence') in ('medium','high') and not result.get('error'))
    known = truth.get('visibility', 'unknown') != 'unknown'
    ground = truth.get('people') if known else None
    return {'file':video['file'], 'room':video['room'], 'scene_id':video['scene_id'], 'split':video['split'],
            'countingMode':'seated_only', 'reported_count':video.get('reported_count'),
            'start_seconds':0, 'requested_seconds':video['observationSeconds'],
            'observed_seconds':times[-1]-times[0] if valid_times else 0,
            'sampled_frames':len(frames), 'dense_window_complete':dense,
            'true_people':ground, 'truth_provenance':truth.get('provenance', 'not supplied'),
            'predicted_people':count if usable else None, 'candidate_people':candidate,
            'candidate_absolute_error_diagnostic_only':abs(candidate-ground) if candidate is not None and ground is not None else None,
            'confidence':result.get('confidence'), 'usable':usable, 'visibility':truth.get('visibility','unknown'),
            'quality_flags':quality, 'method':result.get('method'), 'temporal':result.get('temporal'),
            'inventory':[], 'true_equipment':{}, 'predicted_equipment':{}, 'boxes':None, 'detections':[]}


def run_evaluation(manifest: dict, incoming_dir: Path, output_dir: Path,
                   analyzer: Callable | None = None, progress=None) -> dict:
    """Evaluate snapshots or one dense seated window, never sparse temporal replay."""
    videos = validate_manifest(manifest, incoming_dir)
    if analyzer is None:
        from .vision import analyze_source
        analyzer = analyze_source
    settings = manifest.get("settings", {})
    tolerance = settings.get("attendance_tolerance", 2)
    threshold = settings.get("iou_threshold", .5)
    rows, windows, model_metadata = [], [], []
    for index, video in enumerate(videos):
        times = [observation["time_seconds"] for observation in video["observations"]]
        frames, result = [], {}
        if video['countingMode'] == 'seated_only':
            # Labels and submitted/expected counts are deliberately absent here.
            # One call preserves temporal state across every second of the window.
            result = analyzer(str(video["source"]), video.get("inventory", []), options={
                'counting_mode':'seated_only', 'observation_seconds':video['observationSeconds'],
                'sample_interval_seconds':1, 'max_frames':video['observationSeconds']+1,
                'save_evidence':False,
            })
            if not isinstance(result, dict):
                raise EvaluationError("Analyzer did not return a result object")
            frames.extend(result.get("frames", []))
            windows.append(_window_prediction(video, result))
        else:
            # Legacy object evaluation intentionally requests exact annotations.
            for offset in range(0, len(times), 120):
                selected = times[offset:offset + 120]
                result = analyzer(str(video["source"]), video.get("inventory", []), options={
                    'counting_mode':'all_people', "sample_times": selected,
                    "max_frames": len(selected), "save_evidence": False,
                })
                if not isinstance(result, dict):
                    raise EvaluationError("Analyzer did not return a result object")
                frames.extend(result.get("frames", []))
        remaining = set(range(len(frames)))
        model_metadata.append({key: result[key] for key in ("model", "modelVersion", "engine", "model_metadata", 'postureModel', 'countingMode') if key in result})
        for observation in video["observations"]:
            timestamp = observation["time_seconds"]
            nearest = min(remaining, key=lambda i: abs(frames[i].get("time", -1000) - timestamp), default=None)
            frame = None
            if nearest is not None and abs(frames[nearest].get("time", -1000) - timestamp) <= .5:
                frame = frames[nearest]
                remaining.remove(nearest)
            frame = frame or {}
            usable = bool(frame.get("usable", False))
            if video['countingMode'] == 'seated_only' and frame.get('temporalWarmup', False):
                usable = False
            quality = list(frame.get("qualityFlags", []))
            if not frame:
                quality.append("No frame returned within 0.5 seconds of annotation")
            if result.get("error"):
                quality.append(str(result["error"]))
            count = frame.get('temporalCount') if video['countingMode'] == 'seated_only' else frame.get("people")
            prediction = count if usable and _valid_count(count) else None
            equipment = frame.get("equipment", {})
            if isinstance(equipment, list):
                equipment = {item.get("name", ""): item.get("observed") for item in equipment}
            predicted_equipment = {canonical_name(name): value if usable and _valid_count(value) else None for name, value in equipment.items()}
            known_truth = observation["visibility"] != "unknown"
            detections = frame.get("detections", [])
            if observation.get("boxes") and frame.get("width") and frame.get("height"):
                for box in observation["boxes"]:
                    x, y, w, h = box["bbox"]
                    if x + w > frame["width"] + 1 or y + h > frame["height"] + 1:
                        raise EvaluationError(f"Annotation box outside original image: {video['file']} at {timestamp}s")
            rows.append({"file": video["file"], "room": video["room"], "scene_id": video["scene_id"],
                         "split": video["split"], "time_seconds": timestamp, "observed_time_seconds": frame.get("time"),
                         'countingMode':video['countingMode'], "reported_count": video.get("reported_count"), "true_people": observation.get("people") if known_truth else None,
                         "predicted_people": prediction, "raw_predicted_people": count,
                         "true_equipment": {canonical_name(k): v if known_truth else None for k, v in observation.get("equipment", {}).items()},
                         "predicted_equipment": predicted_equipment, "inventory": video.get("inventory", []),
                         "visibility": observation["visibility"], "usable": usable,
                         "quality_flags": quality, "boxes": observation.get("boxes"), "detections": detections,
                         'box_score_eligible':video['countingMode'] != 'seated_only' or not any(
                             canonical_name(box['class']) == 'person' and box.get('posture') == 'unknown' for box in observation.get('boxes', []))})
        if progress:
            progress(round(100 * (index + 1) / len(videos)), f"Evaluated {index + 1}/{len(videos)} files")
    splits = {v["split"] for v in videos}
    report = {
        "version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "dataset": {"videos": len(videos), "observations": len(rows), 'temporal_windows':len(windows), "scenes": len({v['scene_id'] for v in videos}),
                    "splits": sorted(splits), "has_held_out_test": "test" in splits and "unassigned" not in splits},
        "settings": {"attendance_tolerance": tolerance, "iou_threshold": threshold},
        "metrics": _summarize(rows, tolerance, threshold),
        'window_metrics':_summarize(windows,tolerance,threshold),
        'window_predictions':windows,
        'window_coverage':{'total':len(windows), 'verified':sum(row['usable'] for row in windows),
                           'abstained_or_missing':sum(not row['usable'] for row in windows)},
        'by_counting_mode':{mode:_summarize([r for r in rows if r['countingMode']==mode],tolerance,threshold)
                            for mode in sorted({v['countingMode'] for v in videos})},
        "by_split": {split: _summarize([r for r in rows if r["split"] == split], tolerance, threshold) for split in sorted(splits)},
        "coverage": {"total": len(rows), "usable_predictions": sum(r["usable"] for r in rows),
                     "abstained_or_missing": sum(not r["usable"] for r in rows),
                     "unknown_ground_truth": sum(r["visibility"] == "unknown" for r in rows)},
        "model_metadata": model_metadata,
        "limitations": ["Results apply only to the supplied labelled footage.",
                        "Snapshot alert metrics do not measure sustained-alert delay or alerts per camera-hour.",
                        "All unknown and abstained observations remain in coverage; recall including unverified shows operational missed coverage.",
                        "The declared split and exhaustive annotations must be independently reviewed; the software cannot verify label quality.",
                        "No bounding-box annotations means object-level false-positive/false-negative metrics are unavailable.",
                        "Seated-only evaluation runs every second from zero through observationSeconds in one contiguous inference call; labels never select temporal samples.",
                        "Window metrics score only verified aggregate counts; uncertain candidates remain diagnostic and are not counted as correct.",
                        "User-reported window attendance is not exhaustive per-person posture ground truth. It cannot establish detector FP/FN or posture classification accuracy.",
                        "Standing/seated posture does not establish trainer/student identity. Desk occlusion and standing people leaning over desks remain limitations."],
        "files": ["report.json", "predictions.json", "predictions.csv", "manifest.json"],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("report.json", report), ("predictions.json", rows), ("manifest.json", manifest)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    fields = ["file", "room", "scene_id", "split", "time_seconds", "observed_time_seconds", "reported_count", "true_people", "predicted_people", "raw_predicted_people", "visibility", "usable", "quality_flags", "true_equipment", "predicted_equipment"]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            exported = {key: json.dumps(row[key]) if isinstance(row[key], (list, dict)) else row[key] for key in fields}
            # Stop spreadsheet formula execution in user supplied file/room fields.
            for key, value in exported.items():
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                    exported[key] = "'" + value
            writer.writerow(exported)
    return report
