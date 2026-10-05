"""Dedicated local person detection with overlapping views and duplicate removal.

Attendance never uses the reported count to choose predictions. Models are prepared
separately; a missing model fails clearly instead of falling back to YOLO-World.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import threading

MODEL_DIR = Path(os.environ.get('CENTREWATCH_MODELS_DIR', Path(__file__).resolve().parents[1] / 'models'))
MANIFEST_PATH = MODEL_DIR / 'people.json'
_MODEL = None
_MANIFEST = None
_LOCK = threading.RLock()


class PeopleModelUnavailable(RuntimeError):
    pass


def validate_manifest(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError('Person model manifest must be an object.')
    model_file = value.get('modelFile', '')
    if not isinstance(model_file, str) or Path(model_file).name != model_file or not model_file.endswith('.pt'):
        raise ValueError('Person weights must be a local checkpoint filename.')
    checksum = value.get('sha256', '')
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('Person model checksum is required.')
    if value.get('personClassId', 0) != 0 or value.get('classes') != ['person']:
        raise ValueError('Only the COCO person class is permitted for attendance.')
    result = dict(value)
    for key, default, minimum, maximum in (
        ('imageSize', 960, 320, 1280), ('confidence', .25, .05, .9),
        ('predictorIou', .7, .1, .95), ('iou', .5, .1, .95),
        ('tileFraction', .65, .55, .9), ('tileEdgeMargin', 5, 0, 20),
        ('containmentThreshold', .9, .8, 1),
    ):
        item = value.get(key, default)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or not minimum <= item <= maximum:
            raise ValueError(f'Invalid person detector setting: {key}.')
        if key in ('imageSize', 'tileEdgeMargin') and int(item) != item:
            raise ValueError(f'{key} must be an integer.')
        result[key] = item
    if not isinstance(value.get('tiling', True), bool):
        raise ValueError('tiling must be boolean.')
    result['tiling'] = value.get('tiling', True)
    return result


def model_status() -> dict:
    try:
        manifest = validate_manifest(json.loads(MANIFEST_PATH.read_text()))
        exists = (MODEL_DIR / manifest['modelFile']).is_file()
        dependencies = all(importlib.util.find_spec(name) for name in ('ultralytics', 'torch', 'cv2'))
        return {'ready': exists and dependencies, 'model': manifest.get('baseModel', manifest['modelFile']),
                'manifest': manifest, 'detail': 'Dedicated local person detector available.' if exists and dependencies else 'Run scripts/prepare_people_model.py.'}
    except (OSError, ValueError, TypeError):
        return {'ready': False, 'model': 'Dedicated person detector', 'detail': 'Run scripts/prepare_people_model.py to prepare the attendance model.'}


def _load_model():
    global _MODEL, _MANIFEST
    if _MODEL is not None:
        return _MODEL, _MANIFEST
    state = model_status()
    if not state['ready']:
        raise PeopleModelUnavailable(state['detail'])
    manifest = state['manifest']
    path = MODEL_DIR / manifest['modelFile']
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise PeopleModelUnavailable('Person checkpoint checksum changed. Prepare the model again.')
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(MODEL_DIR / '.ultralytics')
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    from ultralytics.utils.events import events
    events.enabled = False
    model = YOLO(str(path))
    if model.names.get(0) != 'person':
        raise PeopleModelUnavailable('The prepared checkpoint does not expose the expected person class.')
    _MODEL, _MANIFEST = model, manifest
    return _MODEL, _MANIFEST


def overlap(a: list[float], b: list[float]) -> tuple[float, float]:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    if area_a <= 0 or area_b <= 0:
        return 0, 0
    return intersection / (area_a+area_b-intersection), intersection / min(area_a, area_b)


def merge_detections(rows: list[dict], iou: float, containment: float) -> list[dict]:
    """Remove repeated views of the same body before counting, highest score first."""
    kept = []
    for row in sorted(rows, key=lambda r: -r['confidence']):
        if any((lambda pair: pair[0] > iou or pair[1] > containment)(overlap(row['xyxy'], accepted['xyxy'])) for accepted in kept):
            continue
        kept.append(row)
    return kept


def crop_regions(width: int, height: int, fraction: float, tiling=True) -> list[tuple[int, int, int, int]]:
    whole = (0, 0, width, height)
    if not tiling:
        return [whole]
    tile_w, tile_h = round(width * fraction), round(height * fraction)
    return list(dict.fromkeys([whole, (0,0,tile_w,tile_h), (width-tile_w,0,width,tile_h),
                             (0,height-tile_h,tile_w,height), (width-tile_w,height-tile_h,width,height)]))


def cut_at_crop_edge(coords, region, width, height, margin) -> bool:
    x1,y1,x2,y2 = region
    left,top,right,bottom = coords
    return ((x1 > 0 and left < margin) or (y1 > 0 and top < margin)
            or (x2 < width and right > x2-x1-margin) or (y2 < height and bottom > y2-y1-margin))


def detect_people(frame, *, device='cpu', max_detections=300) -> tuple[list[dict], dict]:
    """Run one whole frame and four overlapping crops, then count merged boxes."""
    with _LOCK:
        model, config = _load_model()
        height, width = frame.shape[:2]
        rows, truncated, invalid, rejected_edges = [], False, False, 0
        regions = crop_regions(width, height, config['tileFraction'], config['tiling'])
        for index, (x1,y1,x2,y2) in enumerate(regions):
            prediction = model.predict(source=frame[y1:y2,x1:x2], classes=[0],
                conf=config['confidence'], iou=config['predictorIou'], imgsz=config['imageSize'],
                device=device, max_det=max_detections, verbose=False, save=False)[0]
            if len(prediction.boxes) >= max_detections:
                truncated = True
            for box in prediction.boxes:
                score = float(box.conf.item())
                coords = [float(v) for v in box.xyxy[0].tolist()]
                if (len(coords) != 4 or not all(math.isfinite(v) for v in coords)
                        or not math.isfinite(score) or not 0 <= score <= 1
                        or coords[2] <= coords[0] or coords[3] <= coords[1]):
                    invalid = True
                    continue
                if index and cut_at_crop_edge(coords, (x1,y1,x2,y2), width, height, config['tileEdgeMargin']):
                    rejected_edges += 1
                    continue
                left,top,right,bottom = coords
                rows.append({'label':'person', 'confidence':score,
                             'xyxy':[left+x1,top+y1,right+x1,bottom+y1], 'viewIndex':index,
                             'detector':'dedicated-person'})
        merged = merge_detections(rows, config['iou'], config['containmentThreshold'])
        flags = (['detector_limit'] if truncated else []) + (['invalid_model_output'] if invalid else [])
        return merged, {'model':config.get('baseModel',config['modelFile']), 'sha256':config['sha256'],
                        'confidenceThreshold':config['confidence'], 'imageSize':config['imageSize'],
                        'predictorIou':config['predictorIou'], 'mergeIou':config['iou'],
                        'containmentThreshold':config['containmentThreshold'], 'views':len(regions),
                        'rawCandidates':len(rows), 'duplicatesRemoved':len(rows)-len(merged),
                        'croppedBoundaryCandidatesRejected':rejected_edges, 'qualityFlags':flags}
