"""Local posture evidence, without role, identity, age or uniform recognition.

COCO keypoints provide geometric evidence when visible. A local CLIP encoder
compares fixed posture-only text prompts when desks hide legs. Weak or conflicting
observations stay unknown; temporal callers must not silently treat unknown as seated.
These are pretrained models, not classroom-specific training or calibrated accuracy.
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
MANIFEST_PATH = MODEL_DIR / 'posture.json'
_LOCK = threading.RLock()
_MODELS = None
PROMPT_TEMPLATES = ('a photo of a person {}', 'a CCTV image of a person {}', 'the person in the center is {}')
PROMPTS = tuple(template.format(state) for template in PROMPT_TEMPLATES for state in ('sitting', 'standing'))


class PostureModelUnavailable(RuntimeError):
    pass


def validate_manifest(value):
    if not isinstance(value, dict):
        raise ValueError('Posture manifest must be an object.')
    result = dict(value)
    for key in ('poseModelFile', 'clipModelFile'):
        filename = value.get(key, '')
        if not isinstance(filename, str) or not filename.endswith('.pt') or ':' in filename or '\\' in filename or Path(filename).is_absolute() or '..' in Path(filename).parts:
            raise ValueError('Posture checkpoints must use local paths inside the model directory.')
        checksum = value.get(key.replace('File', 'Sha256'), '')
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError('Posture checkpoint checksums are required.')
    for key, default, minimum, maximum in (('imageSize', 960, 320, 1280), ('poseConfidence', .20, .10, .80),
            ('clipMargin', .012, .005, .10), ('clipStandingMargin', .012, .005, .10), ('clipMinimumSimilarity', .18, .1, .4), ('keypointConfidence', .25, .20, .80)):
        number = value.get(key, default)
        if isinstance(number, bool) or not isinstance(number, (int,float)) or not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f'Invalid posture setting: {key}.')
        result[key] = number
    if int(result['imageSize']) != result['imageSize']:
        raise ValueError('Posture imageSize must be an integer.')
    return result


def model_status():
    try:
        manifest = validate_manifest(json.loads(MANIFEST_PATH.read_text()))
        ready = all((MODEL_DIR / manifest[key]).is_file() for key in ('poseModelFile', 'clipModelFile'))
        ready = ready and all(importlib.util.find_spec(name) for name in ('ultralytics', 'clip', 'torch', 'PIL'))
        return {'ready':bool(ready), 'model':'YOLO11 pose + CLIP posture', 'manifest':manifest,
                'detail':'Local posture models available.' if ready else 'Run scripts/prepare_posture_model.py.'}
    except (OSError, ValueError, TypeError):
        return {'ready':False, 'model':'YOLO11 pose + CLIP posture', 'detail':'Run scripts/prepare_posture_model.py to prepare seated-only analysis.'}


def _load_models(device):
    global _MODELS
    if _MODELS is not None and _MODELS[0] == device:
        return _MODELS[1:]
    state = model_status()
    if not state['ready']:
        raise PostureModelUnavailable(state['detail'])
    config = state['manifest']
    for key in ('poseModelFile', 'clipModelFile'):
        path = MODEL_DIR / config[key]
        if hashlib.sha256(path.read_bytes()).hexdigest() != config[key.replace('File', 'Sha256')]:
            raise PostureModelUnavailable('Posture checkpoint checksum changed. Prepare the model again.')
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(MODEL_DIR / '.ultralytics')
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    from ultralytics.utils.events import events
    events.enabled = False
    import clip
    import torch
    pose = YOLO(str(MODEL_DIR / config['poseModelFile']))
    if pose.task != 'pose' or list(pose.names.values()) != ['person']:
        raise PostureModelUnavailable('Expected an official person pose checkpoint.')
    encoder, preprocess = clip.load(str(MODEL_DIR / config['clipModelFile']), device=device, jit=False)
    encoder.eval()
    with torch.no_grad():
        texts = encoder.encode_text(clip.tokenize(PROMPTS).to(device))
        texts = texts / texts.norm(dim=-1, keepdim=True)
    _MODELS = (device, pose, encoder, preprocess, texts, config)
    return _MODELS[1:]


def _angle(a, b):
    norm = math.hypot(*a) * math.hypot(*b)
    if norm < 1:
        return None
    return math.degrees(math.acos(max(-1., min(1., (a[0]*b[0]+a[1]*b[1])/norm))))


def classify_keypoints(points, threshold=.25):
    """Use only visible shoulder/hip/knee geometry; missing legs are unknown."""
    if len(points) != 17 or any(len(point) != 3 or not all(math.isfinite(v) for v in point) for point in points):
        return {'posture':'unknown', 'score':0., 'reason':'invalid_or_missing_keypoints'}
    votes = []
    for shoulder, hip, knee in ((5,11,13), (6,12,14)):
        s,h,k = points[shoulder], points[hip], points[knee]
        if s[2] < .5 or h[2] < .4 or k[2] < threshold:
            continue
        up = (s[0]-h[0], s[1]-h[1]); thigh = (k[0]-h[0], k[1]-h[1]); torso = math.hypot(*up)
        if torso < 8 or math.hypot(*thigh) < torso*.18:
            continue
        angle = _angle(up, thigh)
        if angle is None:
            continue
        score = min(s[2], h[2], k[2])
        if angle < 125 and thigh[1] < torso*.65:
            votes.append(('seated', score, angle))
        elif angle > 155 and thigh[1] > torso*.6 and abs(thigh[0]) < abs(thigh[1])*.65:
            votes.append(('standing', score, angle))
    labels = {vote[0] for vote in votes}
    if len(labels) != 1:
        return {'posture':'unknown', 'score':0., 'reason':'occluded_or_conflicting_leg_geometry'}
    best = max(votes, key=lambda vote:vote[1])
    return {'posture':best[0], 'score':round(best[1],4), 'reason':'visible_hip_knee_geometry', 'hipAngle':round(best[2],1)}


def match_pose(box, pose_rows, used):
    """Greedy spatial matching, each pose skeleton assigned at most once."""
    from .people import overlap
    choices = []
    for index, row in enumerate(pose_rows):
        if index in used:
            continue
        iou, containment = overlap(box, row['xyxy'])
        if iou >= .35 and containment >= .55:
            choices.append((iou, index, row))
    if not choices:
        return None
    _, index, row = max(choices, key=lambda item:item[0])
    used.add(index)
    return row


def visible_head_anchor(points):
    """Ephemeral head location for association, never a face descriptor/identity."""
    if len(points) != 17:
        return None
    nose = points[0]
    if len(nose) == 3 and all(math.isfinite(v) for v in nose) and nose[2] >= .7 and min(nose[:2]) > 0:
        return {'point':[round(nose[0],2),round(nose[1],2)],'confidence':round(nose[2],4)}
    visible = [point for point in points[1:5] if len(point)==3 and all(math.isfinite(v) for v in point)
               and point[2] >= .7 and min(point[:2]) > 0]
    if len(visible) < 2:
        return None
    return {'point':[round(sum(point[axis] for point in visible)/len(visible),2) for axis in (0,1)],
            'confidence':round(min(point[2] for point in visible),4)}


def combine_posture(pose, margins, similarity, *, threshold=.012, standing_threshold=.012, minimum_similarity=.18, clipped=False):
    """Return evidence categories, never a calibrated probability or person's role."""
    seated = max(margins, default=0.) >= threshold
    standing = min(margins, default=0.) <= -standing_threshold
    visual = 'unknown'
    if similarity >= minimum_similarity and seated != standing:
        visual = 'seated' if seated else 'standing'
    # A partial upper body can look upright even when the person is seated.
    if clipped and visual == 'standing':
        visual = 'unknown'
    geometry = pose['posture']
    if geometry != 'unknown' and visual != 'unknown' and geometry != visual:
        return 'unknown', 0., 'conflicting_pose_and_crop_evidence'
    if geometry != 'unknown':
        return geometry, pose['score'], pose['reason']
    if visual != 'unknown':
        return visual, max(abs(value) for value in margins), 'posture_text_image_margin'
    return 'unknown', 0., 'occluded_or_ambiguous_posture'


def assign_postures(frame, detections, *, device='cpu'):
    """Return copied person rows with posture and metadata; no pixels are stored."""
    if not detections:
        return [], {'model':'YOLO11 pose + CLIP posture', 'seated':0, 'standing':0, 'unknown':0, 'qualityFlags':[]}
    import cv2
    import torch
    from PIL import Image
    with _LOCK:
        pose_model, encoder, preprocess, texts, config = _load_models(device)
        height,width = frame.shape[:2]
        prediction = pose_model.predict(frame, imgsz=config['imageSize'], conf=config['poseConfidence'],
                iou=.5, device=device, verbose=False, save=False)[0]
        pose_rows = [{'xyxy':box.xyxy[0].tolist(), 'keypoints':keypoints}
                     for box,keypoints in zip(prediction.boxes, prediction.keypoints.data.tolist())]
        crops, valid_indices, clipped_indices = [], [], set()
        for index,row in enumerate(detections):
            coords = row.get('xyxy', [])
            if len(coords) != 4 or not all(math.isfinite(v) for v in coords):
                continue
            left,top,right,bottom = coords
            bw,bh = right-left,bottom-top
            if bw <= 2 or bh <= 2:
                continue
            if left <= 2 or top <= 2 or right >= width-2 or bottom >= height-2:
                clipped_indices.add(index)
            for pad in (0., .5):
                x1=max(0,int(left-bw*pad));x2=min(width,int(math.ceil(right+bw*pad)))
                y1=max(0,int(top-bh*.1));y2=min(height,int(math.ceil(bottom+bh*pad)))
                if x2 <= x1 or y2 <= y1:
                    break
                crop = frame[y1:y2,x1:x2]
                crops.append(preprocess(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))))
            else:
                valid_indices.append(index)
        all_scores = []
        with torch.no_grad():
            for start in range(0,len(crops),32):
                features = encoder.encode_image(torch.stack(crops[start:start+32]).to(device))
                features = features / features.norm(dim=-1,keepdim=True)
                all_scores.extend((features @ texts.T).float().cpu().tolist())
        clip_scores = {index:all_scores[offset*2:offset*2+2] for offset,index in enumerate(valid_indices)}
        output, used = [], set()
        for index,row in enumerate(detections):
            matching = match_pose(row.get('xyxy', [0,0,0,0]), pose_rows, used)
            geometry = classify_keypoints(matching['keypoints'],config['keypointConfidence']) if matching else {'posture':'unknown','score':0.,'reason':'no_matching_pose'}
            scores = clip_scores.get(index, [])
            margins = [sum(values[i]-values[i+1] for i in range(0,len(values),2))/len(PROMPT_TEMPLATES) for values in scores]
            similarity = max((max(values) for values in scores), default=0.)
            posture,score,reason = combine_posture(geometry,margins,similarity,
                threshold=config['clipMargin'], standing_threshold=config['clipStandingMargin'], minimum_similarity=config['clipMinimumSimilarity'], clipped=index in clipped_indices)
            head = visible_head_anchor(matching['keypoints']) if matching else None
            output.append({**row, 'posture':posture, 'postureScore':round(score,4), 'postureReason':reason,
                **({'headAnchor': head['point'], 'headAnchorConfidence': head['confidence']} if head else {}),
                'postureEvidence':{'pose':geometry, 'clipMargins':[round(value,5) for value in margins], 'cropClipped':index in clipped_indices}})
        counts = {label:sum(row['posture']==label for row in output) for label in ('seated','standing','unknown')}
        return output, {'model':'YOLO11m pose + CLIP ViT-B/32 posture', **counts,
            'modelVersion':config['poseModelSha256'][:12]+'-'+config['clipModelSha256'][:12],
            'qualityFlags':['posture_unknown'] if counts['unknown'] else [],
            'scoreMeaning':'Pose keypoint confidence or text-image cosine margin; not calibrated accuracy.',
            'limitation':'Classifies visible posture only. Standing students and seated trainers cannot be identified by role.'}
