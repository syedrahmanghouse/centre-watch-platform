"""Short-lived, anonymous spatial support for seated attendance.

These tracks connect nearby detections within one observation window, not people
across sessions. A track is countable only after repeated seated observations.
Missing detections are held briefly; stable visible bodies may reuse recent seated
posture evidence for a bounded period. Standing excludes a track immediately.
The window result uses concurrent occupancy, never a union of track identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics


def _overlap(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    aa, ab = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(aa + ab - intersection, 1e-9), intersection / max(min(aa, ab), 1e-9)


def _anchor(row):
    anchor = row.get('headAnchor')
    if isinstance(anchor, (list, tuple)) and len(anchor) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in anchor):
        return tuple(float(v) for v in anchor)
    x1, y1, x2, y2 = row['xyxy']
    return (x1 + x2) / 2, y1 + (y2 - y1) * .18


def _quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


@dataclass
class _Track:
    id: int
    box: list[float]
    anchor: tuple[float, float]
    first_seen: float
    last_seen: float
    posture: str = 'unknown'
    last_seated: float | None = None
    last_supported: float | None = None
    seated_box: list[float] | None = None
    seated_anchor: tuple[float, float] | None = None
    seated_times: list[float] = field(default_factory=list)
    confirmed: bool = False
    standing_seen: bool = False
    last_standing: float | None = None


class TemporalSeatingTracker:
    """Associate pixel-coordinate person detections sampled in time order.

    update() returns current visible detections plus a separate count of recently
    supported occupants hidden by a short occlusion. ``includedInCount`` is only
    true for visible, confirmed seated detections, including explicitly marked
    recent posture history. Missing detections are never drawn as visible people.
    Posture scores may be geometric evidence or cosine margins, not probabilities;
    classification thresholds belong to the posture classifier.
    """

    def __init__(self, width: int, height: int, grace_seconds: float = 3, min_observations: int = 3,
                 posture_memory_seconds: float = 30):
        if width <= 0 or height <= 0 or not math.isfinite(grace_seconds) or grace_seconds <= 0:
            raise ValueError('Positive frame dimensions and occlusion grace are required.')
        if isinstance(min_observations, bool) or not isinstance(min_observations, int) or min_observations < 2:
            raise ValueError('At least two independent posture observations are required.')
        if not math.isfinite(posture_memory_seconds) or not 0 <= posture_memory_seconds <= 30:
            raise ValueError('Visible posture memory must be between zero and thirty seconds.')
        self.width, self.height = int(width), int(height)
        self.grace_seconds = float(grace_seconds)
        self.min_observations = min_observations
        self.posture_memory_seconds = float(posture_memory_seconds)
        self.confirmation_seconds = max(2., float(min_observations - 1))
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._frames: list[dict] = []
        self._last_time: float | None = None
        self._start_time: float | None = None
        self._reset_time: float | None = None
        self._flags: set[str] = set()

    def reset(self, reason: str = 'camera_motion'):
        """Call when an upstream camera/ROI check detects a changed viewpoint."""
        self._tracks.clear()
        self._reset_time = None
        self._flags.add(reason)

    def _prepare(self, detections):
        rows, invalid = [], False
        for detection in detections:
            if detection.get('label', 'person') != 'person':
                continue
            box = detection.get('xyxy')
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                    or box[2] <= box[0] or box[3] <= box[1]):
                invalid = True
                continue
            row = dict(detection)
            row['xyxy'] = [max(0., min(float(v), self.width if i % 2 == 0 else self.height)) for i, v in enumerate(box)]
            if row['xyxy'][2] <= row['xyxy'][0] or row['xyxy'][3] <= row['xyxy'][1]:
                invalid = True
                continue
            posture = row.get('posture', 'unknown')
            score = row.get('postureScore', 1.)
            if (posture not in ('seated', 'standing', 'unknown') or isinstance(score, bool)
                    or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
                posture = 'unknown'
            row['posture'] = posture
            row['includedInCount'] = False
            rows.append(row)
        # Protect against nearly identical model boxes without merging adjacent
        # seated bodies: both head position and near-total box overlap must agree.
        kept = []
        duplicates = 0
        for row in rows:
            anchor, box = _anchor(row), row['xyxy']
            duplicate = False
            for accepted in kept:
                other, other_anchor = accepted['xyxy'], _anchor(accepted)
                if (_overlap(box, other)[0] >= .82
                        and abs(anchor[0] - other_anchor[0]) < .2 * min(box[2] - box[0], other[2] - other[0])
                        and abs(anchor[1] - other_anchor[1]) < .2 * min(box[3] - box[1], other[3] - other[1])):
                    duplicate = True
                    break
            if duplicate:
                duplicates += 1
            else:
                kept.append(row)
        return kept, invalid, duplicates

    def _association_score(self, track, row):
        box, anchor = row['xyxy'], _anchor(row)
        tw, th = track.box[2] - track.box[0], track.box[3] - track.box[1]
        rw, rh = box[2] - box[0], box[3] - box[1]
        dx, dy = abs(anchor[0] - track.anchor[0]), abs(anchor[1] - track.anchor[1])
        iou, iom = _overlap(track.box, box)
        # Head-position gates distinguish neighbours whose body boxes overlap.
        xgate, ygate = max(8., min(tw, rw) * .65), max(8., min(th, rh) * .55)
        if dx > xgate or dy > ygate or (iou < .12 and math.hypot(dx / xgate, dy / ygate) > .6):
            # A bowed upper body can shorten a detected box enough to move its
            # inferred head anchor. Existing positive seat geometry can support
            # association without inventing a fresh seated classification.
            if row['posture'] == 'standing' or not self._same_seat_geometry(track, row):
                return None
            return iou + .15 * iom - .25
        same = .2 if track.posture == row['posture'] else 0.
        return iou + .15 * iom + same - .25 * (dx / xgate + dy / ygate)

    @staticmethod
    def _forget_posture(track):
        track.confirmed = False
        track.last_seated = track.last_supported = None
        track.seated_times = []
        track.seated_box = track.seated_anchor = None

    @staticmethod
    def _credible_seating_transition(row):
        """A crop that resembles sitting does not prove a standing person sat.

        Require the posture classifier's visible shoulder/hip/knee geometry;
        this evidence is still repeated over time before enabling the track.
        """
        evidence = row.get('postureEvidence')
        pose = evidence.get('pose') if isinstance(evidence, dict) else None
        if not isinstance(pose, dict):
            return False
        angle, score = pose.get('hipAngle'), pose.get('score')
        return (pose.get('posture') == 'seated' and pose.get('reason') == 'visible_hip_knee_geometry'
                and isinstance(angle, (int, float)) and not isinstance(angle, bool)
                and math.isfinite(angle) and 0 <= angle < 125
                and isinstance(score, (int, float)) and not isinstance(score, bool)
                and math.isfinite(score) and .25 <= score <= 1)

    def _standing_veto_donor(self, row, assigned_tracks, timestamp):
        """Transfer only an exclusion across a brief, strongly overlapping split.

        Never transfer seating votes or infer continuity after a long absence.
        A track already matched to another visible person cannot be a donor.
        """
        choices = []
        box = row['xyxy']
        for key, track in self._tracks.items():
            if key in assigned_tracks or not track.standing_seen or timestamp - track.last_seen > self.grace_seconds:
                continue
            iou, containment = _overlap(track.box, box)
            if iou < .4 or containment < .7:
                continue
            tw, th = track.box[2] - track.box[0], track.box[3] - track.box[1]
            rw, rh = box[2] - box[0], box[3] - box[1]
            center_x = abs((track.box[0] + track.box[2] - box[0] - box[2]) / 2)
            center_y = abs((track.box[1] + track.box[3] - box[1] - box[3]) / 2)
            if center_x <= .4 * min(tw, rw) and center_y <= .4 * min(th, rh):
                choices.append((iou, key))
        return self._tracks[max(choices)[1]] if choices else None

    def _recent_seated_geometry(self, track, row, timestamp):
        if (not track.confirmed or track.last_seated is None or track.seated_box is None
                or timestamp - track.last_seated > self.posture_memory_seconds):
            return False
        return self._same_seat_geometry(track, row)

    @staticmethod
    def _same_seat_geometry(track, row):
        """Allow a bowed/moving upper body while its lower seat location stays put.

        These are conservative spatial gates, not a posture classifier. A rise
        of the head/body top, substantial body-height growth, or movement of the
        lower seat anchor invalidates history even when posture is unknown.
        """
        if track.seated_box is None or track.seated_anchor is None:
            return False
        reference, current = track.seated_box, row['xyxy']
        width, height = reference[2] - reference[0], reference[3] - reference[1]
        # Compare with the last classified seated frame, never with successively
        # shifted unknown frames: slow cumulative motion must expire this cache.
        if (abs((current[0] + current[2] - reference[0] - reference[2]) / 2) > max(2., .3 * width)
                or abs(current[3] - reference[3]) > max(2., .3 * height)
                or current[1] <= reference[1] - max(2., .2 * height)
                or current[3] - current[1] > 1.3 * height
                or not .5 <= (current[2] - current[0]) / width <= 2):
            return False
        anchor = _anchor(row)
        return (abs(anchor[0] - track.seated_anchor[0]) <= max(2., .5 * width)
                and anchor[1] > track.seated_anchor[1] - max(2., .2 * height)
                and anchor[1] - track.seated_anchor[1] <= max(2., .65 * height))

    def update(self, timestamp: float, detections: list[dict], *, camera_motion: bool = False) -> dict:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Frame timestamps must be finite, non-negative seconds.')
        timestamp = float(timestamp)
        if self._last_time is not None and timestamp <= self._last_time:
            raise ValueError('Temporal observations must have strictly increasing timestamps.')
        if camera_motion:
            self.reset()
        if self._start_time is None:
            self._start_time = timestamp
        if self._reset_time is None:
            self._reset_time = timestamp
        self._last_time = timestamp
        rows, invalid, duplicates = self._prepare(detections)
        self._tracks = {key: track for key, track in self._tracks.items() if timestamp - track.last_seen <= self.grace_seconds}
        candidates = []
        for key, track in self._tracks.items():
            for index, row in enumerate(rows):
                score = self._association_score(track, row)
                if score is not None:
                    candidates.append((score, key, index))
        assigned_tracks, assigned_rows, associations = set(), set(), {}
        for _, key, index in sorted(candidates, reverse=True):
            if key not in assigned_tracks and index not in assigned_rows:
                associations[index] = key
                assigned_tracks.add(key)
                assigned_rows.add(index)
        for index, row in enumerate(rows):
            key = associations.get(index)
            veto_donor = None
            if key is None:
                veto_donor = self._standing_veto_donor(row, assigned_tracks, timestamp)
                key = self._next_id
                self._next_id += 1
                self._tracks[key] = _Track(key, row['xyxy'], _anchor(row), timestamp, timestamp)
            track = self._tracks[key]
            assigned_tracks.add(key)
            if veto_donor is not None:
                track.standing_seen, track.last_standing = True, veto_donor.last_standing
                self._tracks.pop(veto_donor.id)
            row['standingHistoryTransferred'] = veto_donor is not None
            track.box, track.anchor, track.last_seen = row['xyxy'], _anchor(row), timestamp
            track.posture = row['posture']
            inherited_posture = False
            transition_blocked = (track.standing_seen and track.posture == 'seated'
                                  and not self._credible_seating_transition(row))
            if transition_blocked:
                track.posture = 'unknown'
            if track.posture == 'standing':
                self._forget_posture(track)
                track.standing_seen, track.last_standing = True, timestamp
            elif track.posture == 'seated':
                if (not track.confirmed and track.seated_box is not None
                        and not self._same_seat_geometry(track, row)):
                    self._forget_posture(track)
                track.last_seated = timestamp
                if track.seated_box is None or track.confirmed:
                    track.seated_box, track.seated_anchor = list(row['xyxy']), _anchor(row)
                # Unknown legs do not erase independent positive observations
                # while the body remains visible at the same seat. Votes expire
                # and cannot transfer through standing or a changed seat anchor.
                vote_window = max(self.confirmation_seconds, self.posture_memory_seconds)
                track.seated_times = [time for time in track.seated_times if timestamp - time <= vote_window]
                track.seated_times.append(timestamp)
                if len(track.seated_times) >= self.min_observations and timestamp - track.seated_times[0] >= self.confirmation_seconds:
                    track.confirmed = True
                    track.standing_seen = False
                    track.seated_box, track.seated_anchor = list(row['xyxy']), _anchor(row)
                # Once confirmed, an old timestamp history is unnecessary.
                if track.confirmed:
                    track.seated_times = track.seated_times[-self.min_observations:]
            elif track.confirmed:
                inherited_posture = self._recent_seated_geometry(track, row, timestamp)
                if not inherited_posture:
                    self._forget_posture(track)
            elif track.seated_box is not None and (timestamp - track.last_seated > self.posture_memory_seconds
                                                   or not self._same_seat_geometry(track, row)):
                self._forget_posture(track)
            row['trackId'] = key
            row['includedInCount'] = track.confirmed and (track.posture == 'seated' or inherited_posture)
            row['effectivePosture'] = 'seated' if inherited_posture else track.posture
            row['postureSource'] = ('transition-needs-pose' if transition_blocked else
                                    'recent-seated-history' if inherited_posture else 'current-classifier')
            row['postureTransitionBlocked'] = transition_blocked
            if inherited_posture:
                row['postureHistoryAgeSeconds'] = round(timestamp - track.last_seated, 3)
            if row['includedInCount']:
                track.last_supported = timestamp
        visible = sum(row['includedInCount'] for row in rows)
        counted_ids = {row['trackId'] for row in rows if row['includedInCount']}
        held = [track for track in self._tracks.values() if track.id not in counted_ids and track.confirmed
                and track.posture != 'standing' and track.last_supported is not None and track.last_seated is not None
                and timestamp - track.last_supported <= self.grace_seconds
                and timestamp - track.last_seated <= max(self.grace_seconds, self.posture_memory_seconds)]
        # A shifted detector box may start a fresh track. Its confirmation must
        # not briefly add it to the stale box for the same occupied location.
        # Tracks observed together at least once are never suppressed this way.
        ambiguous = []
        for old in held:
            for key in counted_ids:
                current = self._tracks[key]
                if current.first_seen <= old.last_seen:
                    continue
                widths = old.box[2] - old.box[0], current.box[2] - current.box[0]
                heights = old.box[3] - old.box[1], current.box[3] - current.box[1]
                if (_overlap(old.box, current.box)[0] >= .05
                        and abs(old.anchor[0] - current.anchor[0]) <= .85 * max(widths)
                        and abs(old.anchor[1] - current.anchor[1]) <= .55 * max(heights)):
                    ambiguous.append(old.id)
                    break
        held = [track for track in held if track.id not in ambiguous]
        flags = []
        if invalid:
            flags.append('invalid_model_output')
        if held:
            flags.append('brief_occlusion_recovered')
        if duplicates:
            flags.append('duplicate_person_detection_removed')
        if ambiguous:
            flags.append('ambiguous_spatial_reassociation')
        history_count = sum(row['postureSource'] == 'recent-seated-history' for row in rows)
        if history_count:
            flags.append('posture_from_recent_history')
        transition_count = sum(row['postureTransitionBlocked'] for row in rows)
        if transition_count:
            flags.append('seating_transition_unverified')
        standing_transfers = sum(row['standingHistoryTransferred'] for row in rows)
        if standing_transfers:
            flags.append('standing_history_reassociated')
        warmup = timestamp - self._reset_time < self.confirmation_seconds
        result = {'time': timestamp, 'detections': rows, 'seatedCount': visible + len(held),
                  'visibleSeatedCount': visible, 'recoveredOcclusionCount': len(held),
                  'standingCount': sum(row['posture'] == 'standing' for row in rows),
                  'unknownPostureCount': sum(row['effectivePosture'] == 'unknown' for row in rows),
                  'classifierUnknownPostureCount': sum(row['posture'] == 'unknown' for row in rows),
                  'postureHistoryCount': history_count,
                  'seatingTransitionUncertainCount': transition_count,
                  'standingHistoryTransferCount': standing_transfers,
                  'pendingSeatedCount': sum(row['posture'] == 'seated' and not row['includedInCount'] for row in rows),
                  'ambiguousSpatialCount': len(ambiguous), 'duplicateDetectionsRemoved': duplicates,
                  'personCount': len(rows), 'warmup': warmup, 'qualityFlags': flags}
        self._frames.append({key: value for key, value in result.items() if key != 'detections'})
        return result

    def summary(self) -> dict:
        settled = [frame for frame in self._frames if not frame['warmup']]
        flags = set(self._flags)
        flags.update(flag for frame in self._frames for flag in frame['qualityFlags'])
        duration = 0. if self._last_time is None else self._last_time - self._start_time
        result = {'count': None, 'candidateCount': None, 'confidence': 'low', 'qualityFlags': [],
                  'minCount': None, 'maxCount': None, 'durationSeconds': round(duration, 3),
                  'sampleCount': len(self._frames), 'confirmedSamples': len(settled),
                  'temporalSampleCounts': [frame['seatedCount'] for frame in settled],
                  'graceSeconds': self.grace_seconds, 'confirmationSeconds': self.confirmation_seconds,
                  'postureMemorySeconds': self.posture_memory_seconds,
                  'confirmationWindowSeconds': max(self.confirmation_seconds, self.posture_memory_seconds),
                  'requiredSeatedObservations': self.min_observations,
                  'identity': 'Temporary spatial tracks within this observation only; no facial identification.',
                  'method': f'Median of simultaneously confirmed seated occupants after warm-up; missing detections held up to {self.grace_seconds:g} seconds. A visible body with stable geometry may reuse classified seated posture for up to {self.posture_memory_seconds:g} seconds. Standing is excluded. Track identifiers are never summed.'}
        if not settled:
            flags.add('insufficient_temporal_evidence')
            result['qualityFlags'] = sorted(flags)
            return result
        counts = [frame['seatedCount'] for frame in settled]
        candidate = int(math.floor(statistics.median(counts) + .5))
        spread = max(counts) - min(counts)
        central_min, central_max = _quantile(counts, .05), _quantile(counts, .95)
        central_spread = central_max - central_min
        support = sum(abs(value - candidate) <= 1 for value in counts) / len(counts)
        observations = sum(frame['personCount'] for frame in settled)
        visible_support = sum(frame['personCount'] > 0 for frame in settled) / len(settled)
        unknown_fraction = sum(frame['unknownPostureCount'] for frame in settled) / max(1, observations)
        classifier_unknown_fraction = sum(frame['classifierUnknownPostureCount'] for frame in settled) / max(1, observations)
        history_fraction = sum(frame['postureHistoryCount'] for frame in settled) / max(1, observations)
        transition_fraction = sum(frame['seatingTransitionUncertainCount'] for frame in settled) / max(1, observations)
        transition_frames = sum(frame['seatingTransitionUncertainCount'] > 0 for frame in settled)
        hold_fraction = sum(frame['recoveredOcclusionCount'] for frame in settled) / max(1, sum(counts))
        confidence = 'high'
        if spread:
            flags.add('count_variation')
            confidence = 'medium'
        if unknown_fraction:
            flags.add('posture_uncertain')
            confidence = 'medium'
        if hold_fraction:
            confidence = 'medium'
        if history_fraction:
            confidence = 'medium'
        if 'ambiguous_spatial_reassociation' in flags:
            confidence = 'medium'
        if (len(settled) < 3 or support < .8 or central_spread > max(2, candidate * .2)
                or unknown_fraction > .15 or hold_fraction > .35 or visible_support < .5
                or transition_frames >= self.min_observations
                or flags.intersection({'camera_motion', 'invalid_model_output'})):
            confidence = 'low'
        if len(settled) < 3:
            flags.add('insufficient_temporal_evidence')
        if central_spread > max(2, candidate * .2) or support < .8:
            flags.add('unstable_seated_count')
        if unknown_fraction > .15:
            flags.add('posture_coverage_low')
        if visible_support < .5 or hold_fraction > .35:
            flags.add('insufficient_visible_evidence')
        result.update(count=candidate if confidence != 'low' else None, candidateCount=candidate, confidence=confidence,
                      minCount=min(counts), maxCount=max(counts), qualityFlags=sorted(flags),
                      centralMinCount=round(central_min, 3), centralMaxCount=round(central_max, 3),
                      stableSupportFraction=round(support, 4), unknownPostureFraction=round(unknown_fraction, 4),
                      classifierUnknownPostureFraction=round(classifier_unknown_fraction, 4),
                      postureHistoryFraction=round(history_fraction, 4),
                      seatingTransitionUncertainFraction=round(transition_fraction, 4),
                      seatingTransitionUncertainFrames=transition_frames,
                      occlusionHeldFraction=round(hold_fraction, 4), visibleSupportFraction=round(visible_support, 4))
        return result
