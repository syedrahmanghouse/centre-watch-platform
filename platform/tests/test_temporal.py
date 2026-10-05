import unittest

from server.temporal import TemporalSeatingTracker


def person(x, y=40, posture='seated', width=40, height=60, **kwargs):
    return {'label': 'person', 'xyxy': [x, y, x + width, y + height],
            'confidence': .9, 'posture': posture, 'postureScore': .9, **kwargs}


def seated_pose(x, **kwargs):
    return person(x, postureEvidence={'pose': {'posture': 'seated', 'reason': 'visible_hip_knee_geometry',
                                               'hipAngle': 95., 'score': .6}}, **kwargs)


class TemporalSeatingTests(unittest.TestCase):
    def test_seated_confirmation_excludes_standing_without_inventing_roles(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(6):
            result = tracker.update(time, [person(40), person(110), person(210, posture='standing')])
        self.assertEqual(result['seatedCount'], 2)
        self.assertEqual(result['standingCount'], 1)
        self.assertEqual([row['includedInCount'] for row in result['detections']], [True, True, False])
        self.assertEqual(tracker.summary()['count'], 2)
        self.assertFalse(any('role' in row for row in result['detections']))

    def test_ambiguous_posture_never_becomes_seated_by_waiting(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(8):
            result = tracker.update(time, [person(40, posture='unknown')])
        self.assertEqual(result['seatedCount'], 0)
        self.assertIsNone(tracker.summary()['count'])
        self.assertIn('posture_coverage_low', tracker.summary()['qualityFlags'])

    def test_a_single_seated_false_positive_is_not_confirmed(self):
        tracker = TemporalSeatingTracker(640, 480)
        outputs = [tracker.update(time, [person(40)] if time == 2 else []) for time in range(6)]
        self.assertTrue(all(frame['seatedCount'] == 0 for frame in outputs))
        self.assertIsNone(tracker.summary()['count'])

    def test_occluded_occupant_is_retained_briefly_without_a_visible_box(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40), person(110)])
        frame = tracker.update(3, [person(110)])
        self.assertEqual(frame['seatedCount'], 2)
        self.assertEqual(frame['visibleSeatedCount'], 1)
        self.assertEqual(frame['recoveredOcclusionCount'], 1)
        self.assertEqual(len(frame['detections']), 1)
        restored = tracker.update(4, [person(42), person(110)])
        self.assertEqual(restored['seatedCount'], 2)
        self.assertEqual(restored['recoveredOcclusionCount'], 0)

    def test_long_occlusion_expires_instead_of_accumulating_occupants(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40)])
        for time in range(3, 7):
            result = tracker.update(time, [])
        self.assertEqual(result['seatedCount'], 0)
        self.assertEqual(result['recoveredOcclusionCount'], 0)

    def test_visible_standing_immediately_cancels_seated_support(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40)])
        frame = tracker.update(3, [person(40, posture='standing')])
        self.assertEqual(frame['seatedCount'], 0)
        self.assertEqual(frame['recoveredOcclusionCount'], 0)
        self.assertFalse(tracker.update(4, [seated_pose(40)])['detections'][0]['includedInCount'])
        tracker.update(5, [seated_pose(40)])
        self.assertEqual(tracker.update(6, [seated_pose(40)])['seatedCount'], 1)

    def test_unknown_posture_memory_expires_even_while_detection_stays_visible(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40)])
        frame = tracker.update(3, [person(40, posture='unknown')])
        self.assertEqual(frame['seatedCount'], 1)
        self.assertTrue(frame['detections'][0]['includedInCount'])
        self.assertEqual(frame['postureHistoryCount'], 1)
        self.assertEqual(frame['unknownPostureCount'], 0)
        self.assertEqual(frame['classifierUnknownPostureCount'], 1)
        for time in range(4, 34):
            frame = tracker.update(time, [person(40, posture='unknown')])
        self.assertEqual(frame['seatedCount'], 0)
        self.assertEqual(frame['unknownPostureCount'], 1)

    def test_visible_posture_history_does_not_extend_physical_occlusion_grace(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40)])
        for time in range(3, 15):
            tracker.update(time, [person(40, posture='unknown')])
        frame = tracker.update(15, [])
        self.assertEqual(frame['seatedCount'], 1)
        self.assertEqual(frame['recoveredOcclusionCount'], 1)
        self.assertEqual(frame['visibleSeatedCount'], 0)
        for time in range(16, 19):
            frame = tracker.update(time, [])
        self.assertEqual(frame['seatedCount'], 0)

    def test_geometry_shift_clears_visible_posture_memory(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40)])
        frame = tracker.update(3, [person(40, y=28, height=72, posture='unknown')])
        self.assertEqual(frame['seatedCount'], 0)
        self.assertEqual(frame['unknownPostureCount'], 1)

    def test_slow_drift_is_compared_with_original_seated_geometry(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40)])
        for time in range(3, 20):
            frame = tracker.update(time, [person(40 + time - 2, posture='unknown')])
        self.assertEqual(frame['seatedCount'], 0)

    def test_bowed_upper_body_preserves_visible_seat_anchor(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40)])
        frame = tracker.update(3, [person(36, y=65, height=35, width=48, posture='unknown')])
        self.assertEqual(frame['seatedCount'], 1)
        self.assertEqual(frame['postureHistoryCount'], 1)
        self.assertEqual(frame['unknownPostureCount'], 0)

    def test_three_seated_votes_can_accumulate_through_visible_occluded_legs(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(21):
            frame = tracker.update(time, [person(40, posture='seated' if time in (0, 10, 20) else 'unknown')])
            if time < 20:
                self.assertEqual(frame['seatedCount'], 0)
        self.assertEqual(frame['seatedCount'], 1)

    def test_old_positive_votes_expire_without_becoming_proof_by_waiting(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(42):
            frame = tracker.update(time, [person(40, posture='seated' if time in (0, 20, 40) else 'unknown')])
            self.assertEqual(frame['seatedCount'], 0)

    def test_standing_replacement_cannot_reuse_prior_seated_votes(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40)])
        tracker.update(3, [person(40, posture='standing')])
        for time in range(4, 9):
            frame = tracker.update(time, [seated_pose(40) if time in (5, 8) else person(40, posture='unknown')])
            self.assertEqual(frame['seatedCount'], 0)
        self.assertEqual(tracker.update(9, [seated_pose(40)])['seatedCount'], 1)

    def test_standing_person_leaning_at_desk_is_not_enabled_by_weak_seated_votes(self):
        tracker = TemporalSeatingTracker(640, 480)
        tracker.update(0, [person(40, posture='standing')])
        for time in range(1, 15):
            frame = tracker.update(time, [person(40, y=40 + time % 3, postureReason='posture_text_image_margin',
                                                postureScore=.03)])
            self.assertEqual(frame['seatedCount'], 0)
            self.assertEqual(frame['unknownPostureCount'], 1)
            self.assertEqual(frame['seatingTransitionUncertainCount'], 1)
            self.assertEqual(frame['detections'][0]['posture'], 'seated')
            self.assertEqual(frame['detections'][0]['effectivePosture'], 'unknown')
        self.assertIsNone(tracker.summary()['count'])
        self.assertIn('seating_transition_unverified', tracker.summary()['qualityFlags'])

    def test_standing_person_can_sit_with_repeated_visible_pose_evidence(self):
        tracker = TemporalSeatingTracker(640, 480)
        tracker.update(0, [person(40, posture='standing')])
        for time in range(1, 4):
            frame = tracker.update(time, [seated_pose(40)])
            self.assertEqual(frame['seatedCount'], 1 if time == 3 else 0)
        self.assertEqual(frame['seatingTransitionUncertainCount'], 0)
        self.assertEqual(tracker.update(4, [person(40, posture='unknown')])['postureHistoryCount'], 1)

    def test_one_pose_then_two_weak_crop_votes_cannot_prove_transition(self):
        tracker = TemporalSeatingTracker(640, 480)
        tracker.update(0, [person(40, posture='standing')])
        tracker.update(1, [seated_pose(40)])
        for time in range(2, 10):
            frame = tracker.update(time, [person(40)])
            self.assertEqual(frame['seatedCount'], 0)

    def test_stationary_unknown_after_standing_never_becomes_seated(self):
        tracker = TemporalSeatingTracker(640, 480)
        tracker.update(0, [person(40, posture='standing')])
        for time in range(1, 61):
            frame = tracker.update(time, [person(40, posture='unknown')])
            self.assertEqual(frame['seatedCount'], 0)

    def test_standing_veto_survives_brief_overlapping_track_fragmentation(self):
        tracker = TemporalSeatingTracker(640, 480)
        first = tracker.update(0, [person(40, posture='standing', headAnchor=[60, 45])])
        original_id = first['detections'][0]['trackId']
        for time in range(1, 6):
            frame = tracker.update(time, [person(42, y=42, headAnchor=[61, 100])])
            self.assertEqual(frame['seatedCount'], 0)
            self.assertEqual(frame['seatingTransitionUncertainCount'], 1)
            if time == 1:
                self.assertNotEqual(frame['detections'][0]['trackId'], original_id)
                self.assertEqual(frame['standingHistoryTransferCount'], 1)
        self.assertIn('standing_history_reassociated', tracker.summary()['qualityFlags'])

    def test_standing_veto_is_not_copied_to_another_visible_neighbour(self):
        tracker = TemporalSeatingTracker(640, 480)
        tracker.update(0, [person(40, posture='standing')])
        for time in range(1, 6):
            frame = tracker.update(time, [person(40, posture='standing'), person(70)])
        self.assertEqual(frame['standingCount'], 1)
        self.assertEqual(frame['seatedCount'], 1)
        self.assertEqual(frame['standingHistoryTransferCount'], 0)

    def test_no_identity_or_standing_history_is_assumed_after_long_absence(self):
        tracker = TemporalSeatingTracker(640, 480)
        tracker.update(0, [person(40, posture='standing')])
        for time in range(1, 5):
            tracker.update(time, [])
        for time in range(5, 8):
            frame = tracker.update(time, [person(40)])
        self.assertEqual(frame['seatedCount'], 1)
        self.assertEqual(frame['standingHistoryTransferCount'], 0)

    def test_persistent_unverified_transition_prevents_exact_attendance_claim(self):
        tracker = TemporalSeatingTracker(1200, 600)
        tracker.update(0, [person(40, posture='standing')] + [person(130 + i * 70) for i in range(10)])
        for time in range(1, 10):
            tracker.update(time, [person(40)] + [person(130 + i * 70) for i in range(10)])
        result = tracker.summary()
        self.assertEqual(result['candidateCount'], 10)
        self.assertIsNone(result['count'])
        self.assertLess(result['unknownPostureFraction'], .15)
        self.assertGreaterEqual(result['seatingTransitionUncertainFrames'], 3)

    def test_rising_head_cancels_history_even_if_body_box_stays_put(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40, headAnchor=[60, 58])])
        frame = tracker.update(3, [person(40, posture='unknown', headAnchor=[60, 40])])
        self.assertEqual(frame['seatedCount'], 0)

    def test_pending_votes_do_not_transfer_to_another_seat_location(self):
        tracker = TemporalSeatingTracker(640, 480)
        tracker.update(0, [person(40)])
        tracker.update(1, [person(40)])
        frame = tracker.update(2, [person(57)])
        self.assertEqual(frame['seatedCount'], 0)

    def test_standing_cancels_inherited_posture_immediately(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40)])
        for time in range(3, 9):
            tracker.update(time, [person(40, posture='unknown')])
        frame = tracker.update(9, [person(40, posture='standing')])
        self.assertEqual(frame['seatedCount'], 0)
        self.assertEqual(frame['postureHistoryCount'], 0)

    def test_returning_classified_seated_pose_preserves_supported_visible_track(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40)])
        for time in range(3, 9):
            tracker.update(time, [person(40, posture='unknown')])
        self.assertEqual(tracker.update(9, [person(40)])['seatedCount'], 1)

    def test_classifier_margin_is_not_reinterpreted_as_probability(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(5):
            tracker.update(time, [person(40, postureScore=.02)])
        self.assertEqual(tracker.summary()['count'], 1)

    def test_neighbours_with_overlapping_bodies_remain_separate(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(6):
            frame = tracker.update(time, [person(40 + time % 2, width=70), person(80 - time % 2, width=70)])
        self.assertEqual(frame['seatedCount'], 2)
        self.assertEqual(len({row['trackId'] for row in frame['detections']}), 2)

    def test_five_minute_result_is_not_peak_or_unique_track_union(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(300):
            rows = [person(40), person(110)]
            if 40 <= time < 45:
                rows.append(person(220))
            if 100 <= time < 105:
                rows.append(person(300))
            if 200 <= time < 205:
                rows.append(person(400))
            tracker.update(time, rows)
        result = tracker.summary()
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['maxCount'], 3)
        self.assertEqual(result['sampleCount'], 300)
        self.assertEqual(result['durationSeconds'], 299)

    def test_short_clip_cannot_claim_temporal_confirmation(self):
        tracker = TemporalSeatingTracker(640, 480)
        tracker.update(0, [person(40)])
        tracker.update(1, [person(40)])
        self.assertIsNone(tracker.summary()['count'])
        self.assertIn('insufficient_temporal_evidence', tracker.summary()['qualityFlags'])

    def test_shifted_box_does_not_add_old_held_track_to_new_confirmation(self):
        tracker = TemporalSeatingTracker(640, 480)
        counts = []
        for time in range(8):
            frame = tracker.update(time, [person(40 if time < 3 else 70)])
            counts.append(frame['seatedCount'])
        self.assertEqual(counts[2:], [1] * 6)
        self.assertIn('ambiguous_spatial_reassociation', tracker.summary()['qualityFlags'])

    def test_duplicate_body_boxes_do_not_create_two_seated_occupants(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(6):
            frame = tracker.update(time, [person(40), person(42)])
        self.assertEqual(frame['seatedCount'], 1)
        self.assertEqual(frame['duplicateDetectionsRemoved'], 1)

    def test_previously_visible_neighbours_are_not_treated_as_box_drift(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(3):
            tracker.update(time, [person(40), person(70)])
        frame = tracker.update(3, [person(70)])
        self.assertEqual(frame['seatedCount'], 2)
        self.assertEqual(frame['recoveredOcclusionCount'], 1)

    def test_brief_outlier_does_not_discard_five_minutes_of_stable_support(self):
        tracker = TemporalSeatingTracker(1200, 600)
        for time in range(300):
            rows = [person(40), person(110)]
            if 100 <= time < 103:
                rows.extend(person(200 + index * 60) for index in range(10))
            tracker.update(time, rows)
        result = tracker.summary()
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['maxCount'], 12)
        self.assertEqual(result['centralMaxCount'], 2)
        self.assertNotIn('unstable_seated_count', result['qualityFlags'])

    def test_sustained_occupancy_change_requires_review(self):
        tracker = TemporalSeatingTracker(1200, 600)
        for time in range(300):
            rows = [person(40), person(110)]
            if time >= 150:
                rows.extend(person(200 + index * 60) for index in range(10))
            tracker.update(time, rows)
        self.assertIsNone(tracker.summary()['count'])
        self.assertIn('unstable_seated_count', tracker.summary()['qualityFlags'])

    def test_repeated_timestamp_cannot_manufacture_support(self):
        tracker = TemporalSeatingTracker(640, 480)
        tracker.update(1, [person(40)])
        with self.assertRaises(ValueError):
            tracker.update(1, [person(40)])

    def test_camera_motion_discards_old_spatial_support(self):
        tracker = TemporalSeatingTracker(640, 480)
        for time in range(5):
            tracker.update(time, [person(40)])
        frame = tracker.update(5, [person(140)], camera_motion=True)
        self.assertEqual(frame['seatedCount'], 0)
        self.assertTrue(frame['warmup'])
        for time in range(6, 10):
            tracker.update(time, [person(140)])
        self.assertIsNone(tracker.summary()['count'])
        self.assertIn('camera_motion', tracker.summary()['qualityFlags'])


if __name__ == '__main__':
    unittest.main()
