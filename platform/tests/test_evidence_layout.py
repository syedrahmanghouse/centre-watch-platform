"""Callout layout must preserve readable people and unambiguous associations."""
import unittest

from server.evidence_layout import layout_labels, _segment_crosses


def overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


class EvidenceLayoutTests(unittest.TestCase):
    def assert_labels_clear(self, layout, boxes, padding):
        for placement in layout.placements:
            rect = placement.rect
            self.assertTrue(0 <= rect[0] < rect[2] <= layout.width)
            self.assertTrue(0 <= rect[1] < rect[3] <= layout.height)
            for box in boxes:
                padded = (box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding)
                self.assertFalse(overlap(rect, padded), (rect, padded))
            for other in layout.placements:
                if other.target_index != placement.target_index:
                    self.assertFalse(overlap(rect, other.rect))
            target = boxes[placement.target_index]
            end = placement.connector[-1]
            self.assertTrue(target[0] <= end[0] <= target[2])
            self.assertTrue(target[1] <= end[1] <= target[3])
            self.assertTrue(end[0] in (target[0], target[2]) or end[1] in (target[1], target[3]))

    def test_overlapping_people_keep_all_faces_free_and_number_mapping_stable(self):
        boxes = [(80, 160, 230, 360), (185, 120, 320, 270),
                 (270, 230, 390, 430), (430, 150, 540, 340)]
        sizes = [(100, 30)] * len(boxes)
        result = layout_labels(640, 480, boxes, sizes)
        self.assert_labels_clear(result, boxes, 8)
        self.assertEqual([p.target_index for p in result.placements], list(range(len(boxes))))
        self.assertEqual(result, layout_labels(640, 480, boxes, sizes))
        self.assertEqual(result.height, 480)

    def test_dense_scene_adds_a_label_rail_instead_of_covering_people(self):
        boxes = [(0, 0, 150, 180), (140, 0, 300, 180),
                 (290, 0, 440, 180), (430, 0, 600, 180)]
        result = layout_labels(600, 180, boxes, [(110, 30)] * len(boxes))
        self.assertGreater(result.height, 180)
        self.assert_labels_clear(result, boxes, 8)
        self.assertTrue(all(p.rect[1] >= 188 for p in result.placements))

    def test_narrow_canvas_wraps_fallback_labels_into_multiple_rows(self):
        boxes = [(0, 0, 80, 90)] * 4
        result = layout_labels(80, 90, boxes, [(70, 20)] * 4)
        self.assert_labels_clear(result, boxes, 8)
        self.assertEqual(len({p.rect[1] for p in result.placements}), 4)

    def test_leaders_do_not_cross_people_when_clear_routes_exist(self):
        boxes = [(180, 150, 260, 270), (300, 150, 380, 270)]
        result = layout_labels(560, 400, boxes, [(100, 28)] * len(boxes))
        self.assert_labels_clear(result, boxes, 8)
        for placement in result.placements:
            for start, end in zip(placement.connector, placement.connector[1:]):
                self.assertFalse(any(_segment_crosses(start, end, box) for box in boxes))

    def test_empty_and_invalid_input(self):
        self.assertEqual(layout_labels(320, 200, [], []).placements, ())
        with self.assertRaises(ValueError):
            layout_labels(100, 100, [(0, 0, 20, 20)], [])
        with self.assertRaises(ValueError):
            layout_labels(100, 100, [(0, 0, 20, 20)], [(110, 20)])
        with self.assertRaises(ValueError):
            layout_labels(100, 100, [(0, 0, float('nan'), 20)], [(40, 20)])


if __name__ == '__main__':
    unittest.main()
