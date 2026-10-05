import copy
import unittest
from server.posture import classify_keypoints, combine_posture, validate_manifest, match_pose, visible_head_anchor


def skeleton(hip=(50,60), knee=(80,70), shoulder=(50,20), confidence=.9):
    rows=[[0,0,0] for _ in range(17)]
    for s,h,k in ((5,11,13),(6,12,14)):
        rows[s]=[*shoulder,confidence];rows[h]=[*hip,confidence];rows[k]=[*knee,confidence]
    return rows


class PostureTests(unittest.TestCase):
    def test_visible_bent_thigh_is_seated(self):
        self.assertEqual(classify_keypoints(skeleton())['posture'],'seated')

    def test_visible_vertical_thigh_is_standing(self):
        self.assertEqual(classify_keypoints(skeleton(knee=(51,105)))['posture'],'standing')

    def test_hidden_legs_never_imply_standing(self):
        points=skeleton();points[13][2]=points[14][2]=.05
        self.assertEqual(classify_keypoints(points)['posture'],'unknown')

    def test_conflicting_geometry_stays_unknown(self):
        points=skeleton();points[14]=[51,105,.9]
        self.assertEqual(classify_keypoints(points)['posture'],'unknown')

    def test_pose_crop_conflict_is_not_counted(self):
        pose={'posture':'seated','score':.8,'reason':'geometry'}
        self.assertEqual(combine_posture(pose,[-.025,-.018],.3)[0],'unknown')

    def test_clipped_torso_does_not_create_standing_evidence(self):
        pose={'posture':'unknown','score':0,'reason':'occluded'}
        self.assertEqual(combine_posture(pose,[-.025,-.018],.3,clipped=True)[0],'unknown')

    def test_ambiguous_or_low_similarity_text_evidence_stays_unknown(self):
        pose={'posture':'unknown','score':0,'reason':'occluded'}
        for margins,similarity in (([.004,.001],.3),([.03,-.03],.3),([.03,.02],.10)):
            self.assertEqual(combine_posture(pose,margins,similarity)[0],'unknown')

    def test_pose_matching_does_not_reuse_neighbor_skeleton(self):
        rows=[{'xyxy':[10,10,40,100],'keypoints':[]},{'xyxy':[80,10,110,100],'keypoints':[]}];used=set()
        self.assertIs(match_pose([11,10,41,100],rows,used),rows[0])
        self.assertIsNone(match_pose([11,10,41,100],rows,used))
        self.assertIs(match_pose([80,10,110,100],rows,used),rows[1])

    def test_head_anchor_uses_visible_points_only(self):
        points=skeleton();self.assertIsNone(visible_head_anchor(points))
        points[0]=[50,12,.9]
        self.assertEqual(visible_head_anchor(points)['point'],[50,12])
        points[0]=[0,0,.99];points[1]=[40,12,.8];points[2]=[60,12,.8]
        self.assertEqual(visible_head_anchor(points)['point'],[50,12])
        points[2][2]=.3;self.assertIsNone(visible_head_anchor(points))

    def test_manifest_rejects_remote_path_and_nonfinite_threshold(self):
        value={'poseModelFile':'pose.pt','clipModelFile':'clip/model.pt','poseModelSha256':'a'*64,'clipModelSha256':'b'*64}
        self.assertEqual(validate_manifest(value)['imageSize'],960)
        for field,entry in [('poseModelFile','../pose.pt'),('clipModelFile','https://example.test/model.pt'),('clipMargin',float('nan'))]:
            changed=copy.deepcopy(value);changed[field]=entry
            with self.assertRaises(ValueError):validate_manifest(changed)

if __name__=='__main__':unittest.main()
