"""Regression checks for duplicate counting and crop boundary artifacts."""
import json
import pytest
from server.people import crop_regions, cut_at_crop_edge, merge_detections, validate_manifest


def box(coords, score):
    return {'xyxy':coords,'confidence':score,'label':'person'}


def test_same_person_from_overlapping_crops_is_counted_once():
    rows = [box([100,100,200,260],.8),box([104,102,199,259],.9),box([250,110,330,250],.85)]
    merged = merge_detections(rows,.5,.9)
    assert len(merged) == 2
    assert merged[0]['confidence'] == .9


def test_contained_lower_body_does_not_add_an_extra_person():
    # A torso/full-body duplicate can have low IoU but high containment.
    rows = [box([100,60,180,260],.9),box([112,180,170,250],.35)]
    assert len(merge_detections(rows,.5,.9)) == 1


def test_adjacent_overlapping_students_are_kept():
    rows = [box([100,60,180,220],.9),box([150,50,240,225],.85)]
    assert len(merge_detections(rows,.5,.9)) == 2


def test_internal_crop_cuts_rejected_but_outer_image_edge_preserved():
    assert cut_at_crop_edge([1,30,50,150],(220,0,640,230),640,352,5)
    assert not cut_at_crop_edge([1,30,50,150],(0,0,420,230),640,352,5)
    assert cut_at_crop_edge([40,50,418,180],(0,0,420,230),640,352,5)


def test_overlapping_views_cover_complete_image():
    assert crop_regions(640,352,.65) == [(0,0,640,352),(0,0,416,229),(224,0,640,229),(0,123,416,352),(224,123,640,352)]
    assert crop_regions(640,352,.65,False) == [(0,0,640,352)]


def test_manifest_rejects_remote_paths_invalid_values_and_wrong_class():
    valid={'modelFile':'yolo11m.pt','sha256':'a'*64,'classes':['person'],'personClassId':0}
    assert validate_manifest(valid)['predictorIou'] == .7
    for changed in ({'modelFile':'../bad.pt'}, {'confidence':float('nan')}, {'personClassId':1}, {'imageSize':2048}, {'classes':['chair']}):
        with pytest.raises(ValueError):
            validate_manifest({**valid,**changed})
