import json, pytest
from fqa.engine import (FaceMetrics, RegionObservation, ScoringConfig, assess,
                        best_faces, DEFAULT_RULES)
from fqa.reclassify import safe_slug, export


def _regions(**overrides):
    out = {r: RegionObservation(confidence=1.0, status="visible") for r in
           ("left_eye", "right_eye", "nose", "mouth", "left_cheek",
            "right_cheek", "forehead", "chin")}
    for k, v in overrides.items():
        out[k] = v
    return out


CLEAR = dict(regions=_regions(), landmarks_detected=478, sharpness_normalized=400,
             bbox=(0, 0, 200, 200), brightness=130, contrast=45, det_confidence=0.95)


def mk(**kw):
    return FaceMetrics(face_id="x", image_path="a.jpg", **{**CLEAR, **kw})


CASES = [
    ("clear frontal", mk(), "GOOD"),
    ("side facing", mk(yaw=55), "BAD"),
    ("moderate yaw", mk(yaw=20), "GOOD"),
    ("blurred", mk(sharpness_normalized=40), "BAD"),
    ("mouth covered", mk(regions=_regions(mouth=RegionObservation(status="occluded"))), "REVIEW"),
    ("two essential missing",
     mk(regions=_regions(nose=RegionObservation(status="missing"),
                         mouth=RegionObservation(status="missing"))), "BAD"),
    ("cut off at boundary", mk(boundary_clip_fraction=0.30), "BAD"),
    ("very small", mk(bbox=(0, 0, 40, 40)), "BAD"),
    ("dark image", mk(brightness=12), "REVIEW"),
    ("overexposed", mk(brightness=250), "REVIEW"),
    ("two faces in crop", mk(n_faces_in_crop=2), "REVIEW"),
    ("landmark failure small",
     mk(landmarks_failed=True, landmarks_detected=0, regions={}, bbox=(0, 0, 30, 30)), "BAD"),
    ("low detector confidence", mk(det_confidence=0.30), "BAD"),
]


@pytest.mark.parametrize("label,m,expected", CASES, ids=[c[0] for c in CASES])
def test_classification(label, m, expected):
    assert assess(m, ScoringConfig()).final_category == expected


def test_pose_rule_severity_configurable():
    m = mk(yaw=55)
    c = ScoringConfig(rules={**DEFAULT_RULES, "pose_exceeded": "review"})
    assert assess(m, c).final_category == "REVIEW"


def test_threshold_change_uses_cache_only():
    m = mk(sharpness_normalized=60)
    assert assess(m, ScoringConfig()).final_category == "BAD"
    assert assess(m, ScoringConfig(minimum_sharpness=40)).final_category == "GOOD"


def test_hard_rule_overrides_high_score():
    m = mk(occlusion_fraction=0.9)
    a = assess(m, ScoringConfig())
    assert a.auto_category == "BAD"
    assert any(r["rule"] == "severe_occlusion" for r in a.reasons)


def test_manual_override_preserved():
    m = mk(sharpness_normalized=40, manual_override="GOOD")
    a = assess(m, ScoringConfig())
    assert a.auto_category == "BAD" and a.final_category == "GOOD"


def test_visibility_unmeasurable_renormalises():
    m = mk(landmarks_failed=True, regions={})
    a = assess(m, ScoringConfig())
    assert a.subscores["visibility"] is None
    assert any(r["rule"] == "visibility_unmeasurable" for r in a.reasons)


def test_config_validation_rejects_bad_weights():
    with pytest.raises(ValueError): ScoringConfig(weights={"visibility": -1})
    with pytest.raises(ValueError): ScoringConfig(good_threshold=110)
    with pytest.raises(ValueError): ScoringConfig(review_threshold=90, good_threshold=50)


def test_alias_accepted():
    c = ScoringConfig.from_dict({"minimum_face_width": 90, "maximum_yaw_degrees": 22})
    assert c.min_face_width == 90 and c.max_yaw == 22


def test_best_faces_excludes_review_by_default():
    from fqa.engine import Assessment
    g = Assessment(face_id="g", cluster_id="c1", image_path="", score=90,
                   subscores={}, auto_category="GOOD", reasons=[])
    r = Assessment(face_id="r", cluster_id="c1", image_path="", score=90,
                   subscores={}, auto_category="REVIEW", reasons=[])
    assert best_faces([g, r], 3)["c1"] == [g]
    assert len(best_faces([g, r], 3, include_review=True)["c1"]) == 2