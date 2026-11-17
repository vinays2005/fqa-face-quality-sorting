import json, os, time, pytest
from dataclasses import asdict
from fqa.engine import FaceMetrics, RegionObservation, ScoringConfig, assess_all
from fqa.reclassify import export, write_reports


def _m(fid, cluster="c1", score_sharp=400):
    r = {name: RegionObservation(confidence=1.0, status="visible")
         for name in ("left_eye", "right_eye", "nose", "mouth",
                      "left_cheek", "right_cheek", "forehead", "chin")}
    return FaceMetrics(face_id=fid, image_path=f"/tmp/{fid}.jpg",
                       cluster_id=cluster, regions=r, landmarks_detected=478,
                       sharpness_normalized=score_sharp, bbox=(0, 0, 200, 200),
                       brightness=130, contrast=45, det_confidence=0.95)


def test_threshold_change_does_not_touch_cache(tmp_path):
    cache = tmp_path / "metrics.jsonl"
    with cache.open("w") as f:
        for i in range(10):
            f.write(json.dumps(asdict(_m(f"f{i}"))) + "\n")
    mtime = cache.stat().st_mtime
    metrics = [FaceMetrics.from_dict(json.loads(l)) for l in cache.read_text().splitlines()]
    c1 = ScoringConfig(good_threshold=80)
    c2 = ScoringConfig(good_threshold=60, review_threshold=45)
    assess_all(metrics, c1); assess_all(metrics, c2)
    assert cache.stat().st_mtime == mtime


def test_export_metadata_has_all_faces(tmp_path):
    metrics = [_m(f"f{i}") for i in range(5)]
    cfg = ScoringConfig()
    results = assess_all(metrics, cfg)
    export(results, str(tmp_path), "/tmp", copy_images=False)
    meta = json.loads((tmp_path / "c1" / "good" / "metadata.json").read_text())
    assert len(meta["faces"]) == 5