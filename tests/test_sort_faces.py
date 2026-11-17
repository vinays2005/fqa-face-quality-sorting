import json

import cv2
import numpy as np

from fqa.sort_faces import run_sort


def test_sort_faces_saves_good_crop_only(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = dataset / "frame.jpg"
    frame = np.full((240, 240, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(source), frame)

    regions = {name: {"confidence": 1.0, "in_frame": True,
                      "occluded": 0.0, "status": "visible"}
               for name in ("left_eye", "right_eye", "nose", "mouth",
                            "left_cheek", "right_cheek", "forehead", "chin")}
    metric = {
        "face_id": "face-001", "image_path": str(source),
        "bbox": [40, 40, 200, 200], "image_width": 240, "image_height": 240,
        "det_confidence": 0.99, "regions": regions,
        "landmarks_detected": 468, "landmarks_failed": False,
        "boundary_clip_fraction": 0.0, "sharpness_normalized": 500,
        "yaw": 0, "pitch": 0, "roll": 0, "occlusion_fraction": 0,
        "brightness": 128, "contrast": 40, "n_faces_in_crop": 1,
    }
    manifest_row = {"face_id": "face-001", "image_path": str(source),
                    "bbox": [40, 40, 200, 200]}
    metrics_path = tmp_path / "metrics.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    metrics_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
    config_path = "config/balanced.json"
    output = tmp_path / "output"

    counts = run_sort(str(metrics_path), str(manifest_path), config_path,
                      str(dataset), str(output))

    assert counts["GOOD"] == 1
    assert counts["saved"] == 1
    files = list((output / "good_faces").glob("*.jpg"))
    assert len(files) == 1
    saved = cv2.imread(str(files[0]))
    assert saved.shape[:2] == (160, 160)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert "No ArcFace" in summary["note"]
