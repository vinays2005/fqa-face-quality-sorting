"""Quality-gated ArcFace ONNX embedding extraction.

Only GOOD faces are passed to ArcFace by default. REVIEW and BAD are rejected
(fail-closed). This module does not detect faces, compare identities, or cluster.

Manifest JSONL records: face_id, image_path, bbox [x0,y0,x1,y1], and optionally
face_crop_path or landmarks5 [[x,y], ...] in ORIGINAL IMAGE coordinates.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fqa.engine import FaceMetrics, ScoringConfig, assess

log = logging.getLogger("fqa.arcface")

# Standard ArcFace 112x112 five-point reference template.
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


def align_face(image: np.ndarray, bbox, landmarks5=None) -> np.ndarray:
    """Return a 112x112 BGR face. Uses 5-point alignment when supplied.

    Without landmarks, falls back to a bbox crop resized to 112x112. This fallback
    is less geometrically consistent and should not be used for strict matching.
    """
    h, w = image.shape[:2]
    if landmarks5 is not None:
        pts = np.asarray(landmarks5, dtype=np.float32)
        if pts.shape != (5, 2) or not np.isfinite(pts).all():
            raise ValueError("landmarks5 must contain five finite [x,y] points")
        matrix, _ = cv2.estimateAffinePartial2D(pts, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if matrix is None or not np.isfinite(matrix).all():
            raise ValueError("could not estimate ArcFace alignment transform")
        return cv2.warpAffine(image, matrix, (112, 112), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    x0, y0, x1, y1 = map(float, bbox)
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox is empty or outside image")
    crop = image[y0:y1, x0:x1]
    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)


def preprocess_arcface(face_bgr: np.ndarray, input_layout: str = "NCHW") -> np.ndarray:
    """InsightFace w600k_r50 convention: BGR, float32, (pixel-127.5)/127.5."""
    if face_bgr.shape[:2] != (112, 112):
        face_bgr = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_AREA)
    x = (face_bgr.astype(np.float32) - 127.5) / 127.5
    if input_layout.upper() == "NCHW":
        x = np.transpose(x, (2, 0, 1))[None, ...]
    elif input_layout.upper() == "NHWC":
        x = x[None, ...]
    else:
        raise ValueError("input_layout must be NCHW or NHWC")
    return np.ascontiguousarray(x, dtype=np.float32)


class ArcFaceONNX:
    def __init__(self, model_path: str, providers: list[str] | None = None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install ONNX Runtime: pip install onnxruntime") from exc
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ArcFace ONNX model not found: {model_path}")
        available = ort.get_available_providers()
        requested = providers or (["CUDAExecutionProvider", "CPUExecutionProvider"]
                                  if "CUDAExecutionProvider" in available
                                  else ["CPUExecutionProvider"])
        chosen = [p for p in requested if p in available]
        if not chosen:
            chosen = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=chosen)
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        # Most InsightFace R50 models are [N,3,112,112]. Infer NHWC if needed.
        self.layout = "NHWC" if len(shape) == 4 and shape[-1] == 3 else "NCHW"
        self.output_name = self.session.get_outputs()[0].name
        self.providers = self.session.get_providers()

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        tensor = preprocess_arcface(face_bgr, self.layout)
        raw = self.session.run([self.output_name], {self.input.name: tensor})[0]
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size == 0 or not np.isfinite(vector).all() or norm < 1e-12:
            raise ValueError("ArcFace returned an invalid embedding")
        return vector / norm


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def run_quality_gated_arcface(metrics_path: str, manifest_path: str, config_path: str,
                              model_path: str, output_path: str,
                              include_review: bool = False,
                              providers: list[str] | None = None) -> dict[str, int]:
    cfg = ScoringConfig.from_json(config_path)
    manifest = {str(r["face_id"]): r for r in _load_jsonl(manifest_path)
                if r.get("face_id") is not None}
    metrics = _load_jsonl(metrics_path)
    model = ArcFaceONNX(model_path, providers=providers)
    counts = {"GOOD": 0, "REVIEW": 0, "BAD": 0, "embedded": 0, "failed": 0}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in metrics:
            metric = FaceMetrics.from_dict(row)
            result = assess(metric, cfg)
            category = result.final_category
            counts[category] += 1
            accepted = category == "GOOD" or (include_review and category == "REVIEW")
            record: dict[str, Any] = {
                "face_id": metric.face_id,
                "quality_decision": "PASS" if accepted else "REJECT",
                "quality_category": category,
                "quality_score": result.score,
                "quality_reasons": result.reasons,
                "embedding": None,
                "embedding_dimension": 0,
                "arcface_status": "skipped_quality_gate",
            }
            if accepted:
                try:
                    source = manifest.get(metric.face_id)
                    if source is None:
                        raise ValueError("face_id missing from source manifest")
                    crop_path = source.get("face_crop_path")
                    if crop_path:
                        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
                        if image is None:
                            raise ValueError(f"cannot read face_crop_path: {crop_path}")
                        face = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
                    else:
                        image = cv2.imread(metric.image_path, cv2.IMREAD_COLOR)
                        if image is None:
                            raise ValueError(f"cannot read image: {metric.image_path}")
                        face = align_face(image, metric.bbox, source.get("landmarks5"))
                    embedding = model.embed(face)
                    record.update(embedding=embedding.tolist(),
                                  embedding_dimension=int(embedding.size),
                                  arcface_status="ok")
                    counts["embedded"] += 1
                except Exception as exc:
                    # Do not silently mark an accepted face as embedded.
                    record["arcface_status"] = "error"
                    record["error"] = str(exc)[:500]
                    counts["failed"] += 1
                    log.warning("ArcFace failed for %s: %s", metric.face_id, exc)
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    log.info("ArcFace providers: %s", model.providers)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Quality-gated ArcFace ONNX embeddings")
    parser.add_argument("--metrics", required=True, help="FQA metrics JSONL")
    parser.add_argument("--manifest", required=True, help="Original FQA source manifest JSONL")
    parser.add_argument("--config", required=True, help="Same scoring config used for FQA")
    parser.add_argument("--model", required=True, help="InsightFace-compatible ArcFace ONNX model")
    parser.add_argument("--out", default="output/arcface_embeddings.jsonl")
    parser.add_argument("--include-review", action="store_true",
                         help="Also embed REVIEW faces; default is GOOD only")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    providers = ["CPUExecutionProvider"] if args.cpu_only else None
    counts = run_quality_gated_arcface(args.metrics, args.manifest, args.config,
                                       args.model, args.out, args.include_review, providers)
    print(json.dumps(counts, indent=2))
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
