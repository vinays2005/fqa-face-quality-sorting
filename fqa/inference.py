"""Metric extraction. Runs ONCE per crop; writes FaceMetrics JSONL.

Reuses existing detections when supplied. Does NOT touch identity embeddings or
cluster assignments. Optional models are loaded lazily and released between
stages to fit 4 GB VRAM.
"""
from __future__ import annotations

import gc, json, logging, os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from fqa.engine import FaceMetrics, RegionObservation, ScoringConfig

log = logging.getLogger("fqa.inference")


def _clip_fraction(bbox, W, H) -> float:
    x0, y0, x1, y1 = bbox
    iw = max(0.0, min(x1, W) - max(x0, 0))
    ih = max(0.0, min(y1, H) - max(y0, 0))
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    return 1.0 - (iw * ih) / area


def _sharpness(gray_crop: np.ndarray) -> float:
    if gray_crop.size == 0:
        return 0.0
    norm = cv2.resize(gray_crop, (112, 112), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(norm, cv2.CV_64F).var())


def _brightness_contrast(crop: np.ndarray) -> tuple[float, float]:
    if crop.size == 0:
        return 0.0, 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(g.mean()), float(g.std())


def _pose_from_landmarks(pts: np.ndarray) -> tuple[float, float, float]:
    left_eye, right_eye = pts[33], pts[263]
    nose, chin = pts[1], pts[152]
    eye_mid = (left_eye + right_eye) / 2
    interocular = float(np.linalg.norm(right_eye - left_eye)) + 1e-6
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll = float(np.degrees(np.arctan2(dy, dx)))
    yaw = float(np.degrees(np.arctan2(nose[0] - eye_mid[0], interocular)))
    # Pitch: how far the nose sits, vertically, from its expected resting point
    # roughly halfway between the eye line and the chin on a level face. Scaled
    # by the face's own horizontal size (interocular distance) rather than the
    # eye-to-nose distance, so it reads near 0 for a level face instead of the
    # near-90 degrees a face-height/eye-nose-distance ratio always produces.
    expected_nose_y = eye_mid[1] + 0.5 * (chin[1] - eye_mid[1])
    pitch = float(np.degrees(np.arctan2(nose[1] - expected_nose_y, interocular)))
    return yaw, pitch, roll


class _Landmarker:
    _instance = None

    def __init__(self, model_path: str):
        from mediapipe.tasks.python import vision, BaseOptions
        self._mp = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=40, output_face_blendshapes=True,
                min_face_detection_confidence=0.2, min_face_presence_confidence=0.2))

    @classmethod
    def get(cls, model_path: str):
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    @classmethod
    def release(cls):
        if cls._instance is not None:
            try:
                cls._instance._mp.close()
            except Exception:
                pass
        cls._instance = None
        gc.collect()

    def analyze(self, bgr: np.ndarray, target_bbox=None, crop_offset=None, upscale_to=None):
        """Return landmarks for the detected face best matching target_bbox.

        The input image may contain multiple faces. Selecting face_landmarks[0]
        would silently attach another person's landmarks to this detection.

        upscale_to: if set, bgr is treated as an isolated face crop too small
        for reliable detection (e.g. a ~96px montage cell); it is upscaled to
        this target side length before detection, and crop_offset (x0, y0) is
        added back so returned points stay in the caller's original pixel
        coordinate system.
        """
        from mediapipe import Image as MpImage, ImageFormat
        detect_bgr = bgr
        scale = 1.0
        if upscale_to is not None:
            h, w = bgr.shape[:2]
            scale = max(1.0, upscale_to / max(h, w))
            if scale > 1.0:
                detect_bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))),
                                         interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(detect_bgr, cv2.COLOR_BGR2RGB)
        res = self._mp.detect(MpImage(image_format=ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return None, 0, None

        H, W = detect_bgr.shape[:2]
        candidates = []
        for idx, face in enumerate(res.face_landmarks):
            pts_i = np.array([[lm.x * W, lm.y * H] for lm in face], dtype=np.float32)
            x0, y0 = pts_i.min(axis=0)
            x1, y1 = pts_i.max(axis=0)
            candidates.append((idx, pts_i, (float(x0), float(y0), float(x1), float(y1))))

        chosen = 0
        if upscale_to is not None:
            # The crop was already isolated to one target face before upscaling,
            # so there is no outside bbox to match against: pick whichever
            # candidate sits closest to the crop's center (largest area breaks ties).
            cx, cy = W / 2.0, H / 2.0
            def center_dist(box):
                x0, y0, x1, y1 = box
                bx, by = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                area = max(1e-6, (x1 - x0) * (y1 - y0))
                return (bx - cx) ** 2 + (by - cy) ** 2 - 1e-3 * area
            chosen = min(range(len(candidates)), key=lambda j: center_dist(candidates[j][2]))
        elif target_bbox is not None:
            tx0, ty0, tx1, ty1 = map(float, target_bbox)
            def overlap(box):
                x0, y0, x1, y1 = box
                iw = max(0.0, min(tx1, x1) - max(tx0, x0))
                ih = max(0.0, min(ty1, y1) - max(ty0, y0))
                inter = iw * ih
                union = max(1e-6, (tx1-tx0)*(ty1-ty0) + (x1-x0)*(y1-y0) - inter)
                return inter / union
            chosen = max(range(len(candidates)), key=lambda j: overlap(candidates[j][2]))
            if overlap(candidates[chosen][2]) <= 0:
                return None, 0, None

        idx, pts, _ = candidates[chosen]
        if scale != 1.0:
            pts = pts / scale
        if crop_offset is not None:
            pts = pts + np.array(crop_offset, dtype=np.float32)
        blendshapes = ({c.category_name: c.score
                        for c in res.face_blendshapes[idx]}
                       if res.face_blendshapes and idx < len(res.face_blendshapes) else {})
        return pts, len(pts), blendshapes


REGION_IDX = {
    "left_eye":    [33, 133, 159, 145],
    "right_eye":   [263, 362, 386, 374],
    "nose":        [1, 2, 98, 327],
    "mouth":       [61, 291, 13, 14],
    "left_cheek":  [50, 116, 117],
    "right_cheek": [280, 345, 346],
    "forehead":    [10, 67, 297],
    "chin":        [152, 148, 377],
}


def _regions_from_mesh(pts, bgr, blendshapes, bbox):
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    out = {}
    for name, idx in REGION_IDX.items():
        idx = [i for i in idx if i < len(pts)]
        if not idx:
            out[name] = RegionObservation(status="missing")
            continue
        cx = float(np.mean(pts[idx, 0])); cy = float(np.mean(pts[idx, 1]))
        in_face = x0 <= cx <= x1 and y0 <= cy <= y1
        in_image = 0 <= cx < w and 0 <= cy < h
        if not (in_face and in_image):
            out[name] = RegionObservation(status="out_of_frame", in_frame=False)
            continue
        patch = bgr[max(0, int(cy) - 12):int(cy) + 12,
                    max(0, int(cx) - 12):int(cx) + 12]
        occluded = 0.0
        if patch.size:
            ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
            cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
            skin = ((cr > 133) & (cr < 173) & (cb > 77) & (cb < 127)).mean()
            occluded = float(1.0 - skin)
        if name in ("left_eye", "right_eye"):
            key = "eyeBlinkLeft" if name == "left_eye" else "eyeBlinkRight"
            occluded = max(occluded, blendshapes.get(key, 0) * 0.5)
        status = ("occluded" if occluded > 0.65
                  else "low_confidence" if occluded > 0.4
                  else "visible")
        out[name] = RegionObservation(
            confidence=1.0 - min(1.0, occluded * 0.5),
            occluded=occluded, status=status)
    return out


class CrFiqaAdapter:
    """CR-FIQA wrapped with percentile calibration. Fail-closed."""
    def __init__(self, model_dir, calibration_path):
        self.available = False
        self.calib = None
        if not model_dir or not os.path.isdir(model_dir):
            log.warning("CR-FIQA model_dir not found")
            return
        try:
            import torch
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = torch.jit.load(os.path.join(model_dir, "crfiqa.pt"),
                                        map_location=self.device).eval()
            self.available = True
        except Exception as e:
            log.warning("CR-FIQA load failed: %s", e)
            return
        if calibration_path and os.path.exists(calibration_path):
            with open(calibration_path, encoding="utf-8") as f:
                self.calib = json.load(f)

    def score(self, crop_112):
        if not self.available:
            return None, "unavailable"
        if self.calib is None:
            return None, "uncalibrated"
        t = (self.torch.from_numpy(cv2.cvtColor(crop_112, cv2.COLOR_BGR2RGB))
             .permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0)
        with self.torch.no_grad():
            raw = float(self.model(t).mean())
        try:
            lo, hi = float(self.calib["p05"]), float(self.calib["p95"])
        except (KeyError, TypeError, ValueError):
            return None, "uncalibrated"
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None, "uncalibrated"
        return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0)), "available"


def _extract_one(rec, cfg, lm, crfiqa) -> FaceMetrics:
    path = rec["image_path"]
    cluster = rec.get("cluster_id")
    fid = rec.get("face_id") or Path(path).stem
    base = FaceMetrics(face_id=fid, image_path=path, cluster_id=cluster,
                       bbox=tuple(rec.get("bbox", (0, 0, 0, 0))),
                       det_confidence=float(rec.get("det_confidence", 1.0)))
    img = cv2.imread(path)
    if img is None:
        log.warning("unreadable image %s", path)
        base.landmarks_failed = True
        return base
    H, W = img.shape[:2]
    base.image_width, base.image_height = W, H
    x0, y0, x1, y1 = [int(v) for v in base.bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    crop = img[y0:y1, x0:x1]
    base.boundary_clip_fraction = _clip_fraction(base.bbox, W, H)
    if crop.size == 0:
        base.landmarks_failed = True
        return base
    if cfg.use_grayscale:
        # Flatten color channels for the sharpness/brightness/contrast measurements
        # only. This neutralizes color-only artifacts (colored overlays, tinted
        # lighting) there, but landmark detection and the occlusion heuristic below
        # still need the original color crop: occlusion is read from skin-tone
        # chroma (YCrCb Cr/Cb), which a flattened grayscale image has none of.
        gray_crop = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    else:
        gray_crop = crop
    base.sharpness_normalized = _sharpness(cv2.cvtColor(gray_crop, cv2.COLOR_BGR2GRAY))
    base.brightness, base.contrast = _brightness_contrast(gray_crop)
    if lm is not None:
        crop_w, crop_h = x1 - x0, y1 - y0
        if max(crop_w, crop_h) < 200:
            # Isolated small crop (e.g. a montage grid cell): MediaPipe's own
            # face detector misses faces this small inside a larger frame, so
            # detect on the crop alone, upscaled, instead of the full image.
            pts, n, bs = lm.analyze(crop, upscale_to=320, crop_offset=(x0, y0))
        else:
            pts, n, bs = lm.analyze(img, base.bbox)
        base.landmarks_detected = n
        base.landmarks_failed = pts is None
        if pts is not None and n >= 400:
            base.regions = _regions_from_mesh(pts, img, bs or {}, base.bbox)
            base.yaw, base.pitch, base.roll = _pose_from_landmarks(pts)
            base.occlusion_fraction = float(np.mean(
                [1.0 - o.score() for o in base.regions.values()]))
    else:
        base.landmarks_failed = True
    if crfiqa is not None:
        s, status = crfiqa.score(cv2.resize(crop, (112, 112)))
        base.crfiqa, base.crfiqa_status = s, status
    base.inference_fingerprint = cfg.inference_fingerprint()
    return base


def extract(records, cfg, landmarker_model, crfiqa=None, progress=None):
    lm = _Landmarker.get(landmarker_model) if cfg.use_mediapipe else None
    for i, rec in enumerate(records):
        if progress:
            progress(i, rec.get("face_id", "?"))
        yield _extract_one(rec, cfg, lm, crfiqa)
    _Landmarker.release()


def write_jsonl(metrics, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for m in metrics:
            d = asdict(m)
            f.write(json.dumps(d, default=str) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, out)
    return str(out)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Extract FQA metrics from a manifest")
    p.add_argument("--manifest", required=True,
                   help="JSONL: {face_id, image_path, bbox, det_confidence, cluster_id}")
    p.add_argument("--config", required=True)
    p.add_argument("--landmarker", required=True)
    p.add_argument("--out", default="cache/metrics.jsonl")
    p.add_argument("--crfiqa-dir"); p.add_argument("--crfiqa-calib")
    a = p.parse_args(argv)
    cfg = ScoringConfig.from_json(a.config)
    crf = CrFiqaAdapter(a.crfiqa_dir, a.crfiqa_calib) if cfg.use_crfiqa else None
    with open(a.manifest, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    def progress(i, fid): print(f"\r[{i}] {fid}", end="", flush=True)
    write_jsonl(extract(recs, cfg, a.landmarker, crf, progress=progress), a.out)
    print(f"\nwrote {a.out} ({len(recs)} records)")
    print(f"inference_fingerprint={cfg.inference_fingerprint()}")


if __name__ == "__main__":
    main()