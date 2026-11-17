# Architecture

Two-stage design
Stage A (once): manifest -> fqa.inference -> metrics.jsonl
Stage B (many): metrics.jsonl + config -> fqa.reclassify / dashboard

Stage A is GPU-bound and cached. Stage B is pure arithmetic.

Why split?
The scoring rules must be tunable without re-running MediaPipe or CR-FIQA.
Inference output (FaceMetrics) contains raw values: sharpness, angles,
region observations. Sub-scores, weighted sums and category rules live in
engine.py and read only from those cached values.

Cache contract
metrics.jsonl carries inference_fingerprint, a SHA-256 of
{detector, min_det_confidence, use_mediapipe, use_crfiqa}. The reclassifier
refuses to run if the fingerprint does not match the current config. Any
other config change (thresholds, weights, rules) reclassifies silently.

Component boundaries
fqa/engine.py scoring, config validation, assessment, atomic JSON
fqa/inference.py metric extraction, MediaPipe + CR-FIQA adapters
fqa/reclassify.py export, security (slugs, path containment)
fqa/dashboard.py FastAPI endpoints over cached metrics
fqa/bench.py end-to-end timing + peak VRAM

engine.py imports nothing from inference.py. That is deliberate — the
scoring logic must remain testable without MediaPipe installed.

Non-goals
This package does NOT detect faces, extract ArcFace embeddings, or change
cluster assignments. It consumes what those stages produce.

## Measurement limitations

The current occlusion estimate is a lightweight image-colour/landmark heuristic, not a
trained occlusion detector. It is sensitive to lighting, skin tone, makeup, and camera
processing. Treat occlusion-related scores as a review signal and calibrate against
labelled examples before using them for automatic rejection. Pose estimates from 2D
landmarks are also approximate. The pipeline is a sorting aid, not a biometric identity
or demographic classifier.
