# Troubleshooting

"cache fingerprint ... != config ..."
You changed detector, min_det_confidence, use_mediapipe or use_crfiqa.
These invalidate the cache. Either re-run inference or revert those keys.

"source escapes dataset root"
An image_path in metrics.jsonl is outside --dataset-root. This is a
security guard, not a bug. Pass a wider root or fix the manifest.

"unsafe cluster/file identifier"
cluster_id contains characters outside [A-Za-z0-9_.-], or contains ...
Sanitize before the export step.

Dashboard shows "…" forever
Check FQA_METRICS points at a non-empty JSONL. The startup hook fails
silently if the file is empty.

All GOOD/REVIEW/BAD counts are identical across configs
You are probably hitting the same threshold because review_threshold is
lower than good_threshold but no face falls in the gap. Check mean_score in
the summary output.

CR-FIQA causes CUDA OOM on RTX 2050 (4 GB)
Run bench with --n 50 first. Peak VRAM is reported; if > 3.5 GB, disable
CR-FIQA (use_crfiqa: false) — MediaPipe alone is sufficient for
classification.

Tests fail on Windows with "no module named fqa"
Run from the project root, not from tests/. Or pip install -e ..