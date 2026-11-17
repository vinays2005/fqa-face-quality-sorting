# FQA Control Center

Start the local dashboard from the project root after generating `cache/metrics.jsonl`:

```bash
# Git Bash
source .venv/Scripts/activate
export FQA_METRICS=cache/metrics.jsonl
export FQA_CONFIG=config/balanced.json
export FQA_DATASET_ROOT="D:/JioDataset"
python -m uvicorn fqa.dashboard:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

The page exposes score thresholds, visibility, pose limits, blur/sharpness, face dimensions, resolution, brightness, contrast, occlusion, boundary clipping, detection confidence, scoring weights, region weights, CR-FIQA score settings, and every rule's `OFF` / `REVIEW` / `REJECT` behavior.

- **Apply to current metrics** recalculates GOOD / REVIEW / BAD from cached metrics. It does not run a model or generate embeddings.
- **Save configuration** writes the validated profile to the path in `FQA_CONFIG` (default `config/balanced.json`).
- Changing inference-side settings (`detector`, `min_det_confidence`, `use_mediapipe`, `use_crfiqa`) requires rerunning `fqa.inference`. The dashboard detects stale metric fingerprints and blocks reclassification until fresh metrics are generated.
- Preset buttons load `strict.json`, `balanced.json`, and `relaxed.json`; they do not save until you press Save configuration.
- The dashboard is intentionally bound to `127.0.0.1`. Do not expose it to the public internet; it has no user authentication.

After saving a scoring-only configuration, export GOOD crops with `python -m fqa.sort_faces` using the same config file. REVIEW and BAD crops are not exported by default. No ArcFace inference or embeddings are performed.
