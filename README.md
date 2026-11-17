# FQA — Face Quality Assessment & Intelligent Sorting

Cached-metric FQA pipeline that runs after your existing face detection
and ArcFace clustering. Detects faces? No — consumes them. Extracts metrics
once, then reclassifies instantly under different threshold configs without
re-running any model.

Quick start (Windows PowerShell)
powershell
# 1. Environment
cd path\\to\\FQA_Face_Quality_Sorting_System
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt\n# Optional only if you have a compatible CR-FIQA TorchScript model:\n# install the CUDA-matched PyTorch wheel from pytorch.org, then:\n# pip install -r requirements-crfiqa.txt

# 2. Model weights (not bundled)
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile models/face_landmarker.task

# 3. Produce a manifest from YOUR pipeline
#    (see docs/INTEGRATION.md — one JSON object per line)
#    Place at: cache/manifest.jsonl

# 4. Extract metrics ONCE
python -m fqa.inference `
  --manifest cache/manifest.jsonl `
  --config config/balanced.json `
  --landmarker models/face_landmarker.task `
  --out cache/metrics.jsonl
# Note the printed fingerprint

# 5. Reclassify (fast)
python -m fqa.reclassify `
  --metrics cache/metrics.jsonl `
  --cache-fingerprint <FP> `
  --config config/balanced.json `
  --dataset-root . `
  --out output `
  --top-k 3 `
  --compare config/strict.json

# 6. Dashboard
$env:FQA_METRICS = "cache/metrics.jsonl"
$env:FQA_CONFIG  = "config/balanced.json"
uvicorn fqa.dashboard:app --host 127.0.0.1 --port 8000
Then open http://127.0.0.1:8000.

What's in the box
fqa/engine.py — scoring, classification, config validation, atomic JSON

fqa/inference.py — one-pass metric extraction (MediaPipe + CR-FIQA)

fqa/reclassify.py — CLI, safe export, slug/path guards

fqa/dashboard.py — FastAPI app + /api/* endpoints

fqa/bench.py — GPU timing + peak VRAM

config/ — balanced / strict / relaxed presets

tests/ — engine, security, cache reuse, inference smoke

docs/ — install, usage, architecture, config, integration, models, troubleshooting

What is NOT included
Pretrained model weights (see docs/MODELS.md)\n\nCR-FIQA optional PyTorch dependencies (see requirements-crfiqa.txt)

A face detector or ArcFace embedder — you already have them

Calibrated thresholds — the shipped numbers are defaults; see docs/MODELS.md

Design promise
Once cache/metrics.jsonl exists, changing any threshold is a pass of pure
arithmetic. POST /api/reclassify completes in well under a second on 10k
faces and never touches a model.

License
Source in this package is provided as-is. Model weights carry their own
licenses — check before commercial deployment. InsightFace (SCRFD), CR-FIQA
and MediaPipe each have separate terms.
## Quality gate + ArcFace ONNX (no clustering)

The quality gate runs first. By default, only `GOOD` faces are sent to ArcFace;
`REVIEW` and `BAD` are rejected. This module does not perform identity matching
or clustering.

Install the optional ONNX Runtime dependency:

```bash
python -m pip install -r requirements-arcface.txt
```

Place your compatible ArcFace ONNX model at `models/w600k_r50.onnx`. Run the
existing FQA metric extraction first, then:

```bash
python -m fqa.arcface \
  --metrics cache/metrics.jsonl \
  --manifest cache/manifest.jsonl \
  --config config/balanced.json \
  --model models/w600k_r50.onnx \
  --out output/arcface_embeddings.jsonl
```

The output is JSONL. Each record contains `quality_decision`, `quality_category`,
`quality_score`, rejection reasons, and (for accepted faces) an L2-normalized
ArcFace embedding. `--include-review` explicitly allows REVIEW faces through.

For best ArcFace alignment, add `landmarks5` to each manifest record as five
`[x,y]` points in original-image pixel coordinates ordered as left eye, right
eye, nose tip, left mouth corner, right mouth corner. If your detector already
exports aligned 112x112 face crops, set `face_crop_path` instead. If neither is
provided, the module crops the bounding box and resizes it; this fallback is
less accurate than five-point alignment.

The default model preprocessing is InsightFace-style BGR, 112x112, float32
normalization `(pixel - 127.5) / 127.5`, and L2-normalized output. Confirm your
ONNX model uses this convention. Embeddings are biometric data: restrict access,
avoid public dashboard exposure, and define retention/deletion policies.


## Save GOOD face images only (no embeddings)

This is the recommended quality-gate workflow when you only want clean face crops
saved to disk. It does **not** load ArcFace, generate embeddings, or perform clustering.

1. Run the one-pass metric extraction shown above to create `cache/metrics.jsonl`.
2. Run the sorter:

```bash
python -m fqa.sort_faces \
  --metrics cache/metrics.jsonl \
  --manifest cache/manifest.jsonl \
  --config config/balanced.json \
  --dataset-root "D:/JioDataset" \
  --out output
```

The output is:

```text
output/
  good_faces/          # cropped face images classified GOOD
  quality_report.csv   # every face, decision, score and reason
  summary.json         # counts and workflow note
```

`REVIEW` and `BAD` images are not copied. Only `GOOD` face crops are saved.
The sorter reads the source image and bounding box from the FQA metrics/manifest;
if a manifest record has `face_crop_path`, that face crop is used instead.
Keep `--dataset-root` pointed at the trusted directory containing those files.
The optional `fqa.arcface` module remains separate and is not invoked by this command.

## Interactive configuration dashboard

See [docs/DASHBOARD.md](docs/DASHBOARD.md). The local browser dashboard at `http://127.0.0.1:8000` lets you tune every scoring threshold, score weight, facial-region weight, and rule action, preview the resulting GOOD/REVIEW/BAD counts, and save the profile JSON. Inference-side changes require a fresh inference run. The dashboard does not generate ArcFace embeddings.
