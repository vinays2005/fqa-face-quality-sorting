# FQA — Face Quality Assessment & Intelligent Sorting

Cached-metric FQA pipeline that runs after your existing face detection
and ArcFace clustering. Detects faces? No — consumes them. Extracts metrics
once, then reclassifies instantly under different threshold configs without
re-running any model.

## Quick start (Windows, Git Bash)

```bash
cd FQA_Face_Quality_Sorting_System

# 1. Environment
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

# 2. Model weights (not bundled — see docs/MODELS.md)
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

# 3. Build a manifest from a folder of images.
#    Handles BOTH a folder of separate single-face images AND montage/contact-sheet
#    pages that tile several pre-cropped faces into one image on a uniform grid
#    (it auto-detects which case each file is — see tools/images_to_manifest.py).
python tools/images_to_manifest.py \
  --input-dir images \
  --out cache/manifest.jsonl

# 4. Extract metrics ONCE (slow — runs MediaPipe on every face)
python -m fqa.inference \
  --manifest cache/manifest.jsonl \
  --config config/balanced.json \
  --landmarker models/face_landmarker.task \
  --out cache/metrics.jsonl
# Copy the "inference_fingerprint=..." value it prints — you need it next.

# 5. Classify + export (fast — pure arithmetic, re-run freely with any config)
python -m fqa.reclassify \
  --metrics cache/metrics.jsonl \
  --cache-fingerprint <PASTE_FINGERPRINT_HERE> \
  --config config/balanced.json \
  --out output/quality_report

# 6. Dashboard — tune thresholds live without re-running inference
export FQA_METRICS=cache/metrics.jsonl
export FQA_CONFIG=config/balanced.json
python -m uvicorn fqa.dashboard:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

Step 5 is the one you re-run every time you just want to try different
thresholds — it only needs step 4's `metrics.jsonl` and a `--cache-fingerprint`
that matches it. Only re-run step 4 if you change a setting that affects what
gets *measured* (`detector`, `min_det_confidence`, `use_mediapipe`,
`use_crfiqa`, `use_grayscale` — see table below).

## Editing thresholds

Three ways, same underlying `ScoringConfig` JSON (see `config/*.json`):

1. **Dashboard** (easiest): run step 6 above, drag sliders, click
   "Apply to current metrics" to instantly see new GOOD/REVIEW/BAD counts,
   then "Save configuration" to write it back to the config file.
2. **Edit the JSON directly**: open any file in `config/` and change a number,
   then re-run step 5 pointing `--config` at it.
3. **Copy a preset**: `config/balanced.json` / `strict.json` / `relaxed.json`
   are starting points at different strictness levels — copy one, tweak it,
   point `--config` at your copy.

### What each threshold does

| Field | Meaning | Raise it to... | Lower it to... |
|---|---|---|---|
| `good_threshold` | Minimum weighted score (0–100) to classify a face GOOD. | Accept fewer, higher-quality faces. | Accept more faces as GOOD. |
| `review_threshold` | Minimum score to be REVIEW instead of BAD. | Send more borderline faces to BAD instead of REVIEW. | Rescue more low-scoring faces into REVIEW instead of BAD. |
| `weights.*` (`visibility`, `completeness`, `sharpness`, `pose`, `occlusion`, `resolution`) | How much each sub-score contributes to the final weighted score. Must sum > 0. | Make that signal matter more to the final decision. | Make that signal matter less (set to `0` to ignore it entirely — e.g. when MediaPipe isn't available, so pose/visibility/occlusion can't be measured). |
| `min_visibility` | Minimum visibility sub-score (0–100, from MediaPipe landmark/region detection) before `visibility_below_min` fires. | Demand more of the face be clearly visible. | Tolerate more of the face being hidden/unclear. |
| `minimum_sharpness` | Blur floor (Laplacian-variance based). Below this, `blur_below_min` fires (hard reject by default). | Reject blurrier faces. | Allow blurrier faces through. |
| `sharpness_full_score` | Sharpness value that maps to a perfect 100 sub-score. | Make it harder to earn full sharpness credit. | Make it easier to earn full sharpness credit. |
| `max_yaw` / `max_pitch` / `max_roll` | Max allowed head rotation in degrees (left-right / up-down / tilt) before `pose_exceeded` fires. | Only accept more front-facing faces. | Tolerate more extreme head angles. |
| `min_face_width` / `min_face_height` | Minimum face crop size in pixels before `resolution_below_min` fires. | Reject smaller/more distant faces. | Accept smaller faces. |
| `resolution_full_at` | Face size (px) that earns a perfect resolution sub-score. | Demand bigger faces for full credit. | Easier to earn full resolution credit. |
| `max_occlusion` | Max fraction (0–1) of facial regions allowed to be covered/occluded before `severe_occlusion` fires. | Reject more occluded faces (masks, hands, hair). | Tolerate more occlusion. |
| `min_brightness` / `max_brightness` | Valid mean-brightness range (0–255) before `too_dark`/`too_bright` fire. | Narrow the acceptable lighting range. | Tolerate darker/brighter images. |
| `min_contrast` | Minimum contrast before `low_contrast` fires. | Reject flatter/washed-out images. | Tolerate lower contrast. |
| `max_boundary_clip` | Max fraction of the face bounding box allowed to be clipped by the image edge. | Reject more edge-cropped faces. | Tolerate faces cut off at the frame edge. |
| `min_det_confidence` | Minimum detector confidence before `detection_confidence_low` fires. | Trust only higher-confidence detections. | Accept lower-confidence detections. |
| `crfiqa_weight` | How much the optional CR-FIQA score contributes (0 disables it). | Weight CR-FIQA's opinion more. | Ignore CR-FIQA. |
| `rules.*` | Per-check behavior: `"off"` (ignore), `"review"` (downgrade to REVIEW), or `"reject"` (hard BAD) for each named check (`blur_below_min`, `pose_exceeded`, `severe_occlusion`, `multiple_essential_regions_missing`, `landmarks_failed`, etc). | Set to `"reject"` to make that check disqualifying. | Set to `"review"` or `"off"` to soften/ignore it — e.g. `severe_occlusion: "review"` if your images have overlay text/markers that the occlusion heuristic mistakes for real occlusion. |
| `use_mediapipe` | Whether to run MediaPipe landmarks at all (needed for visibility/pose/occlusion). **Inference-affecting** — requires re-running step 4. | — | Turn off if MediaPipe can't run in your environment; zero out `visibility`/`pose`/`occlusion` weights and relax their rules to compensate (see `config/demo_relaxed.json`). |
| `use_grayscale` | Flattens color before sharpness/brightness/contrast measurement only (landmarks/occlusion still use color). **Inference-affecting.** | — | Use when source images have colored overlays/annotations that would otherwise skew sharpness or lighting scores. |
| `detector` / `min_det_confidence` | Which upstream face detector produced the manifest, and its confidence floor. **Inference-affecting.** | | |

`good_threshold`, `review_threshold`, and every `rules.*`/weight value take
effect immediately on the next reclassify (step 5) — no need to re-run
inference. Fields marked **inference-affecting** change what gets measured,
not just how it's scored, so the cache fingerprint changes and you must
re-run step 4 first; `fqa.reclassify` will refuse a stale cache and tell you
so if you forget.

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
