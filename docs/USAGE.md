# Usage — GOOD face image filtering

All commands assume your virtual environment is activated and your terminal is
inside the project directory.

## 1. Create a manifest from your existing face detector

Create `cache/manifest.jsonl`, one JSON object per detected face:

```json
{"face_id":"frame001_face0","image_path":"D:/data/frames/frame001.jpg","bbox":[12,10,208,220],"det_confidence":0.97}
```

`image_path` points to the original frame and `bbox` is `[x0,y0,x1,y1]` in
that frame's pixel coordinates. A pre-cropped image can instead be supplied as
`face_crop_path`. No cluster ID or embedding is needed.

## 2. Run FQA metric extraction once

```bash
python -m fqa.inference \
  --manifest cache/manifest.jsonl \
  --config config/balanced.json \
  --landmarker models/face_landmarker.task \
  --out cache/metrics.jsonl
```

## 3. Save GOOD face crops only

```bash
python -m fqa.sort_faces \
  --metrics cache/metrics.jsonl \
  --manifest cache/manifest.jsonl \
  --config config/balanced.json \
  --dataset-root "D:/data" \
  --out output
```

Expected output:

```text
output/
  good_faces/          # only face crops classified GOOD
  quality_report.csv   # all decisions, scores and reasons
  summary.json         # decision counts
```

REVIEW and BAD faces are not copied. Originals are never deleted or modified.
The command does not load ArcFace or generate embeddings.

## 4. Optional dashboard

```bash
export FQA_METRICS="cache/metrics.jsonl"
export FQA_CONFIG="config/balanced.json"
export FQA_DATASET_ROOT="D:/data"
uvicorn fqa.dashboard:app --host 127.0.0.1 --port 8000
```

On Windows PowerShell, use `$env:FQA_METRICS = "cache/metrics.jsonl"` syntax.
Open `http://127.0.0.1:8000`. Keep the dashboard bound to localhost.
