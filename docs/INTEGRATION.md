# Integrating with your existing face detector

This workflow only filters face images by visual quality. It does not perform
identity matching, clustering, or ArcFace embedding generation.

## Input contract

Create `cache/manifest.jsonl`, with one JSON object per detected face:

```json
{"face_id":"frame001_face0","image_path":"D:/data/frames/frame001.jpg","bbox":[12,10,208,220],"det_confidence":0.97}
```

Required fields:

- `face_id`: stable unique ID for this detection.
- `image_path`: path to the original image/frame containing the face.
- `bbox`: `[x0, y0, x1, y1]` pixel coordinates in that original image.
- `det_confidence`: detector confidence.

Optional:

- `face_crop_path`: path to an already-extracted face crop. If provided, the
  sorter saves this crop rather than cropping `image_path` using `bbox`.
- `landmarks5`: five facial landmarks in original-image coordinates, if your
  detector supplies them and a later optional ArcFace workflow needs them.

No `cluster_id`, embeddings, or clustering metadata are required.

## Producer example

```python
import json

with open("cache/manifest.jsonl", "w", encoding="utf-8") as stream:
    for face in existing_detections:
        record = {
            "face_id": str(face.id),
            "image_path": str(face.frame_path),
            "bbox": [float(v) for v in face.bbox],
            "det_confidence": float(face.confidence),
        }
        stream.write(json.dumps(record) + "\n")
```

## Processing flow

1. Run `python -m fqa.inference` to calculate quality metrics once.
2. Run `python -m fqa.sort_faces` to evaluate each face and save GOOD crops.
3. Review `output/quality_report.csv` for GOOD, REVIEW, and BAD decisions.

Only GOOD face crops are copied into `output/good_faces/`. REVIEW and BAD
images remain untouched in the source dataset. No ArcFace model is loaded and
no embeddings are produced by this workflow.

## Safety

Keep all source images under the directory passed to `--dataset-root`. The
sorter rejects symlink inputs and paths that resolve outside that directory.
It copies/crops images; it never deletes or modifies source images.

## Integrating with an HDBSCAN clustering step

If your upstream pipeline runs HDBSCAN over face embeddings and produces a
per-detection cluster label, use `tools/cluster_to_manifest.py` to convert
that output straight into `manifest.jsonl` instead of writing a producer
script by hand:

```bash
python tools/cluster_to_manifest.py \
  --input detections.jsonl \
  --out cache/manifest.jsonl
```

Input is one row per detection (`.csv` or `.jsonl`) with `image_path`, `bbox`
(or `x0,y0,x1,y1`), and a `cluster_id` column — HDBSCAN noise (`-1`) is
normalized to the cluster id `"noise"`. `det_confidence`, `face_crop_path`,
and `landmarks5` are optional passthroughs. This script does not run
detection, clustering, or any embedding model; it only reshapes rows you
already have.

Then run the normal flow (`fqa.inference` -> `fqa.sort_faces`) and pass
`--group-by-cluster` to `fqa.sort_faces` to get GOOD crops split back into
one folder per cluster:

```bash
python -m fqa.sort_faces \
  --metrics cache/metrics.jsonl \
  --manifest cache/manifest.jsonl \
  --config config/balanced.json \
  --dataset-root D:/JioDataset \
  --out output \
  --group-by-cluster
```

This produces:

```text
output/
  good_faces/
    0/             # GOOD crops for cluster "0"
    1/
    noise/         # GOOD crops for HDBSCAN noise (-1)
  quality_report.csv   # now includes a cluster_id column
  summary.json         # now includes a by_cluster breakdown
```

Cluster assignments themselves are never changed, merged, or re-scored — this
only decides, per cluster, which already-assigned faces are good enough to
keep.
