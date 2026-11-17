"""Build an FQA manifest.jsonl from HDBSCAN face-clustering output.

Reads per-detection rows produced by an existing detection+clustering pipeline
(one row per face detection, carrying a cluster label) and writes them to the
manifest.jsonl schema documented in docs/INTEGRATION.md. This is a pass-through
layer: it does not run detection, clustering, or any embedding model, and it
does not alter bounding boxes or cluster assignments.

Accepted input: .csv or .jsonl, one row/record per detection.

Required input fields (case-sensitive):
  image_path   - path to the original frame containing the face
  bbox         - either a single field holding "x0,y0,x1,y1" / a JSON list
                 (CSV), or a JSON array (JSONL)
Optional input fields:
  face_id        - stable unique id; generated if missing
  det_confidence - detector confidence; defaults to 1.0 if absent
  cluster_id     - HDBSCAN cluster label; -1 (or blank) is treated as noise
  face_crop_path - path to an already-extracted crop
  landmarks5     - JSON array of five [x, y] points

Output manifest.jsonl rows additionally carry cluster_id (kept as a string,
with HDBSCAN noise normalized to "noise") so downstream fqa.sort_faces can
regroup GOOD crops per cluster.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

NOISE_LABEL = "noise"


def _normalize_cluster_id(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() == "none":
        return None
    if s in ("-1", "-1.0"):
        return NOISE_LABEL
    return s


def _parse_bbox(raw: Any) -> list[float]:
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    elif isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            values = json.loads(s)
        else:
            values = [p.strip() for p in s.split(",")]
    else:
        raise ValueError(f"Unsupported bbox value: {raw!r}")
    if len(values) != 4:
        raise ValueError(f"bbox must have exactly 4 values, got {values!r}")
    return [float(v) for v in values]


def _parse_landmarks5(raw: Any) -> list[list[float]] | None:
    if raw is None or raw == "":
        return None
    values = raw if isinstance(raw, list) else json.loads(raw)
    if len(values) != 5 or any(len(p) != 2 for p in values):
        raise ValueError(f"landmarks5 must be five [x, y] points, got {values!r}")
    return [[float(x), float(y)] for x, y in values]


def _stable_face_id(image_path: str, bbox: list[float], index: int) -> str:
    digest = hashlib.sha256(f"{image_path}|{bbox}|{index}".encode("utf-8")).hexdigest()[:12]
    stem = Path(image_path).stem
    readable = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)[-50:]
    return f"{readable or 'face'}_{index}_{digest}"


def _read_rows(input_path: Path) -> list[dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {input_path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {input_path}:{line_no}")
                rows.append(row)
        return rows
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    raise ValueError(f"Unsupported input extension {suffix!r}; use .csv or .jsonl")


def build_manifest(input_path: Path, output_path: Path) -> dict[str, int]:
    rows = _read_rows(input_path)
    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")

    output_resolved = output_path.resolve(strict=False)
    output_resolved.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    cluster_counts: dict[str, int] = {}
    temp_path = output_resolved.with_name(output_resolved.name + ".tmp")

    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
            for index, row in enumerate(rows):
                image_path = row.get("image_path")
                if not image_path:
                    raise ValueError(f"Row {index}: missing required field image_path")
                bbox_raw = row.get("bbox")
                if bbox_raw is None:
                    bbox_raw = [row.get("x0"), row.get("y0"), row.get("x1"), row.get("y1")]
                    if any(v is None for v in bbox_raw):
                        raise ValueError(f"Row {index}: missing bbox (or x0/y0/x1/y1)")
                bbox = _parse_bbox(bbox_raw)

                face_id = row.get("face_id") or _stable_face_id(str(image_path), bbox, index)
                if face_id in seen_ids:
                    raise ValueError(f"Duplicate face_id after normalization: {face_id!r}")
                seen_ids.add(face_id)

                cluster_id = _normalize_cluster_id(row.get("cluster_id"))
                record: dict[str, Any] = {
                    "face_id": face_id,
                    "image_path": str(image_path),
                    "bbox": bbox,
                    "det_confidence": float(row.get("det_confidence", 1.0)),
                    "cluster_id": cluster_id,
                }
                face_crop_path = row.get("face_crop_path")
                if face_crop_path:
                    record["face_crop_path"] = str(face_crop_path)
                landmarks5 = _parse_landmarks5(row.get("landmarks5"))
                if landmarks5 is not None:
                    record["landmarks5"] = landmarks5

                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                key = cluster_id or "unclustered"
                cluster_counts[key] = cluster_counts.get(key, 0) + 1

        os.replace(temp_path, output_resolved)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return {"total_faces": len(seen_ids), "clusters": len(cluster_counts), **cluster_counts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert HDBSCAN face-clustering output into an FQA manifest.jsonl"
    )
    parser.add_argument("--input", required=True, help="Detections file (.csv or .jsonl) with a cluster label per row")
    parser.add_argument("--out", default="cache/manifest.jsonl", help="Output manifest path")
    args = parser.parse_args(argv)

    try:
        summary = build_manifest(Path(args.input), Path(args.out))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2))
    print(f"Manifest written: {Path(args.out).resolve()}")
    print("No detection, clustering, or ArcFace embeddings were run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
