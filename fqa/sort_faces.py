"""Run the FQA gate and export GOOD face crops only; no embeddings or clustering."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2

from fqa.engine import FaceMetrics, ScoringConfig, assess
from fqa.reclassify import safe_slug


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _safe_source(path_value: str, dataset_root: Path) -> Path:
    raw = Path(path_value)
    if raw.is_symlink():
        raise ValueError(f"Refusing symlink input: {raw}")
    resolved = raw.resolve(strict=True)
    root = dataset_root.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Input path is outside dataset root: {path_value}") from exc
    if not resolved.is_file():
        raise ValueError(f"Input is not a regular file: {path_value}")
    return resolved


def _crop_face(metric: FaceMetrics, manifest_row: dict[str, Any], dataset_root: Path):
    crop_path = manifest_row.get("face_crop_path")
    if crop_path:
        source = _safe_source(str(crop_path), dataset_root)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode face crop: {source}")
        return image, source

    source = _safe_source(metric.image_path, dataset_root)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode source image: {source}")
    x0, y0, x1, y1 = metric.bbox
    h, w = image.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid/empty face bbox for {metric.face_id}: {metric.bbox}")
    return image[y0:y1, x0:x1].copy(), source


def _output_name(face_id: str, source: Path) -> str:
    # Avoid path traversal and collisions while keeping filenames recognizable.
    readable = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in face_id)[:70]
    digest = hashlib.sha256(face_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable or 'face'}_{digest}{source.suffix.lower() or '.jpg'}"


def run_sort(metrics_path: str, manifest_path: str, config_path: str,
             dataset_root: str, output_dir: str, group_by_cluster: bool = False) -> dict[str, int]:
    cfg = ScoringConfig.from_json(config_path)
    root = Path(dataset_root).resolve(strict=True)
    out = Path(output_dir).resolve()
    if out == root or root in out.parents:
        raise ValueError("Output directory must not be inside the input dataset root")
    out.mkdir(parents=True, exist_ok=True)
    good_dir = out / "good_faces"
    good_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_jsonl(manifest_path)
    manifest: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        face_id = str(row.get("face_id", ""))
        if not face_id:
            raise ValueError("Every manifest record must have a non-empty face_id")
        if face_id in manifest:
            raise ValueError(f"Duplicate face_id in manifest: {face_id}")
        manifest[face_id] = row

    metrics_rows = load_jsonl(metrics_path)
    counts = {"GOOD": 0, "REVIEW": 0, "BAD": 0, "saved": 0, "export_failed": 0}
    cluster_counts: dict[str, dict[str, int]] = {}
    report_rows: list[dict[str, Any]] = []
    used_names: set[tuple[str, str]] = set()

    for row in metrics_rows:
        metric = FaceMetrics.from_dict(row)
        result = assess(metric, cfg)
        category = result.final_category
        counts[category] += 1
        cluster_key = safe_slug(metric.cluster_id) if group_by_cluster else None
        if group_by_cluster:
            bucket = cluster_counts.setdefault(cluster_key, {"GOOD": 0, "REVIEW": 0, "BAD": 0, "saved": 0})
            bucket[category] += 1
        exported_path = ""
        error = ""
        if category == "GOOD":
            try:
                source_row = manifest.get(metric.face_id)
                if source_row is None:
                    raise ValueError("face_id not found in manifest")
                crop, source_path = _crop_face(metric, source_row, root)
                filename = _output_name(metric.face_id, source_path)
                dest_dir = good_dir / cluster_key if group_by_cluster else good_dir
                dest_dir.mkdir(parents=True, exist_ok=True)
                name_key = (str(dest_dir), filename)
                if name_key in used_names:
                    raise ValueError("Output filename collision")
                used_names.add(name_key)
                destination = dest_dir / filename
                # Encode then atomically replace, avoiding partially written images.
                ext = destination.suffix if destination.suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"
                ok, encoded = cv2.imencode(ext, crop)
                if not ok:
                    raise ValueError("OpenCV could not encode face crop")
                fd, temp_name = tempfile.mkstemp(prefix=".fqa_", suffix=ext, dir=dest_dir)
                os.close(fd)
                try:
                    with open(temp_name, "wb") as target:
                        target.write(encoded.tobytes())
                        target.flush()
                        os.fsync(target.fileno())
                    os.replace(temp_name, destination)
                finally:
                    if os.path.exists(temp_name):
                        os.unlink(temp_name)
                exported_path = str(destination)
                counts["saved"] += 1
                if group_by_cluster:
                    cluster_counts[cluster_key]["saved"] += 1
            except Exception as exc:
                error = str(exc)[:500]
                counts["export_failed"] += 1

        report_rows.append({
            "face_id": metric.face_id,
            "cluster_id": metric.cluster_id or "",
            "source_image": metric.image_path,
            "decision": category,
            "quality_score": result.score,
            "saved_image": exported_path,
            "reasons": "; ".join(item["detail"] for item in result.reasons),
            "export_error": error,
        })

    csv_path = out / "quality_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report_rows[0]) if report_rows else
                                ["face_id", "cluster_id", "source_image", "decision", "quality_score",
                                 "saved_image", "reasons", "export_error"])
        writer.writeheader()
        writer.writerows(report_rows)

    summary = {**counts, "total": len(report_rows),
               "note": "Only GOOD face crops were saved. REVIEW and BAD images were not copied. No ArcFace or embeddings were run."}
    if group_by_cluster:
        summary["by_cluster"] = cluster_counts
    summary_path = out / "summary.json"
    temp = summary_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temp, summary_path)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description="FQA quality gate: save GOOD face crops only")
    parser.add_argument("--metrics", required=True, help="Metrics JSONL created by fqa.inference")
    parser.add_argument("--manifest", required=True, help="Source manifest JSONL")
    parser.add_argument("--config", default="config/balanced.json")
    parser.add_argument("--dataset-root", required=True, help="Allowed root containing source images/crops")
    parser.add_argument("--out", default="output")
    parser.add_argument("--group-by-cluster", action="store_true",
                        help="Save GOOD crops into good_faces/<cluster_id>/ using cluster_id from the manifest "
                             "(HDBSCAN noise/-1 maps to good_faces/noise/)")
    args = parser.parse_args(argv)
    counts = run_sort(args.metrics, args.manifest, args.config, args.dataset_root, args.out,
                      group_by_cluster=args.group_by_cluster)
    print(json.dumps(counts, indent=2))
    print(f"GOOD face crops: {Path(args.out).resolve() / 'good_faces'}")
    print("No ArcFace model was loaded and no embeddings were generated.")


if __name__ == "__main__":
    main()
