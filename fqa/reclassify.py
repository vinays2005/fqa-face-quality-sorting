"""Reclassify cached metrics. No inference. Safe, non-destructive export."""
from __future__ import annotations

import argparse, csv, json, os, re, shutil, sys
from collections import defaultdict
from pathlib import Path

from fqa.engine import (FaceMetrics, ScoringConfig, assess_all, summarize,
                        compare, best_faces, write_json_atomic)

_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def safe_slug(value, fallback="unclustered"):
    if not value:
        return fallback
    v = str(value).strip()
    if not _SLUG.fullmatch(v) or v in (".", "..") or ".." in v:
        raise ValueError(f"unsafe cluster/file identifier: {value!r}")
    return v


def _within(root_real, path):
    try:
        return os.path.commonpath([os.path.realpath(root_real), os.path.realpath(path)]) == os.path.realpath(root_real)
    except ValueError:  # e.g. different Windows drives
        return False


def load_cache(metrics_path, cache_fp, cfg):
    if cache_fp != cfg.inference_fingerprint():
        raise SystemExit(
            f"cache fingerprint {cache_fp} != config {cfg.inference_fingerprint()}. "
            f"Re-run inference or revert detector/min_det_confidence/use_mediapipe/use_crfiqa.")
    out = []
    with open(metrics_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(FaceMetrics.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as e:
                print(f"skip {metrics_path}:{i}: {e}", file=sys.stderr)
    return out


def safe_copy(src, dest_dir, dataset_root_real, output_root_real):
    # Check the original path before realpath(), otherwise a symlink is resolved
    # and os.path.islink(realpath(src)) can never detect it.
    if os.path.islink(src):
        raise ValueError(f"refusing to copy symlink: {src}")
    src_real = os.path.realpath(src)
    if not _within(dataset_root_real, src_real):
        raise ValueError(f"source escapes dataset root: {src}")
    if not _within(output_root_real, dest_dir):
        raise ValueError(f"destination escapes output root: {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(src_real))
    dest = os.path.join(dest_dir, os.path.basename(src_real))
    n = 1
    while os.path.exists(dest):
        dest = os.path.join(dest_dir, f"{stem}__{n}{ext}")
        n += 1
    shutil.copy2(src_real, dest)
    return dest


def export(assessments, out_dir, dataset_root, copy_images=True):
    out_real = os.path.realpath(out_dir)
    ds_real = os.path.realpath(dataset_root)
    os.makedirs(out_real, exist_ok=True)
    grouped = defaultdict(list)
    for a in assessments:
        cluster = safe_slug(a.cluster_id)
        bucket = safe_slug(a.final_category.lower())
        bucket_dir = os.path.join(out_real, cluster, bucket)
        rec = a.to_dict()
        if copy_images and os.path.exists(a.image_path):
            try:
                rec["exported_path"] = safe_copy(
                    a.image_path, os.path.join(bucket_dir, "images"),
                    ds_real, out_real)
            except ValueError as e:
                rec["exported_path"] = None
                rec["export_error"] = str(e)
        grouped[(cluster, bucket)].append(rec)
    for (cluster, bucket), recs in grouped.items():
        write_json_atomic(os.path.join(out_real, cluster, bucket, "metadata.json"),
                          {"cluster_id": cluster, "category": bucket, "faces": recs})
    by_cluster = defaultdict(list)
    for a in assessments:
        by_cluster[safe_slug(a.cluster_id)].append(a)
    for cluster, items in by_cluster.items():
        write_json_atomic(os.path.join(out_real, cluster, "cluster_metadata.json"),
                          {"cluster_id": cluster,
                           "summary": summarize(items),
                           "faces": [a.to_dict() for a in items]})


def write_reports(assessments, reports_dir):
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "quality_report.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["face_id", "cluster_id", "image_path", "score", "auto_category",
                    "final_category", "visibility", "completeness", "sharpness",
                    "pose", "occlusion", "resolution", "reasons"])
        for a in assessments:
            s = a.subscores
            w.writerow([a.face_id, a.cluster_id or "", a.image_path, a.score,
                        a.auto_category, a.final_category,
                        s.get("visibility"), s.get("completeness"), s.get("sharpness"),
                        s.get("pose"), s.get("occlusion"), s.get("resolution"),
                        "; ".join(r["rule"] for r in a.reasons)])
    write_json_atomic(os.path.join(reports_dir, "quality_report.json"),
                      {"summary": summarize(assessments),
                       "faces": [a.to_dict() for a in assessments]})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--cache-fingerprint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="output")
    p.add_argument("--dataset-root", default=".")
    p.add_argument("--compare")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--include-review-in-best", action="store_true")
    p.add_argument("--no-copy", action="store_true")
    args = p.parse_args(argv)
    cfg = ScoringConfig.from_json(args.config)
    metrics = load_cache(args.metrics, args.cache_fingerprint, cfg)
    results = assess_all(metrics, cfg)
    print(json.dumps(summarize(results), indent=2))
    export(results, args.out, args.dataset_root, copy_images=not args.no_copy)
    write_reports(results, os.path.join(args.out, "reports"))
    if args.compare:
        cfg_b = ScoringConfig.from_json(args.compare)
        if cfg_b.inference_fingerprint() != cfg.inference_fingerprint():
            raise SystemExit("cannot compare configs with different inference fingerprints")
        results_b = assess_all(metrics, cfg_b)
        diff = compare({a.face_id: a for a in results},
                       {a.face_id: a for a in results_b})
        write_json_atomic(os.path.join(args.out, "reports", "config_comparison.json"), diff)
        print(f"\n{args.config} vs {args.compare}: {diff['changed']} faces changed category")
    if args.top_k:
        best = best_faces(results, args.top_k, args.include_review_in_best)
        write_json_atomic(os.path.join(args.out, "reports", "best_faces.json"),
                          {cid: [{"face_id": a.face_id, "score": a.score,
                                  "image_path": a.image_path} for a in v]
                           for cid, v in best.items()})


if __name__ == "__main__":
    main()