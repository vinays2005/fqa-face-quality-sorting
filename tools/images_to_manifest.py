"""Build an FQA manifest.jsonl from a folder of plain image files.

Handles two input shapes, per file, automatically:

1. A "montage" page: several already-cropped face thumbnails tiled into one
   image on a uniform grid (e.g. a debug/report image showing N cluster
   members per row). Detected when the image's width and height are each an
   exact multiple of --cell-size and that grid has more than one cell: the
   image is sliced into one manifest record per grid cell.
2. A single separate face image: one manifest record covering the whole
   image.

No face detector, clustering, or embedding model is run here. The grid split
is pure geometry (fixed cell size), not detection, so it only applies when
the image dimensions actually tile evenly; anything else is treated as one
face per file. Use --force-single to always treat every file as one face
(e.g. a folder that is known to contain only individual crops).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _image_size(path: Path) -> tuple[int, int]:
    import cv2
    import numpy as np

    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    h, w = img.shape[:2]
    return w, h


def _face_id(relative_path: str, suffix: str = "") -> str:
    normalized = relative_path.replace("\\", "/")
    digest = hashlib.sha256((normalized + suffix).encode("utf-8")).hexdigest()[:10]
    stem = Path(normalized).with_suffix("").as_posix()
    readable = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)[-60:]
    return f"{readable or 'face'}{suffix}_{digest}"


def _records_for_image(resolved: Path, relative_path: str, cell_size: int,
                        force_single: bool) -> list[dict]:
    width, height = _image_size(resolved)
    is_grid = (not force_single and cell_size > 0
               and width % cell_size == 0 and height % cell_size == 0
               and (width // cell_size) * (height // cell_size) > 1)
    if not is_grid:
        return [{
            "face_id": _face_id(relative_path),
            "image_path": str(resolved),
            "bbox": [0, 0, width, height],
            "det_confidence": 1.0,
        }]

    cols, rows = width // cell_size, height // cell_size
    records = []
    for row in range(rows):
        for col in range(cols):
            x0, y0 = col * cell_size, row * cell_size
            records.append({
                "face_id": _face_id(relative_path, f"_r{row}c{col}"),
                "image_path": str(resolved),
                "bbox": [x0, y0, x0 + cell_size, y0 + cell_size],
                "det_confidence": 1.0,
                "montage_cell": {"row": row, "col": col, "cell_size": cell_size},
            })
    return records


def build_manifest(input_dir: Path, output_path: Path, cell_size: int,
                    force_single: bool) -> dict[str, int]:
    root = input_dir.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    output_resolved = output_path.resolve(strict=False)
    try:
        output_resolved.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("Manifest output must be outside the input image directory.")

    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file() and not p.is_symlink()
    )
    if not image_paths:
        raise RuntimeError(f"No supported image files found under: {root}")

    temp_path = output_resolved.with_name(output_resolved.name + ".tmp")
    seen_ids: set[str] = set()
    single_files = montage_files = total_faces = skipped = 0

    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
            for image_path in image_paths:
                try:
                    resolved = image_path.resolve(strict=True)
                    resolved.relative_to(root)
                except (OSError, ValueError):
                    skipped += 1
                    continue

                relative_path = resolved.relative_to(root).as_posix()
                try:
                    records = _records_for_image(resolved, relative_path, cell_size, force_single)
                except ValueError:
                    skipped += 1
                    continue

                if len(records) > 1:
                    montage_files += 1
                else:
                    single_files += 1

                for record in records:
                    if record["face_id"] in seen_ids:
                        raise RuntimeError(f"Duplicate generated face_id: {record['face_id']}")
                    seen_ids.add(record["face_id"])
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_faces += 1

        os.replace(temp_path, output_resolved)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return {
        "files_seen": len(image_paths),
        "single_face_files": single_files,
        "montage_files": montage_files,
        "total_faces": total_faces,
        "skipped_files": skipped,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an FQA manifest.jsonl from a folder of images, splitting "
                    "uniform-grid montage pages into one record per face cell."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing images")
    parser.add_argument("--out", default="cache/manifest.jsonl", help="Output JSONL manifest")
    parser.add_argument("--cell-size", type=int, default=96,
                        help="Grid cell side length in pixels for montage detection (default: 96). "
                             "A file whose width and height are both exact multiples of this, with "
                             "more than one cell, is split into a face per cell.")
    parser.add_argument("--force-single", action="store_true",
                        help="Never split: treat every file as exactly one face image")
    args = parser.parse_args(argv)

    if args.cell_size < 1:
        parser.error("--cell-size must be >= 1")

    try:
        summary = build_manifest(Path(args.input_dir), Path(args.out), args.cell_size, args.force_single)
    except (OSError, ValueError, RuntimeError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2))
    print(f"Manifest written: {Path(args.out).resolve()}")
    print("No face detection, clustering, or ArcFace embeddings were run; "
          "montage splitting is fixed-grid geometry only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
