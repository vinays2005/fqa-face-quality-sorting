"""Benchmark inference on YOUR RTX 2050. Do not trust any GPU number you did
not see come out of this script."""
import argparse, json, time
try:
    import torch
except ImportError:
    torch = None
from fqa.inference import extract, CrFiqaAdapter
from fqa.engine import ScoringConfig, write_json_atomic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--landmarker", required=True)
    p.add_argument("--crfiqa-dir"); p.add_argument("--crfiqa-calib")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default="reports/bench.json")
    a = p.parse_args()
    cfg = ScoringConfig(use_crfiqa=bool(a.crfiqa_dir),
                        crfiqa_weight=0.15 if a.crfiqa_dir else 0.0)
    crf = CrFiqaAdapter(a.crfiqa_dir, a.crfiqa_calib) if a.crfiqa_dir else None
    recs = []
    with open(a.metrics, encoding="utf-8") as f:
        for line in f:
            if len(recs) >= a.n: break
            if line.strip(): recs.append(json.loads(line))
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    list(extract(recs, cfg, a.landmarker, crf))
    elapsed = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e6) if cuda_available else 0.0
    write_json_atomic(a.out, {
        "n": len(recs), "elapsed_s": round(elapsed, 2),
        "per_image_ms": round(1000 * elapsed / max(1, len(recs)), 2),
        "peak_vram_mb": round(peak, 1),
        "cuda": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
    })
    print(json.dumps(json.load(open(a.out)), indent=2))


if __name__ == "__main__":
    main()