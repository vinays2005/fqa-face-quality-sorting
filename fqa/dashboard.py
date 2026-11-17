"""Local FastAPI dashboard for cached face-quality classification."""
from __future__ import annotations

import json, os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from fqa.engine import (FaceMetrics, ScoringConfig, assess_all,
                        summarize, compare, best_faces, write_json_atomic)

METRICS = os.environ.get("FQA_METRICS", "cache/metrics.jsonl")
OVERRIDES = os.environ.get("FQA_OVERRIDES", "cache/overrides.json")
DEFAULT_CFG = os.environ.get("FQA_CONFIG", "config/balanced.json")
DATASET_ROOT = os.path.realpath(os.environ.get("FQA_DATASET_ROOT", "."))
DASHBOARD_HTML = Path(__file__).parent.parent / "assets" / "dashboard.html"

app = FastAPI(title="FQA")
_state = {"metrics": [], "overrides": {}, "config": None}


@app.on_event("startup")
def _load():
    metrics = []
    with open(METRICS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metrics.append(FaceMetrics.from_dict(json.loads(line)))
    _state["metrics"] = metrics
    if os.path.exists(OVERRIDES):
        with open(OVERRIDES, encoding="utf-8") as f:
            _state["overrides"] = json.load(f)
    _state["config"] = ScoringConfig.from_json(DEFAULT_CFG)
    for m in _state["metrics"]:
        if (o := _state["overrides"].get(m.face_id)):
            m.manual_override = o


class ReclassifyRequest(BaseModel):
    config: dict
    compare_with: dict | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return DASHBOARD_HTML.read_text(encoding="utf-8")


@app.get("/api/state")
def state():
    cfg = _state["config"]
    results = assess_all(_state["metrics"], cfg)
    stale = any(m.inference_fingerprint and
                m.inference_fingerprint != cfg.inference_fingerprint()
                for m in _state["metrics"])
    return {"summary": summarize(results),
            "clusters": sorted({m.cluster_id for m in _state["metrics"] if m.cluster_id}),
            "config": cfg.to_dict(),
            "inference_fingerprint": cfg.inference_fingerprint(),
            "inference_stale": stale}


@app.post("/api/reclassify")
def reclassify(req: ReclassifyRequest):
    try:
        new_cfg = ScoringConfig.from_dict(req.config)
    except ValueError as e:
        raise HTTPException(400, str(e))
    stale_metrics = any(m.inference_fingerprint and
                        m.inference_fingerprint != _state["config"].inference_fingerprint()
                        for m in _state["metrics"])
    if stale_metrics:
        raise HTTPException(409, "cached metrics are stale for the saved inference settings; rerun fqa.inference")
    if new_cfg.inference_fingerprint() != _state["config"].inference_fingerprint():
        raise HTTPException(409,
            "config changes inference-side keys; re-run inference first")
    before = {a.face_id: a for a in assess_all(_state["metrics"], _state["config"])}
    _state["config"] = new_cfg
    after = {a.face_id: a for a in assess_all(_state["metrics"], new_cfg)}
    result = {"summary": summarize(list(after.values())),
              "changed": compare(before, after)["changed"] if req.compare_with else 0}
    if req.compare_with:
        try:
            other = ScoringConfig.from_dict(req.compare_with)
            if other.inference_fingerprint() != new_cfg.inference_fingerprint():
                raise HTTPException(409, "comparison config changes inference-side settings")
            other_res = assess_all(_state["metrics"], other)
            result["comparison"] = compare(
                {a.face_id: a for a in other_res}, after)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return result


@app.get("/api/presets/{name}")
def get_preset(name: str):
    if name not in {"strict", "balanced", "relaxed"}:
        raise HTTPException(404, "unknown preset")
    path = Path("config") / f"{name}.json"
    try:
        return ScoringConfig.from_json(str(path)).to_dict()
    except (OSError, ValueError) as exc:
        raise HTTPException(500, f"could not load preset: {exc}")


@app.post("/api/config/save")
def save_config(req: ReclassifyRequest):
    """Persist a validated profile. Inference-side changes require fresh metrics."""
    try:
        new_cfg = ScoringConfig.from_dict(req.config)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    target = Path(DEFAULT_CFG)
    # DEFAULT_CFG is an operator-controlled environment setting, never a client path.
    write_json_atomic(str(target), new_cfg.to_dict())
    inference_changed = new_cfg.inference_fingerprint() != _state["config"].inference_fingerprint()
    if not inference_changed:
        _state["config"] = new_cfg
    return {"ok": True, "saved_to": str(target),
            "inference_changed": inference_changed,
            "message": ("Saved. Re-run fqa.inference before applying these inference settings."
                        if inference_changed else "Saved and applied to the loaded metrics.")}


@app.get("/api/faces")
def faces(category: str = Query(..., pattern="^(GOOD|REVIEW|BAD)$"),
          cluster: str | None = None, offset: int = 0, limit: int = 48):
    results = assess_all(_state["metrics"], _state["config"])
    pool = [a for a in results if a.final_category == category
            and (cluster is None or a.cluster_id == cluster)]
    return {"total": len(pool), "items": [
        {**a.to_dict(), "thumb": f"/api/image/{a.face_id}"}
        for a in pool[offset:offset + limit]]}


@app.get("/api/image/{face_id}")
def image(face_id: str):
    m = next((x for x in _state["metrics"] if x.face_id == face_id), None)
    if m is None or not os.path.isfile(m.image_path):
        raise HTTPException(404)
    image_path = os.path.realpath(m.image_path)
    try:
        inside_root = os.path.commonpath([DATASET_ROOT, image_path]) == DATASET_ROOT
    except ValueError:
        inside_root = False
    if not inside_root or os.path.islink(m.image_path):
        raise HTTPException(403, "image path is outside the configured dataset root")
    return FileResponse(image_path)


class OverrideRequest(BaseModel):
    face_id: str
    category: str | None = None


@app.post("/api/override")
def override(req: OverrideRequest):
    if req.category not in (None, "GOOD", "REVIEW", "BAD"):
        raise HTTPException(400, "invalid category")
    if req.category is None:
        _state["overrides"].pop(req.face_id, None)
    else:
        _state["overrides"][req.face_id] = req.category
    write_json_atomic(OVERRIDES, _state["overrides"])
    for m in _state["metrics"]:
        if m.face_id == req.face_id:
            m.manual_override = req.category
    return {"ok": True}


@app.get("/api/best")
def best(k: int = Query(3, ge=1, le=100), include_review: bool = False):
    results = assess_all(_state["metrics"], _state["config"])
    return {cid: [a.face_id for a in v]
            for cid, v in best_faces(results, k, include_review).items()}


@app.get("/api/compare")
def compare_configs(quality: float = 80, visibility: float = 75,
                    quality_b: float = 65, visibility_b: float = 55):
    cfg_a = ScoringConfig(**{**_state["config"].to_dict(),
                             "good_threshold": quality, "min_visibility": visibility})
    cfg_b = ScoringConfig(**{**_state["config"].to_dict(),
                             "good_threshold": quality_b, "min_visibility": visibility_b})
    a = {x.face_id: x for x in assess_all(_state["metrics"], cfg_a)}
    b = {x.face_id: x for x in assess_all(_state["metrics"], cfg_b)}
    return compare(a, b)