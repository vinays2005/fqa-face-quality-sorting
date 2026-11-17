"""Cached-metric Face Quality Assessment: scoring & classification.

Never runs a neural network. Consumes FaceMetrics produced once by
fqa.inference and produces GOOD / REVIEW / BAD. Editing a threshold is a pass
of pure arithmetic over the cache.
"""
from __future__ import annotations

import hashlib, json, os, tempfile
from dataclasses import dataclass, field, fields
from typing import Any, Literal

REGIONS = ("left_eye", "right_eye", "nose", "mouth",
           "left_cheek", "right_cheek", "forehead", "chin")
ESSENTIAL = ("left_eye", "right_eye", "nose", "mouth")
Category = Literal["GOOD", "REVIEW", "BAD"]
RuleState = Literal["off", "review", "reject"]

DEFAULT_WEIGHTS = {"visibility": .30, "completeness": .20, "sharpness": .15,
                   "pose": .15, "occlusion": .10, "resolution": .10}
DEFAULT_REGION_WEIGHTS = {"left_eye": .18, "right_eye": .18, "nose": .16,
                          "mouth": .16, "left_cheek": .08, "right_cheek": .08,
                          "forehead": .08, "chin": .08}
DEFAULT_RULES: dict[str, RuleState] = {
    "blur_below_min": "reject", "resolution_below_min": "reject",
    "face_outside_boundary": "reject", "detection_confidence_low": "reject",
    "severe_occlusion": "reject", "pose_exceeded": "reject",
    "multiple_essential_regions_missing": "reject",
    "essential_region_missing": "review", "landmarks_failed": "review",
    "visibility_unmeasurable": "review", "multiple_faces": "review",
    "visibility_below_min": "review", "incomplete_face": "review",
    "too_dark": "review", "too_bright": "review", "low_contrast": "review",
}
ALIASES = {
    "minimum_quality_score": "good_threshold",
    "maximum_yaw_degrees": "max_yaw",
    "maximum_pitch_degrees": "max_pitch",
    "maximum_roll_degrees": "max_roll",
    "minimum_visibility": "min_visibility",
    "minimum_face_width": "min_face_width",
    "minimum_face_height": "min_face_height",
    "minimum_brightness": "min_brightness",
    "minimum_contrast": "min_contrast",
    "maximum_occlusion": "max_occlusion",
}
INFERENCE_KEYS = ("detector", "min_det_confidence", "use_mediapipe", "use_crfiqa", "use_grayscale")


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _ramp(x, zero_at, full_at):
    if full_at <= zero_at:
        return 100.0 if x >= full_at else 0.0
    return 100.0 * _clamp((x - zero_at) / (full_at - zero_at))


@dataclass
class RegionObservation:
    """One facial region's evidence — NOT a claim about physical area."""
    confidence: float = 0.0
    in_frame: bool = True
    occluded: float = 0.0
    status: str = "missing"

    def score(self) -> float:
        if self.status == "missing" or not self.in_frame:
            return 0.0
        return _clamp(self.confidence) * (1.0 - _clamp(self.occluded))


@dataclass
class FaceMetrics:
    face_id: str
    image_path: str
    cluster_id: str | None = None
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    image_width: int = 0
    image_height: int = 0
    det_confidence: float = 1.0
    regions: dict[str, RegionObservation] = field(default_factory=dict)
    landmarks_detected: int = 0
    landmarks_failed: bool = False
    boundary_clip_fraction: float = 0.0
    sharpness_normalized: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    occlusion_fraction: float = 0.0
    brightness: float = 128.0
    contrast: float = 40.0
    n_faces_in_crop: int = 1
    crfiqa: float | None = None
    crfiqa_status: str = "unavailable"
    manual_override: Category | None = None
    inference_fingerprint: str = ""

    @property
    def face_width(self) -> float: return self.bbox[2] - self.bbox[0]
    @property
    def face_height(self) -> float: return self.bbox[3] - self.bbox[1]
    @property
    def face_size(self) -> float: return min(self.face_width, self.face_height)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FaceMetrics":
        d = dict(d)
        legacy_visible = d.pop("visible", None)
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"{d.get('face_id')}: unknown metric keys {sorted(unknown)}")
        if "bbox" in d:
            d["bbox"] = tuple(d["bbox"])
        if "regions" in d:
            d["regions"] = {k: v if isinstance(v, RegionObservation) else RegionObservation(**v)
                            for k, v in d["regions"].items()}
        elif legacy_visible is not None:
            d["regions"] = {k: RegionObservation(
                confidence=1.0 if v else 0.0,
                status="visible" if v else "missing")
                for k, v in legacy_visible.items()}
        return cls(**d)


@dataclass
class ScoringConfig:
    good_threshold: float = 80.0
    review_threshold: float = 62.0
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    region_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REGION_WEIGHTS))
    rules: dict[str, RuleState] = field(default_factory=lambda: dict(DEFAULT_RULES))
    min_visibility: float = 75.0
    max_boundary_clip: float = 0.02
    require_complete_face: bool = True
    minimum_sharpness: float = 80.0
    sharpness_full_score: float = 400.0
    max_yaw: float = 30.0
    max_pitch: float = 25.0
    max_roll: float = 25.0
    min_face_width: float = 80.0
    min_face_height: float = 80.0
    resolution_full_at: float = 160.0
    max_occlusion: float = 0.25
    min_brightness: float = 35.0
    max_brightness: float = 225.0
    min_contrast: float = 20.0
    crfiqa_weight: float = 0.0
    crfiqa_zero_at: float = 0.0
    crfiqa_full_at: float = 1.0
    detector: str = "scrfd"
    min_det_confidence: float = 0.60
    use_mediapipe: bool = True
    use_crfiqa: bool = False
    use_grayscale: bool = False

    def __post_init__(self):
        self.weights = {**DEFAULT_WEIGHTS, **self.weights}
        self.region_weights = {**DEFAULT_REGION_WEIGHTS, **self.region_weights}
        self.rules = {**DEFAULT_RULES, **self.rules}
        self._validate()

    def _validate(self):
        errs = []
        if not 0 <= self.review_threshold <= self.good_threshold <= 100:
            errs.append("need 0 <= review_threshold <= good_threshold <= 100")
        if set(self.weights) != set(DEFAULT_WEIGHTS):
            errs.append(f"weight keys must be {sorted(DEFAULT_WEIGHTS)}")
        if any(v < 0 for v in self.weights.values()):
            errs.append("weights must be >= 0")
        if sum(self.weights.values()) <= 0:
            errs.append("weights must sum to > 0")
        if not set(self.region_weights) <= set(REGIONS):
            errs.append(f"region_weights keys must be subset of {REGIONS}")
        if any(v < 0 for v in self.region_weights.values()):
            errs.append("region_weights must be >= 0")
        if sum(self.region_weights.values()) <= 0:
            errs.append("region_weights must sum to > 0")
        if set(self.rules) != set(DEFAULT_RULES):
            errs.append(f"rule keys must be {sorted(DEFAULT_RULES)}")
        if set(self.rules.values()) - {"off", "review", "reject"}:
            errs.append(f"invalid rule states: {set(self.rules.values())}")
        for name, lo, hi in [("min_visibility", 0, 100), ("good_threshold", 0, 100),
                             ("review_threshold", 0, 100), ("max_yaw", 1, 90),
                             ("max_pitch", 1, 90), ("max_roll", 1, 90),
                             ("minimum_sharpness", 0, 1e6), ("sharpness_full_score", 0, 1e6),
                             ("min_face_width", 0, 1e5), ("min_face_height", 0, 1e5),
                             ("resolution_full_at", 0, 1e5), ("max_occlusion", 0, 1),
                             ("max_boundary_clip", 0, 1), ("crfiqa_weight", 0, 1)]:
            v = getattr(self, name)
            if not (lo <= v <= hi):
                errs.append(f"{name}={v} outside [{lo}, {hi}]")
        if self.min_brightness >= self.max_brightness:
            errs.append("min_brightness must be < max_brightness")
        if self.use_crfiqa and self.crfiqa_weight == 0:
            errs.append("use_crfiqa=true requires crfiqa_weight > 0")
        if self.crfiqa_weight > 0 and self.crfiqa_full_at <= self.crfiqa_zero_at:
            errs.append("crfiqa_full_at must be > crfiqa_zero_at")
        if errs:
            raise ValueError("invalid ScoringConfig:\n  - " + "\n  - ".join(errs))
        self.weights = {k: v / sum(self.weights.values()) for k, v in self.weights.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for k, v in data.items():
            key = ALIASES.get(k, k)
            if key not in known:
                raise ValueError(f"unknown config key: {k!r}")
            kwargs[key] = v
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str) -> "ScoringConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def inference_fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(
            {k: getattr(self, k) for k in INFERENCE_KEYS}, sort_keys=True
        ).encode()).hexdigest()[:16]

    def scoring_fingerprint(self) -> str:
        d = {k: v for k, v in self.to_dict().items() if k not in INFERENCE_KEYS}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def score_visibility(m: FaceMetrics, c: ScoringConfig) -> float | None:
    if m.landmarks_failed and m.face_size < c.min_face_width:
        return None
    if not m.regions:
        return None
    total = sum(c.region_weights.values())
    got = sum(c.region_weights[r] * obs.score()
              for r, obs in m.regions.items() if r in c.region_weights)
    return 100.0 * got / total if total else None


def score_completeness(m: FaceMetrics, c: ScoringConfig) -> float:
    base = 100.0 * (1.0 - _clamp(m.boundary_clip_fraction))
    if c.require_complete_face:
        fh = m.regions.get("forehead")
        ch = m.regions.get("chin")
        missing_edges = sum(1 for o in (fh, ch) if o is None or o.status != "visible")
        base *= 1.0 - 0.25 * missing_edges
    return max(0.0, base)


def score_sharpness(m: FaceMetrics, c: ScoringConfig) -> float:
    return _ramp(m.sharpness_normalized, c.minimum_sharpness, c.sharpness_full_score)


def score_pose(m: FaceMetrics, c: ScoringConfig) -> float:
    def axis(a, lim): return 100.0 * (1.0 - _clamp(abs(a) / lim))
    return min(axis(m.yaw, c.max_yaw), axis(m.pitch, c.max_pitch), axis(m.roll, c.max_roll))


def score_occlusion(m: FaceMetrics, c: ScoringConfig) -> float:
    if c.max_occlusion <= 0:
        return 100.0 if m.occlusion_fraction <= 0 else 0.0
    return 100.0 * _clamp(1.0 - m.occlusion_fraction / c.max_occlusion)


def score_resolution(m: FaceMetrics, c: ScoringConfig) -> float:
    floor_ = min(c.min_face_width, c.min_face_height)
    return _ramp(m.face_size, floor_, c.resolution_full_at)


SUBSCORES = {"visibility": score_visibility, "completeness": score_completeness,
             "sharpness": score_sharpness, "pose": score_pose,
             "occlusion": score_occlusion, "resolution": score_resolution}


def _missing_essential(m: FaceMetrics) -> list[str]:
    return [r for r in ESSENTIAL
            if (obs := m.regions.get(r)) is None
            or obs.status in ("missing", "out_of_frame", "occluded")]


def _fired_rules(m: FaceMetrics, c: ScoringConfig) -> list[tuple[str, str]]:
    vis = score_visibility(m, c)
    missing = _missing_essential(m)
    checks: list[tuple[str, bool, str]] = [
        ("blur_below_min", m.sharpness_normalized < c.minimum_sharpness,
         f"sharpness {m.sharpness_normalized:.0f} < {c.minimum_sharpness:.0f}"),
        ("resolution_below_min",
         m.face_width < c.min_face_width or m.face_height < c.min_face_height,
         f"{m.face_width:.0f}x{m.face_height:.0f} < {c.min_face_width:.0f}x{c.min_face_height:.0f}"),
        ("face_outside_boundary", m.boundary_clip_fraction > c.max_boundary_clip,
         f"{m.boundary_clip_fraction:.1%} of face outside frame"),
        ("detection_confidence_low", m.det_confidence < c.min_det_confidence,
         f"det {m.det_confidence:.2f} < {c.min_det_confidence:.2f}"),
        ("severe_occlusion", m.occlusion_fraction > c.max_occlusion,
         f"occlusion {m.occlusion_fraction:.0%} > {c.max_occlusion:.0%}"),
        ("essential_region_missing", bool(missing), "missing: " + ", ".join(missing) if missing else ""),
        ("multiple_essential_regions_missing", len(missing) >= 2,
         f"{len(missing)} essential regions missing"),
        ("landmarks_failed", m.landmarks_failed or m.landmarks_detected == 0,
         "no usable landmarks"),
        ("visibility_unmeasurable", vis is None,
         "visibility cannot be determined from this crop"),
        ("multiple_faces", m.n_faces_in_crop > 1, f"{m.n_faces_in_crop} faces in crop"),
        ("visibility_below_min", vis is not None and vis < c.min_visibility,
         f"visibility {vis:.0f}% < {c.min_visibility:.0f}%" if vis is not None else ""),
        ("pose_exceeded", score_pose(m, c) <= 0,
         f"yaw {m.yaw:.0f} pitch {m.pitch:.0f} roll {m.roll:.0f}"),
        ("incomplete_face",
         m.boundary_clip_fraction > 0 and c.require_complete_face,
         "face touches/crosses image edge"),
        ("too_dark", m.brightness < c.min_brightness,
         f"brightness {m.brightness:.0f} < {c.min_brightness:.0f}"),
        ("too_bright", m.brightness > c.max_brightness,
         f"brightness {m.brightness:.0f} > {c.max_brightness:.0f}"),
        ("low_contrast", m.contrast < c.min_contrast,
         f"contrast {m.contrast:.0f} < {c.min_contrast:.0f}"),
    ]
    return [(n, d) for n, fired, d in checks
            if fired and c.rules.get(n, "off") != "off"]


@dataclass
class Assessment:
    face_id: str
    cluster_id: str | None
    image_path: str
    score: float
    subscores: dict[str, float | None]
    auto_category: Category
    reasons: list[dict[str, str]]
    manual_override: Category | None = None
    scoring_fingerprint: str = ""

    @property
    def final_category(self) -> Category:
        return self.manual_override or self.auto_category

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["final_category"] = self.final_category
        return d


def _weighted(subs, weights):
    tw = ts = 0.0
    for k, s in subs.items():
        if s is None:
            continue
        w = weights.get(k, 0.0)
        tw += w
        ts += w * s
    return ts / tw if tw > 0 else 0.0


def assess(m: FaceMetrics, c: ScoringConfig) -> Assessment:
    subs = {name: fn(m, c) for name, fn in SUBSCORES.items()}
    score = _weighted(subs, c.weights)
    if c.crfiqa_weight > 0 and m.crfiqa is not None and m.crfiqa_status == "available":
        alpha = _clamp(c.crfiqa_weight)
        learned = _ramp(m.crfiqa, c.crfiqa_zero_at, c.crfiqa_full_at)
        score = (1 - alpha) * score + alpha * learned
        subs["crfiqa"] = learned
    fired = _fired_rules(m, c)
    hard = [n for n, _ in fired if c.rules[n] == "reject"]
    gates = [n for n, _ in fired if c.rules[n] == "review"]
    if hard:
        cat = "BAD"
    elif score >= c.good_threshold and not gates:
        cat = "GOOD"
    elif score >= c.review_threshold:
        cat = "REVIEW"
    else:
        cat = "BAD"
    reasons = [{"rule": n, "severity": c.rules[n], "detail": d} for n, d in fired]
    if cat == "BAD" and not hard:
        reasons.append({"rule": "score_below_review_threshold", "severity": "reject",
                        "detail": f"{score:.1f} < {c.review_threshold:.1f}"})
    if c.crfiqa_weight > 0 and m.crfiqa_status != "available":
        reasons.append({"rule": "crfiqa_unavailable", "severity": "review",
                        "detail": f"crfiqa_status={m.crfiqa_status}; weight ignored"})
    return Assessment(face_id=m.face_id, cluster_id=m.cluster_id,
                      image_path=m.image_path, score=round(score, 2),
                      subscores={k: (None if v is None else round(v, 2))
                                 for k, v in subs.items()},
                      auto_category=cat, reasons=reasons,
                      manual_override=m.manual_override,
                      scoring_fingerprint=c.scoring_fingerprint())


def assess_all(metrics, c):
    return [assess(m, c) for m in metrics]


def summarize(assessments) -> dict[str, Any]:
    counts = {k: 0 for k in ("GOOD", "REVIEW", "BAD")}
    for a in assessments:
        counts[a.final_category] += 1
    n = len(assessments) or 1
    return {"total": len(assessments), **counts,
            "acceptance_pct": round(100 * counts["GOOD"] / n, 2),
            "rejection_pct": round(100 * counts["BAD"] / n, 2),
            "mean_score": round(sum(a.score for a in assessments) / n, 2)}


def compare(before, after) -> dict[str, Any]:
    moves = [{"face_id": f, "from": before[f].final_category, "to": after[f].final_category}
             for f in before.keys() & after.keys()
             if before[f].final_category != after[f].final_category]
    return {"changed": len(moves), "moves": moves,
            "before": summarize(list(before.values())),
            "after": summarize(list(after.values()))}


def best_faces(assessments, k=1, include_review=False):
    allowed = {"GOOD", "REVIEW"} if include_review else {"GOOD"}
    by: dict[str, list[Assessment]] = {}
    for a in assessments:
        if a.cluster_id and a.final_category in allowed:
            by.setdefault(a.cluster_id, []).append(a)
    return {cid: sorted(v, key=lambda a: a.score, reverse=True)[:k]
            for cid, v in by.items()}


def write_json_atomic(path: str, obj: Any) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise