"""
drawing_views.py
================
Turn one separated, vectorised view PDF into a View object.

    from drawing_views import load_view
    view = load_view("views/drawing_dc699b/drawing_dc699b_view3_left_view_2_1.pdf")
    print(type(view).__name__)        # OrthographicView
    print(view.summary())

load_view() returns the most specific subclass it can justify:
OrthographicView, SectionView, DetailView, AuxiliaryView, PictorialView
or, when the evidence is not enough, UnknownView.

Every value that may be missing on a drawing (label, scale, view type, direction)
is wrapped in Declared(value, source, confidence, evidence), so "read from the
drawing" and "guessed" are never confused.

Needs: pymupdf, numpy.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import pymupdf

T = TypeVar("T")
PT_PER_MM = 72.0 / 25.4
STANDARD_SCALES = [100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]  # drawing : real


# --------------------------------------------------------------------------- enums
class Source(Enum):
    FOUND = "found on drawing"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class Role(Enum):
    VISIBLE = "visible"
    HIDDEN = "hidden"
    CENTER = "center"
    HATCH = "hatch"
    DIMENSION = "dimension"
    OTHER = "other"


class ViewType(Enum):
    ORTHOGRAPHIC = "orthographic"
    SECTION = "section"
    DETAIL = "detail"
    AUXILIARY = "auxiliary"
    PICTORIAL = "pictorial"
    UNKNOWN = "unknown"


class SectionKind(Enum):
    FULL = "full"
    HALF = "half"
    OFFSET = "offset"
    REVOLVED = "revolved"


class ViewDirection(Enum):
    """(look direction, up vector) in model axes: X right, Y depth, Z up.
    FRONT = camera on -Y looking +Y. Change the convention here if your STEP differs."""
    FRONT = ((0, 1, 0), (0, 0, 1))
    REAR = ((0, -1, 0), (0, 0, 1))
    TOP = ((0, 0, -1), (0, 1, 0))
    BOTTOM = ((0, 0, 1), (0, -1, 0))
    RIGHT = ((-1, 0, 0), (0, 0, 1))
    LEFT = ((1, 0, 0), (0, 0, 1))

    @property
    def look(self):
        return np.array(self.value[0], float)

    @property
    def up(self):
        return np.array(self.value[1], float)

    @property
    def right(self):
        return np.cross(self.look, self.up)


def _t(v):
    return tuple(float(round(c, 6)) + 0.0 for c in v)


class UnresolvedViewError(Exception):
    """The view is understood but something needed for a projection is still missing."""


# --------------------------------------------------------------------------- small value types
@dataclass
class Declared(Generic[T]):
    value: Optional[T] = None
    source: Source = Source.UNKNOWN
    confidence: float = 0.0
    evidence: str = ""

    def __bool__(self):
        return self.value is not None

    @classmethod
    def found(cls, value, evidence="", confidence=0.95):
        return cls(value, Source.FOUND, confidence, evidence)

    @classmethod
    def inferred(cls, value, confidence, evidence=""):
        return cls(value, Source.INFERRED, confidence, evidence)

    @classmethod
    def unknown(cls):
        return cls()

    def __str__(self):
        if self.value is None:
            return "unknown"
        v = self.value.value if isinstance(self.value, Enum) else self.value
        return f"{v} [{self.source.name.lower()}, {self.confidence:.2f}]"


@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    role: Role

    @property
    def length(self):
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def angle_deg(self):  # y-up, 0..180
        return math.degrees(math.atan2(-(self.y1 - self.y0), self.x1 - self.x0)) % 180.0


@dataclass
class Curve:
    kind: str               # circle | arc | ellipse | spline
    cx: float
    cy: float
    r: float
    closed: bool
    rms: float              # fit error / r
    bbox: Tuple[float, float, float, float]
    role: Role
    pts: Optional[Any] = None      # sampled points (pt), for arcs/ellipses


@dataclass
class Annotation:
    kind: str               # diameter | radius | chamfer | linear | label | mark | note
    text: str
    value: Optional[float] = None
    count: int = 1
    qualifiers: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)


@dataclass
class GeometrySet:
    segments: List[Segment] = field(default_factory=list)
    curves: List[Curve] = field(default_factory=list)

    def by_role(self, role: Role):
        return ([s for s in self.segments if s.role == role], [c for c in self.curves if c.role == role])

    def outline_bbox(self, gap=6.0, min_rel=0.2):
        """Bounding box of the part outline: connected pieces of visible geometry, ignoring small
        detached pieces (cut-line stubs, stray marks) unless they sit inside the main piece."""
        items = [(s.x0 if s.x0 < s.x1 else s.x1, s.y0 if s.y0 < s.y1 else s.y1,
                  s.x0 if s.x0 > s.x1 else s.x1, s.y0 if s.y0 > s.y1 else s.y1)
                 for s in self.segments if s.role == Role.VISIBLE]
        items += [c.bbox for c in self.curves if c.role == Role.VISIBLE]
        if not items:
            return self.bbox()
        parent = list(range(len(items)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def close(a, b):
            return (max(a[0] - b[2], b[0] - a[2], 0) <= gap) and (max(a[1] - b[3], b[1] - a[3], 0) <= gap)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if close(items[i], items[j]):
                    parent[find(i)] = find(j)
        comps: Dict[int, list] = {}
        for i, b in enumerate(items):
            comps.setdefault(find(i), []).append(b)
        boxes = [(min(b[0] for b in c), min(b[1] for b in c), max(b[2] for b in c), max(b[3] for b in c))
                 for c in comps.values()]
        ext = lambda b: max(b[2] - b[0], b[3] - b[1])
        main = max(boxes, key=ext)
        keep = [b for b in boxes if ext(b) >= min_rel * ext(main)
                or (b[0] >= main[0] and b[1] >= main[1] and b[2] <= main[2] and b[3] <= main[3])]
        return (min(b[0] for b in keep), min(b[1] for b in keep), max(b[2] for b in keep), max(b[3] for b in keep))

    def bbox(self, roles=(Role.VISIBLE, Role.HIDDEN)):
        xs, ys = [], []
        for s in self.segments:
            if s.role in roles:
                xs += [s.x0, s.x1]
                ys += [s.y0, s.y1]
        for c in self.curves:
            if c.role in roles:
                xs += [c.bbox[0], c.bbox[2]]
                ys += [c.bbox[1], c.bbox[3]]
        return (min(xs), min(ys), max(xs), max(ys)) if xs else None


@dataclass
class PlaneSpec:
    normal: Tuple[float, float, float]
    point: Optional[Tuple[float, float, float]] = None   # None = through the part's centre/axis (to be resolved)


@dataclass
class ProjectionSpec:
    kind: str
    look: Tuple[float, float, float]
    up: Tuple[float, float, float]
    scale: Optional[float]
    cut_plane: Optional[PlaneSpec] = None
    notes: str = ""


@dataclass
class FeatureSet:
    """Same structure on the drawing side and the STEP side, in real-world mm."""
    outline_mm: Optional[Tuple[float, float]] = None
    circles_mm: List[float] = field(default_factory=list)      # diameters of drawn full circles
    arc_radii_mm: List[float] = field(default_factory=list)
    diameters_mm: List[float] = field(default_factory=list)    # from dimension text
    linear_mm: List[float] = field(default_factory=list)       # from dimension text
    n_ellipses: int = 0
    hatch_angle_deg: Optional[float] = None
    hatch_spacing_mm: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Discrepancy:
    kind: str
    expected: Any
    found: Any
    detail: str = ""


@dataclass
class ViewReport:
    view_id: str
    status: str = "ok"          # ok | mismatch | unresolved
    score: float = 1.0
    discrepancies: List[Discrepancy] = field(default_factory=list)
    checks: int = 0

    def add(self, kind, expected, found, detail=""):
        self.discrepancies.append(Discrepancy(kind, expected, found, detail))
        self.status = "mismatch"


class StepModel(ABC):
    """Adapter to implement with a CAD kernel (pythonocc, CadQuery...).
    project() must return the same FeatureSet the drawing side produces."""

    @abstractmethod
    def project(self, spec: ProjectionSpec) -> FeatureSet: ...


# --------------------------------------------------------------------------- parsing helpers
def _role_of(layer: str, dashes: str) -> Role:
    l = (layer or "").lower()
    if l.startswith("dim"):
        return Role.DIMENSION
    if l in ("center", "centre", "centerline", "centreline"):
        return Role.CENTER
    if l == "hatch":
        return Role.HATCH
    if l in ("hidden", "dashed"):
        return Role.HIDDEN
    if l in ("visible", "object", "outline", ""):
        d = (dashes or "").strip()
        if d and not d.startswith("[]") and d != "None":
            return Role.HIDDEN
        return Role.VISIBLE
    return Role.OTHER


def _bezier(p1, p2, p3, p4, n=24):
    out = []
    for i in range(n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3
        out.append((a * p1.x + b * p2.x + c * p3.x + d * p4.x, a * p1.y + b * p2.y + c * p3.y + d * p4.y))
    return out


def _fit_circle(pts: np.ndarray):
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0], sol[1]
    r = math.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-12))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)) / r)
    return cx, cy, r, rms


def _curve_from_run(run, role: Role) -> Optional[Curve]:
    pts = []
    for it in run:
        seg = _bezier(it[1], it[2], it[3], it[4])
        pts += seg if not pts else seg[1:]
    P = np.array(pts)
    closed = float(np.hypot(*(P[0] - P[-1]))) < 0.6
    cx, cy, r, rms = _fit_circle(P)
    if rms < 0.01:
        kind = "circle" if closed else "arc"
    else:
        kind = "ellipse" if closed else "spline"
    return Curve(kind, cx, cy, r, closed, rms, (P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()), role, P)


def read_geometry(page) -> GeometrySet:
    g = GeometrySet()
    for d in page.get_drawings():
        role = _role_of(d.get("layer"), str(d.get("dashes")))
        run = []

        def flush():
            nonlocal run
            if run:
                g.curves.append(_curve_from_run(run, role))
                run = []
        for it in d["items"]:
            k = it[0]
            if k == "c":
                run.append(it)
                continue
            flush()
            if k == "l":
                g.segments.append(Segment(it[1].x, it[1].y, it[2].x, it[2].y, role))
            elif k == "re":
                r = it[1]
                pts = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]
                g.segments += [Segment(*pts[i], *pts[i + 1], role) for i in range(4)]
            elif k == "qu":
                q = it[1]
                pts = [q.ul, q.ur, q.lr, q.ll, q.ul]
                g.segments += [Segment(pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y, role) for i in range(4)]
        flush()
    return g


DIA_RE = re.compile(r"^(?:(\d+)\s*[xX×]\s*)?(?:(CBORE|CSK|CSINK)\s+)?Ø\s*(\d+(?:\.\d+)?)(.*)$")
RAD_RE = re.compile(r"^(?:(\d+)\s*[xX×]\s*)?R\s*(\d+(?:\.\d+)?)(.*)$")
CHF_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*°(.*)$")
LIN_RE = re.compile(r"^(\d+(?:\.\d+)?)$")
SECTION_RE = re.compile(r"\bSECTION\s+([A-Z])\s*[-–—]\s*([A-Z])\b")
DETAIL_RE = re.compile(r"\bDETAIL\s+([A-Z])\b")
# a detail named by what it shows, e.g. "PAD FACE D - 10:1 (4x)": a single letter, a dash, then its own scale
NAMED_DETAIL_RE = re.compile(r"\b([A-Z])\s*[-–—]\s*\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?")
PICT_RE = re.compile(r"\b(ISOMETRIC|DIMETRIC|TRIMETRIC|PERSPECTIVE|AXONOMETRIC|OBLIQUE)\b")
ORTH_RE = re.compile(r"\b(TOP|PLAN|FRONT|ELEVATION|BACK|REAR|LEFT|RIGHT|BOTTOM)\b(?:\s+SIDE)?\s+VIEW\b")
SIDE_RE = re.compile(r"\bSIDE\s+VIEW\b")
BARE_RE = re.compile(r"^(TOP|PLAN|FRONT|ELEVATION|BACK|REAR|LEFT|RIGHT|BOTTOM)(?:\s+\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?)?$")
ARROW_RE = re.compile(r"^VIEW\s+([A-Z])\b")
SCALE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)")
DIR_WORDS = {"TOP": "TOP", "PLAN": "TOP", "FRONT": "FRONT", "ELEVATION": "FRONT", "BACK": "REAR",
             "REAR": "REAR", "LEFT": "LEFT", "RIGHT": "RIGHT", "BOTTOM": "BOTTOM"}


def read_spans(page):
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                t = s["text"].strip()
                if t:
                    spans.append({"text": t.replace("⌀", "Ø").replace("∅", "Ø"), "bbox": tuple(s["bbox"]),
                                  "size": s["size"]})
    return spans


def parse_annotation(sp) -> Annotation:
    t, bb = sp["text"], sp["bbox"]
    up = " ".join(t.upper().split())
    if (SECTION_RE.search(up) or DETAIL_RE.search(up) or NAMED_DETAIL_RE.search(up) or PICT_RE.search(up)
            or ORTH_RE.search(up)
            or SIDE_RE.search(up) or ARROW_RE.search(up) or BARE_RE.match(up)):
        return Annotation("label", t, bbox=bb)
    m = DIA_RE.match(t)
    if m:
        rest = m.group(4).strip()
        extra = {"explicit_count": m.group(1) is not None}
        if m.group(2):
            extra["cbore"] = m.group(2)
        dm = re.search(r"[xX×]\s*(\d+(?:\.\d+)?)\s*DEEP", rest.upper())
        if dm:
            extra["depth"] = float(dm.group(1))
        if "THRU" in rest.upper():
            extra["thru"] = True
        return Annotation("diameter", t, float(m.group(3)), int(m.group(1) or 1), rest, extra, bb)
    m = RAD_RE.match(t)
    if m:
        return Annotation("radius", t, float(m.group(2)), int(m.group(1) or 1), m.group(3).strip(),
                          {"explicit_count": m.group(1) is not None}, bb)
    m = CHF_RE.match(t)
    if m:
        return Annotation("chamfer", t, float(m.group(1)), 1, m.group(3).strip(),
                          {"angle_deg": float(m.group(2))}, bb)
    if LIN_RE.match(t):
        return Annotation("linear", t, float(t), bbox=bb)
    if re.fullmatch(r"[A-Z]", t):
        return Annotation("mark", t, bbox=bb)
    return Annotation("note", t, bbox=bb)


def parse_label(spans):
    """First span that names a view. Returns dict or None."""
    found = {}
    for sp in spans:
        up = " ".join(sp["text"].upper().split())
        d = None
        if (m := SECTION_RE.search(up)):
            d = {"type": ViewType.SECTION, "id": m.group(1)}
        elif (m := DETAIL_RE.search(up)) or (m := NAMED_DETAIL_RE.search(up)):
            d = {"type": ViewType.DETAIL, "id": m.group(1)}
        elif PICT_RE.search(up):
            d = {"type": ViewType.PICTORIAL, "kind": PICT_RE.search(up).group(1).lower()}
        elif (m := ORTH_RE.search(up)):
            d = {"type": ViewType.ORTHOGRAPHIC, "direction": ViewDirection[DIR_WORDS[m.group(1)]]}
        elif (m := BARE_RE.match(up)):
            d = {"type": ViewType.ORTHOGRAPHIC, "direction": ViewDirection[DIR_WORDS[m.group(1)]]}
        elif SIDE_RE.search(up):
            d = {"type": ViewType.ORTHOGRAPHIC, "direction": None}
        elif (m := ARROW_RE.search(up)):
            d = {"type": ViewType.AUXILIARY, "id": m.group(1)}
        if d:
            d["text"] = sp["text"]
            sm = SCALE_RE.search(up)
            if sm and float(sm.group(2)) != 0:
                d["scale"] = float(sm.group(1)) / float(sm.group(2))
            return d
    return None


def infer_scale(circles, outline_wh, annots, extra_scales=()):
    """Vote for a standard scale by matching drawn sizes against dimension values."""
    dia = {a.value for a in annots if a.kind == "diameter"}
    lin = {a.value for a in annots if a.kind == "linear"}
    chf = {a.value for a in annots if a.kind == "chamfer"}
    cand_dia = set(dia) | {v + 2 * c for v in dia for c in chf}
    votes: Dict[float, float] = {}

    def vote(measure_pt, value, weight):
        if value <= 0 or measure_pt <= 0:
            return
        s = measure_pt / (value * PT_PER_MM)
        best = min(list(STANDARD_SCALES) + list(extra_scales), key=lambda x: abs(math.log(s / x)))
        if abs(s / best - 1) < 0.02:
            votes[best] = votes.get(best, 0) + weight
    for d_pt in circles:
        for v in cand_dia:
            vote(d_pt, v, 2)
    for m in outline_wh:
        for v in lin | dia:
            vote(m, v, 1)
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    top, w = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    if w < 2 or w < 1.5 * second:
        return None
    conf = min(0.9, 0.5 + 0.1 * w) * (1 - 0.5 * second / w)
    return top, round(conf, 2), f"{w:g} matching size/dimension pairs agree on {_fmt_scale(top)}"


def _fmt_scale(s):
    return f"{s:g}:1" if s >= 1 else f"1:{1 / s:g}"


def hatch_stats(g: GeometrySet):
    segs = [s for s in g.segments if s.role == Role.HATCH and s.length > 1]
    if len(segs) < 3:
        return None
    ang = float(np.median([s.angle_deg for s in segs]))
    n = np.array([-math.sin(math.radians(ang)), -math.cos(math.radians(ang))])  # normal in y-down frame
    offs = sorted(float(np.dot(n, [(s.x0 + s.x1) / 2, (s.y0 + s.y1) / 2])) for s in segs)
    diffs = [b - a for a, b in zip(offs, offs[1:]) if b - a > 0.2]
    return round(ang, 1), (float(np.median(diffs)) if diffs else None)


# --------------------------------------------------------------------------- the classes
class View(ABC):
    """Mother class. One separated view of a drawing."""
    view_type_value: ViewType = ViewType.UNKNOWN

    def __init__(self, view_id, source_path, page_size, geometry, annotations, label_text,
                 label, scale, view_type, section_marks, metrics, warnings, sheet_info=None):
        self.view_id = view_id
        self.source_path = source_path
        self.page_size = page_size
        self.geometry = geometry
        self.annotations = annotations
        self.label_text = label_text
        self.label: Declared[str] = label
        self.scale: Declared[float] = scale
        self.view_type: Declared[ViewType] = view_type
        self.section_marks: Dict[str, List[Tuple[float, float]]] = section_marks
        self.metrics = metrics
        self.warnings: List[str] = warnings
        self.sheet_info = sheet_info or {}
        self.outline_pt = geometry.outline_bbox()

    # ---- what every view can do
    def is_measurable(self) -> bool:
        return True

    @abstractmethod
    def projection_spec(self) -> ProjectionSpec:
        """How the STEP model must be projected to reproduce this view."""

    def extract_features(self) -> FeatureSet:
        s = self.scale.value
        mm = (lambda v: round(float(v) / PT_PER_MM / s, 3)) if s else None
        fs = FeatureSet(metrics=dict(self.metrics))
        if self.outline_pt:
            w = self.outline_pt[2] - self.outline_pt[0]
            h = self.outline_pt[3] - self.outline_pt[1]
            fs.metrics["outline_pt_w"], fs.metrics["outline_pt_h"] = round(w, 2), round(h, 2)
            if mm:
                fs.outline_mm = (mm(w), mm(h))
        vis = [c for c in self.geometry.curves if c.role == Role.VISIBLE]
        if mm:
            fs.circles_mm = sorted(mm(2 * c.r) for c in vis if c.kind == "circle")
            fs.arc_radii_mm = sorted(mm(c.r) for c in vis if c.kind == "arc")
        fs.n_ellipses = sum(1 for c in vis if c.kind == "ellipse")
        fs.diameters_mm = sorted(a.value for a in self.annotations if a.kind == "diameter")
        fs.linear_mm = sorted(a.value for a in self.annotations if a.kind == "linear")
        hs = hatch_stats(self.geometry)
        if hs:
            fs.hatch_angle_deg = hs[0]
            if hs[1] and mm:
                fs.hatch_spacing_mm = mm(hs[1])
        return fs

    def verify(self, step: StepModel, rel_tol=0.01, abs_tol=0.05) -> ViewReport:
        """Template method: projection_spec -> StepModel.project -> compare."""
        rep = ViewReport(self.view_id)
        if not self.scale:
            rep.status = "unresolved"
            rep.discrepancies.append(Discrepancy("scale", None, None, "scale unknown, cannot convert drawing size to mm"))
            return rep
        try:
            spec = self.projection_spec()
        except UnresolvedViewError as e:
            rep.status = "unresolved"
            rep.discrepancies.append(Discrepancy("unresolved", None, None, str(e)))
            return rep
        exp, got = step.project(spec), self.extract_features()
        tol = lambda v: max(abs_tol, rel_tol * abs(v))
        sz_tol = 0.03 if not self.is_measurable() else rel_tol
        if exp.outline_mm and got.outline_mm:
            rep.checks += 1
            for name, e, g in zip(("outline width", "outline height"), exp.outline_mm, got.outline_mm):
                if abs(e - g) > max(abs_tol, sz_tol * abs(e)):
                    rep.add(name, round(e, 3), round(g, 3))
        if self.is_measurable():
            pool = list(exp.circles_mm)
            for d in got.circles_mm:
                rep.checks += 1
                hit = next((e for e in pool if abs(e - d) <= tol(e)), None)
                if hit is None:
                    rep.add("drawn circle", None, d, "no circle of this diameter in the model projection")
                else:
                    pool.remove(hit)
            for e in pool:
                rep.add("missing circle", e, None, "model projection has a circle that is not drawn")
            allowed = list(exp.circles_mm) + list(exp.diameters_mm)
            for v in got.diameters_mm:
                rep.checks += 1
                if not any(abs(v - e) <= tol(e) for e in allowed):
                    rep.add("diameter dimension", None, v, "no matching diameter in the model projection")
        n_bad = len(rep.discrepancies)
        rep.score = round(max(0.0, 1 - n_bad / max(rep.checks, 1)), 3)
        return rep

    # ---- inspection
    def summary(self) -> str:
        f = self.extract_features()
        lines = [f"{type(self).__name__}  ({self.view_id})",
                 f"  type      : {self.view_type}",
                 f"  label     : {self.label_text or '-'}  -> {self.label}",
                 f"  scale     : {_fmt_scale(self.scale.value) + ' [' + self.scale.source.name.lower() + ', ' + format(self.scale.confidence, '.2f') + ']' if self.scale else 'unknown'}",
                 f"  outline   : {f.outline_mm} mm real-size" if f.outline_mm else "  outline   : scale unknown",
                 f"  circles   : {f.circles_mm}  arcs R: {f.arc_radii_mm}  ellipses: {f.n_ellipses}",
                 f"  dims      : diameters {f.diameters_mm}, linear {f.linear_mm}",
                 f"  annots    : {len(self.annotations)}"]
        if f.hatch_angle_deg is not None:
            lines.append(f"  hatch     : {f.hatch_angle_deg} deg, spacing {f.hatch_spacing_mm} mm")
        if self.section_marks:
            lines.append(f"  cut marks : {sorted(self.section_marks)}")
        lines += [f"  extra     : {self._extra()}"] if self._extra() else []
        lines += [f"  WARNING   : {w}" for w in self.warnings]
        return "\n".join(lines)

    def _extra(self) -> str:
        return ""

    def __repr__(self):
        return f"<{type(self).__name__} {self.view_id} type={self.view_type.value}>"


class OrthographicView(View):
    view_type_value = ViewType.ORTHOGRAPHIC

    def __init__(self, *a, direction: Declared = None, **k):
        super().__init__(*a, **k)
        self.direction: Declared[ViewDirection] = direction or Declared.unknown()
        self.convention: Declared[str] = Declared.unknown()      # first / third angle (sheet-level)
        self.neighbours: Dict[str, str] = {}                     # filled by SheetDrawing

    def projection_spec(self):
        if not self.direction:
            raise UnresolvedViewError(f"{self.view_id}: orthographic view but direction unknown "
                                      f"(no label; fit against the STEP projections or use the sheet layout)")
        d = self.direction.value
        return ProjectionSpec("orthographic", _t(d.look), _t(d.up), self.scale.value)

    def _extra(self):
        return f"direction {self.direction}"


class SectionView(View):
    view_type_value = ViewType.SECTION

    def __init__(self, *a, section_id: Optional[str] = None, **k):
        super().__init__(*a, **k)
        self.section_id = section_id
        self.section_kind: Declared[SectionKind] = Declared.inferred(SectionKind.FULL, 0.5, "default assumption")
        self.parent_view: Optional[View] = None
        self.cutting_plane: Optional[PlaneSpec] = None
        self.look: Optional[np.ndarray] = None
        self.cut_line_3d: Optional[np.ndarray] = None
        self.hatch = hatch_stats(self.geometry)         # (angle, spacing_pt) or None

    def projection_spec(self):
        if self.cutting_plane is None or self.look is None or self.cut_line_3d is None:
            raise UnresolvedViewError(f"{self.view_id}: section {self.section_id}-{self.section_id} needs its parent "
                                      f"view (the one carrying the cut line); run SheetDrawing.link_views()")
        d = self.parent_view.direction.value
        L3 = self.cut_line_3d
        # a section drawn above/below its parent has the cut line horizontal and "up" = towards the parent's viewer;
        # one drawn beside it has the cut line vertical and "up" along the cut line
        if abs(float(np.dot(L3, d.right))) >= abs(float(np.dot(L3, d.up))):
            up = -d.look
        else:
            up = L3 * (1.0 if np.dot(L3, d.up) >= 0 else -1.0)
        return ProjectionSpec("section", _t(self.look), _t(up), self.scale.value, self.cutting_plane,
                              "cut position along the normal not resolved (assumed through the part centre)")

    def _extra(self):
        p = self.parent_view.view_id if self.parent_view else "not linked"
        return f"section {self.section_id}-{self.section_id}, kind {self.section_kind}, parent: {p}"


class DetailView(View):
    view_type_value = ViewType.DETAIL

    def __init__(self, *a, detail_id: Optional[str] = None, **k):
        super().__init__(*a, **k)
        self.detail_id = detail_id
        self.parent_view: Optional[View] = None
        self.region = None                       # circled area in the parent, resolved by link_views()

    @property
    def magnification(self):
        return self.scale.value

    def projection_spec(self):
        if self.parent_view is None or not isinstance(self.parent_view, OrthographicView):
            raise UnresolvedViewError(f"{self.view_id}: detail {self.detail_id} needs a linked parent view")
        spec = self.parent_view.projection_spec()
        return ProjectionSpec("detail", spec.look, spec.up, self.scale.value, None,
                              f"crop of parent view, region {self.region}")

    def _extra(self):
        return f"detail {self.detail_id}, magnification {self.scale}"


class AuxiliaryView(View):
    view_type_value = ViewType.AUXILIARY

    def __init__(self, *a, arrow_id: Optional[str] = None, **k):
        super().__init__(*a, **k)
        self.arrow_id = arrow_id
        self.parent_view: Optional[View] = None
        self.face_normal: Optional[Tuple[float, float, float]] = None

    def projection_spec(self):
        raise UnresolvedViewError(f"{self.view_id}: auxiliary view {self.arrow_id}: viewing direction comes from "
                                  f"an arrow/inclined face in another view (not resolved yet)")


class PictorialView(View):
    view_type_value = ViewType.PICTORIAL

    def __init__(self, *a, projection_kind: str = "unknown", **k):
        super().__init__(*a, **k)
        self.projection_kind: Declared[str] = (Declared.found(projection_kind, "label")
                                               if projection_kind != "unknown" else Declared.unknown())
        self.camera_angles: Declared[Tuple[float, float]] = Declared.unknown()   # azimuth, elevation (deg)

    def is_measurable(self):
        return False

    def projection_spec(self):
        if self.projection_kind.value != "isometric":
            raise UnresolvedViewError(f"{self.view_id}: pictorial view of kind "
                                      f"{self.projection_kind.value!r}, camera unknown")
        az, el = (45.0, 35.264)
        self.camera_angles = Declared.inferred((az, el), 0.4, "standard isometric; real camera must be fitted")
        cam = np.array([math.cos(math.radians(el)) * math.sin(math.radians(az)),
                        -math.cos(math.radians(el)) * math.cos(math.radians(az)), math.sin(math.radians(el))])
        return ProjectionSpec("pictorial", _t(-cam), (0.0, 0.0, 1.0), self.scale.value, None,
                              "camera assumed (az 45, el 35.264); fit against the drawn silhouette")

    def _extra(self):
        return f"projection {self.projection_kind}, measurable: no"


class UnknownView(View):
    view_type_value = ViewType.UNKNOWN

    def __init__(self, *a, candidates: Optional[List[Tuple[ViewType, float]]] = None, **k):
        super().__init__(*a, **k)
        self.candidates = candidates or []

    def projection_spec(self):
        raise UnresolvedViewError(f"{self.view_id}: view type unresolved; candidates {self.candidates}")

    def _extra(self):
        return f"candidates {[(t.value, s) for t, s in self.candidates]}"


# --------------------------------------------------------------------------- classification
def classify(label, g: GeometrySet, hatch, metrics):
    """-> (ViewType, confidence, evidence, source)"""
    if label:
        return label["type"], 0.95, f"label '{label['text']}'", Source.FOUND
    evidence = []
    if hatch:
        return ViewType.SECTION, 0.6, "unlabelled but has hatching", Source.INFERRED
    if metrics["tessellation"] > 0.5:
        return ViewType.PICTORIAL, 0.7, (f"unlabelled; {metrics['tessellation']:.0%} of visible segments are "
                                          f"short and off-axis (tessellated curves)"), Source.INFERRED
    n_circ = sum(1 for c in g.curves if c.role == Role.VISIBLE and c.kind in ("circle", "arc"))
    if metrics["axis_share"] > 0.4 or n_circ:
        return ViewType.ORTHOGRAPHIC, 0.6, (f"unlabelled; axis-aligned lines ({metrics['axis_share']:.0%}) and "
                                             f"{n_circ} true circles/arcs, no hatching"), Source.INFERRED
    return ViewType.UNKNOWN, 0.0, "not enough evidence", Source.UNKNOWN


def geometry_metrics(g: GeometrySet):
    vis = [s for s in g.segments if s.role == Role.VISIBLE]
    tot = sum(s.length for s in vis) or 1.0

    def off_axis(s):
        a = s.angle_deg % 90
        return min(a, 90 - a) > 2.0
    axis_len = sum(s.length for s in vis if not off_axis(s))
    short_off = sum(1 for s in vis if s.length < 4 and off_axis(s))
    return {"n_visible_segments": len(vis), "axis_share": axis_len / tot,
            "tessellation": short_off / len(vis) if vis else 0.0}


# --------------------------------------------------------------------------- public API
def load_view(path: str) -> View:
    """Path to one separated view PDF -> the matching View subclass."""
    doc = pymupdf.open(path)
    page = doc[0]
    g = read_geometry(page)
    spans = read_spans(page)
    annots = [parse_annotation(s) for s in spans]
    label = parse_label(spans)
    metrics = geometry_metrics(g)
    hatch = hatch_stats(g)
    warnings: List[str] = []

    # letters used as cut-line markers (same letter at least twice)
    marks: Dict[str, List[Tuple[float, float]]] = {}
    for a in annots:
        if a.kind == "mark":
            b = a.bbox
            marks.setdefault(a.text, []).append(((b[0] + b[2]) / 2, -(b[1] + b[3]) / 2))
    marks = {k: v for k, v in marks.items() if len(v) >= 2}

    vtype, conf, ev, src = classify(label, g, hatch, metrics)
    vt = Declared(vtype, src, conf, ev)

    # scale: label first, geometry as a cross-check / fallback
    outline = g.outline_bbox()
    wh = ((outline[2] - outline[0]), (outline[3] - outline[1])) if outline else ()
    circ = [2 * c.r for c in g.curves if c.role == Role.VISIBLE and c.kind == "circle"]
    side = _find_sidecar(path)
    sheet_scale = side.get("sheet_scale") if side else None
    inf = infer_scale(circ, wh, annots, [sheet_scale] if sheet_scale else ())
    if label and "scale" in label:
        scale = Declared.found(label["scale"], f"label '{label['text']}'")
        if inf and abs(inf[0] / label["scale"] - 1) > 0.03 and inf[1] >= 0.6:
            warnings.append(f"label says {_fmt_scale(label['scale'])} but drawn sizes match {_fmt_scale(inf[0])}")
    elif inf:
        scale = Declared.inferred(inf[0], inf[1], inf[2])
    elif sheet_scale:
        scale = Declared.inferred(sheet_scale, 0.4, "title-block scale of the sheet (not confirmed by this view)")
    else:
        scale = Declared.unknown()

    lab = Declared.found(label["text"], "text on the view") if label else Declared.unknown()
    common = dict(view_id=os.path.splitext(os.path.basename(path))[0], source_path=os.path.abspath(path),
                  page_size=(page.rect.width, page.rect.height), geometry=g, annotations=annots,
                  label_text=label["text"] if label else None, label=lab, scale=scale, view_type=vt,
                  section_marks=marks, metrics=metrics, warnings=warnings)

    if vtype == ViewType.ORTHOGRAPHIC:
        if label and label.get("direction"):
            d = Declared.found(label["direction"], f"label '{label['text']}'")
        else:
            d = Declared.unknown()
        view = OrthographicView(**common, direction=d)
    elif vtype == ViewType.SECTION:
        view = SectionView(**common, section_id=label.get("id") if label else None)
    elif vtype == ViewType.DETAIL:
        view = DetailView(**common, detail_id=label.get("id"))
    elif vtype == ViewType.AUXILIARY:
        view = AuxiliaryView(**common, arrow_id=label.get("id"))
    elif vtype == ViewType.PICTORIAL:
        view = PictorialView(**common, projection_kind=label.get("kind", "unknown") if label else "unknown")
    else:
        view = UnknownView(**common, candidates=[])

    if side:
        view.sheet_info = side
    if isinstance(view, OrthographicView) and not view.direction:
        view.warnings.append("direction not determinable from this view alone (unlabelled)")
    return view


def _find_sidecar(path):
    name = os.path.basename(path)
    for js in glob.glob(os.path.join(os.path.dirname(path), "*_views.json")):
        try:
            data = json.load(open(js))
        except Exception:
            continue
        for v in data.get("views", []):
            if v.get("file") == name:
                sc = None
                txt = data.get("sheet_info_texts", [])
                for i, t in enumerate(txt[:-1]):
                    if t.strip().upper() == "SCALE" and (m := SCALE_RE.search(txt[i + 1])) and float(m.group(2)):
                        sc = float(m.group(1)) / float(m.group(2))
                return {"sheet_scale": sc, "source_drawing": data.get("source"), "outline_bbox_sheet_pt": v.get("outline_bbox_sheet_pt"),
                        "extent_bbox_sheet_pt": v.get("extent_bbox_sheet_pt"),
                        "offset_to_view_pt": v.get("offset_to_view_pt")}
    return None


class SheetDrawing:
    """Owns the views of one drawing and links the ones that refer to each other."""

    def __init__(self, views: List[View], projection_convention: Declared = None):
        self.views = views
        self.projection_convention = projection_convention or Declared.unknown()

    @classmethod
    def from_folder(cls, folder: str):
        return cls([load_view(p) for p in sorted(glob.glob(os.path.join(folder, "*.pdf")))])

    def link_views(self):
        for v in self.views:
            if isinstance(v, SectionView) and v.section_id:
                parents = [p for p in self.views if p is not v and v.section_id in p.section_marks]
                if not parents:
                    v.warnings.append(f"no view carries the cut marks '{v.section_id}'")
                    continue
                p = parents[0]
                v.parent_view = p
                if isinstance(p, OrthographicView) and p.direction:
                    (x0, y0), (x1, y1) = p.section_marks[v.section_id][:2]
                    L = np.array([x1 - x0, y1 - y0], float)
                    L /= np.linalg.norm(L)
                    d = p.direction.value
                    L3 = d.right * L[0] + d.up * L[1]
                    n = np.cross(L3, d.look)
                    n /= np.linalg.norm(n)
                    v.cutting_plane = PlaneSpec(_t(n))
                    v.cut_line_3d = L3
                    v.look = -n          # keep the -normal half; arrows on the cut line decide the sign
                    v.warnings.append("viewing sign (+/-) of the section taken from convention, not from the arrows")
                else:
                    v.warnings.append(f"parent '{p.view_id}' direction unknown; cutting plane unresolved")
            elif isinstance(v, DetailView) and v.detail_id:
                for p in self.views:
                    if p is not v and any(a.kind == "mark" and a.text == v.detail_id for a in p.annotations):
                        v.parent_view = p
                        break
        return self

    def verify(self, step: StepModel) -> Dict[str, ViewReport]:
        return {v.view_id: v.verify(step) for v in self.views}


def view_to_dict(view: View) -> dict:
    """Everything extracted from one view, as plain JSON-serialisable data."""
    f = view.extract_features()
    dec = lambda d: None if not d else {"value": (d.value.value if isinstance(d.value, Enum) else d.value),
                                        "source": d.source.name.lower(), "confidence": d.confidence,
                                        "evidence": d.evidence}
    out = {
        "view_id": view.view_id, "class": type(view).__name__, "view_type": dec(view.view_type),
        "label": view.label_text, "scale": dec(view.scale),
        "outline_mm": list(f.outline_mm) if f.outline_mm else None,
        "circles_diameter_mm": f.circles_mm, "arc_radii_mm": f.arc_radii_mm, "ellipses": f.n_ellipses,
        "hatch": {"angle_deg": f.hatch_angle_deg, "spacing_mm": f.hatch_spacing_mm} if f.hatch_angle_deg is not None else None,
        "cut_marks": sorted(view.section_marks),
        "annotations": [{"kind": a.kind, "text": a.text, "value": a.value, "count": a.count,
                         "qualifiers": a.qualifiers, **a.extra} for a in view.annotations],
        "warnings": view.warnings, "sheet": view.sheet_info or None,
    }
    if hasattr(view, "direction"):
        out["direction"] = dec(view.direction)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv
    for p in args:
        v = load_view(p)
        print(json.dumps(view_to_dict(v), indent=1, ensure_ascii=False) if as_json else v.summary())
        print()