"""JSON contracts between the agents: one summary per view, one feature tree per part.

Agents name a model in their YAML (`output_schema: TopSummary`); the harness sends
its JSON Schema to the model and validates the reply with it. Validation errors go
back to the model so it can fix its answer.

Pass `context={"views": [...]}` to model_validate to also check that evidence
points at views that exist.
"""
from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

# --------------------------------------------------------------------------- operations
# Fixed list, from the operations that appear on mechanical detail drawings. Anything
# else (gear teeth, weldments, surfaces, engraving) is not an operation: it goes in
# FeatureTree.unsupported, so the part is flagged instead of guessed. Sheet metal is
# built from a flat blank (base_flange) folded along its bend lines (src/sheet_metal.py).
Op = Literal[
    # core shape
    "extrude_boss", "revolve_boss", "extrude_cut", "revolve_cut", "boolean",
    "sweep", "loft", "shell", "rib", "draft", "thicken",
    # holes and edges
    "hole", "counterbore_hole", "countersink_hole", "tapped_hole", "external_thread",
    "slot", "keyway", "chamfer", "fillet", "knurl",
    # sheet metal
    "base_flange", "bend", "edge_flange",
]

# Parameters each operation needs. A tuple means "at least one of these".
REQUIRED_PARAMS: dict[str, list[str | tuple[str, ...]]] = {
    "extrude_boss": ["profile", "height"],
    "extrude_cut": ["profile", ("depth", "thru")],
    "revolve_boss": ["profile"],
    "revolve_cut": ["profile"],
    "boolean": ["operation"],
    "shell": ["wall"],
    "rib": ["thickness"],
    "draft": ["angle"],
    "thicken": ["thickness"],
    "hole": ["d", ("depth", "thru")],
    "counterbore_hole": ["d", "cbore_d", "cbore_depth", ("depth", "thru")],
    "countersink_hole": ["d", "csk_d", "csk_angle", ("depth", "thru")],
    "tapped_hole": ["thread", ("depth", "thru")],
    "external_thread": ["thread", "length"],
    "slot": ["width", "length", ("depth", "thru")],
    "keyway": ["width", "depth", "length"],
    "chamfer": ["size"],
    "fillet": ["r"],
    "base_flange": ["profile", "thickness"],   # profile: the flat blank [[x, y], ...]
    "bend": ["line", "angle", "direction"],     # line: [[x1, y1], [x2, y2]] across the flat blank
    "edge_flange": ["length", "angle"],
}

CHECK_RE = re.compile(r"^(bbox|volume|projection:[a-z0-9_,]+|cylinder:[a-z0-9_.=,]+|count:[a-z0-9_.=,]+)$")
SHEET_SOURCES = {"title_block", "notes"}

ParamValue = Union[float, int, bool, str, list[float], list[list[float]]]
Axis = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def check_views(values: list[str], info: ValidationInfo) -> None:
    known = (info.context or {}).get("views")
    if known is None:
        return
    allowed = set(known) | SHEET_SOURCES
    bad = sorted({v for v in values if v not in allowed})
    if bad:
        raise ValueError(f"unknown view(s) {bad}; use one of {sorted(allowed)}")


# --------------------------------------------------------------------------- view summaries
class Dimension(Strict):
    text: str = Field(description="Exactly as written on the drawing, e.g. '4x Ø6.6 THRU'")
    value: float
    kind: Literal["linear", "diameter", "radius", "angle", "depth", "position", "chamfer", "thread"]
    direction: Literal["X", "Y", "Z", "none"] = "none"
    refers_to: str = Field(description="Which edges or feature it measures, in words")
    tolerance: str | None = None


class Hypothesis(Strict):
    """A feature this view suggests, with what supports it."""
    name: str = Field(description="Short snake_case name, e.g. corner_mounting_holes")
    op: Op
    description: str
    params: dict[str, ParamValue] = Field(default_factory=dict, description="Values this view gives, in mm/deg")
    count: int = 1
    evidence: list[str] = Field(default_factory=list, description="Dimension texts and callouts supporting it")
    confidence: float = Field(ge=0, le=1)


class ViewSummary(Strict):
    view: str
    label: str
    scale: str | None = None
    horizontal: Literal["X", "Y", "Z", "unknown"]
    vertical: Literal["X", "Y", "Z", "unknown"]
    outline: str = Field(description="Overall silhouette in words, with its overall sizes")
    dimensions: list[Dimension]
    hypotheses: list[Hypothesis]
    cannot_tell: list[str] = Field(description="What this view does not define, e.g. depths, hidden sides")
    notes: list[str] = Field(default_factory=list)


class FrontSummary(ViewSummary):
    view: Literal["front"]


class SideSummary(ViewSummary):
    view: Literal["side"]
    side: Literal["left", "right", "unknown"]


class Pattern(Strict):
    type: Literal["single", "rectangular", "linear", "circular", "mirror"]
    count: int = 1
    nx: int | None = None
    ny: int | None = None
    dx: float | None = None
    dy: float | None = None
    pitch_diameter: float | None = None
    start_angle_deg: float | None = None
    centre: str = Field(description="What the pattern is centred on: part centre, a boss, an offset point")


class TopSummary(ViewSummary):
    view: Literal["top"]
    patterns: list[Pattern] = Field(default_factory=list)


class SectionSummary(ViewSummary):
    view: Literal["section"]
    cutting_plane: str = Field(description="Name, plane (XZ/YZ/XY), what it passes through")
    revolved: bool = Field(description="True if the section is a half-profile of a turned part")
    solid_regions: list[str] = Field(description="Hatched regions: base, wall, flange... with sizes")


class DetailSummary(ViewSummary):
    view: Literal["detail"]
    parent_view: str
    region: str


class IsometricSummary(Strict):
    view: Literal["isometric"]
    label: str
    part_type: Literal["turned", "prismatic", "plate", "bracket", "housing", "mixed", "other"]
    summary: str
    resting_face: str
    symmetry: str
    hypotheses: list[Hypothesis] = Field(description="Features you can see; params stay empty here")
    cannot_tell: list[str]

    @field_validator("hypotheses")
    @classmethod
    def no_measurements(cls, hyps: list[Hypothesis]) -> list[Hypothesis]:
        measured = [h.name for h in hyps if any(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                for v in h.params.values())]
        if measured:
            raise ValueError(f"isometric views are not measured; remove numeric params from {measured}")
        return hyps


# --------------------------------------------------------------------------- feature tree
class Evidence(Strict):
    view: str = Field(description="View id from split_views, or 'title_block' / 'notes'")
    text: str = Field(description="Exactly as written on the drawing")
    role: str = Field(description="The param this supports, e.g. 'd' or 'height'")
    derived: str | None = Field(None, description="Arithmetic if the value is computed, e.g. '25 - 4'")


class Placement(Strict):
    """One instance of a feature that is repeated irregularly (not a regular pattern), e.g. two pads on one
    leg and two on the other."""
    at: list[float] = Field(min_length=3, max_length=3, description="x, y, z of this instance in the part frame, mm")
    direction: Axis | None = Field(None, description="Extrude, hole or boss direction of this instance")
    long_axis: Axis | None = Field(None, description="Slots and obround profiles: the direction their length runs")


class Feature(Strict):
    id: str = Field(pattern=r"^F\d{2}$")
    name: str
    op: Op
    params: dict[str, ParamValue]
    on: str = Field(description="Face or plane the feature starts from, e.g. 'top face of F01' or 'XY plane'")
    position: list[float] | None = Field(None, min_length=3, max_length=3,
                                         description="x, y, z of the feature origin in the part frame, mm")
    direction: Axis | None = Field(None, description="Extrude, hole or revolve axis direction")
    long_axis: Axis | None = Field(None, description="Slots and obround profiles: the direction their length runs")
    pattern: Pattern | None = None
    extends_into: str | None = Field(None, pattern=r"^F\d{2}$", description="Id of the round body this feature "
                                     "merges into (a support under a barrel, a boss on a cylinder): the harness then "
                                     "computes its height up to that body's axis; leave the height out")
    placements: list[Placement] = Field(default_factory=list, description="Every instance with its own "
                                        "position, when the copies are not a regular pattern; then leave "
                                        "`position` and `pattern` empty")
    evidence: list[Evidence]
    depends_on: list[str] = Field(default_factory=list)
    checks: list[str] = Field(description="bbox | volume | projection:<views> | cylinder:d=..,count=..,thru")
    state: Literal["not_started", "built", "passing", "failing", "unsupported"] = "not_started"

    @field_validator("checks")
    @classmethod
    def known_checks(cls, checks: list[str]) -> list[str]:
        bad = [c for c in checks if not CHECK_RE.match(c)]
        if bad:
            raise ValueError(f"unknown check(s) {bad}")
        return checks

    @field_validator("evidence")
    @classmethod
    def evidence_views(cls, evidence: list[Evidence], info: ValidationInfo) -> list[Evidence]:
        check_views([e.view for e in evidence], info)
        return evidence

    @model_validator(mode="after")
    def required_params(self) -> Feature:
        missing = []
        for need in REQUIRED_PARAMS.get(self.op, []):
            options = need if isinstance(need, tuple) else (need,)
            if self.extends_into and options == ("height",):
                continue  # computed by the harness up to the body it merges into
            if not any(o in self.params for o in options):
                missing.append(" or ".join(options))
        if missing:
            raise ValueError(f"{self.id} ({self.op}) is missing param(s): {', '.join(missing)}")
        # sheet-metal operations are placed by their profile and bend line, in the blank's frame
        if self.position is None and not self.placements and self.op not in ("base_flange", "bend", "edge_flange"):
            raise ValueError(f"{self.id} has no position: give `position` [x, y, z], or `placements` with one "
                             f"entry per instance")
        if self.placements and self.pattern and self.pattern.type != "single":
            raise ValueError(f"{self.id}: use either a regular `pattern` or explicit `placements`, not both")
        if self.op == "slot" and not (self.long_axis or (self.placements and all(pl.long_axis
                                                                                    for pl in self.placements))):
            raise ValueError(f"{self.id} (slot) needs `long_axis`: the direction its length runs")
        return self


class Assumption(Strict):
    id: str = Field(pattern=r"^A\d+$")
    text: str
    covers: list[str] = Field(description="'<feature id>.<param>' for each value it supplies, e.g. 'F03.depth'")


class Unsupported(Strict):
    text: str = Field(description="The callout or feature as on the drawing")
    reason: str
    evidence: list[str] = Field(default_factory=list)


class Frame(Strict):
    origin: str = Field(description="e.g. 'centre of the bottom face'")
    x: str = Field(description="e.g. 'front-view horizontal'")
    z: Literal["up"] = "up"


class FeatureTree(Strict):
    frame: Frame
    features: list[Feature] = Field(min_length=1)
    assumptions: list[Assumption] = Field(default_factory=list)
    unsupported: list[Unsupported] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self) -> FeatureTree:
        errors, seen = [], set()
        for f in self.features:
            if f.id in seen:
                errors.append(f"duplicate id {f.id}")
            for dep in f.depends_on:
                if dep not in seen:
                    errors.append(f"{f.id} depends on {dep}, which is not an earlier feature")
            seen.add(f.id)
        covered = {ref for a in self.assumptions for ref in a.covers}
        for ref in covered:
            if ref.split(".")[0] not in seen:
                errors.append(f"assumption covers {ref}, but there is no feature {ref.split('.')[0]}")
        for f in self.features:  # a position is a number too: it needs a source
            located = f.placements or (f.position and any(abs(c) > 1e-9 for c in f.position))
            sourced = any(e.role.startswith(("position", "placement")) for e in f.evidence) \
                or any(ref.startswith((f"{f.id}.position", f"{f.id}.placement")) for ref in covered)
            if located and not sourced:
                errors.append(f"{f.id} position has no evidence with role 'position' (quote the dimensions that "
                              f"locate it and, in `derived`, the face they are measured from) and no assumption "
                              f"covering '{f.id}.position'")
        ids = {f.id for f in self.features}
        for f in self.features:
            if f.extends_into and (f.extends_into not in ids or f.extends_into == f.id):
                errors.append(f"{f.id}.extends_into = {f.extends_into}: there is no other feature with that id")
        for f in self.features:  # every number needs a source
            roles = {e.role for e in f.evidence} | ({"height"} if f.extends_into else set())
            for key, value in f.params.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) \
                        and key not in roles and f"{f.id}.{key}" not in covered:
                    errors.append(f"{f.id}.{key} = {value} has no evidence with role '{key}' and no assumption")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class Overview(Strict):
    title: str
    drawing_number: str | None = None
    units: str = "mm"
    scale: str | None = None
    projection: Literal["first_angle", "third_angle", "unknown"]
    part_type: Literal["turned", "prismatic", "plate", "bracket", "housing", "mixed", "other"]
    description: str = Field(description="What the part looks like, in a few sentences")
    function: str = Field(description="Best guess at what it is for, stated as a guess")
    notes: list[str] = Field(default_factory=list)


class PartPlan(Strict):
    """The main agent's answer."""
    overview: Overview
    feature_tree: FeatureTree
    uncertainties: list[str] = Field(default_factory=list)


class BuiltFeature(Strict):
    id: str = Field(pattern=r"^F\d{2}$")
    state: Literal["built", "approximated", "skipped"]
    note: str = ""


class BuildReport(Strict):
    """The builder agent's answer."""
    status: Literal["built", "partial", "failed"]
    features: list[BuiltFeature]
    notes: list[str] = Field(default_factory=list)


class Issue(Strict):
    """One difference between the drawing and the built model, in one view."""
    kind: Literal["missing", "extra", "wrong_size", "wrong_position", "wrong_orientation"]
    description: str = Field(description="What differs, in words a modeller can act on")
    drawing: str = Field(description="What the drawing shows, quoting dimension text exactly where there is one")
    model: str = Field(description="What the model projection shows instead")
    feature: str | None = Field(None, pattern=r"^F\d{2}$", description="Feature id at fault; null if no feature exists")
    owner: Literal["plan", "build"] = Field(description="plan: the feature tree is wrong; build: the tree is "
                                                        "right but the model does not follow it")
    fix: str = Field(description="The change to make, e.g. 'F02.height: 3 -> 40, support rises to the barrel axis'")
    severity: Literal["high", "low"] = Field(description="high: changes the part's shape or a dimension; "
                                                         "low: cosmetic or drafting convention")
    evidence: list[str] = Field(default_factory=list, description="Metrics or callout checks that show it")


class ValidationReport(Strict):
    """A validation agent's answer for one view."""
    view: str
    verdict: Literal["pass", "minor", "fail"]
    orientation: Literal["matches", "mirrored", "rotated", "unknown"]
    matches: list[str] = Field(description="What agrees with the drawing, with feature ids, e.g. 'F07 4x Ø10 holes'")
    issues: list[Issue]
    confidence: float = Field(ge=0, le=1)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verdict_matches_issues(self) -> ValidationReport:
        high = [i for i in self.issues if i.severity == "high"]
        if self.verdict == "pass" and self.issues and high:
            raise ValueError("verdict 'pass' with high-severity issues; use 'fail'")
        if self.verdict == "fail" and not high:
            raise ValueError("verdict 'fail' needs at least one high-severity issue")
        return self


def _flat(text: str) -> str:
    return re.sub(r"\s+", "", text.upper().replace("⌀", "Ø").replace("∅", "Ø"))


class Ruling(Strict):
    """The arbiter's decision on one cross-check item."""
    item: str = Field(pattern=r"^C\d+$", description="The item id, e.g. C3")
    decision: Literal["resolved", "already_covered", "unresolved"] = Field(
        description="resolved: you settled it from the drawing; already_covered: another view's hypothesis already "
                    "describes it (say which); unresolved: the drawing does not settle it")
    fact: str = Field(description="The settled fact in part axes, precise enough to model from, e.g. 'support block: "
                                  "X -34..34 (68), Y -18..18 (36), from the plate top Z=9 up into the barrel'")
    values: dict[str, float] = Field(default_factory=dict, description="Named values, e.g. {'support.size_x': 68}")
    evidence: list[str] = Field(default_factory=list, description="Drawing text exactly as printed that proves it")
    views: list[str] = Field(default_factory=list, description="View ids you checked")
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def quotes_the_drawing(self, info: ValidationInfo) -> Ruling:
        texts = (info.context or {}).get("texts")
        if self.decision == "resolved" and not self.evidence:
            raise ValueError(f"{self.item}: a resolved item must quote the drawing text that proves it")
        if texts and self.evidence:
            sheet = [_flat(x) for x in texts]
            missing = [e for e in self.evidence if not any(_flat(e) in s or s in _flat(e) for s in sheet)]
            if missing:
                raise ValueError(f"{self.item}: evidence {missing} is not text on the drawing; quote dimensions and "
                                 f"callouts exactly as printed (read_drawing_text lists them)")
        return self


class ArbiterReport(Strict):
    """The arbiter's answer: one ruling per cross-check item."""
    rulings: list[Ruling]
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- path agent (one region of a view)
PathOp = Literal[
    "extrude_boss", "revolve_boss", "extrude_cut", "revolve_cut", "boolean", "sweep", "loft", "shell", "rib",
    "draft", "thicken", "hole", "counterbore_hole", "countersink_hole", "tapped_hole", "external_thread", "slot",
    "keyway", "chamfer", "fillet", "knurl", "base_flange", "bend", "edge_flange", "flat_pattern", "none",
]


class PathDimension(Strict):
    """A dimension or callout that sizes or places this region."""
    text: str = Field(description="Exactly as printed, e.g. 'R25', 'Ø7 THRU', '47.8'")
    value: float | None = Field(None, description="The number in mm or degrees, if any")
    kind: Literal["length", "diameter", "radius", "angle", "position", "thickness", "count", "note", "other"]
    measures: str = Field(description="What it measures on this region, e.g. 'width of P3', 'radius of the arc on "
                                      "P1's right edge', 'position of the hole from the left edge'")


class PathEdge(Strict):
    """One stretch of the region's boundary, in order round it."""
    kind: Literal["line", "arc", "circle", "spline", "unclear"]
    where: str = Field(description="Where it is, e.g. 'top edge', 'lower-right corner', 'shared with P2'")
    boundary_of: Literal["part_outline", "bend_line", "inner_edge", "hole", "annotation", "unclear"] = Field(
        description="What this stretch is on the drawing: the part's outer outline, a bend line, an edge between "
                    "two faces, a hole, or an annotation line (dimension, leader, extension, centre line)")
    dimension: str | None = Field(None, description="The text of the dimension that sizes this stretch, if any")


class PathNeighbour(Strict):
    region: str = Field(pattern=r"^P\d+$")
    relation: Literal["bend", "edge_between_faces", "step", "same_surface", "contains", "inside",
                      "annotation_split", "unclear"] = Field(
        description="bend: a bend line separates them; edge_between_faces: they are two faces meeting at an edge "
                    "(pictorial view); step: a change of level; same_surface: one face cut in two by an annotation "
                    "or centre line; contains / inside: one lies inside the other (a hole in a face); "
                    "annotation_split: the boundary between them is only a dimension or leader line")
    note: str = ""


class PathCreate(Strict):
    """How to make this region in CAD."""
    op: PathOp
    params: dict[str, ParamValue] = Field(default_factory=dict, description="Values in mm / degrees, named as in "
                                                                            "the feature tree (d, r, width, length, "
                                                                            "height, thickness, angle, ...)")
    sketch: str = Field("", description="The profile to sketch, in words with numbers, e.g. 'rectangle 61 x 18 "
                                        "with R2 corners on the left'")
    on: str = Field("", description="Where it starts / which face or plane, in words")
    depends_on: list[str] = Field(default_factory=list, description="Regions it is built on, e.g. ['P1']")


class PathReport(Strict):
    """The path agent's analysis of one region (path) of one view."""
    view: str
    region: str = Field(pattern=r"^P\d+$")
    role: Literal["part_face", "panel", "hole_or_cutout", "section_cut", "background", "annotation",
                  "unclear"] = Field(description="part_face: a face of the part (pictorial or orthographic view); "
                                                 "panel: a panel of a sheet-metal flat blank, between bend lines; "
                                                 "hole_or_cutout: an opening in the part; section_cut: cut material "
                                                 "in a section; background: empty paper enclosed by lines (between "
                                                 "dimension or extension lines); annotation: inside text, an "
                                                 "arrow or a symbol")
    description: str = Field(description="What this region is, in one or two sentences")
    shape: Literal["rectangle", "rounded_rectangle", "circle", "ring", "slot", "polygon", "arc_band", "irregular",
                   "other"]
    edges: list[PathEdge] = Field(default_factory=list)
    dimensions: list[PathDimension] = Field(default_factory=list)
    size_mm: list[float] = Field(default_factory=list, description="Overall width and height of the region in mm "
                                                                   "when the dimensions give them")
    neighbours: list[PathNeighbour] = Field(default_factory=list)
    create: PathCreate | None = Field(None, description="Leave out for background and annotation regions")
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def create_for_geometry(self) -> PathReport:
        if self.role in ("part_face", "panel", "hole_or_cutout", "section_cut") and self.create is None:
            raise ValueError(f"{self.region}: a {self.role} needs `create` (how to make it)")
        return self


class CornerVerdict(Strict):
    """One traced corner, judged from its patch."""
    corner: str = Field(pattern=r"^V\d+$", description="The corner id written on its patch, e.g. V7")
    verdict: Literal["real_corner", "on_edge", "artifact"] = Field(
        description="real_corner: the part's outline really turns here (a corner, or where a line meets an arc); "
                    "on_edge: the point lies on the part's edge but the edge does not turn here (along a straight "
                    "line or a smooth arc); artifact: the blue trace is not on the part's edge here (it follows an "
                    "arrowhead, a leader, a dimension or extension line, a centre line or text, or cuts a corner)")
    reason: str = Field(description="What you see in the patch, in a few words")


class CornerCheck(Strict):
    """The answer of the corner check: one verdict per corner shown."""
    corners: list[CornerVerdict]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def every_corner_once(self, info: ValidationInfo) -> CornerCheck:
        shown = (info.context or {}).get("corners")
        given = [c.corner for c in self.corners]
        if shown:
            missing = [c for c in shown if c not in given]
            extra = sorted({c for c in given if c not in shown})
            twice = sorted({c for c in given if given.count(c) > 1})
            if missing or extra or twice:
                raise ValueError(f"answer every corner shown exactly once; missing {missing}, unknown {extra}, "
                                 f"answered twice {twice}")
        return self


class Objection(Strict):
    claim: str = Field(description="What is wrong with the model, in one sentence")
    evidence: str = Field(description="What the drawing shows against what the model shows, and in which view")
    feature: str | None = Field(None, description="The feature id it concerns, if any")
    kind: Literal["mirrored", "orientation", "missing_feature", "extra_feature", "wrong_face", "wrong_count",
                  "bend_direction", "wrong_size", "wrong_profile", "other"]
    severity: Literal["high", "low"] = Field(description="high: the part's shape is wrong; low: a detail")


class RedTeamReport(Strict):
    """The red team's answer: the strongest reasons the finished model is wrong."""
    verdict: Literal["objections", "no_objection"]
    objections: list[Objection] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self) -> RedTeamReport:
        if self.verdict == "objections" and not self.objections:
            raise ValueError("verdict 'objections' needs at least one objection")
        return self


MODELS: dict[str, type[BaseModel]] = {m.__name__: m for m in (
    FrontSummary, TopSummary, SideSummary, SectionSummary, DetailSummary, IsometricSummary,
    FeatureTree, PartPlan, BuildReport, ValidationReport, ArbiterReport, PathReport, CornerCheck, RedTeamReport,
)}
