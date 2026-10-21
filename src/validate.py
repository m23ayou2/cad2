"""Compare a built STEP with every view of its drawing; the evidence for the validation agents.

    python src/validate.py <split views folder> <model.step> <out folder> [drawing.pdf]

For each view: project the model as built (no re-orientation) with hidden lines
removed, cut it for a section, align it on the drawn edges, and measure how far
the two are apart. Writes, per view, the drawing / model / overlay images, the
aligned points (for zooming), and validation.json with:

- metrics: mean and 95th-percentile distances, the share of drawn edges the model covers;
- missing / extra regions: where drawn edges have no model edge nearby, and the reverse,
  as boxes in the view's model axes (e.g. "horizontal = +X, vertical = +Z");
- callout checks: each Ø, R and linear dimension of the view matched against the solid;
- pose: whether another axis-aligned turn or mirror of the model fits the views better.

Sections are located from their cutting-plane line on the parent view (which may itself
be a section): the plane's position from where the line is drawn, the viewing direction
from the line's arrowheads (read from the drawing PDF). When no arrowhead is found, the
viewing direction is a convention and the section is flagged `side_known: false`.

Runs in its own process because OpenCascade can crash on a bad shape.
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pymupdf  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import pose_solver as ps  # noqa: E402
import step_checks  # noqa: E402
from views import (PT_PER_MM, Declared, DetailView, OrthographicView, PictorialView,  # noqa: E402
                   SectionView, ViewDirection, load_view)

AXES = ["X", "Y", "Z"]
UNIT_AXES = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]


# --------------------------------------------------------------------------- sections
def sheet_marks(views):
    """Single-letter marks of every view, in sheet coordinates (PDF points, y down)."""
    marks = {}
    for v in views:
        off = (v.sheet_info or {}).get("offset_to_view_pt")
        if not off:
            continue
        for a in v.annotations:
            if a.kind == "mark":
                b = a.bbox
                marks.setdefault(a.text, []).append(np.array([(b[0] + b[2]) / 2 - off[0], (b[1] + b[3]) / 2 - off[1]]))
    return marks


def arrowheads(drawing_pdf):
    """Small filled triangles of the drawing: (tip, unit direction, anchor), sheet coordinates. On a
    cutting-plane line each arrowhead ends a thin stub whose other end lies on the line: that end is the
    anchor (None when there is no stub)."""
    out = []
    if not drawing_pdf:
        return out
    with pymupdf.open(drawing_pdf) as doc:
        drawings = doc[0].get_drawings()
        stubs = [(np.array([it[1].x, it[1].y]), np.array([it[2].x, it[2].y]))
                 for d in drawings if d.get("fill") is None for it in d["items"] if it[0] == "l"]
        for d in drawings:
            r = d["rect"]
            if d.get("fill") is None or r.width > 15 or r.height > 15 or r.width + r.height < 3:
                continue
            pts = []
            for it in d["items"]:
                if it[0] == "l":
                    for p in (it[1], it[2]):
                        q = np.array([p.x, p.y])
                        if not any(np.allclose(q, s, atol=0.05) for s in pts):
                            pts.append(q)
            if len(pts) != 3:
                continue
            # the tip is the vertex opposite the shortest side
            sides = [np.linalg.norm(pts[(i + 1) % 3] - pts[(i + 2) % 3]) for i in range(3)]
            tip = pts[int(np.argmin(sides))]
            base = (sum(pts) - tip) / 2
            direction = tip - base
            if np.linalg.norm(direction) < 1e-6:
                continue
            direction /= np.linalg.norm(direction)
            anchor = None
            for a, b in stubs:
                for end, other in ((a, b), (b, a)):
                    # the stub runs from the line to the tip, along the arrow
                    if np.linalg.norm(end - tip) < 0.6 and np.linalg.norm(other - tip) > 3 \
                            and np.dot(tip - other, direction) > 0.9 * np.linalg.norm(tip - other):
                        anchor = other
            out.append((tip, direction, anchor))
    return out


def frame_of(view, frames):
    """(right, up, look) of a view whose orientation is known, else None."""
    if view.view_id in frames:
        f = frames[view.view_id]
        b = ps.view_basis(f["look"], f["up"])
        return b[0], b[1], np.array(f["look"], float)
    if isinstance(view, OrthographicView) and view.direction:
        d = view.direction.value
        return d.right, d.up, d.look
    return None


def to_3d(vec2, right, up):
    """A sheet vector (x right, y down) drawn in a view -> model direction."""
    return right * vec2[0] - up * vec2[1]


def outline_box(view):
    return (view.sheet_info or {}).get("outline_bbox_sheet_pt")


def infer_directions(views, first_angle=True):
    """Give unlabelled orthographic views a direction from where they sit on the sheet (first-angle
    projection: the view below an elevation looks down, the view to its right looks from the left, ...).
    The reference is a labelled view, or else a section cut from the unlabelled view: that section is an
    elevation (Z up, X to the right), which is how the plan frames the part. Detached stubs (cut-line ends
    split off as their own 'view') are ignored."""
    boxes = [outline_box(v) for v in views if outline_box(v)]
    biggest = max((max(b[2] - b[0], b[3] - b[1]) for b in boxes), default=0)
    marks = sheet_marks(views)
    sign = 1.0 if first_angle else -1.0
    for _ in range(2):
        for u in views:
            ub = outline_box(u)
            if not isinstance(u, OrthographicView) or u.direction or not ub \
                    or max(ub[2] - ub[0], ub[3] - ub[1]) < 0.25 * biggest:
                continue
            references = []
            for r in views:
                rb = outline_box(r)
                if r is u or not rb:
                    continue
                if isinstance(r, OrthographicView) and r.direction:
                    d = r.direction.value
                    references.append((r, d.right, d.up))
                elif isinstance(r, SectionView) and r.section_id and any(
                        ub[0] - 30 <= p[0] <= ub[2] + 30 and ub[1] - 30 <= p[1] <= ub[3] + 30
                        for p in marks.get(r.section_id, [])):
                    references.append((r, np.array([1.0, 0, 0]), np.array([0, 0, 1.0])))
            for r, right, up in references:
                rb = outline_box(r)
                x_overlap = min(ub[2], rb[2]) - max(ub[0], rb[0])
                y_overlap = min(ub[3], rb[3]) - max(ub[1], rb[1])
                if x_overlap > 0.3 * min(ub[2] - ub[0], rb[2] - rb[0]):      # stacked vertically
                    look = -up * sign if ub[1] > rb[1] else up * sign          # below: seen from above
                elif y_overlap > 0.3 * min(ub[3] - ub[1], rb[3] - rb[1]):    # side by side
                    look = right * sign if ub[0] > rb[0] else -right * sign    # right of it: seen from the left
                else:
                    continue
                match = next((d for d in ViewDirection if np.allclose(d.look, look)), None)
                if match:
                    u.direction = Declared.inferred(match, 0.7, f"sheet layout relative to {r.view_id}")
                    u.warnings.append(f"direction {match.name} inferred from the sheet layout ({r.view_id})")
                    break


def locate_sections(views, drawing_pdf):
    """Cutting plane, position and viewing direction of every section, from its cut line on a parent view.
    A section can be the parent of another section, so this runs in passes."""
    marks, arrows = sheet_marks(views), arrowheads(drawing_pdf)
    frames, todo = {}, [v for v in views if isinstance(v, SectionView) and v.section_id]
    for _ in range(3):
        for sec in list(todo):
            pts = marks.get(sec.section_id, [])
            if len(pts) < 2:
                continue
            p0 = pts[0]
            p1 = max(pts[1:], key=lambda p: np.linalg.norm(p - p0))
            # the arrowhead next to each letter: its direction, and its stub's anchor on the cut line
            near = []
            for p in (p0, p1):
                cand = [(np.linalg.norm(tip - p), d, anchor) for tip, d, anchor in arrows
                        if np.linalg.norm(tip - p) < 30]
                near.append(min(cand, key=lambda c: c[0]) if cand else None)
            if all(n is not None and n[2] is not None for n in near) and \
                    np.linalg.norm(near[0][2] - near[1][2]) > 5:
                p0, p1 = near[0][2], near[1][2]  # exact points on the line, not the letters beside it
            mid = (p0 + p1) / 2
            parents = []
            for v in views:
                ob = (v.sheet_info or {}).get("outline_bbox_sheet_pt")
                if v is sec or not ob or frame_of(v, frames) is None:
                    continue
                dist = math.hypot(max(ob[0] - mid[0], mid[0] - ob[2], 0), max(ob[1] - mid[1], mid[1] - ob[3], 0))
                parents.append((dist, v))
            if not parents:
                continue
            parent = min(parents, key=lambda p: p[0])[1]
            right, up, plook = frame_of(parent, frames)
            line = to_3d(p1 - p0, right, up)
            line /= np.linalg.norm(line)
            normal = np.cross(line, plook)
            normal /= np.linalg.norm(normal)
            # viewing direction: the arrowheads next to the letters, perpendicular to the cut line
            votes = [float(np.dot(to_3d(n[1], right, up), normal)) for n in near if n is not None]
            side_known = bool(votes) and all(abs(v) > 0.5 for v in votes) and len({np.sign(v) for v in votes}) == 1
            look = normal * (np.sign(votes[0]) if side_known else -1.0)
            # where the plane lies: the cut line's offset from the parent outline's centre, along the cut normal
            ob = parent.sheet_info["outline_bbox_sheet_pt"]
            centre = np.array([(ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2])
            scale = parent.scale.value or 1.0
            offset_mm = float(np.dot(to_3d(mid - centre, right, up), -look)) / PT_PER_MM / scale
            frames[sec.view_id] = {"look": tuple(look), "up": None, "parent": parent.view_id,
                                   "cut_normal": tuple(-look), "cut_offset": offset_mm, "side_known": side_known}
            frames[sec.view_id]["up"] = standard_up(look)
            todo.remove(sec)
    return frames


def standard_up(look):
    """A section is drawn the way the standard view from its direction is: Z up when looking sideways,
    +Y up when looking down (as in a top view), and so on."""
    look = np.array(look, float)
    for d in ViewDirection:
        if np.allclose(d.look, look, atol=1e-6):
            return tuple(float(c) for c in d.up)
    return tuple(next(a for a in UNIT_AXES if abs(np.dot(a, look)) < 0.5))


def fit_section_up(shape, view, frame):
    """Keep the standard orientation of the section unless the drafter clearly drew it turned in its
    plane: another of the 4 'up' directions must fit at least twice as well and cover the drawing.
    Turning a view in its plane cannot hide a mirror. Returns (up, note)."""
    A = ps.drawing_points(view)
    scores = {}
    for up in UNIT_AXES:
        if abs(np.dot(up, frame["look"])) > 0.5:
            continue
        B = ps.model_view_points(shape, {**frame, "up": tuple(up)})
        if len(B):
            m = compare(A, B)[0]
            scores[tuple(float(c) for c in up)] = (m["mean_mm"], m["drawn_covered"])
    standard = tuple(float(c) for c in frame["up"])
    if not scores or standard not in scores:
        return frame["up"], None
    best = min(scores, key=lambda u: scores[u][0])
    if best != standard and scores[best][0] < 0.5 * scores[standard][0] and scores[best][1] >= 0.95:
        return best, f"drawn turned in its plane: 'up' is {axis_name(np.array(best))}, not the standard " \
                     f"{axis_name(np.array(standard))}"
    return standard, None


def view_frames(views_dir, drawing_pdf=None):
    """Each view's axes in the part frame, from the drawing alone (labels, sheet layout, cut-line arrows), for
    the analysis agents and the planner: {view pdf stem: {horizontal, vertical, looking_along, source}}.
    Uses the same conventions as the validation, so planned positions and the checks agree."""
    views = []
    for pdf in sorted(glob.glob(os.path.join(views_dir, "*_view*.pdf"))):
        try:
            views.append(load_view(pdf))
        except Exception:
            continue
    infer_directions(views)
    frames = locate_sections(views, drawing_pdf)
    out = {}
    for v in views:
        if isinstance(v, SectionView) and v.view_id in frames:
            f = frames[v.view_id]
            look, up = f["look"], f["up"]
            source = "cutting-plane line and its arrows" if f["side_known"] else \
                "cutting-plane line; no arrowhead found, so left and right may be swapped"
        elif isinstance(v, OrthographicView) and v.direction:
            d = v.direction.value
            look, up = d.look, d.up
            source = "view label" if v.direction.source.name == "FOUND" else v.direction.evidence
        else:
            continue
        basis = ps.view_basis(look, up)
        out[v.view_id] = {"horizontal": axis_name(basis[0]), "vertical": axis_name(basis[1]),
                          "looking_along": axis_name(np.array(look, float)), "source": source}
    return out


def fit_isometric(shape, view, views):
    """Measure an isometric view: a drawing uses one of the standard isometric directions, drawn either as a true
    projection or as an 'isometric drawing' (axes at full length, 1.22x larger). Keep the cameras that look down
    from the front view's side, and the direction and size that fit the drawn edges best. Returns
    (spec, drawing points, model points, note) or None when nothing fits. The camera is chosen by the fit, so a
    mirror could hide in it: this view is never used for the pose check."""
    try:
        A0 = ps.drawing_points(view)
    except Exception:
        return None
    front = next((v for v in views if isinstance(v, OrthographicView) and v.direction
                  and v.direction.value.name == "FRONT"), None)
    best = None
    for sx, sy in itertools.product((1, -1), repeat=2):
        look = np.array([sx, sy, -1.0]) / math.sqrt(3)       # looking down at the part
        if front is not None and np.dot(look, front.direction.value.look) <= 0:
            continue                                         # the camera is on the front view's side
        spec = {"look": tuple(look), "up": (0.0, 0.0, 1.0)}
        B = ps.model_view_points(shape, spec)
        if len(B) == 0:
            continue
        for factor, style in ((1.0, "true isometric projection"), (1 / 1.2247, "isometric drawing (1.22x)")):
            m = compare(A0 * factor, B)[0]
            if best is None or m["mean_mm"] < best[0]["mean_mm"]:
                best = (m, spec, A0 * factor, B, style)
    if best is None or best[0]["drawn_covered"] < 0.5:
        return None
    m, spec, A, B, style = best
    spec = {**spec, "side_known": False}
    note = (f"isometric camera fitted to the drawn edges ({style}, looking along {np.round(spec['look'], 3).tolist()}); "
            "measured like the other views, but not used to judge mirrors")
    return spec, A, B, note


# --------------------------------------------------------------------------- comparison
def align(A, B, iterations=12, start=None):
    """Translate the drawing points A onto the model points B. Starts from the bounding-box centres (or from
    `start`, an offset found by a wider search), then refines with the median offset of the closest pairs, so a
    few stray marks cannot drag the fit."""
    t = (B.min(0) + B.max(0)) / 2 - (A.min(0) + A.max(0)) / 2 if start is None else np.asarray(start, float)
    tree = cKDTree(B)
    size = float(np.max(B.max(0) - B.min(0))) or 1.0
    # coarse search first: the closest-pair refinement alone can lock onto a wrong offset when long edges in
    # one direction agree (vertical edges fit at any vertical shift) and a stray mark moved the starting box
    sample = A[:: max(1, len(A) // 1500)]
    span = max(1.0, 0.04 * size)
    score = lambda s: float(np.percentile(tree.query(sample + t + s)[0], 60))
    coarse = np.linspace(-span, span, 17)
    best = np.array(min(((dx, dy) for dx in coarse for dy in coarse), key=lambda s: score(np.array(s))))
    fine = np.linspace(-(coarse[1] - coarse[0]), coarse[1] - coarse[0], 11)
    best = min((best + np.array((dx, dy)) for dx in fine for dy in fine), key=score)
    t = t + best
    for _ in range(iterations):
        d, j = tree.query(A + t)
        keep = d < max(np.percentile(d, 70), 0.01 * size)
        if keep.sum() < 10:
            break
        step = np.median(B[j[keep]] - (A[keep] + t), axis=0)
        t = t + step
        if np.linalg.norm(step) < 1e-3:
            break
    return A + t


def clusters(points, distances, tol, min_points=8):
    """Groups of points farther than tol from the other set, as boxes."""
    far = points[distances > tol]
    if len(far) == 0:
        return []
    tree = cKDTree(far)
    link = max(tol * 2, 1.0)
    label = -np.ones(len(far), int)
    n = 0
    for i in range(len(far)):
        if label[i] >= 0:
            continue
        stack, label[i] = [i], n
        while stack:
            k = stack.pop()
            for m in tree.query_ball_point(far[k], link):
                if label[m] < 0:
                    label[m] = n
                    stack.append(m)
        n += 1
    out = []
    for c in range(n):
        P = far[label == c]
        if len(P) < min_points:
            continue
        lo, hi = P.min(0), P.max(0)
        out.append({"from": [round(float(lo[0]), 1), round(float(lo[1]), 1)],
                    "to": [round(float(hi[0]), 1), round(float(hi[1]), 1)],
                    "length_mm": round(float(len(P) * 0.1), 1)})
    return sorted(out, key=lambda r: -r["length_mm"])[:8]


def compare(A, B, tol=None, start=None):
    """A drawn, B model (view plane, mm). Returns metrics, regions and the aligned drawing points. `tol` overrides
    the default tolerance, e.g. for a scanned drawing whose edges are only accurate to about a pixel; `start` is an
    initial offset for the alignment."""
    A = align(A, B, start=start)
    da = cKDTree(B).query(A)[0]
    db = cKDTree(A).query(B)[0]
    size = float(np.max(B.max(0) - B.min(0)))
    if tol is None:
        tol = max(0.03, 0.003 * size)  # drawn edges are exact: references fit to ~0.02 mm
    metrics = {
        "tolerance_mm": round(tol, 2),
        "mean_mm": round(float((da.mean() + db.mean()) / 2), 3),
        "drawn_to_model_p95_mm": round(float(np.percentile(da, 95)), 2),
        "model_to_drawn_p95_mm": round(float(np.percentile(db, 95)), 2),
        "drawn_covered": round(float((da <= tol).mean()), 3),
        "model_covered": round(float((db <= tol).mean()), 3),
    }
    return metrics, clusters(A, da, tol), clusters(B, db, tol), A


def axis_name(vec):
    i = int(np.argmax(np.abs(vec)))
    return ("+" if vec[i] > 0 else "-") + AXES[i]


def spec_of(view):
    sp = view.projection_spec()
    return {"kind": sp.kind, "look": sp.look, "up": sp.up,
            "cut_normal": sp.cut_plane.normal if sp.cut_plane else None}


# --------------------------------------------------------------------------- callouts
def check_callouts(shape, view, look, tol=0.05):
    """Ø / R callouts of one view against the solid's cylinders, using the view's axis: a counted callout
    ('4x Ø6.6') labels circles, i.e. cylinders seen end-on (axis along the look) in an ordinary view, and
    cylinders cut lengthwise (axis in the cut plane) in a section. So the axial Ø3 bore is not counted as
    one of the '2x Ø3 RADIAL' holes drawn as circles in a side view."""
    inventory = step_checks.cylinder_inventory(shape)
    look = np.array(look, float)
    rows, seen = [], set()
    for a in view.annotations:
        if a.kind not in ("diameter", "radius") or a.text in seen:
            continue
        seen.add(a.text)
        r = a.value / 2 if a.kind == "diameter" else a.value
        same = [c for c in inventory if abs(c["radius"] - r) <= tol]
        if isinstance(view, SectionView):
            in_view = [c for c in same if abs(np.dot(c["axis"], look)) < 0.1]
        else:
            in_view = [c for c in same if abs(np.dot(c["axis"], look)) > 0.9]
        ok, why = True, []
        if a.kind == "diameter" and a.extra.get("explicit_count", False):
            if len(in_view) != a.count:
                ok = False
                why.append(f"{len(in_view)} cylinder(s) of this size in this view's direction, drawing says {a.count}x")
            hits = in_view
        else:
            hits = in_view or same
            if not hits:
                ok = False
                why.append("no cylinder of this size")
        if a.extra.get("depth") is not None and hits:
            depths = sorted({round(c["depth"], 2) for c in hits})
            if not any(abs(d - a.extra["depth"]) <= tol for d in depths):
                ok = False
                why.append(f"depth {depths} vs {a.extra['depth']}")
        if a.extra.get("thru") and hits and a.kind == "diameter":
            if any(step_checks.has_bottom_face(shape, c) for c in hits):
                ok = False
                why.append("marked THRU but the hole has a bottom")
        rows.append((a.text, ok, "; ".join(why) or f"{len(hits)} cylinder(s) r={r:g}"))
    return rows


# --------------------------------------------------------------------------- images
def plot(path, A, B, title, horizontal, vertical, missing=(), extra=()):
    fig, ax = plt.subplots(figsize=(8, 8 * max(0.4, min(2.0, np.ptp(B[:, 1]) / max(np.ptp(B[:, 0]), 1e-6)))))
    if A is not None:
        ax.plot(A[:, 0], A[:, 1], ".", color="black", ms=1.2, label="drawing")
    if B is not None:
        ax.plot(B[:, 0], B[:, 1], ".", color="red", ms=1.0, label="model")
    for r, color in [(r, "orange") for r in missing] + [(r, "purple") for r in extra]:
        (x0, y0), (x1, y1) = r["from"], r["to"]
        pad = 0.02 * max(np.ptp(B[:, 0]), np.ptp(B[:, 1]))
        ax.add_patch(plt.Rectangle((x0 - pad, y0 - pad), x1 - x0 + 2 * pad, y1 - y0 + 2 * pad,
                                   fill=False, color=color, lw=1.5))
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)
    ax.set_xlabel(f"{horizontal} (mm)")
    ax.set_ylabel(f"{vertical} (mm)")
    ax.set_title(title, fontsize=10)
    if A is not None and B is not None:
        ax.legend(loc="upper right", fontsize=8, markerscale=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def drawing_png(view, path):
    with pymupdf.open(view.source_path) as doc:
        page = doc[0]
        zoom = min(6.0, 1100 / max(page.rect.width, page.rect.height))
        page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).save(path)


# --------------------------------------------------------------------------- pose
def axis_turns(size):
    """Axis-aligned turns and mirrors worth testing. Swapping two axes changes the bounding box unless
    those sizes are nearly equal, and the bounding box is already checked, so only swaps between
    near-equal sizes are kept; the 7 sign flips (mirrors and 180° turns) are always kept."""
    out = []
    for perm in itertools.permutations(range(3)):
        if any(abs(size[perm[i]] - size[i]) > 0.02 * max(size) for i in range(3)):
            continue
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[i, p] = s
            out.append(M)
    return out


def pose_check(shape, measured, identity_total):
    """Score the other orientations on the measured views; report if one fits clearly better than as built.
    Sections whose viewing direction is a convention are left out: they cannot tell a mirror."""
    measured = [m for m in measured if m[1].get("side_known", True)]
    identity_total = sum(m[3] for m in measured)
    lo, hi = ps.bbox(shape)
    scores = []
    for M in axis_turns(hi - lo):
        if np.allclose(M, np.eye(3)):
            continue
        try:
            cand = ps.transform(shape, M)
            total = 0.0
            for view, spec, A, _ in measured:
                B = ps.model_view_points(cand, spec)
                if len(B) == 0:
                    raise ValueError("empty projection")
                total += compare(A, B)[0]["mean_mm"]
        except Exception:
            continue
        scores.append((total, M))
    scores.sort(key=lambda s: s[0])
    best_total, best = scores[0] if scores else (float("inf"), np.eye(3))
    better = best_total < 0.6 * identity_total and identity_total - best_total > 0.3
    mirror = bool(round(np.linalg.det(best)) == -1)
    return {
        "as_built_mean_mm": round(identity_total, 3),
        "best_other_mean_mm": round(best_total, 3),
        "best_other": ps.describe_rotation(best),
        "best_other_is_mirror": mirror,
        "better_orientation_found": bool(better),
        "verdict": ("another orientation fits clearly better: the model is "
                    + ("mirrored" if mirror else "turned") + " relative to the drawing")
                   if better else "the model as built fits best",
    }


# --------------------------------------------------------------------------- main
def run(views_dir, step_path, out_dir, drawing_pdf=None):
    os.makedirs(out_dir, exist_ok=True)
    started = time.monotonic()
    views = []
    for pdf in sorted(glob.glob(os.path.join(views_dir, "*_view*.pdf"))):
        try:
            views.append(load_view(pdf))
        except Exception as error:
            print(f"cannot load {pdf}: {error}", file=sys.stderr)
    infer_directions(views)
    frames = locate_sections(views, drawing_pdf)
    shape = ps.load_step(step_path)
    out = {"step": step_path, "views": [], "pose": None}
    measured = []

    for view in views:
        entry = {"view_id": view.view_id, "class": type(view).__name__, "label": view.label_text}
        out["views"].append(entry)
        try:
            if isinstance(view, DetailView):
                raise ValueError("detail views are compared through their parent view")
            if isinstance(view, SectionView):
                if view.view_id not in frames:
                    raise ValueError(f"section {view.section_id}-{view.section_id}: its cutting-plane line was "
                                     f"not found on another view")
                frame = frames[view.view_id]
                frame["up"], turned = fit_section_up(shape, view, frame)
                if turned:
                    entry["orientation_note"] = turned
                spec = {k: frame[k] for k in ("look", "up", "cut_normal", "cut_offset")}
                spec["side_known"] = frame["side_known"]
                entry.update(parent_view=frame["parent"], side_known=frame["side_known"],
                             cut_offset_mm=round(frame["cut_offset"], 2))
                if not frame["side_known"]:
                    entry["side_note"] = ("no arrowhead found on the cutting-plane line: the viewing direction is "
                                          "a convention, so left and right in this view may be swapped")
            else:
                spec = spec_of(view)
                if isinstance(view, OrthographicView) and view.direction.source.name == "INFERRED":
                    entry["direction_note"] = view.direction.evidence
            basis = ps.view_basis(spec["look"], spec["up"])
            entry["horizontal"], entry["vertical"] = axis_name(basis[0]), axis_name(basis[1])
            entry["looking_along"] = axis_name(np.array(spec["look"], float))
            if spec.get("cut_normal") is not None:
                entry["cut_plane_normal"] = axis_name(np.array(spec["cut_normal"], float))
            B = ps.model_view_points(shape, spec)
            if len(B) == 0:
                raise ValueError("the model projection is empty")
        except Exception as error:
            entry["unresolved"] = str(error)
            continue

        stem = os.path.join(out_dir, view.view_id)
        drawing_png(view, stem + ".drawing.png")
        entry["images"] = {"drawing": stem + ".drawing.png", "model": stem + ".model.png"}
        title = f"{view.label_text or view.view_id}: model as built, looking along {entry['looking_along']}"
        if isinstance(view, PictorialView):
            fitted = fit_isometric(shape, view, views)
            if fitted is None:
                entry["note"] = "pictorial view: the camera could not be fitted, so it is compared by eye only"
                plot(stem + ".model.png", None, B, title, entry["horizontal"] + " (approx.)", entry["vertical"] + " (approx.)")
                continue
            spec, A, B, how = fitted
            basis = ps.view_basis(spec["look"], spec["up"])
            entry.update(horizontal=axis_name(basis[0]) + " (iso)", vertical=axis_name(basis[1]) + " (iso)",
                         looking_along=f"{np.round(spec['look'], 3).tolist()}", note=how, side_known=False)
            title = f"{view.label_text or view.view_id}: model as built, fitted isometric camera"
        try:
            A = A if isinstance(view, PictorialView) else ps.drawing_points(view)
        except Exception as error:
            entry["unresolved"] = f"drawing edges not readable: {error}"
            plot(stem + ".model.png", None, B, title, entry["horizontal"], entry["vertical"])
            continue
        metrics, missing, extra, A_aligned = compare(A, B)
        entry.update(metrics=metrics, missing_regions=missing, extra_regions=extra)
        plot(stem + ".model.png", None, B, title, entry["horizontal"], entry["vertical"])
        plot(stem + ".overlay.png", A_aligned, B, f"{view.label_text or view.view_id}: drawing (black) vs model (red); "
             "orange = drawn, missing in model; purple = in model, not drawn",
             entry["horizontal"], entry["vertical"], missing, extra)
        entry["images"]["overlay"] = stem + ".overlay.png"
        np.savez_compressed(stem + ".points.npz", drawing=A_aligned, model=B)
        entry["points"] = stem + ".points.npz"
        rows = check_callouts(shape, view, spec["look"]) + step_checks.check_linear(shape, view.annotations)
        entry["callouts"] = [{"text": t, "ok": ok, "detail": why} for t, ok, why in rows]
        measured.append((view, spec, A, metrics["mean_mm"]))

    fits_well = all(v["metrics"]["drawn_covered"] >= 0.95 for v in out["views"] if v.get("metrics"))
    if measured and fits_well:
        out["pose"] = {"verdict": "the model as built fits every measured view; no other orientation tested",
                       "better_orientation_found": False}
    elif measured:
        try:
            out["pose"] = pose_check(shape, measured, 0.0)
        except Exception as error:
            out["pose"] = {"verdict": f"pose check failed: {error}", "better_orientation_found": False}
    out["seconds"] = round(time.monotonic() - started, 1)
    with open(os.path.join(out_dir, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    return out


if __name__ == "__main__":
    result = run(*sys.argv[1:5])
    for v in result["views"]:
        m = v.get("metrics")
        extra = f" [section of {v['parent_view']}, side {'known' if v['side_known'] else 'UNKNOWN'}, " \
                f"offset {v['cut_offset_mm']} mm]" if "parent_view" in v else ""
        print(f"{v['view_id']:36s} " + (f"h={v['horizontal']} v={v['vertical']} mean {m['mean_mm']} mm, "
                                        f"drawn covered {m['drawn_covered']:.0%}, missing {len(v['missing_regions'])}, "
                                        f"extra {len(v['extra_regions'])}" if m else v.get("unresolved") or v.get("note", ""))
              + extra)
        for c in v.get("callouts", []):
            if not c["ok"]:
                print(f"      callout failing: {c['text']}: {c['detail']}")
    print("pose:", (result["pose"] or {}).get("verdict"), f"({result['seconds']} s)")
