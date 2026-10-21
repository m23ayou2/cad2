"""Snap a sheet-metal plan's blank to the scan's own flat pattern.

On a scan the planner rebuilds the flat blank from the dimensions it reads, and gets its overall size right but
often its shape wrong (missing notches and corners, a panel cut short), so the folded part falls apart. The local
segmentation (src/local_parts.py, saved in the run's parts/ folder) has the flat pattern's panels, bend lines and
holes exactly, in PDF points. snap_blank puts them in the plan:

1. the blank outline is the union of the flat view's panels; its bend lines and hole centres come along;
2. it is fitted onto the plan: one of 8 turns or mirrors, a scale and a shift. With holes, on the holes: the scale
   from the diameters (exact callouts), turn and shift from the planned centres (read from dimensions). Without
   holes, on the planner's own profile: the scale from the two bounding boxes, a shift to the same corner;
3. the plan's base_flange profile becomes that outline, each bend's line the nearest local bend line, each hole's
   centre the nearest local hole of a similar size, and the fixed point a point inside the largest panel.

The planner's ops, bend directions, angles, radii, hole sizes and thickness are kept.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

TURNS = [np.array(m, float) for m in ([[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],
                                      [[-1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [0, -1]], [[0, -1], [-1, 0]])]
MIN_IOU = 0.5      # the fitted outline must overlap the planner's profile this much, else nothing is changed
MAX_ASPECT = 0.08  # the two bounding-box scales may differ by this share at most


def flat_pattern(parts_dir: Path) -> dict | None:
    """The cleanest flat view's blank in its own PDF points: {"view", "outline", "bends": [(p, q)], "holes":
    [(centre, d)], "panels": [polygon]}, or None when no flat view has panels."""
    views = json.loads((parts_dir / "views.json").read_text())
    flats = [v for v in views if v["kind"] not in ("isometric", "other")
             and any(p["feature"] == "panel" for p in v["parts"])]
    if not flats:
        return None
    # a view whose panels were moved from another is a copy; prefer the original
    v = min(flats, key=lambda v: (bool(v.get("transferred_from")), len(v["parts"])))
    panels = [Polygon(p["polygon_pt"]).buffer(0) for p in v["parts"] if p["feature"] == "panel"]
    # panels are a drawn line (and, after corrections, a pixel or two) apart: bridge gaps up to 3 pt
    union = unary_union([p.buffer(1.5, join_style=2) for p in panels]).buffer(-1.5, join_style=2)
    if union.geom_type != "Polygon":
        union = max(union.geoms, key=lambda g: g.area)
    outline = list(union.exterior.simplify(0.25).coords)[:-1]
    bends = []
    for p in v["parts"]:
        if p["feature"] == "bend":
            q = np.array(p["polygon_pt"], float)
            if len(q) == 4:       # a line drawn as a thin rectangle: its long middle line
                bends.append(((q[0] + q[3]) / 2, (q[1] + q[2]) / 2))
    holes = []
    for p in v["parts"]:
        if p["feature"] == "hole":
            q = np.array(p["polygon_pt"], float)
            c = q.mean(0)
            holes.append((c, 2 * float(np.linalg.norm(q - c, axis=1).mean())))
    return {"view": v["id"], "outline": np.array(outline), "bends": bends, "holes": holes, "panels": panels}


def fit(outline: np.ndarray, target: np.ndarray) -> dict | None:
    """The turn, scale and shift that best put `outline` (PDF points) on `target` (the planner's profile, mm)."""
    goal = Polygon(target).buffer(0)
    tmin, tmax = target.min(0), target.max(0)
    best = None
    for T in TURNS:
        pts = outline @ T.T
        lo, hi = pts.min(0), pts.max(0)
        sx, sy = (tmax - tmin) / np.maximum(hi - lo, 1e-9)
        if abs(sx / sy - 1) > MAX_ASPECT:
            continue
        s = (sx + sy) / 2
        shift = tmin - lo * s
        moved = Polygon(pts * s + shift).buffer(0)
        iou = moved.intersection(goal).area / moved.union(goal).area
        if best is None or iou > best["iou"]:
            best = {"T": T, "s": float(s), "shift": shift, "iou": float(iou), "by": "outline"}
    return best


def fit_holes(flat: dict, tree: dict) -> dict | None:
    """The turn, scale and shift that best put the drawn holes on the plan's holes: the scale from the diameters
    (callouts are exact), the turn and shift from the centres, each drawn hole paired with a planned one of a
    similar size. None when the plan and the drawing do not have the same two or more holes, or the best pairing
    leaves holes further apart than a tenth of the blank."""
    planned = [(np.array(h["position"][:2], float), float(h["params"]["d"])) for h in tree["features"]
               if h["op"] == "hole" and h.get("position") and h["params"].get("d")]
    drawn = flat["holes"]
    if len(planned) < 2 or len(planned) != len(drawn) or len(planned) > 7:
        return None
    s = float(np.median(np.array(sorted(d for _, d in planned)) / np.array(sorted(d for _, d in drawn))))
    size = s * float(np.linalg.norm(np.ptp(flat["outline"], axis=0)))
    best = None
    for T in TURNS:
        moved = [c @ T.T * s for c, _ in drawn]
        for perm in itertools.permutations(range(len(planned))):
            if any(abs(drawn[i][1] * s / planned[j][1] - 1) > 0.25 for i, j in enumerate(perm)):
                continue
            shift = np.mean([planned[j][0] - moved[i] for i, j in enumerate(perm)], axis=0)
            err = float(np.mean([np.linalg.norm(moved[i] + shift - planned[j][0]) for i, j in enumerate(perm)]))
            if best is None or err < best["error_mm"]:
                best = {"T": T, "s": s, "shift": shift, "error_mm": err, "by": "holes"}
    return best if best and best["error_mm"] < 0.1 * size else None


def snap_blank(tree: dict, parts_dir: Path) -> list[str]:
    """Snap the plan's blank, bends and holes to the scan's flat pattern, in place. Returns what was changed
    (empty when there is no flat pattern, no base_flange, or the fit is too poor to trust)."""
    flange = next((f for f in tree["features"] if f["op"] == "base_flange"), None)
    if flange is None or not (parts_dir / "views.json").exists():
        return []
    flat = flat_pattern(parts_dir)
    target = np.array(flange["params"].get("profile") or [], float)
    if flat is None or len(target) < 3:
        return []
    f = fit_holes(flat, tree) or fit(flat["outline"], target)
    if f is None or (f["by"] == "outline" and f["iou"] < MIN_IOU):
        return [f"blank not snapped: no hole fit, and the best fit of the {flat['view']} outline has IoU "
                f"{f['iou'] if f else 0:.2f} with the planned profile"]
    mm = lambda p: np.asarray(p, float) @ f["T"].T * f["s"] + f["shift"]
    r = lambda p: [round(float(x), 2) for x in p]
    how = (f"holes {f['error_mm']:.1f} mm from the planned centres on average" if f["by"] == "holes"
           else f"IoU {f['iou']:.2f} with the planned profile")
    notes = [f"blank snapped to the flat pattern in {flat['view']} (fitted on the {how}; {f['s']:.3f} mm per pt)"]
    flange["params"]["profile"] = [r(p) for p in mm(flat["outline"])]
    largest = max(flat["panels"], key=lambda p: p.area)
    flange["params"]["fixed"] = r(mm(np.array(largest.representative_point().coords[0])))

    lines = [(mm(p), mm(q)) for p, q in flat["bends"]]
    used = set()
    for b in (x for x in tree["features"] if x["op"] == "bend"):
        line = b["params"].get("line")
        if not line or not lines:
            continue
        mid = np.mean(np.array(line, float), axis=0)
        k = min((k for k in range(len(lines)) if k not in used),
                key=lambda k: np.linalg.norm((lines[k][0] + lines[k][1]) / 2 - mid), default=None)
        if k is None:
            continue
        used.add(k)
        p, q = lines[k]
        b["params"]["line"] = [r(p), r(q)]
        if b.get("position"):
            b["position"] = r([*((p + q) / 2), b["position"][2] if len(b["position"]) > 2 else 0.0])
        notes.append(f"{b['id']} line -> {b['params']['line']}")

    holes = [(mm(c), d * f["s"]) for c, d in flat["holes"]]
    used = set()
    for h in (x for x in tree["features"] if x["op"] == "hole" and x.get("position")):
        d = float(h["params"].get("d") or 0)
        at = np.array(h["position"][:2], float)
        cands = [k for k in range(len(holes)) if k not in used and (not d or abs(holes[k][1] / d - 1) < 0.25)]
        k = min(cands, key=lambda k: np.linalg.norm(holes[k][0] - at), default=None)
        if k is None:
            continue
        used.add(k)
        h["position"] = r([*holes[k][0], h["position"][2] if len(h["position"]) > 2 else 0.0])
        notes.append(f"{h['id']} centre -> {h['position'][:2]} (drawn Ø{holes[k][1]:.1f})")

    for a in tree.get("assumptions", []):
        if any(c.startswith(flange["id"]) for c in a.get("covers", [])):
            a["text"] += f" [harness: {notes[0]}]"
    return notes
