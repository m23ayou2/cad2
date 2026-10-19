"""Compare a built STEP with the views of a scanned drawing (the scans read by scan_view).

    python src/raster_check.py <drawing.pdf> <model.step> <scans.json> <out folder>

The same check as validate.py, for sheets with no vector content. For each scanned view the model is projected
from that view's direction (hidden lines removed, cut through the centre for a section, since a scan gives no cut
position) and compared with the view's part edges as separated by raster_lines (annotations removed). Scale:
- orthographic views and sections use the scan's measured mm per pixel (from its dimension values) when there is
  one, otherwise the scale that fits the model's size (marked as fitted, not measured);
- a scanned view's direction and its turn on the sheet are what the planner said, which can be wrong (a view along
  a turned part's axis taken for a front view, a section drawn with the axis horizontal): orthographic views and
  sections also try the six standard directions, each turned by 0, 90, 180 and 270 degrees on the sheet, and keep
  the best fit, reporting the given orientation's fit alongside;
- isometric views try the four standard isometric cameras on the front side and keep the best, at the scale that
  fits the model's size;
- flat patterns are not compared: the model is formed.

Writes validation.json in validate.py's format (views with metrics, missing / extra regions, images, points), so
the validation agents and the correction loop work unchanged.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

import pose_solver as ps
import raster_lines as rl
import validate as vc
from views import ViewDirection

ORTHOGRAPHIC = {"front", "top", "side", "section", "detail"}
TIME_BUDGET_S = 420     # the harness gives the whole check 600 s: views left when this is spent are skipped


def shaded(gray: np.ndarray) -> bool:
    """A shaded render (an isometric picture), not a line drawing: most of its ink is mid-grey, not black strokes."""
    ink = gray < 235
    return bool(ink.sum()) and float(((gray > 70) & (gray < 215))[ink].mean()) > 0.5


def part_points(gray: np.ndarray, cache: str | None = None) -> np.ndarray:
    """The view's part edges, as points in pixels with y up (annotations and centre lines removed). A shaded render
    has no strokes to separate: its edges are where the shading changes sharply (Canny). The separation takes a
    minute on a crowded view, so the mask is kept in `cache` (the scan's folder) for the next rounds."""
    path = os.path.join(cache, "check_part.png") if cache else None
    if path and os.path.isfile(path):
        part = cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 0
    elif shaded(gray):
        part = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 110) > 0
    else:
        sep = rl.separate_structure(gray)
        axes, _ = rl.dashed_runs(sep["part"])
        part = sep["part"] & ~axes
    if path and not os.path.isfile(path) and part.shape == gray.shape:
        cv2.imwrite(path, part.astype(np.uint8) * 255)
    ys, xs = np.nonzero(part)
    pts = np.stack([xs, -ys], 1).astype(float)
    if len(pts) > 20000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
    return pts


def sample(points: np.ndarray, n: int) -> np.ndarray:
    return points[np.random.default_rng(0).choice(len(points), n, replace=False)] if len(points) > n else points


def quick_fit(drawn: np.ndarray, model: np.ndarray, tol: float) -> tuple[float, np.ndarray]:
    """A fast fit for ranking candidate cameras and scales (validate.compare takes seconds): both point sets
    sampled, a wide coarse search for the offset (a dimension stack on one side moves the drawing's box far off the
    part's), a few median steps, then the share of each set within tol of the other, summed (0 to 2), and the
    offset found."""
    a, b = sample(drawn, 600), sample(model, 3000)
    tree = cKDTree(b)
    size = extent(b) or 1.0
    t = (b.min(0) + b.max(0)) / 2 - (a.min(0) + a.max(0)) / 2
    # every candidate offset scored at once: a distance map of the model's edges, looked up at the shifted points
    cell = max(size / 400, 1e-6)
    lo = b.min(0) - 0.5 * size
    shape = (np.ceil((b.max(0) - lo + 0.5 * size) / cell).astype(int) + 1)[::-1]
    img = np.ones(shape, np.uint8)
    ij = ((b - lo) / cell).astype(int)
    img[ij[:, 1], ij[:, 0]] = 0
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 3) * cell
    def costs(offsets: np.ndarray) -> np.ndarray:
        p = ((a[None, :, :] + t + offsets[:, None, :] - lo) / cell).astype(int)
        p[..., 0] = p[..., 0].clip(0, shape[1] - 1)
        p[..., 1] = p[..., 1].clip(0, shape[0] - 1)
        return np.median(dist[p[..., 1], p[..., 0]], axis=1)
    grid = np.linspace(-0.35 * size, 0.35 * size, 29)
    offsets = np.array([(dx, dy) for dx in grid for dy in grid])
    best = offsets[int(np.argmin(costs(offsets)))]
    step = grid[1] - grid[0]
    fine = np.array([(dx, dy) for dx in np.linspace(-step, step, 7) for dy in np.linspace(-step, step, 7)]) + best
    t = t + fine[int(np.argmin(costs(fine)))]
    for _ in range(6):
        d, j = tree.query(a + t)
        keep = d < max(np.percentile(d, 60), 0.01 * size)
        if keep.sum() < 10:
            break
        t = t + np.median(b[j[keep]] - (a[keep] + t), axis=0)
    a = a + t
    return float((tree.query(a)[0] <= tol).mean() + (cKDTree(a).query(b)[0] <= tol).mean()), t


def extent(points: np.ndarray) -> float:
    return float(np.max(points.max(0) - points.min(0))) if len(points) else 0.0


def check_view(shape, scan: dict, gray: np.ndarray, sheet_px_per_mm: float | None = None
               ) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    kind, direction = scan.get("kind", "unknown"), scan.get("direction", "none")
    if kind == "detail":
        return {"unresolved": "detail view: a magnified piece of the part, not comparable with the whole model's "
                              "projection; judge it by eye"}, None, None
    drawn_px = part_points(gray, scan.get("folder"))
    if len(drawn_px) < 50:
        return {"unresolved": "too few part edges found in this scan"}, None, None
    candidates = []
    if kind in ORTHOGRAPHIC and direction in ViewDirection.__members__:
        # the given direction first, then every standard one: each is also tried turned on the sheet (below)
        for name in [direction] + [n for n in ("FRONT", "REAR", "TOP", "BOTTOM", "LEFT", "RIGHT") if n != direction]:
            d = ViewDirection[name]
            spec = {"look": tuple(d.look), "up": tuple(d.up), "given": name == direction}
            if kind == "section":
                spec["cut_normal"] = tuple(-np.array(d.look))
            candidates.append((f"{name.lower()} {'section' if kind == 'section' else 'view'}", spec))
    elif kind == "isometric":
        for sz, height in ((-1, "above"), (1, "below")):
            for sx, side in ((1, "left"), (-1, "right")):
                look = (sx / math.sqrt(3), 1 / math.sqrt(3), sz / math.sqrt(3))
                candidates.append((f"isometric, front-{side}, from {height}", {"look": look, "up": (0, 0, 1)}))
    else:
        reason = "flat pattern: the model is formed, so it is not compared" if kind == "flat_pattern" else \
            f"no direction for this view (kind {kind!r}, direction {direction!r})"
        return {"unresolved": reason}, None, None

    best, given = None, None
    turns = (0, 90, 180, 270) if kind in ORTHOGRAPHIC else (0,)
    for caption, spec in candidates:
        try:
            projected = ps.model_view_points(shape, spec)
        except ValueError:                    # a section cut that removes nothing from this side
            continue
        if len(projected) == 0:
            continue
        px_per_mm = scan.get("px_per_mm") if kind in ORTHOGRAPHIC else None
        hows = ["measured"] if px_per_mm else (["sheet"] if sheet_px_per_mm else []) + [
            # leaders and centre lines beyond the outline only make the drawing look larger: try larger scales too
            f"fitted*{f}" for f in (1.0, 1.1, 1.2, 1.35, 1.5)]
        for turn, how in ((t, h) for t in turns for h in hows):
            # the view drawn turned on the sheet by `turn` degrees anticlockwise
            a = math.radians(turn)
            rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]).round(12)
            model = projected @ rot.T
            # the scan's own scale; else the sheet's (views are usually drawn at one scale), or one fitted to the
            # model's size: leaders and centre lines beyond the outline make the fitted one too small
            scale = (1 / px_per_mm if how == "measured" else 1 / sheet_px_per_mm if how == "sheet" else
                     float(how.split("*")[1]) * extent(model) / max(extent(drawn_px), 1e-6))
            measured_scale = not how.startswith("fitted")
            drawn = drawn_px * scale
            tol = max(1.5 * scale, 0.005 * extent(model))   # a scanned edge is accurate to about a pixel
            # rank the candidates quickly (a full comparison takes seconds); the best one is compared in full below
            score, offset = quick_fit(drawn, model, tol)
            label = caption + (f", drawn turned {turn}°" if turn else "")
            found = (score, offset, drawn, model, label, spec, measured_scale, scale, rot, tol)
            if best is None or score > best[0]:
                best = found
            if spec.get("given") and turn == 0 and (given is None or score > given[0]):
                given = found
    if best is None:
        return {"unresolved": "the model projection is empty"}, None, None
    _, offset, drawn, model, caption, spec, measured_scale, scale, rot, tol = best
    metrics, missing, extra, aligned = vc.compare(drawn, sample(model, 12000), tol=tol, start=offset)
    b = ps.view_basis(spec["look"], spec["up"])
    basis = [rot[0, 0] * b[0] + rot[0, 1] * b[1], rot[1, 0] * b[0] + rot[1, 1] * b[1]]
    entry = {"metrics": metrics, "missing_regions": missing, "extra_regions": extra,
             "horizontal": vc.axis_name(basis[0]), "vertical": vc.axis_name(basis[1]),
             "looking_along": vc.axis_name(np.array(spec["look"], float)),
             "camera": caption, "scale_mm_per_px": round(scale, 5),
             "note": ("scanned view: part edges separated by image processing; scale "
                      + ("measured from the dimension values" if measured_scale and scan.get("px_per_mm") else
                         "the sheet's, measured on another view" if measured_scale else
                         "fitted to the model's size (not measured: sizes are not checked, only shape)")
                      + (". Sections are cut through the part's centre: a scan gives no cut position"
                         if kind == "section" else "")
                      + (". Edges of a shaded render, found where the shading changes: shading bands give extra "
                         "edges, so only the outline and the main edges count" if shaded(gray) else ""))}
    if given is not None and given is not best:
        entry["orientation"] = {
            "given": f"{direction.lower()}, not turned", "fits_better": caption,
            "given_score": round(given[0] / 2, 3), "best_score": round(best[0] / 2, 3),
            "note": "The model fits this scan better from another direction or turned on the sheet. Either the scan's "
                    "kind/direction was given wrong (common on scans: a view along a turned part's axis, a section "
                    "drawn with the axis horizontal), or the model is built in the wrong orientation. The comparison "
                    "above uses the better fit."}
    return entry, aligned, model


def run(drawing: str, step: str, scans_json: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    shape = ps.load_step(step)
    scans = json.load(open(scans_json, encoding="utf-8"))
    measured = [s["px_per_mm"] for s in scans if s.get("px_per_mm") and s.get("kind") in ORTHOGRAPHIC]
    sheet_px_per_mm = float(np.median(measured)) if measured else None
    out = {"step": step, "views": [], "scanned": True,
           "pose": {"verdict": "not checked on a scanned drawing", "better_orientation_found": False}}
    started = time.monotonic()
    for s in scans:
        entry = {"view_id": s["id"], "class": s.get("kind"), "label": s.get("label") or s.get("kind")}
        out["views"].append(entry)
        if time.monotonic() - started > TIME_BUDGET_S:
            entry["unresolved"] = f"not compared: the check ran out of time ({TIME_BUDGET_S} s for all views)"
            continue
        gray = rl.render(drawing, s["box"])
        stem = os.path.join(out_dir, s["id"])
        cv2.imwrite(stem + ".drawing.png", gray)
        try:
            result, drawn, model = check_view(shape, s, gray, sheet_px_per_mm)
        except Exception as error:
            entry["unresolved"] = f"projection failed: {error}"
            continue
        entry.update(result)
        if drawn is None:
            continue
        entry["images"] = {"drawing": stem + ".drawing.png", "model": stem + ".model.png",
                           "overlay": stem + ".overlay.png"}
        vc.plot(stem + ".model.png", None, model, f"{entry['label']}: model, {entry['camera']}",
                entry["horizontal"], entry["vertical"])
        vc.plot(stem + ".overlay.png", drawn, model, f"{entry['label']}: scanned edges (black) vs model (red); "
                "orange = drawn, missing in model; purple = in model, not drawn",
                entry["horizontal"], entry["vertical"], entry["missing_regions"], entry["extra_regions"])
        np.savez_compressed(stem + ".points.npz", drawing=drawn, model=model)
        entry["points"] = stem + ".points.npz"
        entry["callouts"] = []
    with open(os.path.join(out_dir, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    return out


if __name__ == "__main__":
    result = run(*sys.argv[1:5])
    for v in result["views"]:
        m = v.get("metrics")
        print(f"{v['view_id']:6s} {v.get('class') or '':12s} " + (
            f"{v['camera']}: mean {m['mean_mm']} mm, drawn covered {m['drawn_covered']:.0%}, model covered "
            f"{m['model_covered']:.0%}" if m else v.get("unresolved", "")))
