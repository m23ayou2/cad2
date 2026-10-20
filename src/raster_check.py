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
- flat patterns are not compared: the model is formed;
- a view of kind "auto" (cropped from the sheet, not labelled) tries every standard direction, turned on the
  sheet, and every isometric camera, and is reported as the kind that fits best.

How the fit is judged (the scene-matching method of GeoInt's terrain and pose tools, adapted):
- the scale is fixed when it is known (measured on the view, or on another view of the sheet); only a view with no
  measured scale searches it;
- separability: the best camera's score is reported with the runner-up's and the gap between them, and the fit in
  pixels; a gap within the noise means the view cannot tell the candidates apart (`separation.decided` false);
- landmarks: hole centres found in the drawing are matched to the model's cylinders seen end-on (same diameter),
  which places the drawing on the model more precisely than its edges; each match is checked by leaving it out
  (a misread or misplaced hole stands out), and the landmark offset starts the final alignment;
- model edges under annotations (dimension, extension and leader lines, text) are not counted as missing: they are
  hidden on the drawing, not absent from it;
- images for the validators are filled, not dotted: the drawing's regions in colour, the model's projection filled
  into faces in the same pixel frame, their silhouettes overlaid (both / drawn only / model only), and, with
  landmarks, a match sheet with a line per matched hole.

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
import regions as rg
import step_checks
import validate as vc
from views import ViewDirection

ORTHOGRAPHIC = {"front", "top", "side", "section", "detail"}
TIME_BUDGET_S = 420     # the harness gives the whole check 600 s: views left when this is spent are skipped


def shaded(gray: np.ndarray) -> bool:
    """A shaded render (an isometric picture), not a line drawing: most of its ink is mid-grey, not black strokes."""
    ink = gray < 235
    return bool(ink.sum()) and float(((gray > 70) & (gray < 215))[ink].mean()) > 0.5


def edges_and_annotations(gray: np.ndarray, cache: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The view's part edges (annotations and centre lines removed) and its annotations (dimension, extension and
    leader lines, arrowheads, text), as masks. A shaded render has no strokes to separate: its edges are where the
    shading changes sharply (Canny) and it has no annotations. The separation takes up to a minute on a crowded
    view, so both masks are kept in `cache` (the scan's folder) for the next rounds."""
    part_path = os.path.join(cache, "check_part.png") if cache else None
    annot_path = os.path.join(cache, "check_annot.png") if cache else None
    if part_path and annot_path and os.path.isfile(part_path) and os.path.isfile(annot_path):
        part = cv2.imread(part_path, cv2.IMREAD_GRAYSCALE) > 0
        annot = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE) > 0
    elif shaded(gray):
        part = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 110) > 0
        annot = np.zeros_like(part)
    else:
        sep = rl.separate_structure(gray)
        axes, _ = rl.dashed_runs(sep["part"])
        part = sep["part"] & ~axes
        annot = sep["dimension"] | sep["extension"] | sep["text"]      # arrowheads are in the dimension layer
    if part.shape != gray.shape:
        return part, np.zeros_like(part)
    if part_path:
        cv2.imwrite(part_path, part.astype(np.uint8) * 255)
        cv2.imwrite(annot_path, annot.astype(np.uint8) * 255)
    return part, annot


def part_points(part: np.ndarray) -> np.ndarray:
    """Part edge pixels as points with y up."""
    ys, xs = np.nonzero(part)
    pts = np.stack([xs, -ys], 1).astype(float)
    if len(pts) > 20000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
    return pts


def to_pixels(points_mm: np.ndarray, offset: np.ndarray, scale: float) -> np.ndarray:
    """View-plane mm (as aligned: drawing px * scale + offset) back to the drawing's pixels (x right, y down)."""
    p = (points_mm - offset) / scale
    return np.stack([p[:, 0], -p[:, 1]], 1)


# --------------------------------------------------------------------------- landmarks
def drawn_holes(part: np.ndarray) -> list[dict]:
    """Circles on the drawing's part edges (holes, bores, bosses seen end-on), in pixels."""
    import raster_vector
    return raster_vector.find_holes(part)


def model_holes(shape, spec: dict, rot: np.ndarray) -> list[dict]:
    """The model's cylinders seen end-on from this camera (axis along the view direction): their centres in the
    view plane (same frame as the projected edges, after the turn `rot`) and diameters."""
    look = np.array(spec["look"], float)
    look /= np.linalg.norm(look)
    b = ps.view_basis(spec["look"], spec["up"])
    out = []
    for c in step_checks.cylinder_inventory(shape):
        if abs(float(np.dot(c["axis"], look))) < 0.98:
            continue
        p = np.array(c["key"][2], float)
        centre = rot @ np.array([p @ b[0], p @ b[1]])
        out.append({"centre": centre, "d": 2 * c["radius"]})
    return out


def register_landmarks(drawn: list[dict], model: list[dict], scale: float, tol: float) -> dict | None:
    """Match drawn hole centres to model hole centres of the same diameter by one offset (RANSAC over pairs), then
    check each match by leaving it out: the offset from the other matches predicts where that hole should be, and a
    hole far from its prediction is suspect (misread, misplaced, or a different hole). None if fewer than two match."""
    if len(drawn) < 2 or len(model) < 2:
        return None
    dpts = [(np.array([h["centre"][0], -h["centre"][1]]) * scale, h["d_px"] * scale, h) for h in drawn]
    same = lambda dd, md: abs(dd - md) <= max(0.15 * md, 1.0)
    best = None
    for dp, dd, _ in dpts:
        for m in model:
            if not same(dd, m["d"]):
                continue
            t = m["centre"] - dp
            pairs, used = [], set()
            for i, (dp2, dd2, _) in enumerate(dpts):
                cands = [(float(np.linalg.norm(dp2 + t - m2["centre"])), j) for j, m2 in enumerate(model)
                         if j not in used and same(dd2, m2["d"])]
                if cands:
                    dist, j = min(cands)
                    if dist <= 3 * tol:
                        pairs.append((i, j)); used.add(j)
            if best is None or len(pairs) > len(best):
                best = pairs
    if not best or len(best) < 2:
        return None
    diffs = np.array([model[j]["centre"] - dpts[i][0] for i, j in best])
    offset = np.median(diffs, axis=0)
    matches = []
    for k, (i, j) in enumerate(best):
        others = np.delete(diffs, k, axis=0)
        t_k = np.median(others, axis=0)
        residual = float(np.linalg.norm(dpts[i][0] + t_k - model[j]["centre"]))
        matches.append({"drawn_px": [round(v, 1) for v in dpts[i][2]["centre"]], "drawn_d_mm": round(dpts[i][1], 2),
                        "model_mm": [round(float(v), 2) for v in model[j]["centre"]], "model_d_mm": round(model[j]["d"], 2),
                        "residual_mm": round(residual, 2), "suspect": residual > 3 * tol})
    rms = float(np.sqrt(np.mean([np.linalg.norm(dpts[i][0] + offset - model[j]["centre"]) ** 2 for i, j in best])))
    return {"offset": offset, "matches": matches, "rms_mm": round(rms, 3),
            "drawn_holes": len(drawn), "model_holes": len(model)}


# --------------------------------------------------------------------------- filled images
def raster_model(polylines: list, offset: np.ndarray, scale: float, shape: tuple) -> np.ndarray:
    """The model's projection drawn straight from the STEP into the drawing's pixel frame: each visible edge (hidden
    lines removed) as a continuous line, like a drawing view (white paper, black lines)."""
    img = np.full(shape, 255, np.uint8)
    for line in polylines:
        if len(line) < 2:
            continue
        px = np.round(to_pixels(line, offset, scale) * 4).astype(np.int32)       # 2 bits of sub-pixel precision
        cv2.polylines(img, [px], False, 0, 2, cv2.LINE_AA, shift=2)
    return img


def silhouette(regions_list: list[dict], shape: tuple) -> np.ndarray:
    """The union of regions, with the strokes between them: the area the part covers in this view."""
    m = np.zeros(shape, np.uint8)
    for r in regions_list:
        m |= r["mask"].astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0


def filled_comparison(gray: np.ndarray, annot: np.ndarray, model: list, offset: np.ndarray, scale: float,
                      stem: str, landmarks: dict | None) -> tuple[dict, dict]:
    """Filled images for the validators and the silhouette metrics. Returns (images, metrics)."""
    drawn_regions = rg.fill_regions(gray)
    rg.annotation_share(gray, drawn_regions, annot)
    # paper boxed in by dimension lines is not part of the part; but on a view with hardly any annotations (an
    # isometric) a long thin edge can pass for one, so there every region counts
    ink = gray < rg.INK
    dimensioned = annot[ink].mean() > 0.25 if ink.any() else False
    part_regions = [r for r in drawn_regions if not dimensioned or r["annotation_share"] < 0.6]
    model_ink = raster_model(model, offset, scale, gray.shape)
    model_regions = rg.fill_regions(model_ink)
    images = {"drawing": stem + ".drawing_filled.png", "model": stem + ".model_filled.png",
              "overlay": stem + ".overlay_filled.png"}
    cv2.imwrite(images["drawing"], rg.label_image(gray, drawn_regions))
    cv2.imwrite(images["model"], rg.label_image(model_ink, model_regions))
    ds, ms = silhouette(part_regions, gray.shape), silhouette(model_regions, gray.shape)
    over = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(float)
    over = 255 - (255 - over) * 0.5
    for mask, colour in ((ds & ms, (120, 200, 120)), (ds & ~ms, (60, 160, 255)), (ms & ~ds, (220, 120, 200))):
        over[mask] = over[mask] * 0.35 + np.array(colour) * 0.65
    over = over.astype(np.uint8)
    over[model_ink < 128] = (150, 0, 150)
    rg._text(over, "green: both   orange: drawn, not in model   purple: in model, not drawn", (8, 22), 0.5)
    cv2.imwrite(images["overlay"], over)
    union = int((ds | ms).sum())
    metrics = {"silhouette_iou": round(float((ds & ms).sum()) / union, 3) if union else None,
               "drawn_only_share": round(float((ds & ~ms).sum()) / max(1, int(ds.sum())), 3),
               "model_only_share": round(float((ms & ~ds).sum()) / max(1, int(ms.sum())), 3),
               "drawn_faces": len(part_regions), "model_faces": len(model_regions)}
    if landmarks and landmarks.get("matches"):
        left = rg.label_image(gray, drawn_regions)
        right = rg.label_image(model_ink, model_regions)
        sheet = np.hstack([left, np.full((gray.shape[0], 20, 3), 255, np.uint8), right])
        for k, m in enumerate(landmarks["matches"]):
            colour = tuple(int(c) for c in rg.PALETTE[k % len(rg.PALETTE)][::-1]) if not m["suspect"] else (0, 0, 230)
            a = tuple(int(v) for v in m["drawn_px"])
            b = to_pixels(np.array([m["model_mm"]], float), offset, scale)[0]
            b = (int(b[0]) + gray.shape[1] + 20, int(b[1]))
            cv2.circle(sheet, a, 7, colour, 2)
            cv2.circle(sheet, b, 7, colour, 2)
            cv2.line(sheet, a, b, colour, 1, cv2.LINE_AA)
        images["matches"] = stem + ".matches.png"
        cv2.imwrite(images["matches"], sheet)
    return images, metrics


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
    part, annot = edges_and_annotations(gray, scan.get("folder"))
    drawn_px = part_points(part)
    if len(drawn_px) < 50:
        return {"unresolved": "too few part edges found in this scan"}, None, None
    candidates = []
    card = scan.get("card_camera")
    if card and kind not in ("flat_pattern", "isometric"):
        # the frame card fixes this view's camera: compare from exactly that camera. A free search would let a
        # wrong orientation of the whole part pass by finding a camera that hides it
        spec = {"look": tuple(card["look"]), "up": tuple(card["up"]), "turns": (0,), "given": True,
                "source": "card"}
        if kind == "section":
            spec["cut_normal"] = tuple(-np.array(card["look"], float))
        candidates.append((f"the frame card's camera (looking along {card.get('looks_along', card['look'])})", spec))
    elif kind == "auto":
        # nobody said what this view is: every standard direction (turned on the sheet) and every isometric camera
        for name, cls in (("FRONT", "front"), ("REAR", "front"), ("TOP", "top"), ("BOTTOM", "top"),
                          ("LEFT", "side"), ("RIGHT", "side")):
            d = ViewDirection[name]
            candidates.append((f"{name.lower()} view", {"look": tuple(d.look), "up": tuple(d.up), "class": cls,
                                                        "turns": (0, 90, 180, 270)}))
        for sz, height in ((-1, "above"), (1, "below")):
            for sx, side in ((1, "left"), (-1, "right")):
                look = (sx / math.sqrt(3), 1 / math.sqrt(3), sz / math.sqrt(3))
                candidates.append((f"isometric, front-{side}, from {height}",
                                   {"look": look, "up": (0, 0, 1), "class": "isometric", "turns": (0,)}))
    elif kind in ORTHOGRAPHIC and direction in ViewDirection.__members__:
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

    best, given, per_camera = None, None, {}
    turns = (0, 90, 180, 270) if kind in ORTHOGRAPHIC else (0,)
    for caption, spec in candidates:
        try:
            projected = ps.model_view_points(shape, spec)
        except ValueError:                    # a section cut that removes nothing from this side
            continue
        if len(projected) == 0:
            continue
        px_per_mm = scan.get("px_per_mm") if kind in ORTHOGRAPHIC else None
        # a known scale is used as it is: searching it as well lets a wrong camera absorb its error through the scale
        hows = (["measured"] if px_per_mm else ["sheet"] if sheet_px_per_mm else
                # no scale at all: leaders and centre lines beyond the outline only make the drawing look larger
                [f"fitted*{f}" for f in (1.0, 1.1, 1.2, 1.35, 1.5)])
        for turn, how in ((t, h) for t in spec.get("turns", turns) for h in hows):
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
            per_camera[label] = max(per_camera.get(label, 0.0), score)
            if spec.get("given") and turn == 0 and (given is None or score > given[0]):
                given = found
    if best is None:
        return {"unresolved": "the model projection is empty"}, None, None
    _, offset, drawn, model, caption, spec, measured_scale, scale, rot, tol = best
    # landmarks (holes seen end-on) place the drawing on the model more precisely than its edges
    landmarks = None
    if spec.get("class", kind) != "isometric" and measured_scale:     # positions mean nothing at a guessed scale
        landmarks = register_landmarks(drawn_holes(part), model_holes(shape, spec, rot), scale, tol)
        if landmarks:
            # use their offset only when enough holes agree and it fits the edges at least as well as the edges' own
            agree = len(landmarks["matches"]) >= 3 and landmarks["rms_mm"] <= 2 * tol
            tree = cKDTree(sample(model, 4000))
            cover = lambda t: float((tree.query(sample(drawn, 3000) + t)[0] <= tol).mean())
            better = agree and cover(landmarks["offset"]) >= cover(offset) - 0.02
            landmarks["used"] = bool(better)
            landmarks["note"] = ("holes matched to the model's end-on cylinders; their common offset placed the "
                                 "drawing" if better else "too few holes agree, or their offset fits the edges "
                                 "worse: the drawing was placed by its edges")
            if better:
                offset = landmarks["offset"]
    # model edges hidden under annotations on the drawing are not missing from it
    px = to_pixels(model, offset, scale).round().astype(int)
    inside = (px[:, 0] >= 0) & (px[:, 0] < annot.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < annot.shape[0])
    hidden = np.zeros(len(model), bool)
    near = cv2.dilate(annot.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    hidden[inside] = near[px[inside, 1], px[inside, 0]] & ~cv2.dilate(part.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)[px[inside, 1], px[inside, 0]]
    visible = model[~hidden] if (~hidden).sum() > 50 else model
    metrics, missing, extra, aligned = vc.compare(drawn, sample(visible, 12000), tol=tol, start=offset)
    final_offset = (aligned[0] - drawn[0]) if len(drawn) else offset
    ranked = sorted(per_camera.items(), key=lambda kv: -kv[1])
    runner = next(((lab, sc) for lab, sc in ranked if lab != caption), None)
    fit_px = metrics["mean_mm"] / scale
    separation = {"best": caption, "best_score": round(ranked[0][1] / 2, 3) if ranked else None,
                  "runner_up": runner[0] if runner else None, "runner_up_score": round(runner[1] / 2, 3) if runner else None,
                  "gap": round((ranked[0][1] - runner[1]) / 2, 3) if runner else None,
                  "fit_px": round(fit_px, 2)}
    separation["decided"] = bool(runner is None or separation["gap"] >= 0.05)
    separation["note"] = ("scores are the mean of the drawn and model coverage (0..1); the gap is how far the best "
                          "camera is ahead of the next one. Not decided: this view cannot tell those cameras apart, so "
                          "it says nothing about the orientation" if not separation["decided"] else
                          "the best camera is clearly ahead of the others")
    b = ps.view_basis(spec["look"], spec["up"])
    basis = [rot[0, 0] * b[0] + rot[0, 1] * b[1], rot[1, 0] * b[0] + rot[1, 1] * b[1]]
    stem = scan.get("_stem")
    images, filled = {}, {}
    if stem:   # the projection again, as lines this time, turned like the drawing
        lines = [np.asarray(l) @ rot.T for l in ps.model_view_points(shape, spec, polylines=True)]
        images, filled = filled_comparison(gray, annot, lines, final_offset, scale, stem, landmarks)
    metrics = {**metrics, **filled, "hidden_by_annotations": round(float(hidden.mean()), 3)}
    entry = {"metrics": metrics, "missing_regions": missing, "extra_regions": extra, "separation": separation,
             **({"landmarks": {k: v for k, v in landmarks.items() if k != "offset"}} if landmarks else {}),
             "_images": images,
             "horizontal": vc.axis_name(basis[0]), "vertical": vc.axis_name(basis[1]),
             "looking_along": vc.axis_name(np.array(spec["look"], float)),
             "camera": caption, "scale_mm_per_px": round(scale, 5),
             "camera_source": spec.get("source", "search"),
             **({"class": spec["class"]} if spec.get("class") else {}),
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


# --------------------------------------------------------------------------- one pose for the whole part
def rotations() -> list[np.ndarray]:
    """The 24 axis-aligned rotations of the part (proper rotations: no mirror)."""
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, (j, sg) in enumerate(zip(perm, signs)):
                m[i, j] = sg
            if np.linalg.det(m) > 0:
                out.append(m)
    return out


def describe_rotation(r: np.ndarray) -> str:
    names = ["+X", "+Y", "+Z"]
    where = lambda v: ("+" if v[int(np.argmax(np.abs(v)))] > 0 else "-") + "XYZ"[int(np.argmax(np.abs(v)))]
    moved = [f"{n} -> {where(r @ np.eye(3)[i])}" for i, n in enumerate(names) if where(r @ np.eye(3)[i]) != n]
    return "turn the whole part so that " + ", ".join(moved)


def joint_pose(shape, views: list[dict], budget_s: float) -> dict:
    """The orientation of the whole part. The frame card fixes each orthographic view's camera relative to the
    part, and the model is built in the card, so card and model turn together: the orthographic views only check
    that the model follows the card (their own comparison above). What fixes the part's absolute orientation is the
    isometric views, drawn by convention from a camera with Z up: the part is tried in each of the 24 axis-aligned
    rotations (the card turning with it) and scored on the isometrics, each from the best of the four standard
    isometric cameras. The card's pose stands unless another rotation is clearly better and better on every
    isometric (GeoInt's robust camera check: no single view decides)."""
    started = time.monotonic()
    isos = [v for v in views if v["kind"] == "isometric"]
    if not isos:
        return {"verdict": "not checked: no isometric view fixes the absolute orientation",
                "better_orientation_found": False}
    memo = {}

    def project(look, up):
        key = (tuple(np.round(look, 4)), tuple(np.round(up, 4)))
        if key not in memo:
            try:
                memo[key] = sample(ps.model_view_points(shape, {"look": tuple(look), "up": tuple(up)}), 4000)
            except Exception:
                memo[key] = np.zeros((0, 2))
        return memo[key]

    def view_score(v, look, up):
        model = project(look, up)
        if len(model) < 20:
            return 0.0
        best = 0.0
        for f in (1.0, 1.15, 1.3, 1.5):
            scale = f * extent(model) / max(extent(v["drawn"]), 1e-6)
            tol = max(1.5 * scale, 0.005 * extent(model))
            best = max(best, quick_fit(sample(v["drawn"], 1500) * scale, model, tol)[0] / 2)
        return best

    cams = [(np.array((sx, 1, sz)) / math.sqrt(3), np.array((0, 0, 1.0))) for sx in (-1, 1) for sz in (-1, 1)]
    rows = []
    for r in rotations():
        if time.monotonic() - started > budget_s:
            return {"verdict": "not checked: out of time", "better_orientation_found": False}
        # the part turned by r, seen from the fixed standard cameras = the part seen from cameras turned by r^T
        rows.append((r, {v["id"]: max(view_score(v, r.T @ lk, r.T @ up) for lk, up in cams) for v in isos}))
    # a view whose score hardly changes whichever way the part is turned (a dark shaded underside, say) cannot tell
    # orientations apart: it gets no vote. The others must all agree.
    spread = {v["id"]: max(row[1][v["id"]] for row in rows) - min(row[1][v["id"]] for row in rows) for v in isos}
    voters = [k for k, sp in spread.items() if sp >= 0.10] or list(spread)
    mean = lambda per: float(np.mean([per[k] for k in voters]))
    card_row = next(row for row in rows if np.allclose(row[0], np.eye(3)))
    best_r, best = max(rows, key=lambda row: mean(row[1]))
    gain = mean(best) - mean(card_row[1])
    every = all(best[k] - card_row[1][k] >= 0.05 for k in voters)
    found = (not np.allclose(best_r, np.eye(3))) and gain >= 0.10 and every
    result = {"verdict": (describe_rotation(best_r) + ": the isometric views fit clearly better" if found else
                          "the frame card's orientation fits the isometric views best" if np.allclose(best_r, np.eye(3))
                          else "another orientation scores a little better, but not clearly or not on every "
                               "isometric: the card stands"),
              "better_orientation_found": bool(found),
              "card_pose_score": round(mean(card_row[1]), 3), "best_pose_score": round(mean(best), 3),
              "card_pose_per_view": {k: round(v, 3) for k, v in card_row[1].items()},
              "best_pose_per_view": {k: round(v, 3) for k, v in best.items()},
              "views_used": voters, "views_without_a_vote": [k for k in spread if k not in voters],
              "seconds": round(time.monotonic() - started, 1)}
    if found:
        result["rotation"] = best_r.round().astype(int).tolist()
        result["rotation_text"] = describe_rotation(best_r)
        result["supporting_views"] = voters
    return result


def run(drawing: str, step: str, scans_json: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    shape = ps.load_step(step)
    scans = json.load(open(scans_json, encoding="utf-8"))
    measured = [s["px_per_mm"] for s in scans if s.get("px_per_mm") and s.get("kind") in ORTHOGRAPHIC]
    sheet_px_per_mm = float(np.median(measured)) if measured else None
    out = {"step": step, "views": [], "scanned": True,
           "pose": {"verdict": "not checked on a scanned drawing", "better_orientation_found": False}}
    started = time.monotonic()
    joint = []
    for s in scans:
        entry = {"view_id": s["id"], "class": s.get("kind"), "label": s.get("label") or s.get("kind")}
        out["views"].append(entry)
        if time.monotonic() - started > TIME_BUDGET_S:
            entry["unresolved"] = f"not compared: the check ran out of time ({TIME_BUDGET_S} s for all views)"
            continue
        # a view cropped beforehand comes with its own image (only its ink, no neighbouring view's)
        gray = cv2.imread(s["image"], cv2.IMREAD_GRAYSCALE) if s.get("image") else rl.render(drawing, s["box"])
        stem = os.path.join(out_dir, s["id"])
        cv2.imwrite(stem + ".drawing.png", gray)
        try:
            result, drawn, model = check_view(shape, {**s, "_stem": stem}, gray, sheet_px_per_mm)
        except Exception as error:
            entry["unresolved"] = f"projection failed: {error}"
            continue
        filled = result.pop("_images", {})
        entry.update(result)
        if s.get("card_camera") or s.get("kind") == "isometric":
            joint.append({"id": s["id"], "kind": s.get("kind"), "look": (s.get("card_camera") or {}).get("look"),
                          "up": (s.get("card_camera") or {}).get("up"),
                          "scale": (1 / s["px_per_mm"]) if s.get("px_per_mm") else None,
                          "drawn": part_points(edges_and_annotations(gray, s.get("folder"))[0])})
        if drawn is None:
            continue
        # the validators see the filled images (faces in colour, silhouettes overlaid); the edge plots stay on disk
        entry["images"] = filled or {"drawing": stem + ".drawing.png", "model": stem + ".edges_model.png",
                                     "overlay": stem + ".edges_overlay.png"}
        vc.plot(stem + ".edges_model.png", None, model, f"{entry['label']}: model, {entry['camera']}",
                entry["horizontal"], entry["vertical"])
        vc.plot(stem + ".edges_overlay.png", drawn, model, f"{entry['label']}: scanned edges (black) vs model (red); "
                "orange = drawn, missing in model; purple = in model, not drawn",
                entry["horizontal"], entry["vertical"], entry["missing_regions"], entry["extra_regions"])
        np.savez_compressed(stem + ".points.npz", drawing=drawn, model=model)
        entry["points"] = stem + ".points.npz"
        entry["callouts"] = []
    # the orientation of the whole part, judged on all views at once from the frame card's cameras
    if joint:
        remaining = max(30.0, TIME_BUDGET_S + 120 - (time.monotonic() - started))
        out["pose"] = joint_pose(shape, joint, remaining)
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
