"""Turn a scanned view into vector data, as views.py does for vector PDFs (prototype, tier-4 scans).

raster_lines.separate_structure() splits the ink into part edges, dimension lines, extension lines, arrowheads and
text. Removing the extension lines also removes the stretches of part edge that lie on the same straight line (a
plate edge that carries on as an extension line). This module:

1. restores those stretches: where two loose ends of the part layer face each other in line, with ink between them,
   the ink between them is part edge again (an outline runs from one corner arc to the next);
2. vectorises the result: the outline as a closed polygon, the holes as circles, and every dimension as the line
   between its two arrow tips with the box of its value text, for a vision model to read the value.

All sizes are in pixels; a scale (mm per pixel) comes from the dimension values once they are read.

    uv run python src/raster_vector.py <drawing.pdf> <out folder> x0 y0 x1 y1
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np

import raster_lines as rl


# --------------------------------------------------------------------------- thinning and loose ends
def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: one-pixel-wide centre lines of the strokes."""
    img = mask.astype(np.uint8).copy()
    img = np.pad(img, 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = [np.roll(np.roll(img, -dy, 0), -dx, 1) for dy, dx in
                 ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))]
            # p2..p9 clockwise from north
            b = sum(p)
            seq = p + [p[0]]
            a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                c = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            remove = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & c
            if remove.any():
                img[remove] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


def fragment_kinds(sk: np.ndarray, labels: np.ndarray, n: int, straight_min=25) -> dict:
    """Which skeleton fragments can anchor an outline edge: curved ones (corner arcs, curves: only part edges are
    curved here, annotation lines are straight), however short, and long straight ones. Short straight pieces, such
    as centre-mark crosses on extension lines, cannot."""
    kinds = {}
    for i in range(1, n):
        ys, xs = np.nonzero(labels == i)
        if len(xs) < 4:
            continue
        pts = np.stack([xs, ys], 1).astype(float)
        s = np.linalg.svd(pts - pts.mean(0), compute_uv=False)
        spread = s[1] / math.sqrt(len(pts))            # rms distance from the best straight line
        if spread > 0.6:
            kinds[i] = "curved"
        elif len(pts) >= straight_min:
            kinds[i] = "straight"
    return kinds


def loose_ends(skel: np.ndarray) -> list[dict]:
    """Endpoints of the skeleton (one neighbour), with the direction the stroke was heading when it stopped. Only
    ends of fragments that can anchor an outline edge (see fragment_kinds)."""
    sk = skel.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sk, connectivity=8)
    kinds = fragment_kinds(sk, labels, n)
    neighbours = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    ys, xs = np.nonzero((sk == 1) & (neighbours == 1))
    ends = []
    for x, y in zip(xs, ys):
        if labels[y, x] not in kinds:
            continue
        # walk back along the stroke to get its heading: a few pixels only, so a short arc gives its end tangent
        path, prev, cur = [(x, y)], None, (x, y)
        for _ in range(5):
            nxt = [(cur[0] + dx, cur[1] + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if (dx or dy) and 0 <= cur[1] + dy < sk.shape[0] and 0 <= cur[0] + dx < sk.shape[1]
                   and sk[cur[1] + dy, cur[0] + dx] and (cur[0] + dx, cur[1] + dy) != prev
                   and (cur[0] + dx, cur[1] + dy) not in path]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
        if len(path) < 3:
            continue
        d = np.array(path[0], float) - np.array(path[-1], float)
        ends.append({"at": np.array([x, y], float), "heading": d / (np.linalg.norm(d) or 1),
                     "fragment": int(labels[y, x]), "kind": kinds[labels[y, x]]})
    return ends


def coverage(ink: np.ndarray, p, q) -> float:
    n = max(2, int(np.linalg.norm(q - p)))
    xs = np.linspace(p[0], q[0], n).round().astype(int).clip(0, ink.shape[1] - 1)
    ys = np.linspace(p[1], q[1], n).round().astype(int).clip(0, ink.shape[0] - 1)
    hit = np.zeros(n, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            hit |= ink[(ys + dy).clip(0, ink.shape[0] - 1), (xs + dx).clip(0, ink.shape[1] - 1)]
    return float(hit.mean())


def restore_edges(part: np.ndarray, ink: np.ndarray, max_turn_deg=20) -> tuple[np.ndarray, list]:
    """Bridge facing loose ends of the part layer along inked straight stretches."""
    skel = thin(part)
    ends = loose_ends(skel)
    cos_turn = math.cos(math.radians(max_turn_deg))
    restored, bridges, used = part.copy(), [], set()
    for i, e in enumerate(ends):
        best = None
        for j, f in enumerate(ends):
            if j <= i or j in used or i in used or e["fragment"] == f["fragment"] and np.linalg.norm(e["at"] - f["at"]) < 30:
                continue
            gap = f["at"] - e["at"]
            dist = float(np.linalg.norm(gap))
            if dist < 3:
                continue
            d = gap / dist
            # both ends must head into the gap, along it: the stroke carries on straight from one to the other
            if np.dot(e["heading"], d) < cos_turn or np.dot(f["heading"], -d) < cos_turn:
                continue
            if coverage(ink, e["at"], f["at"]) < 0.92:
                continue
            if best is None or dist < best[0]:
                best = (dist, j)
        if best is not None:
            j = best[1]
            used |= {i, j}
            p, q = ends[i]["at"], ends[j]["at"]
            cv2.line(restored.view(np.uint8), tuple(int(v) for v in p), tuple(int(v) for v in q), 1, 2)
            bridges.append([p.round(1).tolist(), q.round(1).tolist()])
    return restored & ink | part, bridges


# --------------------------------------------------------------------------- vectorise
def outline_and_holes(part: np.ndarray) -> dict:
    """The outline is the boundary of the region the part's strokes enclose; holes are the small round regions
    enclosed inside it."""
    strokes = cv2.morphologyEx(part.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    h, w = strokes.shape
    # flood the background from the border: what it cannot reach is enclosed by strokes
    flood = strokes.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 1)
    enclosed = (flood == 0).astype(np.uint8)          # empty pixels the border cannot reach
    region = (enclosed | strokes).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"closed": False}
    outer = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(outer, 1.0, True).reshape(-1, 2)
    # holes: enclosed empty pockets that are round
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(enclosed, 8)
    holes, inside = [], np.zeros_like(region)
    cv2.drawContours(inside, [outer], -1, 1, -1)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 12 or area > 0.05 * cv2.contourArea(outer):
            continue
        comp = (labels == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = cv2.arcLength(cs[0], True)
        circularity = 4 * math.pi * area / (perim * perim) if perim else 0
        if circularity < 0.75 or not inside[int(centroids[i][1]), int(centroids[i][0])]:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cs[0])
        # the pocket is the inside of the hole's stroke: add half a stroke to get the drawn diameter
        holes.append({"centre": [round(cx, 1), round(cy, 1)], "d_px": round(2 * r + 2, 1),
                      "circularity": round(circularity, 2)})
    # the outline is closed if the largest region has a real interior (a ring of strokes, not a tangle)
    interior = cv2.contourArea(outer) - strokes[inside.astype(bool)].sum()
    return {"closed": bool(interior > 0.3 * cv2.contourArea(outer)), "outline": polygon.tolist(),
            "outline_bbox": list(cv2.boundingRect(outer)), "area_px": float(cv2.contourArea(outer)), "holes": holes}


def dimensions(sep: dict, text: np.ndarray) -> list[dict]:
    """A dimension is two arrowheads on one line, facing each other: its length is tip to tip, whatever the ink
    between them looks like. An arrowhead with no partner is a leader. Each dimension gets the box of the value text
    nearest its middle, for a vision model to read."""
    arrows = sep.get("dimension_runs", [])
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8)), 8)
    out, paired = [], set()
    for i, a in enumerate(arrows):
        if i in paired:
            continue
        d = np.array(a["direction"], float)               # from the arrowhead along its line
        best = None
        for j, b in enumerate(arrows):
            if j == i or j in paired:
                continue
            gap = np.array(b["centre"], float) - np.array(a["centre"], float)
            along = float(np.dot(gap, d))
            across = abs(float(gap[0] * d[1] - gap[1] * d[0]))
            facing = float(np.dot(np.array(b["direction"], float), -d)) > 0.95
            if along > 5 and across < 3 and facing and (best is None or along < best[0]):
                best = (along, j)
        if best is None:
            continue
        j = best[1]
        paired |= {i, j}
        b = arrows[j]
        # tips: half an arrowhead beyond each centre, outward
        tip_a = np.array(a["centre"], float) - d * a["length"] / 2
        tip_b = np.array(b["centre"], float) + d * b["length"] / 2
        mid = (tip_a + tip_b) / 2
        box = None
        if n > 1:
            k = 1 + int(np.argmin([np.linalg.norm(centroids[m] - mid) for m in range(1, n)]))
            if np.linalg.norm(centroids[k] - mid) < 60:
                x, y, w, h, _ = stats[k]
                box = [int(x), int(y), int(x + w), int(y + h)]
        out.append({"from": tip_a.round(1).tolist(), "to": tip_b.round(1).tolist(),
                    "length_px": round(float(np.linalg.norm(tip_b - tip_a)), 1),
                    "orientation": "horizontal" if abs(d[0]) > 0.9 else "vertical" if abs(d[1]) > 0.9 else "angled",
                    "text_box": box})
    leaders = [{"at": np.array(a["centre"]).round(1).tolist(), "direction": np.round(a["direction"], 3).tolist()}
               for i, a in enumerate(arrows) if i not in paired]
    return out, leaders


def find_holes(part: np.ndarray, min_r=4, max_r=40) -> list[dict]:
    """Circles on the part layer (holes), kept only where ink covers most of the circumference."""
    img = cv2.GaussianBlur(np.where(part, 0, 255).astype(np.uint8), (3, 3), 0)
    found = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=2 * min_r, param1=100, param2=12,
                             minRadius=min_r, maxRadius=max_r)
    holes = []
    if found is None:
        return holes
    dil = cv2.dilate(part.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for x, y, r in found[0]:
        a = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        xs = (x + r * np.cos(a)).round().astype(int).clip(0, part.shape[1] - 1)
        ys = (y + r * np.sin(a)).round().astype(int).clip(0, part.shape[0] - 1)
        cover = float(dil[ys, xs].mean())
        if cover < 0.8 or any(math.dist((x, y), h["centre"]) < r for h in holes):
            continue
        holes.append({"centre": [round(float(x), 1), round(float(y), 1)], "d_px": round(float(2 * r), 1),
                      "coverage": round(cover, 2)})
    return holes


# --------------------------------------------------------------------------- flat blanks (sheet metal)
def fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares circle (centre, radius, largest distance of the points from it)."""
    a = np.c_[pts, np.ones(len(pts))]
    sol = np.linalg.lstsq(a, -(pts ** 2).sum(1), rcond=None)[0]
    centre = -sol[:2] / 2
    r = math.sqrt(max(centre @ centre - sol[2], 1e-9))
    return centre, r, float(np.abs(np.linalg.norm(pts - centre, axis=1) - r).max())


def outline_segments(contour: np.ndarray, eps=1.5, max_dev=1.0, max_r=600) -> list[dict]:
    """A closed contour as lines and arcs: the polygon's vertices (approxPolyDP), then runs of three or more short
    edges that one circle fits are merged into an arc."""
    c = contour.reshape(-1, 2).astype(float)
    idx = {tuple(p): i for i, p in enumerate(contour.reshape(-1, 2).tolist())}
    poly = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2).tolist()
    vi = sorted(idx[tuple(p)] for p in poly)
    if len(vi) < 3:
        return []
    # start at the sharpest corner, so no arc is cut in two at the start
    def turn(k):
        a, b, d = c[vi[k - 1]], c[vi[k]], c[vi[(k + 1) % len(vi)]]
        u, v = b - a, d - b
        return abs(math.atan2(u[0] * v[1] - u[1] * v[0], u @ v))
    s = max(range(len(vi)), key=turn)
    vi = vi[s:] + vi[:s]
    n = len(c)
    span = lambda i, j: c[[k % n for k in range(vi[i], vi[j] + (n if vi[j] <= vi[i] else 0) + 1)]]
    segs, i, m = [], 0, len(vi)
    while i < m:
        best = None
        for j in range(i + 3, m + 1):              # j == m closes back to the start
            pts = span(i, j % m)
            centre, r, dev = fit_circle(pts)
            if dev > max_dev or r > max_r:
                break
            best = (j, centre, r, pts)
        if best:
            j, centre, r, pts = best
            mid = pts[len(pts) // 2]
            sweep = (pts[0] - centre, pts[-1] - centre)
            side = float((pts[-1] - pts[0])[0] * (mid - pts[0])[1] - (pts[-1] - pts[0])[1] * (mid - pts[0])[0])
            segs.append({"arc": {"from": pts[0].tolist(), "to": pts[-1].tolist(), "mid": mid.tolist(),
                                 "centre": centre.round(1).tolist(), "radius": round(r, 1),
                                 "sweep_deg": round(math.degrees(abs(math.atan2(
                                     sweep[0][0] * sweep[1][1] - sweep[0][1] * sweep[1][0], sweep[0] @ sweep[1])))
                                     if side else 0, 1)}})
            i = j
        else:
            segs.append({"line": [c[vi[i]].tolist(), c[vi[(i + 1) % m]].tolist()]})
            i += 1
    return segs


def chord(sil: np.ndarray, p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The line through p and q, clipped to the stretch of the silhouette that contains its middle."""
    d = (q - p) / (np.linalg.norm(q - p) or 1)
    mid = (p + q) / 2
    ends = []
    for sign in (-1, 1):
        t = 0.0
        while True:
            x, y = (mid + sign * (t + 1) * d).round().astype(int)
            if not (0 <= x < sil.shape[1] and 0 <= y < sil.shape[0]) or not sil[y, x]:
                break
            t += 1
        ends.append(mid + sign * t * d)
    return ends[0], ends[1]


def blank(gray: np.ndarray, sep: dict, leaders: list[dict]) -> dict:
    """A flat blank drawn without dimensions (a sheet-metal flat pattern): its panels are the regions the outline
    and the bend lines enclose, so the bend lines are the straight strokes two panels share. Returns the silhouette
    outline as lines and arcs, the bend lines (clipped to the blank), the holes inside it and, for each leader whose
    arrow points at a bend line, that bend line. A copy crowded with dimensions encloses false panels between its
    extension lines: read the blank from a copy without dimensions."""
    ink = gray < 200
    strokes = cv2.morphologyEx((ink & ~sep["text"]).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    h, w = strokes.shape
    flood = strokes.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    enclosed = (flood == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed, 4)
    if n < 2:
        return {}
    largest = stats[1:, 4].max()
    panels = sorted((i for i in range(1, n) if stats[i, 4] >= max(200, 0.03 * largest)), key=lambda i: -stats[i, 4])
    pid = {lab: f"P{k}" for k, lab in enumerate(panels, 1)}       # P1 = the largest
    grown = {i: cv2.dilate((labels == i).astype(np.uint8), np.ones((7, 7), np.uint8)) for i in panels}
    sil = np.zeros_like(enclosed)
    for g in grown.values():
        sil |= g
    # small pockets a leader encloses against the outline or a bend line: part of the blank when panels lie on
    # most of their surround (just across the strokes), outside it when the background does
    in_panels = np.isin(labels, panels)
    for i in range(1, n):
        if i in panels:
            continue
        pocket = (labels == i).astype(np.uint8)
        ring = cv2.dilate(pocket, np.ones((9, 9), np.uint8)).astype(bool) & ~cv2.dilate(
            pocket, np.ones((5, 5), np.uint8)).astype(bool) & ~strokes.astype(bool)
        if ring.sum() and in_panels[ring].mean() > 0.5:
            sil |= cv2.dilate(pocket, np.ones((7, 7), np.uint8))
    sil = cv2.erode(sil, np.ones((5, 5), np.uint8))   # grown by 3, back by 2: the outline stroke's centre
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outer = max(contours, key=cv2.contourArea)
    inside = np.zeros_like(sil)
    cv2.drawContours(inside, [outer], -1, 1, -1)
    bends = []
    for a in panels:
        for b in panels:
            if b <= a:
                continue
            shared = (grown[a] & grown[b]).astype(bool)
            if shared.sum() < 20:
                continue
            ys, xs = np.nonzero(shared)
            pts = np.stack([xs, ys], 1).astype(float)
            mu = pts.mean(0)
            _, sv, vt = np.linalg.svd(pts - mu, full_matrices=False)
            if sv[1] / math.sqrt(len(pts)) > 2.0:      # not one straight stroke
                continue
            t = (pts - mu) @ vt[0]
            if t.max() - t.min() < 10:
                continue
            p, q = chord(inside.astype(bool), mu + vt[0] * t.min(), mu + vt[0] * t.max())
            bends.append({"id": f"B{len(bends) + 1}", "from": p.round(1).tolist(), "to": q.round(1).tolist(),
                          "panels": [pid[a], pid[b]]})
    holes = [hole for hole in find_holes(ink & ~sep["text"])
             if inside[int(hole["centre"][1]), int(hole["centre"][0])]
             and cv2.pointPolygonTest(outer, tuple(hole["centre"]), True) > hole["d_px"] / 2]
    for l in leaders:
        tip = np.array(l["at"], float)
        near = []
        for bl in bends:
            p, q = np.array(bl["from"]), np.array(bl["to"])
            t = np.clip((tip - p) @ (q - p) / max((q - p) @ (q - p), 1e-9), 0, 1)
            near.append((float(np.linalg.norm(p + t * (q - p) - tip)), bl["id"]))
        if near and min(near)[0] < 15:
            l["bend"] = min(near)[1]
    x, y, bw, bh = cv2.boundingRect(outer)
    import outline_fit
    panel_info = []
    for lab in panels:
        inside_pt = deepest_point(labels == lab)
        panel_info.append({"id": pid[lab], "area_px": int(stats[lab, 4]), "inside": inside_pt,
                           "bends": [b["id"] for b in bends if pid[lab] in b["panels"]]})
    image = label_image(gray, [labels == lab for lab in panels], [pid[lab] for lab in panels],
                        [(np.array(b["from"]), np.array(b["to"]), f"{b['id']}: {b['panels'][0]}|{b['panels'][1]}")
                         for b in bends])
    return {"bbox": [x, y, bw, bh], "segments": outline_segments(outer), "bend_lines": bends, "holes": holes,
            "panels": panel_info, "contour": outer.reshape(-1, 2).tolist(),
            "corners": outline_fit.vertices(outer), "_image": image}


PALETTE = [(120, 180, 240), (140, 220, 140), (240, 170, 120), (200, 150, 230), (110, 220, 220), (230, 220, 110),
           (240, 140, 180), (170, 170, 250), (160, 230, 190), (250, 200, 150), (190, 210, 120), (150, 190, 210)]


def deepest_point(mask: np.ndarray) -> list[float]:
    """The point of a region farthest from its edges: a safe point inside it, and where its id is written."""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return [float(x), float(y)]


def label_image(gray: np.ndarray, masks: list, ids: list[str], lines: list = ()) -> np.ndarray:
    """The view with each region filled in its own colour and its id written inside it, and labelled lines (bend
    lines) drawn over them: what the parts are and which touches which, at a glance."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(float)
    for k, (m, rid) in enumerate(zip(masks, ids)):
        img[m] = img[m] * 0.35 + np.array(PALETTE[k % len(PALETTE)]) * 0.65
    img = img.astype(np.uint8)
    for m, rid in zip(masks, ids):
        x, y = (int(v) for v in deepest_point(m))
        cv2.putText(img, rid, (x - 12, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, rid, (x - 12, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    for p, q, text in lines:
        cv2.line(img, tuple(p.astype(int)), tuple(q.astype(int)), (0, 0, 220), 2)
        m = ((p + q) / 2).astype(int)
        cv2.putText(img, text, (m[0] + 5, m[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, text, (m[0] + 5, m[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
    return img


def faces(gray: np.ndarray, sep: dict) -> dict:
    """The faces of a pictorial (isometric) view: each region its outline and edges enclose, in its own colour with
    an id (F1 = the largest). On a formed part they show which panels lie flat, which rise, and where they meet.
    Letters and small loops are left out. Not for shaded renders (their shading is not ink)."""
    ink = (gray < 200) & ~sep["text"]
    strokes = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    h, w = strokes.shape
    flood = strokes.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((flood == 0).astype(np.uint8), 4)
    if n < 2:
        return {}
    largest = stats[1:, 4].max()
    keep = sorted((i for i in range(1, n) if stats[i, 4] >= max(300, 0.01 * largest)), key=lambda i: -stats[i, 4])
    masks = [labels == i for i in keep]
    ids = [f"F{k}" for k in range(1, len(keep) + 1)]
    # which faces meet (share an edge stroke)
    grown = [cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) for m in masks]
    touching = {ids[a]: [ids[b] for b in range(len(keep)) if b != a and (grown[a] & grown[b]).sum() >= 20]
                for a in range(len(keep))}
    return {"faces": [{"id": i, "area_px": int(stats[lab, 4]), "touches": touching[i]} for i, lab in zip(ids, keep)],
            "_image": label_image(gray, masks, ids)}


def corner_image(gray: np.ndarray, contour: list, corners: list[int], pieces: list | None = None) -> np.ndarray:
    """The view with its traced outline (blue) and numbered corners V1..Vn, enlarged so the ids are readable; with
    `pieces`, the fitted outline in red over it."""
    k = 2
    img = cv2.cvtColor(cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    img = (255 - (255 - img) * 0.45).astype(np.uint8)            # fade the drawing so the marks stand out
    c = np.array(contour, float)
    cv2.polylines(img, [(c * k).astype(np.int32)], True, (255, 120, 0), 1)
    if pieces is not None:
        import outline_fit
        pts = np.array(outline_fit.sample(pieces), float) * k
        cv2.polylines(img, [pts.astype(np.int32)], True, (0, 0, 220), 2)
    centre = c.mean(0)
    for n, i in enumerate(corners, 1):
        p = c[i] * k
        cv2.circle(img, tuple(p.astype(int)), 4, (0, 140, 0), -1)
        out = p - centre * k
        out = out / (np.linalg.norm(out) or 1)
        label = p + out * 16 - np.array([10, -5])
        cv2.putText(img, f"V{n}", tuple(label.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 110, 0), 1, cv2.LINE_AA)
    return img


def corner_patches(gray: np.ndarray, contour: list, corners: list[int], half: int = 40, zoom: int = 3,
                   columns: int = 6) -> np.ndarray:
    """One close-up per corner, in a grid: the drawing around the corner (2 * half pixels square, enlarged), the
    traced outline thin in blue so the ink under it stays visible, the corner as a large green dot, its neighbours
    as small grey dots, and its id above. For a vision model to judge every corner in one look."""
    c = np.array(contour, float)
    h, w = gray.shape
    pad = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_CONSTANT, value=255)
    tiles = []
    for n, i in enumerate(corners, 1):
        x, y = c[i].round().astype(int)
        crop = pad[y:y + 2 * half, x:x + 2 * half]
        tile = cv2.cvtColor(cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC),
                            cv2.COLOR_GRAY2BGR)
        local = ((c - [x - half, y - half]) * zoom).astype(np.int32)
        cv2.polylines(tile, [local], True, (255, 90, 0), 1, cv2.LINE_AA)
        for j in corners:
            if j != i:
                cv2.circle(tile, tuple(local[j]), 3, (150, 150, 150), -1)
        cv2.circle(tile, tuple(local[i]), 6, (0, 170, 0), 2)
        head = np.full((26, tile.shape[1], 3), 255, np.uint8)
        cv2.putText(head, f"V{n}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 110, 0), 2, cv2.LINE_AA)
        tile = np.vstack([head, tile])
        tiles.append(cv2.copyMakeBorder(tile, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(200, 200, 200)))
    rows = []
    for r in range(0, len(tiles), columns):
        row = tiles[r:r + columns]
        row += [np.full_like(tiles[0], 255)] * (columns - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows) if rows else np.full((10, 10, 3), 255, np.uint8)


def vectorise(gray: np.ndarray, flat: bool = False, pictorial: bool = False) -> dict:
    sep = rl.separate_structure(gray)
    ink = gray < 200
    # centre lines and cutting-plane lines (dash-dot runs) are axes, not edges
    axes, _ = rl.dashed_runs(sep["part"])
    part = sep["part"] & ~axes
    restored, bridges = restore_edges(part, ink)
    shape = outline_and_holes(restored)
    shape["holes"] = find_holes(restored)
    dims, leaders = dimensions(sep, sep["text"])
    flat_blank = blank(gray, sep, leaders) if flat else {}
    ink = gray < 235
    shaded = bool(ink.sum()) and float(((gray > 70) & (gray < 215))[ink].mean()) > 0.5   # as raster_check.shaded
    view_faces = faces(gray, sep) if pictorial and not shaded else {}
    return {"separation": sep, "part": restored, "axes": axes, "bridges": bridges, "shape": shape,
            "dimensions": dims, "leaders": leaders, "blank": flat_blank, "faces": view_faces}


def save(result: dict, gray: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    over = np.full(gray.shape + (3,), 255, np.uint8)
    over[result["separation"]["dimension"] | result["separation"]["extension"]] = (200, 200, 255)
    over[result["axes"]] = (180, 220, 180)
    over[result["part"]] = (0, 0, 0)
    for p, q in result["bridges"]:
        cv2.line(over, tuple(int(v) for v in p), tuple(int(v) for v in q), (0, 170, 0), 2)
    shape = result["shape"]
    if shape.get("outline"):
        cv2.polylines(over, [np.array(shape["outline"], np.int32)], True, (255, 120, 0), 1)
    for h in shape.get("holes", []):
        cv2.circle(over, tuple(int(v) for v in h["centre"]), int(h["d_px"] / 2), (0, 140, 255), 1)
    for d in result["dimensions"]:
        cv2.line(over, tuple(int(v) for v in d["from"]), tuple(int(v) for v in d["to"]), (0, 0, 230), 1)
        if d["text_box"]:
            x0, y0, x1, y1 = d["text_box"]
            cv2.rectangle(over, (x0, y0), (x1, y1), (200, 0, 200), 1)
    b = result.get("blank") or {}
    if b.get("contour"):
        cv2.polylines(over, [np.array(b["contour"], np.int32)], True, (255, 0, 0), 1)
    for bl in b.get("bend_lines", []):
        p, q = (tuple(int(v) for v in bl[k]) for k in ("from", "to"))
        cv2.line(over, p, q, (200, 0, 200), 2)
        cv2.putText(over, bl["id"], (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1)
    for hole in b.get("holes", []):
        cv2.circle(over, tuple(int(v) for v in hole["centre"]), int(hole["d_px"] / 2), (0, 140, 255), 2)
    cv2.imwrite(os.path.join(out_dir, "vector_overlay.png"), over)
    if b.get("_image") is not None:
        cv2.imwrite(os.path.join(out_dir, "panels.png"), b["_image"])
    if (result.get("faces") or {}).get("_image") is not None:
        cv2.imwrite(os.path.join(out_dir, "faces.png"), result["faces"]["_image"])
    # the part edges raster_check compares with (part layer minus centre lines), so it need not separate again
    cv2.imwrite(os.path.join(out_dir, "check_part.png"),
                (result["separation"]["part"] & ~result["axes"]).astype(np.uint8) * 255)
    with open(os.path.join(out_dir, "vector.json"), "w") as f:
        json.dump({"shape": shape, "bridges": result["bridges"], "dimensions": result["dimensions"],
                   "leaders": result["leaders"],
                   "blank": {k: v for k, v in b.items() if k != "contour" and not k.startswith("_")},
                   "faces": (result.get("faces") or {}).get("faces", [])}, f,
                  indent=1)


if __name__ == "__main__":
    pdf, out_dir, *box = sys.argv[1:]
    g = rl.render(pdf, [float(v) for v in box] if box else None)
    r = vectorise(g, flat=os.environ.get("FLAT") == "1")
    save(r, g, out_dir)
    s = r["shape"]
    print(f"outline closed: {s.get('closed')}, {len(s.get('outline', []))} points, bbox {s.get('outline_bbox')}; "
          f"holes: {[h['d_px'] for h in s.get('holes', [])]}; bridges: {len(r['bridges'])}; "
          f"dimensions: {len(r['dimensions'])}")
