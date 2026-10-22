"""Parts of each view and their matches across views, from the pixels alone: no LLM call (tier-4 scans).

A local stand-in for steps 2-3 of part_views (colour the parts of each view, match parts across views). It writes
the same files: views.json (parts with polygon_pt / holes_pt), matches.json (groups), views/vK_colored.png,
views/vK_matched.png, matched_all.png.

1. segment:  regions.fill_regions fills every closed region of a view; each region is then classified:
             face   a large region (inscribed width at least FACE_WIDTH of the widest): a panel of a flat pattern,
                    a face of an isometric;
             hole   isometric: a small region enclosed by one face; flat views: a circle found on the ink (centre
                    lines cut a drawn hole into quarters, so its regions are useless);
             text   a letter's counter: a small region with no neighbour, lying on the paper (dropped);
             bend   the strip joining two faces (isometric) or the line two panels share (flat pattern);
             other  everything else (side faces seen edge-on, ...).
2. transfer: a flat view cut into strips by dimension lines gets its panels from a cleaner view of the same flat
             pattern, registered onto its ink (rotation by 90 degree steps, mirror, scale, shift).
3. match:    each view's faces form a graph: two faces are neighbours when they touch, directly or through one
             strip. The reference view (the flat view with most faces) is matched to every other view by the
             mapping that best keeps hole counts, the size order of holes and the neighbours. Holes then follow
             their faces (ordered by their distances to the bends), bends their pair of faces.

    uv run python local_parts_demo.py cases/tier4/01/drawing.pdf [--views runs/<run>/views.json]
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from src import regions

FACE_WIDTH = 0.4      # a face is at least this share of the widest region's inscribed width
MIN_SUPPORT = 0.85    # share of a registered outline (or circle) that must lie on ink
MAX_CHANCE = 0.45     # ... and at most this share at random poses (a dense view has ink everywhere)
MAX_FACES = 7         # the matching tries every mapping between the largest faces of two views
PALETTE = [(0.90, 0.10, 0.10), (0.10, 0.45, 0.90), (0.10, 0.70, 0.20), (0.95, 0.55, 0.00),
           (0.60, 0.20, 0.80), (0.00, 0.70, 0.70), (0.85, 0.20, 0.60), (0.55, 0.40, 0.10),
           (0.40, 0.60, 0.00), (0.20, 0.20, 0.60), (0.95, 0.80, 0.00), (0.50, 0.50, 0.50)]


# --------------------------------------------------------------------------- views
def load_views(drawing: Path, views_json: Path | None = None) -> tuple[list[dict], float]:
    """The views of the sheet, each with its crop ("image", sheet pixels). Boxes come from `views_json` (a
    part_views run) when given, else from regions.view_crops. Returns (views, sheet pixels per PDF point)."""
    gray = regions.render(str(drawing))
    with pymupdf.open(drawing) as doc:
        zoom = gray.shape[1] / doc[0].rect.width
    views = []
    if views_json:
        for v in json.loads(Path(views_json).read_text()):
            x0, y0, x1, y1 = (int(round(c * zoom)) for c in v["rect_pt"])
            views.append({"id": v["id"], "sheet": v.get("sheet", 1), "name": v.get("name", ""),
                          "kind": v.get("kind", "other"), "rect_pt": v["rect_pt"], "image": gray[y0:y1, x0:x1].copy()})
    else:
        for k, v in enumerate(regions.view_crops(gray), 1):
            views.append({"id": f"v{k}", "sheet": 1, "name": "", "kind": guess_kind(v["image"]),
                          "rect_pt": [round(c / zoom, 2) for c in v["box"]], "image": v["image"]})
    return views, zoom


def guess_kind(gray: np.ndarray) -> str:
    """isometric when under 60% of the length of long straight lines runs horizontal or vertical, else
    orthographic."""
    found = cv2.createLineSegmentDetector().detect(gray)[0]
    if found is None:
        return "other"
    s = found.reshape(-1, 4)
    length = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
    s, length = s[length >= 25], length[length >= 25]       # circles break into short pieces at every angle
    if not len(s):
        return "other"
    ang = np.degrees(np.arctan2(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0])) % 90
    straight = (np.minimum(ang, 90 - ang) < 4)
    return "orthographic" if (length * straight).sum() >= 0.6 * length.sum() else "isometric"


def is_flat(view: dict) -> bool:
    return view["kind"] not in ("isometric", "other")


# --------------------------------------------------------------------------- segmentation
def segment(gray: np.ndarray, kind: str) -> list[dict]:
    """The view's regions, each classified (see the module docstring). Each has "id" (local, r1, r2, ...), "mask",
    "class", "width" (inscribed diameter, px), "touches" (local ids) and, for a face, "holes" (list of dicts with
    "mask" or "circle" (cx, cy, r), "diameter" px, "centre")."""
    flat = kind not in ("isometric", "other")
    regs = regions.fill_regions(gray)
    for k, r in enumerate(regs, 1):
        r["fid"], r["id"] = r["id"], f"r{k}"
    rename = {r["fid"]: r["id"] for r in regs}
    for r in regs:
        r["touches"] = [rename[t] for t in r["touches"]]
        r["shared_lines"] = [{**s, "with": rename[s["with"]]} for s in r["shared_lines"]]
        r["width"] = 2.0 * float(cv2.distanceTransform(r["mask"].astype(np.uint8), cv2.DIST_L2, 3).max())
    strokes = cv2.morphologyEx((gray < regions.INK).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    paper = _paper(strokes)
    lab = np.zeros(gray.shape, np.int32)
    for k, r in enumerate(regs, 1):
        lab[r["mask"]] = k
    lab[paper] = -1
    by_id = {r["id"]: r for r in regs}
    for r in regs:
        ring = (cv2.dilate(r["mask"].astype(np.uint8), np.ones((13, 13), np.uint8)) > 0) & ~r["mask"] & (strokes == 0)
        vals = lab[ring]
        vals = vals[vals != 0]
        r["paper_share"] = float((vals == -1).mean()) if len(vals) else 0.0
        others = vals[vals > 0]
        if len(others) and len(vals):
            top = np.bincount(others).argmax()
            r["enclosed_by"] = regs[top - 1]["id"] if (others == top).sum() >= 0.9 * len(vals) else None
        else:
            r["enclosed_by"] = None
        r["class"] = "other"
    for r in regs:
        if not r["touches"] and r["paper_share"] > 0.5:
            r["class"] = "text"
    # iso: a hole is a small roundish region inside one bigger region
    if not flat:
        for r in regs:
            host = by_id.get(r["enclosed_by"]) if r["enclosed_by"] else None
            if host is not None and r["area_px"] < 0.25 * host["area_px"] and _circularity(r["mask"]) > 0.3:
                r["class"] = "hole"
    candidates = [r for r in regs if r["class"] == "other"]
    wmax = max((r["width"] for r in candidates), default=0)
    amax = max((r["area_px"] for r in candidates), default=0)
    for r in candidates:
        if r["width"] >= FACE_WIDTH * wmax and r["area_px"] >= 0.03 * amax:
            r["class"] = "face"
    faces = [r for r in regs if r["class"] == "face"]
    for f in faces:
        f["holes"] = []
    if flat:
        lab_faces = np.zeros(gray.shape, np.int32)
        for k, f in enumerate(faces, 1):
            lab_faces[f["mask"]] = k
        for cx, cy, rad in find_circles(gray):
            k = _host(lab_faces, cx, cy, rad)
            if k:
                faces[k - 1]["holes"].append({"circle": (cx, cy, rad), "diameter": 2 * rad, "centre": (cx, cy)})
                for r in regs:          # regions inside the circle are pieces of the hole, not parts
                    if r["class"] != "face" and math.dist(r["inside"], (cx, cy)) < rad:
                        r["class"] = "text"
    else:
        for r in regs:
            if r["class"] == "hole" and by_id[r["enclosed_by"]]["class"] == "face":
                by_id[r["enclosed_by"]]["holes"].append({"mask": r["mask"], "region": r["id"],
                                                        "diameter": _feret(r["mask"]), "centre": tuple(r["inside"])})
            elif r["class"] == "hole":
                r["class"] = "other"
    return [r for r in regs if r["class"] != "text"]


def _paper(strokes: np.ndarray) -> np.ndarray:
    n, lab = cv2.connectedComponents((strokes == 0).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, border[border > 0])


def _circularity(mask: np.ndarray) -> float:
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnt)
    return 4 * math.pi * mask.sum() / per ** 2 if per else 0.0


def _feret(mask: np.ndarray) -> float:
    """Largest extent of a region: for a hole seen at an angle, its ellipse's major axis (the true diameter)."""
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.vstack(cnt)).reshape(-1, 2).astype(float)
    return float(max(np.linalg.norm(hull[:, None] - hull[None], axis=2).max(), 1.0))


def _host(lab: np.ndarray, cx: float, cy: float, rad: float) -> int:
    """The face round a circle: the label most samples just outside it fall in (0 if none holds 60%)."""
    h, w = lab.shape
    hits = []
    for a in np.linspace(0, 2 * math.pi, 48, endpoint=False):
        x, y = int(round(cx + (rad + 6) * math.cos(a))), int(round(cy + (rad + 6) * math.sin(a)))
        if 0 <= x < w and 0 <= y < h and lab[y, x] > 0:
            hits.append(lab[y, x])
    if not hits:
        return 0
    k = np.bincount(hits).argmax()
    return int(k) if hits.count(k) >= 0.6 * 48 else 0


def _on_circle(ink: np.ndarray, cx: float, cy: float, r: float, n: int = 120) -> float:
    """Share of the circle's points that fall on ink."""
    h, w = ink.shape
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    xs = np.clip((cx + r * np.cos(a)).round().astype(int), 0, w - 1)
    ys = np.clip((cy + r * np.sin(a)).round().astype(int), 0, h - 1)
    return float(ink[ys, xs].mean())


def find_circles(gray: np.ndarray) -> list[tuple[float, float, float]]:
    """Drawn circles (holes on a flat view). Hough gives many candidates (arcs of fillets too); each is nudged onto
    the ink and kept when ink lies all round it, its inside is paper (crossed by centre lines at most) and so is a
    thin ring just outside it (a fillet's arc runs on into straight edges, a hole's circle does not)."""
    h, w = gray.shape
    ink = gray < regions.INK
    thick = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    found = cv2.HoughCircles(cv2.GaussianBlur(gray, (5, 5), 1.2), cv2.HOUGH_GRADIENT, dp=1, minDist=4, param1=100,
                             param2=10, minRadius=4, maxRadius=int(0.12 * min(h, w)))
    kept = []
    for cx, cy, r in ([] if found is None else found.reshape(-1, 3)):
        if _on_circle(thick, cx, cy, r, 48) < 0.6:
            continue
        best = (_on_circle(thick, cx, cy, r), cx, cy, r)
        for step in (1.0, 0.5):
            for _ in range(8):
                s0, x0, y0, r0 = best
                moves = [(x0 + dx, y0 + dy, r0 + dr) for dx, dy, dr in
                         ((step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0), (0, 0, step), (0, 0, -step))]
                top = max((_on_circle(thick, *m), *m) for m in moves)
                if top[0] <= s0:
                    break
                best = top
        s, cx, cy, r = best
        if s < 0.9 or max(_on_circle(ink, cx, cy, r + k) for k in (3, 4, 5)) > 0.35 \
                or np.mean([_on_circle(ink, cx, cy, r * q) for q in (0.45, 0.6, 0.75)]) > 0.25:
            continue
        kept.append((float(cx), float(cy), float(r)))
    # largest first; a circle centred inside a bigger one is a piece of it (a quarter between centre lines)
    out = []
    for c in sorted(kept, key=lambda c: -c[2]):
        if not any(math.dist(c[:2], k[:2]) < k[2] for k in out):
            out.append(c)
    return out


# --------------------------------------------------------------------------- transfer between flat views
def outline_points(faces: list[dict], holes: bool = True, n: int = 800) -> np.ndarray:
    """Points on the outline of the faces' union (bend lines between panels closed), and on their holes."""
    union = np.zeros(faces[0]["mask"].shape, np.uint8)
    for f in faces:
        union |= f["mask"].astype(np.uint8)
    # onto the middle of the drawn line, and over the notches leader lines cut into the panels
    union = cv2.dilate(union, np.ones((3, 3), np.uint8))
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cnt, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = [c.reshape(-1, 2) for c in cnt]
    if holes:
        for f in faces:
            for hl in f["holes"]:
                cx, cy, r = hl["circle"]
                a = np.linspace(0, 2 * math.pi, 36, endpoint=False)
                pts.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1))
    pts = np.vstack(pts).astype(float)
    return pts[np.linspace(0, len(pts) - 1, min(n, len(pts))).astype(int)]


DIHEDRAL = [np.array(m, float) for m in ([[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],
                                         [[-1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [0, -1]], [[0, -1], [-1, 0]])]


def register(src_pts: np.ndarray, dst_gray: np.ndarray) -> dict:
    """The similarity (one of 8 rotations/mirrors, scale, shift) that puts the most outline points on ink in
    dst. Coarse search at quarter resolution, then refined. Returns {"A": 2x3 affine src->dst, "support"}."""
    ink = (dst_gray < regions.INK).astype(np.uint8)
    near = cv2.dilate(ink, np.ones((5, 5), np.uint8)).astype(np.float32)
    c = src_pts.mean(0)

    def best_shift(pts, field, step):
        x0, y0 = pts.min(0)
        tpl_pts = ((pts - (x0, y0)) / step).round().astype(int)
        tw, th = tpl_pts.max(0) + 1
        if tw >= field.shape[1] or th >= field.shape[0]:
            return -1, None
        tpl = np.zeros((th, tw), np.float32)
        tpl[tpl_pts[:, 1], tpl_pts[:, 0]] = 1
        res = cv2.matchTemplate(field, tpl, cv2.TM_CCORR)
        _, mx, _, loc = cv2.minMaxLoc(res)
        return mx / tpl.sum(), (loc[0] * step - x0, loc[1] * step - y0)

    small = cv2.resize(near, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small = (small > 0.05).astype(np.float32)
    best = (-1, None)
    for D in DIHEDRAL:
        for s in np.geomspace(0.4, 2.5, 60):
            pts = (src_pts - c) @ D.T * s
            sc, shift = best_shift(pts, small, 4)
            if sc > best[0]:
                best = (sc, (D, s, shift))
    _, (D, s, _) = best
    fine = (-1, None)
    for s2 in s * np.linspace(0.96, 1.04, 17):
        for rot in np.radians(np.linspace(-2, 2, 5)):
            R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]]) @ D
            pts = (src_pts - c) @ R.T * s2
            sc, shift = best_shift(pts, near, 1)
            if sc > fine[0]:
                fine = (sc, (R, s2, shift))
    sc, (R, s2, shift) = fine
    M = R * s2
    t = np.array(shift) - M @ c
    A = np.hstack([M, t[:, None]])
    h, w = ink.shape
    on = cv2.dilate(ink, np.ones((5, 5), np.uint8)) > 0

    def support_at(M, t):
        moved = src_pts @ M.T + t
        inside = (moved[:, 0] >= 0) & (moved[:, 0] < w) & (moved[:, 1] >= 0) & (moved[:, 1] < h)
        xs, ys = moved[inside, 0].astype(int), moved[inside, 1].astype(int)
        return float(on[ys, xs].sum() / len(moved))

    # chance: the same outline at random shifts and turns, kept inside the view where possible
    rng = np.random.default_rng(0)
    chance = []
    for k in range(24):
        Mk = DIHEDRAL[k % 8] @ M
        lo, hi = (src_pts @ Mk.T).min(0), (src_pts @ Mk.T).max(0)
        tk = np.array([rng.uniform(-lo[0], max(-lo[0], w - hi[0])), rng.uniform(-lo[1], max(-lo[1], h - hi[1]))])
        chance.append(support_at(Mk, tk))
    return {"A": A, "support": support_at(M, t), "chance": float(np.median(chance))}


def transfer(src: list[dict], dst_gray: np.ndarray, A: np.ndarray) -> list[dict]:
    """The source view's faces moved onto the destination view (holes found again on its own ink)."""
    h, w = dst_gray.shape
    faces = []
    for f in src:
        if f["class"] != "face":
            continue
        m = cv2.warpAffine(f["mask"].astype(np.uint8), A, (w, h), flags=cv2.INTER_NEAREST) > 0
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        faces.append({"id": f["id"], "mask": m, "class": "face", "area_px": int(m.sum()), "inside": regions.deepest_point(m),
                      "holes": [], "touches": [], "shared_lines": [], "transferred_from": f["id"]})
    lab = np.zeros((h, w), np.int32)
    for k, f in enumerate(faces, 1):
        lab[f["mask"]] = k
    for cx, cy, rad in find_circles(dst_gray):
        k = _host(lab, cx, cy, rad)
        if k:
            faces[k - 1]["holes"].append({"circle": (cx, cy, rad), "diameter": 2 * rad, "centre": (cx, cy)})
    grown = {f["id"]: cv2.dilate(f["mask"].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0 for f in faces}
    for a, b in itertools.combinations(faces, 2):
        shared = grown[a["id"]] & grown[b["id"]]
        if shared.sum() >= 20:
            a["touches"].append(b["id"])
            b["touches"].append(a["id"])
            ys, xs = np.nonzero(shared)
            line = _line_of(np.stack([xs, ys], 1).astype(float))
            if line:
                for x, y in ((a, b), (b, a)):
                    x["shared_lines"].append({"with": y["id"], "from": line[0], "to": line[1]})
    return faces


def _line_of(pts: np.ndarray):
    mu = pts.mean(0)
    _, sv, vt = np.linalg.svd(pts - mu, full_matrices=False)
    t = (pts - mu) @ vt[0]
    if sv[1] / math.sqrt(len(pts)) <= 2.5 and t.max() - t.min() >= 15:
        return (mu + vt[0] * t.min()).round(1).tolist(), (mu + vt[0] * t.max()).round(1).tolist()
    return None


# --------------------------------------------------------------------------- view graph
def face_graph(regs: list[dict]) -> dict:
    """Faces and their neighbours. Two faces are neighbours when they touch, or when one non-face region touches
    both: that region (the one sharing straight lines with both, else the largest) is their bend. Returns
    {"faces": [ids], "adj": {id: set}, "bends": {frozenset(a, b): region id or a line}, "contact": {(a, b): pts}}."""
    by_id = {r["id"]: r for r in regs}
    faces = [r["id"] for r in regs if r["class"] == "face"]
    adj = {f: set() for f in faces}
    bends, contact = {}, {}
    for a, b in itertools.combinations(faces, 2):
        ra, rb = by_id[a], by_id[b]
        if b in ra["touches"]:
            line = next((s for s in ra["shared_lines"] if s["with"] == b), None)
            bends[frozenset((a, b))] = {"line": (line["from"], line["to"])} if line else {"line": None}
            pts = _contact(ra["mask"], rb["mask"])
            contact[(a, b)], contact[(b, a)] = pts, pts
        else:
            mids = [by_id[m] for m in set(ra["touches"]) & set(rb["touches"])
                    if m in by_id and by_id[m]["class"] == "other"]
            if not mids:
                continue
            lined = lambda m: {s["with"] for s in m["shared_lines"]} >= {a, b}
            mid = max(mids, key=lambda m: (lined(m), m["area_px"]))
            bends[frozenset((a, b))] = {"region": mid["id"]}
            contact[(a, b)] = _contact(ra["mask"], mid["mask"])
            contact[(b, a)] = _contact(rb["mask"], mid["mask"])
        adj[a].add(b)
        adj[b].add(a)
    return {"faces": faces, "adj": adj, "bends": bends, "contact": contact, "by_id": by_id}


def _contact(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    k = np.ones((9, 9), np.uint8)
    ys, xs = np.nonzero((cv2.dilate(a.astype(np.uint8), k) > 0) & (cv2.dilate(b.astype(np.uint8), k) > 0))
    return np.stack([xs, ys], 1).astype(float)


# --------------------------------------------------------------------------- matching
CUES = {"holes": 3.0, "neighbours": 2.0, "hole_size": 1.0}   # weights of the match score's terms


def match_faces(ref: dict, other: dict, cues: dict = CUES, only=None) -> tuple[dict, float, list]:
    """The mapping ref face -> other face (or None) with the best score; also its score and the runners-up
    [(score, mapping)], best first. The score adds, with the weights in `cues`:
      holes       a matched pair has the same number of holes (else a penalty per hole of difference);
      neighbours  two matched faces are neighbours in both views, or in neither (neighbours through a face left
                  unmatched count: it is likely a wide bend);
      hole_size   the ratio of hole sizes of two matched faces is the same in both views (a hole's largest extent
                  is its diameter in any parallel projection);
    and takes off one for every face left unmatched, more for a large one. `only(mapping)`, if given, limits the
    search to the mappings it accepts."""
    rf, of = (sorted(g["faces"], key=lambda f: -g["by_id"][f]["area_px"])[:MAX_FACES] for g in (ref, other))
    amax = max((other["by_id"][f]["area_px"] for f in of), default=1)
    results = []

    def holes(g, f):
        return sorted(h["diameter"] for h in g["by_id"][f]["holes"])

    def score(m):
        s = 0.0
        used = {b for b in m.values() if b}
        spare = [u for u in of if u not in used]
        for a, b in m.items():
            if b is None:
                s -= 1
                continue
            na, nb = len(holes(ref, a)), len(holes(other, b))
            s += cues["holes"] * (1 if na == nb else -abs(na - nb))
        pairs = [(a, b) for a, b in m.items() if b is not None]
        for (a1, b1), (a2, b2) in itertools.combinations(pairs, 2):
            ra = a2 in ref["adj"][a1]
            rb = b2 in other["adj"][b1] or any({b1, b2} <= other["adj"][u] for u in spare)
            s += cues["neighbours"] * (1 if ra and rb else -1 if ra != rb else 0)
            ha, hb = holes(ref, a1), holes(ref, a2)
            ka, kb = holes(other, b1), holes(other, b2)
            if ha and hb and ka and kb:
                d = abs(math.log(max(ha) / max(hb)) - math.log(max(ka) / max(kb)))
                s += cues["hole_size"] * max(-3.0, 1 - 5 * d)
        s -= sum(1 + 2 * other["by_id"][u]["area_px"] / amax for u in spare)
        return s

    def search(i, used, m):
        if i == len(rf):
            if only is None or only(m):
                results.append((score(m), dict(m)))
            return
        for b in [None] + [b for b in of if b not in used]:
            m[rf[i]] = b
            search(i + 1, used | ({b} if b else set()), m)
        del m[rf[i]]

    search(0, set(), {})
    results.sort(key=lambda r: -r[0])
    return results[0][1], results[0][0], results[1:6]


def absorb_bends(ref: dict, g: dict, fmap: dict) -> None:
    """A wide bend seen face-on passes for a face. An unmatched face touching two matched faces that are
    neighbours in the reference (but not here) is their bend: make it one, and the two faces neighbours."""
    used = set(fmap.values())
    for a, b in itertools.combinations([f for f in fmap if fmap[f]], 2):
        x, y = fmap[a], fmap[b]
        if b not in ref["adj"][a] or y in g["adj"][x]:
            continue
        mids = [u for u in g["faces"] if u not in used and {x, y} <= g["adj"][u]]
        if not mids:
            continue
        u = max(mids, key=lambda u: g["by_id"][u]["area_px"])
        g["faces"].remove(u)
        g["by_id"][u]["class"] = "other"
        for n in g["adj"].pop(u):
            g["adj"][n].discard(u)
            g["bends"].pop(frozenset((u, n)), None)
        g["adj"][x].add(y)
        g["adj"][y].add(x)
        g["bends"][frozenset((x, y))] = {"region": u}
        # u may be a neighbour through a bend region rather than by touching: keep face_graph's contact points
        g["contact"][(x, y)] = g["contact"][(x, u)]
        g["contact"][(y, x)] = g["contact"][(y, u)]
        used.add(u)


def match_holes(ref: dict, other: dict, a: str, b: str, fmap: dict) -> list[tuple[int, int]]:
    """Pairs (i in a's holes, j in b's holes). Holes are told apart by their distances to the bends to matched
    neighbours (the order of distances to a line in a plane survives any parallel projection) and by size."""
    ha, hb = ref["by_id"][a]["holes"], other["by_id"][b]["holes"]
    if not ha or not hb:
        return []
    nbrs = [(n, fmap[n]) for n in ref["adj"][a] if fmap.get(n) and fmap[n] in other["adj"][b]]

    def dists(g, f, hs, n):
        pts = g["contact"][(f, n)]
        return [float(np.min(np.linalg.norm(pts - np.array(h["centre"]), axis=1))) for h in hs]

    feats = [(dists(ref, a, ha, n), dists(other, b, hb, m)) for n, m in nbrs]
    best, best_s = [], -1e9
    small, large = (ha, hb) if len(ha) <= len(hb) else (hb, ha)
    for perm in itertools.permutations(range(len(large)), len(small)):
        pairs = list(zip(range(len(small)), perm)) if small is ha else [(p, i) for i, p in enumerate(perm)]
        s = 0.0
        for (i1, j1), (i2, j2) in itertools.combinations(pairs, 2):
            for da, db in feats:
                s += np.sign(da[i1] - da[i2]) * np.sign(db[j1] - db[j2])
            s += 0.5 * np.sign(ha[i1]["diameter"] - ha[i2]["diameter"]) * np.sign(hb[j1]["diameter"] - hb[j2]["diameter"])
        if s > best_s:
            best, best_s = pairs, s
    return best


# --------------------------------------------------------------------------- parts, groups, images
def mask_polygon(mask: np.ndarray, zoom: float) -> list[list[float]]:
    m = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = cv2.approxPolyDP(max(cnt, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
    return [[float(x) / zoom, float(y) / zoom] for x, y in c]


def circle_polygon(cx, cy, r, zoom) -> list[list[float]]:
    return [[(cx + r * math.cos(a * math.pi / 24)) / zoom, (cy + r * math.sin(a * math.pi / 24)) / zoom] for a in range(48)]


def line_polygon(p, q, zoom, half=2.0) -> list[list[float]]:
    p, q = np.array(p, float), np.array(q, float)
    d = (q - p) / (np.linalg.norm(q - p) or 1)
    n = np.array([-d[1], d[0]]) * half
    return [((x / zoom).tolist()) for x in (p + n, q + n, q - n, p - n)]


def view_parts(view: dict, graph: dict, zoom: float) -> list[dict]:
    """part_views-style parts of one view: faces (with holes_pt), holes, bends (strips or lines), other regions.
    Each also keeps "key": what it is in the view graph (face id, (face id, hole index), or the bend's pair)."""
    parts, by_id = [], graph["by_id"]
    bend_regions = {v["region"]: k for k, v in graph["bends"].items() if "region" in v}
    hole_regions = {h["region"] for f in graph["faces"] for h in by_id[f]["holes"] if "region" in h}

    def hole_poly(h):
        return circle_polygon(*h["circle"], zoom) if "circle" in h else mask_polygon(h["mask"], zoom)

    for f in graph["faces"]:
        r = by_id[f]
        parts.append({"name": f"face {f}", "feature": "panel", "key": f, "polygon_pt": mask_polygon(r["mask"], zoom),
                      "holes_pt": [hole_poly(h) for h in r["holes"]], "hidden": False,
                      "notes": f"{len(r['holes'])} holes" + (f"; moved from {view['transferred_from']}"
                                                              if view.get("transferred_from") else "")})
        for i, h in enumerate(r["holes"]):
            parts.append({"name": f"hole {i + 1} of {f}", "feature": "hole", "key": (f, i), "polygon_pt": hole_poly(h),
                          "holes_pt": [], "hidden": False, "notes": f"diameter {h['diameter'] / zoom:.1f} pt"})
    for pair, b in graph["bends"].items():
        a, c = sorted(pair)
        if "region" in b:
            poly = mask_polygon(by_id[b["region"]]["mask"], zoom)
        elif b.get("line"):
            poly = line_polygon(*b["line"], zoom)
        else:
            continue
        parts.append({"name": f"bend {a}-{c}", "feature": "bend", "key": pair, "polygon_pt": poly, "holes_pt": [],
                      "hidden": False, "notes": ""})
    for r in by_id.values():
        if r["class"] == "other" and r["id"] not in bend_regions and r["id"] not in hole_regions:
            parts.append({"name": f"region {r['id']}", "feature": "other", "key": r["id"],
                          "polygon_pt": mask_polygon(r["mask"], zoom), "holes_pt": [], "hidden": False, "notes": ""})
    for p in parts:
        p["outline"] = {"polygon": [[round(x * zoom, 1), round(y * zoom, 1)] for x, y in p["polygon_pt"]]}
    return parts


def draw(gray: np.ndarray, items: list[tuple[str, list, list, tuple]], zoom: float) -> np.ndarray:
    """(label, polygon_pt, holes_pt, rgb 0-1) drawn over the view, like part_views.draw_view."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fill = img.copy()
    for _, poly, holes, rgb in items:
        bgr = tuple(int(255 * v) for v in rgb[::-1])
        rings = [np.array(r, float) * zoom for r in [poly, *holes]]
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(mask, [rings[0].round().astype(np.int32)], 1)
        for hole in rings[1:]:
            cv2.fillPoly(mask, [hole.round().astype(np.int32)], 0)
        fill[mask > 0] = bgr
    img = cv2.addWeighted(img, 0.62, fill, 0.38, 0)
    for _, poly, holes, rgb in items:
        bgr = tuple(int(255 * v) for v in rgb[::-1])
        for r in [poly, *holes]:
            cv2.polylines(img, [(np.array(r) * zoom).round().astype(np.int32)], True, bgr, 2, cv2.LINE_AA)
    for label, poly, _, rgb in items:
        m = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(m, [(np.array(poly) * zoom).round().astype(np.int32)], 1)
        x, y = (int(v) for v in regions.deepest_point(m > 0)) if m.any() else (0, 0)
        bgr = tuple(int(255 * v) for v in rgb[::-1])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 4), (255, 255, 255), -1)
        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 4), bgr, 1)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)
    return img


def side_by_side(images: list[np.ndarray], names: list[str], height: int = 700) -> np.ndarray:
    tiles = []
    for img, name in zip(images, names):
        t = cv2.resize(img, (round(img.shape[1] * height / img.shape[0]), height), interpolation=cv2.INTER_AREA)
        t = cv2.copyMakeBorder(t, 40, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, name, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(t)
    return np.hstack(tiles)


# --------------------------------------------------------------------------- run
def run(drawing: Path, run_dir: Path, views_json: Path | None = None) -> dict:
    (run_dir / "views").mkdir(parents=True, exist_ok=True)
    views, zoom = load_views(drawing, views_json)
    sheet = regions.render(str(drawing))
    cv2.imwrite(str(run_dir / "sheet.png"), cv2.resize(sheet, None, fx=1600 / max(sheet.shape), fy=1600 / max(sheet.shape),
                                                      interpolation=cv2.INTER_AREA))
    timing = {}
    for v in views:
        t = time.time()
        v["regions"] = segment(v["image"], v["kind"])
        timing[v["id"]] = {"segment_s": round(time.time() - t, 2)}

    # transfer: a flat view with far more regions than a cleaner flat view of the same pattern takes its panels
    flats = [v for v in views if is_flat(v) and any(r["class"] == "face" for r in v["regions"])]
    for dst in flats:
        for src in sorted(flats, key=lambda v: len(v["regions"])):
            if src is dst or len(dst["regions"]) < 2 * len(src["regions"]) or src.get("transferred_from"):
                continue
            t = time.time()
            reg = register(outline_points([r for r in src["regions"] if r["class"] == "face"]), dst["image"])
            timing[dst["id"]]["register_s"] = round(time.time() - t, 2)
            print(f"{dst['id']}: registered {src['id']} onto it, {reg['support']:.0%} of its outline on ink "
                  f"({reg['chance']:.0%} at random poses)", flush=True)
            # an outline of straight lines fits anywhere on a dimensioned view: the source's holes must land on
            # circles of the same size too
            src_holes = [h["circle"] for r in src["regions"] if r["class"] == "face" for h in r["holes"]]
            A, scale = reg["A"], math.sqrt(abs(np.linalg.det(reg["A"][:, :2])))
            dst_circles = find_circles(dst["image"]) if src_holes else []
            landed = sum(any(math.dist(A[:, :2] @ (cx, cy) + A[:, 2], c[:2]) < 4 and abs(r * scale - c[2]) < 0.2 * r * scale
                             for c in dst_circles) for cx, cy, r in src_holes)
            print(f"   {landed}/{len(src_holes)} of {src['id']}'s holes land on circles", flush=True)
            if reg["support"] >= MIN_SUPPORT and reg["chance"] <= MAX_CHANCE and src_holes and landed == len(src_holes):
                dst["regions"] = transfer(src["regions"], dst["image"], reg["A"])
                dst["transferred_from"], dst["registration"] = src["id"], reg
                break

    graphs = {v["id"]: face_graph(v["regions"]) for v in views}
    # reference: the cleanest flat view (fewest regions per face), so every other view is matched to clean panels
    ref = min(views, key=lambda v: (not is_flat(v), not graphs[v["id"]]["faces"],
                                    len(v["regions"]) / max(1, len(graphs[v["id"]]["faces"]))))
    rg = graphs[ref["id"]]
    maps, alts = {ref["id"]: {f: f for f in rg["faces"]}}, {}
    for v in views:
        if v is ref:
            continue
        t = time.time()
        if v.get("transferred_from") == ref["id"]:
            maps[v["id"]] = {f: f for f in rg["faces"] if f in graphs[v["id"]]["faces"]}
        else:
            maps[v["id"]], best, runners = match_faces(rg, graphs[v["id"]])
            alts[v["id"]] = {"score": best, "runners_up": [[s, m] for s, m in runners]}
            absorb_bends(rg, graphs[v["id"]], maps[v["id"]])
        timing[v["id"]]["match_s"] = round(time.time() - t, 3)

    # parts, numbered P1, P2, ... across views as part_views does
    n = 0
    key_to_label = {}
    for v in views:
        v["parts"] = view_parts(v, graphs[v["id"]], zoom)
        for p in v["parts"]:
            n += 1
            p["id"] = p["local_id"] = f"P{n}"
            key_to_label[(v["id"], p["key"])] = p["id"]
        g = graphs[v["id"]]
        for p in v["parts"]:     # a panel's neighbours through bends, as part labels
            if p["feature"] == "panel":
                p["neighbours"] = sorted(key_to_label[(v["id"], f)] for f in g["adj"].get(p["key"], ())
                                         if (v["id"], f) in key_to_label)

    groups = []
    for f in rg["faces"]:
        members = [(v["id"], maps[v["id"]].get(f)) for v in views if maps[v["id"]].get(f)]
        nh = len(rg["by_id"][f]["holes"])
        role = "end tab" if len(rg["adj"][f]) == 1 else "panel"
        groups.append({"kind": "face", "name": f"{role} with {nh} hole{'s' * (nh != 1)} ({ref['id']} {f})",
                       "members": [(vid, key_to_label[(vid, k)]) for vid, k in members]})
        # holes of this face, following the hole pairing in each view
        for i, h in enumerate(rg["by_id"][f]["holes"]):
            hm = [(ref["id"], key_to_label[(ref["id"], (f, i))])]
            for vid, k in members:
                if vid == ref["id"]:
                    continue
                pairs = match_holes(rg, graphs[vid], f, k, maps[vid]) if not views_by(views, vid).get("transferred_from") \
                    else _nearest_holes(rg, graphs[vid], f, k, views_by(views, vid)["registration"]["A"])
                j = next((j for i2, j in pairs if i2 == i), None)
                if j is not None:
                    hm.append((vid, key_to_label[(vid, (k, j))]))
            groups.append({"kind": "hole", "name": f"hole {i + 1} of {f} ({h['diameter'] / zoom:.1f} pt in {ref['id']})",
                           "members": hm})
    for pair in rg["bends"]:
        a, b = sorted(pair)
        members = []
        for v in views:
            m = maps[v["id"]]
            if m.get(a) and m.get(b):
                key = frozenset((m[a], m[b]))
                if key in graphs[v["id"]]["bends"] and (v["id"], key) in key_to_label:
                    members.append((v["id"], key_to_label[(v["id"], key)]))
        groups.append({"kind": "bend", "name": f"bend between {a} and {b}", "members": members})
    # faces of other views that matched nothing get their own group
    for v in views:
        used = set(maps[v["id"]].values())
        for f in graphs[v["id"]]["faces"]:
            if f not in used:
                groups.append({"kind": "face", "name": f"unmatched face {f} of {v['id']}",
                               "members": [(v["id"], key_to_label[(v["id"], f)])]})
    out_groups = []
    for k, g in enumerate([g for g in groups if g["members"]], 1):
        by_view = {}
        for vid, label in g["members"]:
            by_view.setdefault(vid, []).append(label)
        out_groups.append({"id": f"G{k}", "name": g["name"], "kind": g["kind"],
                           "members": [{"view": vid, "parts": ps} for vid, ps in by_view.items()],
                           "confidence": None, "reason": "local geometry: hole counts, neighbours through bends"})
    matches = {"groups": out_groups, "reference_view": ref["id"], "face_maps": maps, "alternatives": alts,
               "timing": timing}

    # files
    owner = {}
    for i, g in enumerate(out_groups):
        for m in g["members"]:
            for label in m["parts"]:
                owner[label] = (g["id"], PALETTE[i % len(PALETTE)])
    matched = []
    for v in views:
        cv2.imwrite(str(run_dir / "views" / f"{v['id']}.png"), v["image"])
        colored = draw(v["image"], [(p["id"], p["polygon_pt"], p["holes_pt"], PALETTE[i % len(PALETTE)])
                                    for i, p in enumerate(v["parts"]) if p["feature"] != "hole"], zoom)
        cv2.imwrite(str(run_dir / "views" / f"{v['id']}_colored.png"), colored)
        img = draw(v["image"], [(owner[p["id"]][0], p["polygon_pt"], p["holes_pt"], owner[p["id"]][1])
                                for p in sorted(v["parts"], key=lambda p: p["feature"] == "hole") if p["id"] in owner], zoom)
        cv2.imwrite(str(run_dir / "views" / f"{v['id']}_matched.png"), img)
        matched.append(img)
    cv2.imwrite(str(run_dir / "matched_all.png"), side_by_side(matched, [v["id"] for v in views]))
    out_views = [{k: v[k] for k in ("id", "sheet", "name", "kind", "rect_pt")} |
                 {"parts": [{k: p[k] for k in p if k != "key"} for p in v["parts"]],
                  "transferred_from": v.get("transferred_from")} for v in views]
    (run_dir / "views.json").write_text(json.dumps(out_views, indent=2))
    (run_dir / "matches.json").write_text(json.dumps(matches, indent=2, default=_jsonable))
    return {"views": out_views, "matches": matches, "zoom": zoom}


def views_by(views, vid):
    return next(v for v in views if v["id"] == vid)


def _nearest_holes(ref: dict, other: dict, a: str, b: str, A: np.ndarray) -> list[tuple[int, int]]:
    """Holes of a view whose panels were moved from the reference: pair each with the nearest moved hole."""
    pairs = []
    for i, h in enumerate(ref["by_id"][a]["holes"]):
        p = A[:, :2] @ np.array(h["centre"]) + A[:, 2]
        cands = [(math.dist(p, k["centre"]), j) for j, k in enumerate(other["by_id"][b]["holes"])]
        if cands and min(cands)[0] < 10:
            pairs.append((i, min(cands)[1]))
    return pairs


def _jsonable(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(type(o))
