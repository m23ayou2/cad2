"""Separate a scanned drawing's part edges from its annotations (prototype for tier-4 scans).

A vision model confuses part outlines with dimension lines, and a plain grey threshold cannot split them:
anti-aliasing makes the edge of every stroke grey. What does split them is the darkness at the *core* of a stroke.
On these scans part edges are drawn darker (core grey about 120-145) than dimension, extension and leader lines and
their text (about 175-180). So:

1. core:   pixels darker than CORE are the cores of part strokes; grow them by a pixel, within the ink, to get the
           whole stroke back (curves and all);
2. specks: drop small isolated marks (dimension arrowheads and stray dark dots) that survive the threshold;
3. axes:   centre lines and cutting-plane lines are dash-dot runs; they go to their own layer: useful (they mark axes
           and cuts) but they are not edges of the part.

Everything else that is ink is annotation.

    uv run python src/raster_lines.py <drawing.pdf> <out folder> [x0 y0 x1 y1]

writes part.png (part edges only), axes.png, annotations.png, overlay.png (part black, axes green, annotations
light red) and lines.json.
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np
import pymupdf

SHEET_PX = 3072
INK = 215        # grey below which a pixel is ink at all
CORE = 155       # grey below which a pixel is the core of a part edge
SPECK = 12       # px: isolated marks smaller than this are arrowheads or dots, not edges


def render(pdf: str, box=None) -> np.ndarray:
    with pymupdf.open(pdf) as doc:
        page = doc[0]
        zoom = SHEET_PX / max(page.rect.width, page.rect.height)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY, alpha=False)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    if box:
        h, w = gray.shape
        gray = gray[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)].copy()
    return gray


def part_strokes(gray: np.ndarray) -> np.ndarray:
    ink = gray < INK
    core = (gray < CORE).astype(np.uint8)
    strokes = cv2.dilate(core, np.ones((3, 3), np.uint8)).astype(bool) & ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats(strokes.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= SPECK
    return keep[labels]


def angle(s) -> float:
    return math.degrees(math.atan2(s[3] - s[1], s[2] - s[0])) % 180


def dashed_runs(strokes: np.ndarray, max_dash=40, max_gap=30) -> tuple[np.ndarray, np.ndarray]:
    """Centre and cutting-plane lines: runs of at least three short collinear pieces with gaps. Returns the axes
    mask and the segments found on the strokes."""
    img = np.where(strokes, 0, 255).astype(np.uint8)
    found = cv2.createLineSegmentDetector().detect(img)[0]
    segs = np.zeros((0, 4)) if found is None else found.reshape(-1, 4)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) if len(segs) else np.zeros(0)
    short = [i for i in range(len(segs)) if lengths[i] < max_dash]
    axes = np.zeros(strokes.shape, np.uint8)
    for i in short:
        s = segs[i]
        d = (s[2:] - s[:2]) / (lengths[i] or 1)
        normal = np.array([-d[1], d[0]])
        run = [j for j in short if abs((angle(segs[j]) - angle(s) + 90) % 180 - 90) < 3
               and abs(np.dot(segs[j][:2] - s[:2], normal)) < 2 and abs(np.dot(segs[j][2:] - s[:2], normal)) < 2]
        if len(run) < 3:
            continue
        # the pieces must be spread along the line with gaps, not stacked on top of each other
        along = sorted(np.dot(segs[j][:2] - s[:2], d) for j in run)
        if max(b - a for a, b in zip(along, along[1:])) > max_gap * 4:
            continue
        for j in run:
            cv2.line(axes, tuple(int(v) for v in segs[j][:2]), tuple(int(v) for v in segs[j][2:]), 1, 4)
    return axes.astype(bool), segs


def separate(gray: np.ndarray) -> dict:
    ink = gray < INK
    strokes = part_strokes(gray)
    axes, segs = dashed_runs(strokes)
    axes &= strokes
    part = strokes & ~axes
    annotation = ink & ~strokes
    overlay = np.full(gray.shape + (3,), 255, np.uint8)
    overlay[annotation] = (170, 170, 255)
    overlay[axes] = (0, 170, 0)
    overlay[part] = (0, 0, 0)
    return {"part": part, "axes": axes, "annotation": annotation, "overlay": overlay, "segments": segs,
            "pixels": {"ink": int(ink.sum()), "part": int(part.sum()), "axes": int(axes.sum()),
                       "annotation": int(annotation.sum())}}


def main(pdf: str, out_dir: str, box=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    gray = render(pdf, box)
    r = separate(gray)
    for name in ("part", "axes", "annotation"):
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), np.where(r[name], 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, "overlay.png"), r["overlay"])
    with open(os.path.join(out_dir, "lines.json"), "w") as f:
        json.dump({"pixels": r["pixels"], "core_threshold": CORE, "speck_px": SPECK}, f)
    return r


if __name__ == "__main__":
    args = sys.argv[1:]
    box = [float(v) for v in args[2:6]] if len(args) >= 6 else None
    print(main(args[0], args[1], box)["pixels"])


# =========================================================================== structural separation
# For sheets where dimension lines are drawn like part edges (same darkness, same width), only structure tells them
# apart: (1) arrowheads mark dimension lines, (2) extension lines cross an arrow tip at right angles.

def text_mask(ink: np.ndarray, glyph=(5, 30), min_glyphs=2) -> np.ndarray:
    """Clusters of glyph-sized marks: dimension values and callouts."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    glyphs = np.zeros(ink.shape, np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if glyph[0] <= max(w, h) <= glyph[1] and min(w, h) >= 3:
            glyphs[labels == i] = 1
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(glyphs, np.ones((7, 7), np.uint8)), 8)
    mask = np.zeros(ink.shape, np.uint8)
    for i in range(1, n):
        if cv2.connectedComponents((glyphs * (labels == i)).astype(np.uint8), connectivity=8)[0] - 1 >= min_glyphs:
            mask[labels == i] = 1
    return mask.astype(bool)


def find_arrowheads(ink: np.ndarray) -> list[dict]:
    """Arrowheads are the only filled marks: where the ink is thicker than a line stroke (distance to the
    background at least 1.5 px). Each blob's axis is its longest extent; the tip is the narrow end."""
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    thick = (dist >= 1.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(thick, np.ones((3, 3), np.uint8)), 8)
    heads = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (3 <= max(w, h) <= 25 and area <= 200):
            continue
        # the arrowhead's own ink, around the thick core
        sub = ink[max(0, y - 4):y + h + 4, max(0, x - 4):x + w + 4] & \
            (cv2.dilate((labels == i).astype(np.uint8), np.ones((7, 7), np.uint8))[max(0, y - 4):y + h + 4,
                                                                                 max(0, x - 4):x + w + 4] > 0)
        ys, xs = np.nonzero(sub)
        if len(xs) < 6:
            continue
        pts = np.stack([xs + max(0, x - 4), ys + max(0, y - 4)], 1).astype(float)
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c)
        axis = vt[0]                                  # along the arrow
        t = (pts - c) @ axis
        ends = [pts[np.argmin(t)], pts[np.argmax(t)]]
        # the tip is the end where the blob is narrowest
        width = lambda end: np.sum(np.abs((pts - end) @ axis) < 2)
        tip, base = (ends[0], ends[1]) if width(ends[0]) < width(ends[1]) else (ends[1], ends[0])
        heads.append({"centre": c, "tip": tip, "axis": (tip - base) / (np.linalg.norm(tip - base) or 1),
                      "length": float(np.linalg.norm(tip - base))})
    return heads


def follow(ink: np.ndarray, start, direction, max_gap=3, limit=4000) -> list:
    """Pixels of a straight ink run from `start` along `direction`, stepping 0.5 px, until a gap of more than
    `max_gap` px. Looks one pixel either side of the line, for anti-aliasing."""
    d = np.array(direction, float) / np.linalg.norm(direction)
    nrm = np.array([-d[1], d[0]])
    h, w = ink.shape
    pts, gap, p = [], 0.0, np.array(start, float)
    for _ in range(int(limit * 2)):
        hit = None
        for o in (0, -1, 1):
            q = (p + nrm * o).round().astype(int)
            if 0 <= q[0] < w and 0 <= q[1] < h and ink[q[1], q[0]]:
                hit = q
                break
        if hit is None:
            gap += 0.5
            if gap > max_gap:
                break
        else:
            gap = 0.0
            pts.append(tuple(hit))
        p = p + d * 0.5
    return pts


def separate_structure(gray: np.ndarray) -> dict:
    ink = gray < 200
    text = text_mask(ink)
    heads = find_arrowheads(ink & ~text)
    dimension = np.zeros(ink.shape, bool)
    extension = np.zeros(ink.shape, bool)
    # thin annotation lines are anti-aliased light: follow them on a looser ink threshold, so they do not break up
    lines = (gray < 235) & ~text
    # (1) dimension lines: from each arrowhead, back along its axis (away from the tip) while the ink continues
    dim_dirs, dim_runs = [], []
    for a in heads:
        # the blob's own axis is only approximate (a crossing extension line skews it), and which end is the base
        # is not always clear on a small blob: search directions within 30 degrees of the axis, both ways, and keep
        # the longest straight ink run. The line leaves the arrowhead from its base.
        base = math.atan2(a["axis"][1], a["axis"][0])
        best, best_dir = [], None
        for sense in (0.0, math.pi):
            for k in range(-20, 21):
                th = base + sense + math.radians(1.5 * k)
                d = np.array([math.cos(th), math.sin(th)])
                run = follow(lines, a["centre"], d)
                if len(run) > len(best):
                    best, best_dir = run, d
        if best_dir is not None and len(best) > 2 * max(a["length"], 4):
            for x, y in best:
                dimension[y, x] = True
            dim_runs.append({"points": [a["centre"].tolist()] + [list(map(int, q)) for q in best],
                             "direction": best_dir, "centre": a["centre"], "length": a["length"]})
            # the tip is on the other side of the centre from the line
            dim_dirs.append({**a, "axis": -best_dir, "tip": a["centre"] - best_dir * a["length"] / 2})
    # (2) extension lines: straight ink runs through an arrow tip, at right angles to its dimension line. The tip
    # is only known to a pixel or two, so look a few pixels either side of it along the dimension line and keep
    # the longest perpendicular run on each side.
    free = lines & ~cv2.dilate(dimension.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for a in dim_dirs:
        perp = np.array([-a["axis"][1], a["axis"][0]])
        for sign in (1, -1):
            best = []
            for off in np.arange(-3, 6.01, 0.5):
                start = a["tip"] + a["axis"] * off + perp * sign * 2.5   # clear of the dimension line itself
                run = follow(free, start, perp * sign, max_gap=4)
                if len(run) > len(best):
                    best = run
            if len(best) >= 8:
                for x, y in best:
                    extension[y, x] = True
    thick = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ink
    dimension, extension = thick(dimension), thick(extension) & ~thick(dimension)
    arrows = np.zeros(ink.shape, bool)
    for a in heads:
        cv2.circle(arrows.view(np.uint8), tuple(int(v) for v in a["centre"]), int(max(3, a["length"] / 2 + 1)), 1, -1)
    arrows &= ink
    part = ink & ~text & ~dimension & ~extension & ~arrows
    overlay = np.full(gray.shape + (3,), 255, np.uint8)
    overlay[part] = (0, 0, 0)
    overlay[extension] = (0, 140, 255)
    overlay[dimension] = (0, 0, 230)
    overlay[text & ink] = (230, 120, 0)
    overlay[arrows] = (200, 0, 200)
    return {"part": part, "dimension": dimension, "extension": extension, "text": text & ink, "arrows": heads,
            "overlay": overlay, "dimension_runs": dim_runs}


def main_structure(pdf: str, out_dir: str, box=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    gray = render(pdf, box)
    r = separate_structure(gray)
    cv2.imwrite(os.path.join(out_dir, "overlay.png"), r["overlay"])
    cv2.imwrite(os.path.join(out_dir, "part.png"), np.where(r["part"], 0, 255).astype(np.uint8))
    return r
