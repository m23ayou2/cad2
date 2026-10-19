"""Views and regions of a drawing sheet, from its pixels alone (scans and vector PDFs alike).

1. view_crops: the views on the sheet. The sheet frame and the title block's rules (long straight lines) are
   removed, nearby ink is merged into blobs (a view with its dimension stacks), and blobs that are only letters
   (notes) or lie in the title block are dropped. Each crop keeps only its own blob's ink, so a neighbouring view
   that overlaps its box does not leak into it.
2. fill_regions: every region a view's lines enclose, filled from inside up to the ink: a face, a panel, a hole,
   or (on a dimensioned view) a strip between dimension lines. Each gets an id, P1 = the largest, the regions it
   touches, and the straight strokes it shares with a neighbour (on a flat blank, its bend lines).
3. label_image / focus_image: the view with every region coloured and labelled, and one region picked out, for a
   vision model.

    uv run python src/regions.py <drawing.pdf> <out folder>
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import pymupdf

SHEET_PX = 3072          # long side of the rendered sheet
INK = 200                # darker than this is ink

PALETTE = [(120, 180, 240), (140, 220, 140), (240, 170, 120), (200, 150, 230), (110, 220, 220), (230, 220, 110),
           (240, 140, 180), (170, 170, 250), (160, 230, 190), (250, 200, 150), (190, 210, 120), (150, 190, 210)]


def render(pdf: str) -> np.ndarray:
    with pymupdf.open(pdf) as doc:
        page = doc[0]
        zoom = SHEET_PX / max(page.rect.width, page.rect.height)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


# --------------------------------------------------------------------------- views
def view_crops(gray: np.ndarray) -> list[dict]:
    """The views on the sheet: [{"id": "V1", "box": [x0, y0, x1, y1] (pixels), "image": the crop, with only this
    view's ink}], top to bottom, left to right."""
    h, w = gray.shape
    ink = (gray < INK).astype(np.uint8)
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.25 * w), 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.25 * h))))
    rules = cv2.dilate(horiz | vert, np.ones((5, 5), np.uint8))
    inner = np.ones_like(ink)
    ys, xs = np.nonzero(horiz | vert)
    if len(xs):                                    # inside the frame, past its zone strip
        m = int(0.012 * w)
        inner[:] = 0
        inner[ys.min() + m:ys.max() - m, xs.min() + m:xs.max() - m] = 1
        corner = (xs.max() - m - 5, ys.max() - m - 5)  # the title block sits in the frame's lower-right corner
    else:
        corner = None
    body = ink & ~rules & inner
    k = max(15, int(0.02 * w))
    blobs = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(blobs, 8)
    _, lab_ink, st_ink, _ = cv2.connectedComponentsWithStats(body, 8)
    found = []
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        if bw * bh < 0.004 * w * h:
            continue
        mine = (lab == i) & (body > 0)
        strokes = np.unique(lab_ink[mine])
        longest = max((max(st_ink[j, 2], st_ink[j, 3]) for j in strokes if j), default=0)
        if longest < max(60, 0.12 * max(bw, bh)):           # only letters: notes
            continue
        found.append((i, [int(x), int(y), int(x + bw), int(y + bh)]))
    title = [b for i, b in found if corner and b[0] <= corner[0] <= b[2] and b[1] <= corner[1] <= b[3]]
    inside = lambda b, t: b[0] >= t[0] - 5 and b[1] >= t[1] - 5 and b[2] <= t[2] + 5 and b[3] <= t[3] + 5
    views = [(i, b) for i, b in found if not any(inside(b, t) for t in title)]
    views.sort(key=lambda v: (round(v[1][1] / (0.1 * h)), v[1][0]))
    out = []
    for k, (i, b) in enumerate(views, 1):
        x0, y0, x1, y1 = b
        keep = (lab[y0:y1, x0:x1] == i)
        crop = np.where(keep, gray[y0:y1, x0:x1], 255).astype(np.uint8)
        out.append({"id": f"V{k}", "box": b, "image": crop})
    return out


# --------------------------------------------------------------------------- regions
def fill_regions(gray: np.ndarray, min_area: int = 300) -> list[dict]:
    """Every region the view's lines enclose: [{"id": "P1", "mask", "area_px", "box", "inside" (a point deep
    inside it), "touches": [ids], "shared_lines": [{"with": id, "from", "to"}] (straight strokes it shares with a
    neighbour)}], P1 = the largest. Letters and specks are left out by area."""
    ink = (gray < INK).astype(np.uint8)
    strokes = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    h, w = strokes.shape
    flood = strokes.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    n, lab, st, _ = cv2.connectedComponentsWithStats((flood == 0).astype(np.uint8), 4)
    if n < 2:
        return []
    largest = st[1:, 4].max()
    keep = sorted((i for i in range(1, n) if st[i, 4] >= max(min_area, 0.005 * largest)), key=lambda i: -st[i, 4])
    ids = {lab_i: f"P{k}" for k, lab_i in enumerate(keep, 1)}
    grown = {i: cv2.dilate((lab == i).astype(np.uint8), np.ones((7, 7), np.uint8)) for i in keep}
    regions = []
    for i in keep:
        x, y, bw, bh, area = st[i]
        mask = lab == i
        regions.append({"id": ids[i], "mask": mask, "area_px": int(area), "box": [int(x), int(y), int(x + bw), int(y + bh)],
                        "inside": deepest_point(mask), "touches": [], "shared_lines": []})
    by_id = {r["id"]: r for r in regions}
    for a in keep:
        for b in keep:
            if b <= a:
                continue
            shared = (grown[a] & grown[b]).astype(bool)
            if shared.sum() < 20:
                continue
            by_id[ids[a]]["touches"].append(ids[b])
            by_id[ids[b]]["touches"].append(ids[a])
            ys, xs = np.nonzero(shared)
            pts = np.stack([xs, ys], 1).astype(float)
            mu = pts.mean(0)
            _, sv, vt = np.linalg.svd(pts - mu, full_matrices=False)
            t = (pts - mu) @ vt[0]
            if sv[1] / math.sqrt(len(pts)) <= 2.0 and t.max() - t.min() >= 15:      # one straight stroke
                p, q = (mu + vt[0] * t.min()).round(1).tolist(), (mu + vt[0] * t.max()).round(1).tolist()
                by_id[ids[a]]["shared_lines"].append({"with": ids[b], "from": p, "to": q})
                by_id[ids[b]]["shared_lines"].append({"with": ids[a], "from": p, "to": q})
    return regions


def deepest_point(mask: np.ndarray) -> list[float]:
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return [float(x), float(y)]


# --------------------------------------------------------------------------- images
def _text(img, text, at, scale=0.6, colour=(0, 0, 0)):
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def label_image(gray: np.ndarray, regions: list[dict]) -> np.ndarray:
    """The view with every region in its own colour and its id written inside it."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(float)
    for k, r in enumerate(regions):
        img[r["mask"]] = img[r["mask"]] * 0.35 + np.array(PALETTE[k % len(PALETTE)]) * 0.65
    img = img.astype(np.uint8)
    for r in regions:
        x, y = (int(v) for v in r["inside"])
        _text(img, r["id"], (x - 12, y + 6))
    return img


def focus_image(gray: np.ndarray, region: dict, margin: float = 1.0, min_side: int = 500) -> np.ndarray:
    """One region picked out: the drawing around it (its box grown by `margin` on every side, so the dimensions
    beside it are in), the region tinted, its outline drawn, enlarged so small text is readable."""
    h, w = gray.shape
    x0, y0, x1, y1 = region["box"]
    mx, my = int((x1 - x0) * margin) + 80, int((y1 - y0) * margin) + 80
    x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(float)
    m = region["mask"]
    img[m] = img[m] * 0.5 + np.array((80, 200, 255)) * 0.5
    img = img.astype(np.uint8)
    # outline the region as a whole: close the thin notches leaders and centre lines cut into it
    closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 120, 255), 1, cv2.LINE_AA)
    crop = img[y0:y1, x0:x1]
    zoom = max(1.0, min(3.0, min_side / max(1, min(crop.shape[:2]))))
    crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    _text(crop, region["id"], (10, 30), 0.9, (0, 90, 200))
    return crop


if __name__ == "__main__":
    pdf, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    sheet = render(pdf)
    for v in view_crops(sheet):
        regions = fill_regions(v["image"])
        cv2.imwrite(os.path.join(out, f"{v['id']}.png"), v["image"])
        cv2.imwrite(os.path.join(out, f"{v['id']}_regions.png"), label_image(v["image"], regions))
        print(v["id"], v["box"], f"{len(regions)} regions")
