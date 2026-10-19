"""Fit a traced outline with the lines and arcs a vision model names (deterministic; used by trace_outline).

The outline of a scanned view is traced as a closed pixel contour with numbered corners V1..Vn (approxPolyDP
vertices, in contour order). The model says what each stretch between two corners is: a straight edge, an arc
(with its labelled radius when the drawing gives one), or leaves it as traced. Each stretch is fitted to the
contour pixels with a robust fit (a leader crossing the edge, a jog in the trace, are outliers), consecutive
pieces are joined where they meet (tangent arcs touch, lines intersect), and the result is a closed outline of
exact primitives in pixels, with a fit report per stretch.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

INLIER_PX = 1.5        # a scanned edge is accurate to about a pixel


def vertices(contour: np.ndarray, eps: float = 3.0) -> list[int]:
    """Corner indices into the contour (approxPolyDP vertices), in contour order."""
    import cv2
    c = np.asarray(contour, np.int32).reshape(-1, 1, 2)
    index = {tuple(p): i for i, p in enumerate(c.reshape(-1, 2).tolist())}
    return sorted(index[tuple(p)] for p in cv2.approxPolyDP(c, eps, True).reshape(-1, 2).tolist())


def _stretch(contour: np.ndarray, i: int, j: int) -> np.ndarray:
    """Contour points from index i to index j, going forward (wrapping round)."""
    n = len(contour)
    k = (j - i) % n or n
    return contour[[(i + m) % n for m in range(k + 1)]]


# --------------------------------------------------------------------------- robust fits
def fit_line(pts: np.ndarray) -> dict:
    keep = np.ones(len(pts), bool)
    for _ in range(4):
        mu = pts[keep].mean(0)
        d = np.linalg.svd(pts[keep] - mu, full_matrices=False)[2][0]
        normal = np.array([-d[1], d[0]])
        res = np.abs((pts - mu) @ normal)
        keep = res <= max(INLIER_PX, float(np.percentile(res, 70)))
    return {"kind": "line", "point": mu, "dir": d, "residuals": res}


def fit_arc(pts: np.ndarray, radius_px: float | None) -> dict:
    a = np.c_[pts, np.ones(len(pts))]
    sol = np.linalg.lstsq(a, -(pts ** 2).sum(1), rcond=None)[0]
    centre = -sol[:2] / 2
    r0 = math.sqrt(max(centre @ centre - sol[2], 1e-9))
    if radius_px is not None and abs(r0 - radius_px) > 0.2 * radius_px:
        # start from the chord: centre on the side the stretch bulges away from, at the given radius
        p, q, mid = pts[0], pts[-1], pts[len(pts) // 2]
        half = max(np.linalg.norm(q - p) / 2, 1e-6)
        m = (p + q) / 2
        normal = np.array([-(q - p)[1], (q - p)[0]]) / (2 * half)
        if (mid - m) @ normal > 0:
            normal = -normal
        centre = m + normal * math.sqrt(max(radius_px ** 2 - half ** 2, 0.0))
    if radius_px is None:
        f = lambda x: np.linalg.norm(pts - x[:2], axis=1) - x[2]
        x = least_squares(f, [*centre, r0], loss="soft_l1", f_scale=INLIER_PX).x
        centre, r = x[:2], float(x[2])
    else:
        f = lambda x: np.linalg.norm(pts - x, axis=1) - radius_px
        centre, r = least_squares(f, centre, loss="soft_l1", f_scale=INLIER_PX).x, float(radius_px)
    res = np.abs(np.linalg.norm(pts - centre, axis=1) - r)
    return {"kind": "arc", "centre": centre, "r": r, "residuals": res}


def project(prim: dict, p: np.ndarray) -> np.ndarray:
    if prim["kind"] == "line":
        return prim["point"] + ((p - prim["point"]) @ prim["dir"]) * prim["dir"]
    if prim["kind"] == "arc":
        v = p - prim["centre"]
        return prim["centre"] + v / (np.linalg.norm(v) or 1) * prim["r"]
    return p                                   # traced: no primitive, the corner stays where it is


def junction(a: dict, b: dict, near: np.ndarray) -> tuple[np.ndarray, float]:
    """Where two consecutive pieces meet, near their shared corner: alternating projections converge to their
    intersection (or to the closest points of two that do not quite touch, e.g. a tangent arc)."""
    x = near.astype(float)
    for _ in range(60):
        x = project(b, project(a, x))
    pa, pb = project(a, x), project(b, x)
    return (pa + pb) / 2, float(np.linalg.norm(pa - pb))


def fit_outline(contour, corner_idx: list[int], spans: list[dict], px_per_mm: float) -> dict:
    """spans: [{"from": "V3", "to": "V7", "kind": "line" | "arc" | "traced", "radius": mm or None,
    "corner_radius": mm or None}], in contour order. corner_radius rounds the corner where this span (a line) meets
    the next one (a line), for a small labelled corner radius that has no corners of its own in the trace.
    Stretches no span covers stay as traced. Returns the closed outline as pieces in pixels and a report."""
    contour = np.asarray(contour, float).reshape(-1, 2)
    m = len(corner_idx)
    ids = {f"V{k + 1}": k for k in range(m)}
    given = {}
    for s in spans:
        if s.get("from") not in ids or s.get("to") not in ids:
            raise ValueError(f"span {s}: corners must be ids V1..V{m} from the outline image")
        a, b = ids[s["from"]], ids[s["to"]]
        if a == b:
            raise ValueError(f"span {s}: from and to are the same corner")
        if a in given:
            raise ValueError(f"two spans start at {s['from']}")
        kind = s.get("kind", "line")
        if kind not in ("line", "arc", "traced"):
            raise ValueError(f"span {s}: kind must be line, arc or traced")
        given[a] = (b, kind, s.get("radius"), s.get("corner_radius"))
    # walk round the outline from corner to corner: a given span jumps to its end, a gap stays traced
    pieces, k, start = [], 0, None
    # start at a corner where a span starts, so no span is split by the wrap-round
    k = min(given) if given else 0
    first, seen = k, 0
    while True:
        b, kind, radius, corner = given.get(k, ((k + 1) % m, "traced", None, None))
        span = (b - k) % m or m
        seen += span
        if seen > m:
            raise ValueError("spans overlap: list them in order round the outline, each starting where the last ended "
                             "or after it, as numbered on the outline image")
        pts = _stretch(contour, corner_idx[k], corner_idx[b])
        if kind == "line":
            prim = fit_line(pts)
        elif kind == "arc":
            prim = fit_arc(pts, float(radius) * px_per_mm if radius else None)
        else:
            prim = {"kind": "traced", "points": pts, "residuals": np.zeros(len(pts))}
        prim.update({"from": k, "to": b, "pts": pts, "label_radius": radius, "corner_radius": corner})
        pieces.append(prim)
        k = b
        if k == first:
            break
    # join consecutive pieces
    n = len(pieces)
    for i, p in enumerate(pieces):
        q = pieces[(i + 1) % n]
        near = contour[corner_idx[p["to"]]]
        if p["kind"] == "traced" and q["kind"] == "traced":
            point, gap = near, 0.0
        else:
            point, gap = junction(p, q, near)
        p["end"], q["start"], p["gap_px"] = point, point, gap
    # rounded corners between two lines
    rounded = []
    for i, p in enumerate(pieces):
        rounded.append(p)
        q = pieces[(i + 1) % n]
        r = (p.get("corner_radius") or 0) * px_per_mm
        if not r or p["kind"] != "line" or q["kind"] != "line":
            continue
        x = p["end"]
        d1 = p["start"] - x
        d2 = q["end"] - x
        d1, d2 = d1 / (np.linalg.norm(d1) or 1), d2 / (np.linalg.norm(d2) or 1)
        theta = math.acos(float(np.clip(d1 @ d2, -1, 1)))
        if theta < 1e-3 or theta > math.pi - 1e-3:
            continue
        t = r / math.tan(theta / 2)
        bis = (d1 + d2) / np.linalg.norm(d1 + d2)
        centre = x + bis * r / math.sin(theta / 2)
        t1, t2 = x + d1 * t, x + d2 * t
        mid = centre - bis * r
        p["end"], q["start"] = t1, t2
        rounded.append({"kind": "arc", "centre": centre, "r": r, "start": t1, "end": t2, "pts": np.array([t1, mid, t2]),
                        "residuals": np.zeros(0), "from": p["to"], "to": p["to"], "label_radius": r / px_per_mm,
                        "gap_px": 0.0, "fillet": True})
    pieces = rounded
    report = []
    for p in pieces:
        res = p["residuals"]
        if p.get("fillet"):
            report.append({"corner": f"V{p['from'] + 1}", "kind": "corner_radius",
                           "radius_mm": round(p["r"] / px_per_mm, 2), "ok": True})
            continue
        entry = {"from": f"V{p['from'] + 1}", "to": f"V{p['to'] + 1}", "kind": p["kind"]}
        if p["kind"] != "traced":
            inl = res <= INLIER_PX
            entry |= {"fit_rms_mm": round(float(np.sqrt(np.mean(res[inl] ** 2))) / px_per_mm, 3) if inl.any() else None,
                      "on_edge": round(float(inl.mean()), 2),
                      "moved_corner_mm": round(float(np.linalg.norm(p["end"] - p["pts"][-1])) / px_per_mm, 2)}
        if p["kind"] == "arc":
            entry["radius_mm"] = round(p["r"] / px_per_mm, 2)
            entry["radius"] = "labelled" if p["label_radius"] else "fitted"
        if p["gap_px"] > 2 * INLIER_PX:
            entry["gap_to_next_mm"] = round(p["gap_px"] / px_per_mm, 2)
        entry["ok"] = p["kind"] == "traced" or (entry["on_edge"] >= 0.6 and p["gap_px"] <= 3 * INLIER_PX)
        report.append(entry)
    return {"pieces": pieces, "report": report}


def sample(pieces: list[dict], step_deg: float = 5.0) -> list[np.ndarray]:
    """The closed outline as points (pixels): each piece from its start, arcs sampled every step_deg."""
    out = []
    for p in pieces:
        s, e = p["start"], p["end"]
        if p["kind"] == "line":
            out.append(s)
        elif p["kind"] == "traced":
            out.append(s)
            out += list(p["pts"][1:-1][:: max(1, len(p["pts"]) // 40)])
        else:
            c, r = p["centre"], p["r"]
            mid = c + (p["pts"][len(p["pts"]) // 2] - c) / (np.linalg.norm(p["pts"][len(p["pts"]) // 2] - c) or 1) * r
            a0, am, a1 = (math.atan2(*(v - c)[::-1]) for v in (s, mid, e))
            sweep = (a1 - a0) % (2 * math.pi)
            if (am - a0) % (2 * math.pi) > sweep:      # the arc runs the other way round
                sweep -= 2 * math.pi
            steps = max(2, int(abs(math.degrees(sweep)) / step_deg))
            out += [c + r * np.array([math.cos(a0 + sweep * k / steps), math.sin(a0 + sweep * k / steps)])
                    for k in range(steps)]
    return out


def arcs(pieces: list[dict]) -> list[dict]:
    """The arcs as start, middle, end and centre (pixels), for a profile with true arcs."""
    out = []
    for p in pieces:
        if p["kind"] == "arc":
            c, r = p["centre"], p["r"]
            v = p["pts"][len(p["pts"]) // 2] - c
            out.append({"from": p["start"], "mid": c + v / (np.linalg.norm(v) or 1) * r, "to": p["end"],
                        "centre": c, "r": r})
    return out
