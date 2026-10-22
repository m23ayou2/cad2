"""The views of a scanned drawing with no flat pattern, made ready for the view subagents (main.py's views route).

The vision model has boxed the views (part_views.find_views). Each one is cropped into its own PDF with a margin,
so dimension text at its edge stays in the crop; the dimension values of every orthographic view and section are
transcribed (one call per view: the values only, by the direction of their dimension line in the image); and the
views' axes in the part frame are worked out from the values they share: two image axes that print the same
values (MIN_SHARED or more) measure the same part axis. This catches a view drawn turned on the sheet, and gives a
scan's views the axes that validate.view_frames gives a vector drawing's.

Writes, in the run folder: views/<id>.pdf, parts/views.json (read by the raster validation), parts/dims.json and
parts/axes.json. The calls are traced in parts/trace.jsonl, with their cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf

from src import part_views
from src.llm import ask_json, image_part

CROP_MARGIN_PT = 20  # dimension text at a view's edge must stay in its crop (a cut "5" was read as "6")
MATCH_MM = 0.15      # two printed values this close are the same dimension
MIN_SHARED = 2       # values two image axes must share to measure the same part axis
MIN_VALUE_MM = 2     # smaller values (thicknesses, small offsets) recur everywhere and link nothing

DIMS_PROMPT = """This image is one view cropped from a scanned engineering drawing, {w} x {h} pixels. List every
linear dimension value printed in it, grouped by the direction of its dimension line in the image: "horizontal"
(the dimension line runs left-right) or "vertical" (it runs up-down). Leave out diameters (Ø), radii (R), angles
(°), slanted dimensions and notes. Copy each number exactly as printed; do not interpret it.
Reply with JSON only: {{"horizontal": [120.5, 30], "vertical": [42.2]}}"""

UNIT = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}


def prepare(drawing: Path, run_dir: Path, found: list[dict], model: str) -> dict[str, dict]:
    """Crop the views, transcribe their dimension values and work out their axes; returns the axes by view id
    (views whose axes could not be worked out are left out)."""
    parts = run_dir / "parts"
    (run_dir / "views").mkdir(exist_ok=True)
    with pymupdf.open(drawing) as doc:
        for v in found:
            x0, y0, x1, y1 = v["rect_pt"]
            m = CROP_MARGIN_PT
            box = pymupdf.Rect(x0 - m, y0 - m, x1 + m, y1 + m) & doc[v["sheet"] - 1].rect
            part_views.crop_view(doc, {**v, "rect_pt": list(box)}, run_dir / "views" / f"{v['id']}.pdf")
    (parts / "views.json").write_text(json.dumps([{**v, "parts": []} for v in found], indent=2))
    dims = {v["id"]: transcribe(run_dir / "views" / f"{v['id']}.pdf", parts / "trace.jsonl", model)
            for v in found if v["kind"] in ("orthographic", "section")}
    axes = solve_axes(dims)
    (parts / "dims.json").write_text(json.dumps(dims, indent=2))
    (parts / "axes.json").write_text(json.dumps(axes, indent=2))
    return axes


def transcribe(view_pdf: Path, trace: Path, model: str) -> dict[str, list[float]]:
    """One call: the linear dimension values of a view, by the direction of their dimension line in the image."""
    with pymupdf.open(view_pdf) as doc:
        png, _ = part_views.render(doc[0], 2000)
        w, h = round(doc[0].rect.width), round(doc[0].rect.height)
    data, _ = ask_json([{"type": "text", "text": DIMS_PROMPT.format(w=w, h=h)}, image_part(png)],
                       f"dims_{view_pdf.stem}", trace, model=model)
    out = {}
    for key in ("horizontal", "vertical"):
        values = []
        for x in data.get(key) or []:
            try:
                values.append(float(str(x).replace(",", ".")))
            except ValueError:
                pass
        out[key] = values
    return out


def shared(a: list[float], b: list[float]) -> list[float]:
    return sorted({x for x in a if x > MIN_VALUE_MM and any(abs(x - y) <= MATCH_MM for y in b)})


def solve_axes(dims: dict[str, dict]) -> dict[str, dict]:
    """Two image axes that print the same dimension values measure the same part axis. The view linked to the
    most others is the plan (image right = X, image up = Y); a view linked to a placed one takes that axis on its
    linked image axis, and Z on the other (Y when the shared axis is Z)."""
    ids, links = list(dims), []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            for ka in ("horizontal", "vertical"):
                for kb in ("horizontal", "vertical"):
                    s = shared(dims[a][ka], dims[b][kb])
                    if len(s) >= MIN_SHARED:
                        links.append((a, ka, b, kb, s))
    if not links:
        return {}
    degree = {v: sum(v in (link[0], link[2]) for link in links) for v in ids}
    anchor = max(ids, key=lambda v: (degree[v], len(dims[v]["horizontal"]) + len(dims[v]["vertical"])))
    axes = {anchor: {"horizontal": "X", "vertical": "Y", "evidence": "linked to the most other views: the plan"}}
    changed = True
    while changed:
        changed = False
        for a, ka, b, kb, s in links:
            for (u, ku), (v, kv) in (((a, ka), (b, kb)), ((b, kb), (a, ka))):
                if u in axes and v not in axes:
                    axis = axes[u][ku]
                    other = "vertical" if kv == "horizontal" else "horizontal"
                    axes[v] = {kv: axis, other: "Z" if axis in ("X", "Y") else "Y",
                               "evidence": f"its {kv} values {s} are {ku} in {u}"}
                    changed = True
    return axes


def view_frame(axes: dict) -> dict:
    """A view's frame (main.View.frame) from its solved axes; values are taken to grow to the right and upward,
    as ordinate chains do."""
    look = -np.cross(UNIT[axes["horizontal"]], UNIT[axes["vertical"]])
    i = int(np.argmax(np.abs(look)))
    return {"horizontal": "+" + axes["horizontal"], "vertical": "+" + axes["vertical"],
            "looking_along": ("+" if look[i] > 0 else "-") + "XYZ"[i],
            "source": f"dimension values shared with the other views ({axes['evidence']}); signs assume the "
                      "values grow to the right and upward"}


def subagent(kind: str, name: str) -> str:
    """The view subagent suggested for a view the vision model found (its kind and name are guesses)."""
    name = (name or "").upper()
    if kind in ("section", "detail", "isometric"):
        return kind
    if any(w in name for w in ("TOP", "BOTTOM", "PLAN")):
        return "top"
    return "side" if any(w in name for w in ("SIDE", "LEFT", "RIGHT", "END")) else "front"
