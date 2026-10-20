"""Split a drawing into views, colour the parts of each view, then match parts across views.

Pipeline (each step writes its output to the run folder):
  1. sheet_N.png            every PDF sheet rendered for the LLM
  2. views.json             LLM finds each view's bounding box on the sheet
  3. views/vK.pdf           each view cropped from the original PDF (vectors and scans kept intact)
  4. three LLM calls per view, each seeing the previous answer drawn on the view:
     views/vK_points.png    points where the object's edges meet or end
     views/vK_graph.png/json   edges (line, arc, circle) joining those points
     views/vK_parts.json    parts, each a loop of point ids or a circle (with optional holes)
     views/vK_colored.pdf/png   parts drawn as coloured polygons, labelled P1, P2, ...
     (numbering continues from one view to the next)
  5. matches.json           the matching agent groups labels that are the same physical part
     views/vK_matched.png   every view coloured by matched group, so a group has one colour everywhere
     matched_all.png        all matched views side by side
"""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pymupdf

from src.agent.matcher import match_parts
from src.llm import ask_json, image_part

SHEET_MAX_PX = 1600
VIEW_MAX_PX = 1200
PALETTE = [(0.90, 0.10, 0.10), (0.10, 0.45, 0.90), (0.10, 0.70, 0.20), (0.95, 0.55, 0.00),
           (0.60, 0.20, 0.80), (0.00, 0.70, 0.70), (0.85, 0.20, 0.60), (0.55, 0.40, 0.10),
           (0.40, 0.60, 0.00), (0.20, 0.20, 0.60), (0.95, 0.80, 0.00), (0.50, 0.50, 0.50)]

VIEWS_PROMPT = """This image is sheet {sheet} of an engineering drawing, {w} x {h} pixels.
Find every view of the part on it: orthographic views, sections, details, isometric or pictorial views,
flat patterns. Ignore the border, title block, notes, revision table and logos.
Coordinates: {coords}.
For each view give a bounding box [x0, y0, x1, y1] (left, top, right, bottom) that contains the whole view
together with its dimension lines, dimension text and view label. Boxes must not overlap other views.
Reply with JSON only:
{{"views": [{{"name": "SECTION A-A", "kind": "orthographic|section|detail|isometric|flat_pattern|other",
  "box": [x0, y0, x1, y1]}}]}}"""

VIEW_INTRO = """The first image is one view cropped from an engineering drawing: "{name}" ({kind}),
{w} x {h} pixels. The second image is the whole sheet, for context only.
We trace the object in this view in three steps: points, then edges joining them, then parts.
All coordinates refer to the FIRST image: {coords}.
"""

POINTS_PROMPT = VIEW_INTRO + """
Step 1 of 3, points only, no lines yet. Place a point at every place where edges of the object
meet or end: corners, junctions where three or more lines meet, ends of hidden (dashed) lines, points
where an arc or fillet meets a straight line (tangent points), ends of bend lines, and the centre of
every circle, hole and arc. Only object geometry: no points on dimension lines, arrows, text, leaders,
centre-line ends or hatching. Put each point exactly where the drawn lines meet.
Reply with JSON only:
{{"points": [{{"id": "1", "xy": [x, y], "kind": "corner|junction|tangent|center|line_end"}}]}}"""

EDGES_PROMPT = VIEW_INTRO + """
Step 2 of 3. The first image now shows your points as red numbered dots. Connect them: give one edge
for every object edge drawn between two of the points.
- line: straight edge from one point to another.
- arc: curved edge from one point to another; "via" is a point [x, y] on the arc about halfway along.
- circle: a full circle (hole, round outline); "center" is its centre point, "radius" in pixels.
Mark dashed edges hidden and sheet-metal bend lines bend_line. If a point is missing, add it in
"new_points" with an id continuing the numbering, and use it.
Reply with JSON only:
{{"new_points": [{{"id": "41", "xy": [x, y], "kind": "corner"}}],
 "edges": [{{"type": "line|arc|circle", "from": "1", "to": "2", "via": [x, y], "center": "7", "radius": 20,
   "hidden": false, "bend_line": false}}]}}"""

PARTS_PROMPT = VIEW_INTRO + """
Step 3 of 3. The first image now shows your points (red numbers) joined by your edges (blue; dashed
blue = hidden, green = bend line). Split the object into its distinct parts: each part is one
recognisable physical region or feature, such as a flange, boss, shaft step, bore, hole, slot, tab,
bend, rib, fillet zone or flat panel. Holes and bores are parts too. Split a flat pattern at its bend
lines into separate panels. A part may be partly visible, hidden, cut by a section or seen edge-on.
Give each part's outline using your points:
- {{"loop": ["1", "2", "3", ...]}}: the point numbers in order around the outline, following edges;
- {{"circle": "7"}}: a round outline, given by the centre point of its circle.
A part with openings inside it (a ring, a plate with holes) lists them in "holes", in the same form.
Also give the dimensions printed on the drawing that belong to the part.
Reply with JSON only:
{{"parts": [{{"id": "p1", "name": "flange", "feature": "flange|boss|hole|bore|slot|tab|bend|panel|rib|chamfer|fillet|other",
  "outline": {{"loop": ["1", "2", "3", "4"]}}, "holes": [{{"circle": "9"}}],
  "dimensions": ["Ø36", "4"], "hidden": false, "notes": ""}}]}}"""


def coords_text(w: int, h: int) -> str:
    return f"pixels of a {w} x {h} image, x from the left edge, y from the top edge"


def to_px(xy: list) -> list[float]:
    return [float(xy[0]), float(xy[1])]


def render(page: pymupdf.Page, max_px: int, max_zoom: float = 6.0) -> tuple[bytes, float]:
    zoom = min(max_px / max(page.rect.width, page.rect.height), max_zoom)
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).tobytes("png"), zoom


def find_views(doc: pymupdf.Document, run: Path, trace: Path) -> list[dict]:
    views = []
    for pno, page in enumerate(doc):
        png, zoom = render(page, SHEET_MAX_PX)
        (run / f"sheet_{pno + 1}.png").write_bytes(png)
        w, h = round(page.rect.width * zoom), round(page.rect.height * zoom)
        prompt = VIEWS_PROMPT.format(sheet=pno + 1, w=w, h=h, coords=coords_text(w, h))
        data, _ = ask_json([{"type": "text", "text": prompt}, image_part(png)], f"views_sheet{pno + 1}", trace)
        pad = 0.01 * max(page.rect.width, page.rect.height)
        for view in data.get("views", []):
            if len(view.get("box") or []) != 4:
                continue
            (x0, y0), (x1, y1) = (to_px(view["box"][:2]), to_px(view["box"][2:]))
            x0, x1, y0, y1 = min(x0, x1) / zoom, max(x0, x1) / zoom, min(y0, y1) / zoom, max(y0, y1) / zoom
            rect = pymupdf.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad) & page.rect
            if rect.is_empty or rect.width < 10 or rect.height < 10:
                continue
            views.append({"id": f"v{len(views) + 1}", "sheet": pno + 1, "name": view.get("name", ""),
                          "kind": view.get("kind", "other"), "rect_pt": [round(c, 2) for c in rect]})
    return views


def crop_view(doc: pymupdf.Document, view: dict, out: Path) -> None:
    """Copy the view's region of the sheet into its own PDF page, keeping vectors and full-res scans."""
    rect = pymupdf.Rect(view["rect_pt"])
    dst = pymupdf.open()
    page = dst.new_page(width=rect.width, height=rect.height)
    page.show_pdf_page(page.rect, doc, view["sheet"] - 1, clip=rect)
    dst.save(out)


def draw_graph(view_pdf: Path, zoom: float, points: dict, edges: list, out_png: Path) -> bytes:
    """Draw the model's points (and edges, if any) over the view, in the view's pixel frame."""
    with pymupdf.open(view_pdf) as doc:
        page = doc[0]
        unit = min(page.rect.width, page.rect.height) / 120
        pt = {k: pymupdf.Point(x / zoom, y / zoom) for k, (x, y) in points.items()}
        for e in edges:
            color = (0.1, 0.7, 0.2) if e.get("bend_line") else (0.1, 0.35, 0.95)
            dashes = f"[{unit * 2} {unit}] 0" if e.get("hidden") else None
            if e.get("type") == "circle" and e.get("center") in pt and e.get("radius"):
                page.draw_circle(pt[e["center"]], e["radius"] / zoom, color=color, width=unit / 3, dashes=dashes)
            elif e.get("from") in pt and e.get("to") in pt:
                via = [pymupdf.Point(e["via"][0] / zoom, e["via"][1] / zoom)] \
                    if e.get("type") == "arc" and e.get("via") else []
                page.draw_polyline([pt[e["from"]], *via, pt[e["to"]]], color=color,
                                   width=unit / 3, dashes=dashes)
        for k, p in pt.items():
            page.draw_circle(p, unit * 0.7, color=(0.9, 0.1, 0.1), fill=(0.9, 0.1, 0.1))
            page.insert_text(p + (unit, -unit), k, fontsize=unit * 2.6, fontname="hebo", color=(0.85, 0, 0))
        png, _ = render(page, VIEW_MAX_PX)
    out_png.write_bytes(png)
    return png


def outline_px(spec: dict, points: dict, edges: list) -> list[list[float]]:
    """Turn {"loop": [ids]} or {"circle": centre_id} into a pixel polygon; [] if it cannot be built."""
    if "circle" in (spec or {}):
        circle = next((e for e in edges if e.get("type") == "circle" and e.get("center") == spec["circle"]
                       and e.get("radius")), None)
        if circle is None or spec["circle"] not in points:
            return []
        (cx, cy), r = points[spec["circle"]], circle["radius"]
        return [[cx + r * math.cos(a * math.pi / 24), cy + r * math.sin(a * math.pi / 24)] for a in range(48)]
    ids = [i for i in (spec or {}).get("loop", []) if i in points]
    arcs = {frozenset((e["from"], e["to"])): e["via"] for e in edges
            if e.get("type") == "arc" and e.get("via") and e.get("from") and e.get("to")}
    poly = []
    for a, b in zip(ids, ids[1:] + ids[:1]):
        poly.append(list(points[a]))
        if frozenset((a, b)) in arcs:
            poly.append(list(arcs[frozenset((a, b))]))
    return poly if len(ids) >= 3 or len(poly) >= 3 else []


def label_parts(view: dict, run: Path, trace: Path) -> list[dict]:
    """Points -> edges -> parts, three LLM calls; each one sees the previous answer drawn on the view."""
    vdir, vid = run / "views", view["id"]
    with pymupdf.open(vdir / f"{vid}.pdf") as doc:
        png, zoom = render(doc[0], VIEW_MAX_PX)
        w, h = round(doc[0].rect.width * zoom), round(doc[0].rect.height * zoom)
    (vdir / f"{vid}.png").write_bytes(png)
    sheet = image_part((run / f"sheet_{view['sheet']}.png").read_bytes())
    fmt = dict(name=view["name"], kind=view["kind"], w=w, h=h, coords=coords_text(w, h))

    def read_points(items: list) -> dict:
        return {str(p["id"]): to_px(p["xy"]) for p in items
                if isinstance(p.get("xy"), list) and len(p["xy"]) == 2}

    def call(prompt: str, image: bytes, step: str) -> dict:
        data, _ = ask_json([{"type": "text", "text": prompt.format(**fmt)}, image_part(image), sheet],
                           f"{step}_{vid}", trace)
        return data

    data = call(POINTS_PROMPT, png, "points")
    points = read_points(data.get("points", []))
    points_png = draw_graph(vdir / f"{vid}.pdf", zoom, points, [], vdir / f"{vid}_points.png")

    data = call(EDGES_PROMPT, points_png, "edges")
    points.update(read_points(data.get("new_points", [])))
    edges = []
    for e in data.get("edges", []):
        e = {**e, **{k: str(e[k]) for k in ("from", "to", "center") if e.get(k) is not None}}
        if isinstance(e.get("via"), list) and len(e["via"]) == 2:
            e["via"] = to_px(e["via"])
        else:
            e.pop("via", None)
        edges.append(e)
    graph_png = draw_graph(vdir / f"{vid}.pdf", zoom, points, edges, vdir / f"{vid}_graph.png")
    (vdir / f"{vid}_graph.json").write_text(json.dumps({"zoom": zoom, "points": points, "edges": edges}, indent=1))

    data = call(PARTS_PROMPT, graph_png, "parts")
    parts = []
    for part in data.get("parts", []):
        outer = outline_px(part.get("outline"), points, edges)
        if not outer:
            print(f"[parts_{vid}] dropped {part.get('id')}: outline {part.get('outline')} not buildable", flush=True)
            continue
        holes = [h for h in (outline_px(s, points, edges) for s in part.get("holes", [])) if h]
        parts.append({**part, "local_id": part.get("id"),
                      "polygon_pt": [[x / zoom, y / zoom] for x, y in outer],
                      "holes_pt": [[[x / zoom, y / zoom] for x, y in hole] for hole in holes]})
    return parts


def draw_view(view_pdf: Path, regions: list[tuple[str, list, list, tuple]], out_png: Path) -> None:
    """Draw (label, outer polygon, hole polygons, colour) regions, in points, over a copy of the view PDF."""
    with pymupdf.open(view_pdf) as doc:
        page = doc[0]
        size = max(6.0, min(page.rect.width, page.rect.height) / 22)
        for label, poly, holes, color in regions:
            shape = page.new_shape()
            for ring in [poly, *holes]:
                shape.draw_polyline([pymupdf.Point(*p) for p in ring] + [pymupdf.Point(*ring[0])])
            shape.finish(color=color, fill=color, fill_opacity=0.35, stroke_opacity=0.9,
                         width=max(0.6, size / 10), closePath=True, even_odd=True)
            shape.commit()
        for label, poly, holes, color in regions:
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            box = pymupdf.Rect(cx - size * 0.1, cy - size, cx + size * 0.62 * len(label), cy + size * 0.3)
            page.draw_rect(box, color=color, fill=(1, 1, 1), fill_opacity=0.85, width=0.5)
            page.insert_text((cx, cy), label, fontsize=size, fontname="hebo", color=color)
        doc.save(out_png.with_suffix(".pdf"))
        png, _ = render(page, 1600)
    out_png.write_bytes(png)


def side_by_side(images: list[Path], out: Path, height: int = 700) -> None:
    pixmaps = [pymupdf.Pixmap(str(p)) for p in images]
    dst = pymupdf.open()
    widths = [pm.width * height / pm.height for pm in pixmaps]
    page = dst.new_page(width=sum(widths) + 20 * (len(widths) + 1), height=height + 40)
    x = 20
    for path, w in zip(images, widths):
        page.insert_image(pymupdf.Rect(x, 20, x + w, 20 + height), filename=str(path))
        page.insert_text((x, 15), path.stem.split("_")[0], fontsize=12)
        x += w + 20
    page.get_pixmap(alpha=False).save(out)


def run(drawing: Path, out_root: Path = Path("runs"), run_dir: Path | None = None) -> Path:
    """Run the whole pipeline; outputs go to run_dir, or to a new folder under out_root."""
    if run_dir is None:
        run_dir = out_root / f"{drawing.parent.parent.name}-{drawing.parent.name}-parts-{datetime.now():%Y%m%d-%H%M%S}"
    (run_dir / "views").mkdir(parents=True)
    trace = run_dir / "trace.jsonl"
    print(f"Run folder: {run_dir}", flush=True)

    with pymupdf.open(drawing) as doc:
        views = find_views(doc, run_dir, trace)
        if not views:
            raise ValueError("No views found on the drawing")
        for view in views:
            crop_view(doc, view, run_dir / "views" / f"{view['id']}.pdf")
    print(f"{len(views)} views: " + ", ".join(f"{v['id']}={v['name']}" for v in views), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        per_view = list(pool.map(lambda v: label_parts(v, run_dir, trace), views))

    # Global numbering: P1, P2 ... in view order, continuing from one view to the next.
    n = 0
    for view, parts in zip(views, per_view):
        for part in parts:
            n += 1
            part["id"] = f"P{n}"
        view["parts"] = parts
        (run_dir / "views" / f"{view['id']}_parts.json").write_text(json.dumps(parts, indent=2))
        draw_view(run_dir / "views" / f"{view['id']}.pdf",
                  [(p["id"], p["polygon_pt"], p["holes_pt"], PALETTE[i % len(PALETTE)])
                   for i, p in enumerate(parts)],
                  run_dir / "views" / f"{view['id']}_colored.png")
    (run_dir / "views.json").write_text(json.dumps(views, indent=2))

    matches = match_parts(views, run_dir, trace)
    (run_dir / "matches.json").write_text(json.dumps(matches, indent=2))

    owner = {}
    for i, group in enumerate(matches["groups"]):
        for member in group["members"]:
            for label in member["parts"]:
                owner[label] = (group["id"], PALETTE[i % len(PALETTE)])
    matched_pngs = []
    for view in views:
        regions = [(owner[p["id"]][0], p["polygon_pt"], p["holes_pt"], owner[p["id"]][1])
                   for p in view["parts"] if p["id"] in owner]
        out = run_dir / "views" / f"{view['id']}_matched.png"
        draw_view(run_dir / "views" / f"{view['id']}.pdf", regions, out)
        matched_pngs.append(out)
    side_by_side(matched_pngs, run_dir / "matched_all.png")

    costs = [json.loads(line).get("cost") for line in trace.read_text().splitlines()]
    total = sum(c for c in costs if c is not None)
    print(f"{n} parts in {len(views)} views -> {len(matches['groups'])} matched groups. "
          f"LLM calls: {len(costs)}, total cost ${total:.4f}", flush=True)
    return run_dir
