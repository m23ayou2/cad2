"""Draw2CAD: drawing -> view analyses -> feature tree -> STEP.

    uv run python main.py cases/tier2/03/drawing.pdf [--view] [-v]
    uv run python main.py --build runs/<run>          # rebuild the STEP from an existing plan

The main agent (agents/main.md) reads the sheet, splits it into views with
src/split.py, calls one analysis agent (agents/analysis/<type>.md) per view, and
returns a feature tree. The builder agent (agents/builder.md) turns the tree into
a CadQuery script, runs it in its own process, and fixes it until it is one valid
solid. src/validate.py then projects the model into every view of the drawing,
and one validation agent per view (agents/validation/<type>.md) reports what
matches and what must change; plan issues go back to the main agent, build
issues to the builder, for up to MAX_CORRECTIONS rounds. Agents are Markdown
files with YAML settings; tools are the typed Python functions below; every
answer is validated against the pydantic model its YAML names (src/schema.py). Every run is saved under runs/<drawing>-<time>/: split
views, one summary per view, plan.json, feature_tree.json, build attempts,
model.py, model.step, validation/round_N/ (projections, images, reports, that
round's model), viewer.html, trace.jsonl, and result.json.
"""
from __future__ import annotations

import argparse
import base64
import inspect
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import types
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

import pymupdf
import yaml
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import conflicts  # noqa: E402
import schema  # noqa: E402
import split as splitter  # noqa: E402
import views as view_reader  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import validate as view_checks  # noqa: E402
import visualization  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402

MODEL = "openai/gpt-6-luna" 
# MODEL = "anthropic/claude-haiku-5.5" 
AGENTS_DIR = ROOT / "agents"
RUNS_DIR = ROOT / "runs"
REPAIR_TURNS = 2
CAD_TIMEOUT_S = 120
VALIDATE_TIMEOUT_S = 600
MAX_CORRECTIONS = 2  # validate -> correct rounds after the first build
SubagentName = Literal["front", "top", "side", "section", "detail", "isometric"]

log = logging.getLogger("draw2cad")


# --------------------------------------------------------------------------- agents
@dataclass
class Agent:
    name: str
    settings: dict
    prompt: str

    @property
    def tools(self) -> list[str]:
        return self.settings.get("tools") or []

    @property
    def output_model(self) -> type[BaseModel] | None:
        name = self.settings.get("output_schema")
        if name is None:
            return None
        if name not in schema.MODELS:
            raise ValueError(f"agents/{self.name}.md: unknown output_schema {name!r}; use one of {sorted(schema.MODELS)}")
        return schema.MODELS[name]

    @property
    def system_prompt(self) -> str:
        if not self.output_model:
            return self.prompt
        spec = json.dumps(self.output_model.model_json_schema(), indent=1)
        return (f"{self.prompt}\n\n# Output schema\n\nYour final reply is one JSON object that follows this "
                f"JSON Schema exactly, with every required key and no extra keys. The harness validates it and "
                f"sends back any problems.\n\n```json\n{spec}\n```")


def load_agent(name: str) -> Agent:
    text = (AGENTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not match:
        raise ValueError(f"agents/{name}.md has no YAML front matter")
    settings = yaml.safe_load(match.group(1)) or {}
    if settings.get("model", MODEL) != MODEL:
        log.warning("agents/%s.md asks for %s; using %s", name, settings["model"], MODEL)
    return Agent(name, settings, match.group(2).strip())


# --------------------------------------------------------------------------- run state
@dataclass
class View:
    id: str
    label: str
    pdf: Path | None
    png: Path
    suggested_subagent: str | None
    summary: dict = field(default_factory=dict)
    frame: dict | None = None  # axes of the view in the part frame, from validate.view_frames
    sheet_box: list[float] | None = None  # where the view is on the sheet: x0, y0, x1, y1 as page fractions
    scan: dict | None = None  # scanned sheets: this view's scan (kind, direction, box), from scan_view
    outline_pt: list[float] | None = None  # the part outline alone (no dimensions), sheet points


@dataclass
class Run:
    drawing: Path
    dir: Path
    client: OpenAI
    max_cost_usd: float
    cost_usd: float = 0.0
    cost_unknown_calls: int = 0
    views: dict[str, View] = field(default_factory=dict)
    summaries: list[dict] = field(default_factory=list)
    build_attempts: int = 0
    cross_check: dict | None = None  # conflicts between the view analyses and the arbiter's rulings
    scans: dict = field(default_factory=dict)  # scanned sheets: regions read by scan_view, by id (S1, S2, ...)
    scanned: bool = False  # split_views found no vector content: the sheet is a scan

    def __post_init__(self):
        self.build_attempts = len(list((self.dir / "build").glob("attempt_*.py")))  # continue on --build

    def trace(self, event: str, **data):
        record = {"t": datetime.now().isoformat(timespec="milliseconds"), "event": event, **data}
        with (self.dir / "trace.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @property
    def over_budget(self) -> bool:
        return self.cost_usd >= self.max_cost_usd


@dataclass
class Ctx:
    """What a tool can see: the run, and for a subagent, the view it analyses or validates."""
    run: Run
    view: View | None = None
    check: dict | None = None  # validate.py's entry for the view being validated


@dataclass
class ToolOutput:
    data: Any
    images: list[tuple[str, bytes]] = field(default_factory=list)  # (caption, png)


# --------------------------------------------------------------------------- rendering helpers
def png_of(page: pymupdf.Page, clip: pymupdf.Rect | None = None, long_side: int = 1600) -> bytes:
    rect = clip or page.rect
    zoom = min(8.0, long_side / max(rect.width, rect.height))
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False).tobytes("png")


def region(page: pymupdf.Page, x0: float, y0: float, x1: float, y1: float) -> pymupdf.Rect:
    r = page.rect
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if x1 - x0 < 0.02 or y1 - y0 < 0.02:
        raise ValueError("Region too small: give fractions of the page, e.g. x0=0.1 y0=0.2 x1=0.5 y1=0.6")
    return pymupdf.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height, r.x0 + x1 * r.width, r.y0 + y1 * r.height)


def text_spans(page: pymupdf.Page) -> list[dict]:
    """Text with its position as fractions of the page (x0, y0, x1, y1), top-left origin."""
    w, h = page.rect.width, page.rect.height
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip().replace("⌀", "Ø").replace("∅", "Ø")
                if text:
                    b = span["bbox"]
                    out.append({"text": text, "bbox": [round(b[0] / w, 3), round(b[1] / h, 3),
                                                       round(b[2] / w, 3), round(b[3] / h, 3)],
                                "vertical": abs(line["dir"][0]) < 0.7})
    return out


def suggest_subagent(view: view_reader.View) -> str | None:
    if isinstance(view, view_reader.SectionView):
        return "section"
    if isinstance(view, view_reader.DetailView):
        return "detail"
    if isinstance(view, view_reader.PictorialView):
        return "isometric"
    if isinstance(view, view_reader.OrthographicView) and view.direction:
        return {"FRONT": "front", "REAR": "front", "TOP": "top", "BOTTOM": "top",
                "LEFT": "side", "RIGHT": "side"}[view.direction.value.name]
    return None


# --------------------------------------------------------------------------- tools
Fraction = Annotated[float, "Fraction of the page, 0 to 1, origin at the top-left corner"]


def view_drawing(ctx: Ctx,
                 x0: Fraction = 0.0, y0: Fraction = 0.0, x1: Fraction = 1.0, y1: Fraction = 1.0,
                 page: Annotated[int, "Sheet number, starting at 1"] = 1) -> ToolOutput:
    """Look at the full drawing sheet, or zoom into a region of it for more detail."""
    with pymupdf.open(ctx.run.drawing) as doc:
        sheets = len(doc)
        if not 1 <= page <= sheets:
            raise ValueError(f"The drawing has {sheets} sheet(s)")
        p = doc[page - 1]
        full = (x0, y0, x1, y1) == (0.0, 0.0, 1.0, 1.0)
        image = png_of(p, None if full else region(p, x0, y0, x1, y1))
    caption = f"Sheet {page}" + ("" if full else f", region x {x0:.2f}-{x1:.2f}, y {y0:.2f}-{y1:.2f}")
    return ToolOutput({"sheet": page, "sheets": sheets, "shown": caption}, [(caption, image)])


def read_drawing_text(ctx: Ctx) -> dict:
    """Read every text item on the drawing with its position. Empty for scanned drawings."""
    with pymupdf.open(ctx.run.drawing) as doc:
        sheets = [{"sheet": i + 1, "texts": text_spans(p)} for i, p in enumerate(doc)]
    if not any(s["texts"] for s in sheets):
        return {"sheets": sheets, "note": "No text layer: this drawing is a scan. Read values from the image."}
    return {"sheets": sheets}


def split_views(ctx: Ctx) -> ToolOutput:
    """Find the views on the sheet, save each one, and summarise what was extracted from each."""
    run = ctx.run
    out_dir = run.dir / "views"
    try:
        splitter.split(str(run.drawing), str(out_dir))
    except Exception as error:  # split.py needs vector geometry; scans have none
        log.warning("split.py failed (%s: %s); using the whole sheet as one view", type(error).__name__, error)
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / "sheet.png"
        with pymupdf.open(run.drawing) as doc:
            png.write_bytes(png_of(doc[0]))
        # keep any views already scanned with scan_view: split_views may be called after them
        run.views = {"sheet": View("sheet", "WHOLE SHEET", None, png, None)} | \
            {k: v for k, v in run.views.items() if v.scan}
        run.scanned = True
        run.trace("split", ok=False, error=str(error))
        try:  # crop the scanned views and label their regions now (views/scans/), for checking and validation
            scan_views(run)
        except Exception as crop_error:
            log.warning("could not crop the scanned views: %s", crop_error)
        return ToolOutput({
            "views": [{"id": "sheet", "label": "WHOLE SHEET", "suggested_subagent": None}],
            "note": ("The sheet could not be split automatically (probably a scan). For each view you saw, call "
                     "scan_view with a box around it and its dimensions, its kind and its direction. Each scan "
                     "becomes a view with its own id (S1, S2, ...). Read the dimension values from the crops it "
                     "returns and call set_dimension_values. Then call call_subagent once per scanned view, with "
                     "view_id = its scan id, and put in `context` the checked dimensions (values, directions, "
                     "scale) and the leaders from its scan."),
        })

    listing, meta = index_views(run)
    run.trace("split", ok=True, views=listing)
    overview = (out_dir / f"{run.drawing.stem}_overview.png").read_bytes()
    return ToolOutput({"views": listing, "sheet_info_texts": meta.get("sheet_info_texts", [])},
                      [("Sheet with the detected views boxed and numbered", overview)])


def index_views(run: Run) -> tuple[list[dict], dict]:
    """Read the split views from disk into run.views (also used by --build, without splitting again)."""
    out_dir = run.dir / "views"
    stem = run.drawing.stem
    meta = json.loads((out_dir / f"{stem}_views.json").read_text(encoding="utf-8"))
    run.views = {}
    listing = []
    for entry in meta["views"]:
        pdf = out_dir / entry["file"]
        view_id = pdf.stem.removeprefix(f"{stem}_")
        suggestion = None
        try:
            loaded = view_reader.load_view(str(pdf))
            info, suggestion = view_reader.view_to_dict(loaded), suggest_subagent(loaded)
            if getattr(loaded, "direction", None):
                info["direction"]["value"] = loaded.direction.value.name
        except Exception as error:
            log.warning("views.py could not read %s: %s", pdf.name, error)
            info = {"error": str(error)}
        view = View(view_id, entry["label"], pdf, pdf.with_suffix(".png"), suggestion, info)
        with pymupdf.open(run.drawing) as doc:
            w, h = doc[0].rect.width, doc[0].rect.height
        view.outline_pt = entry.get("outline_bbox_sheet_pt")
        box = entry.get("extent_bbox_sheet_pt")
        if box:
            view.sheet_box = [round(box[0] / w, 3), round(box[1] / h, 3), round(box[2] / w, 3), round(box[3] / h, 3)]
        run.views[view_id] = view
        listing.append({
            "id": view_id, "label": entry["label"], "suggested_subagent": view.suggested_subagent,
            "class": info.get("class"), "scale": (info.get("scale") or {}).get("value"),
            "outline_mm": info.get("outline_mm"), "texts": entry["texts"], "warnings": info.get("warnings", []),
        })
    try:
        frames = view_checks.view_frames(str(out_dir), str(run.drawing))
    except Exception as error:  # orientation is a help for the agents, not a requirement
        log.warning("could not work out the view orientations: %s", error)
        frames = {}
    for item in listing:
        view = run.views[item["id"]]
        view.frame = frames.get(view.pdf.stem)
        if view.frame:
            item["axes"] = view.frame
            if view.suggested_subagent is None and view.frame["looking_along"] in ("-Z", "+Z"):
                view.suggested_subagent = item["suggested_subagent"] = "top"
            elif view.suggested_subagent is None:
                view.suggested_subagent = item["suggested_subagent"] = (
                    "front" if view.frame["looking_along"] in ("+Y", "-Y") else "side")
    return listing, meta


def call_subagent(ctx: Ctx,
                  agent: Annotated[SubagentName, "The subagent matching the view type"],
                  view_id: Annotated[str, "A view id returned by split_views"],
                  context: Annotated[str, "What the subagent cannot see in its crop: projection type, "
                                          "cutting-plane line, neighbouring views, which view to analyse"] = "") -> dict:
    """Run the subagent responsible for one view and return its JSON analysis."""
    run = ctx.run
    view = run.views.get(view_id)
    if view is None:
        raise ValueError(f"Unknown view_id {view_id!r}; call split_views first. Known: {sorted(run.views)}")
    if run.over_budget:
        raise RuntimeError(f"Run budget of ${run.max_cost_usd:.2f} reached; finish with what you have")
    sub = load_agent(f"analysis/{agent}")
    intro = [f"View id: {view.id}", f"Label on the sheet: {view.label}"]
    if view.frame:
        f = view.frame
        intro.append(f"Axes of this view in the part frame (worked out from the drawing: {f['source']}): image right "
                     f"= {f['horizontal']}, image up = {f['vertical']}, you are looking along {f['looking_along']}. "
                     "Use these signs for every position and direction you report; they override the generic axis "
                     "description in your instructions.")
    if view.summary:
        intro.append("Extracted automatically from the vector view by views.py (`source` says whether a value "
                      "was found on the drawing or inferred):\n```json\n"
                      + json.dumps(view.summary, ensure_ascii=False, indent=1) + "\n```")
    if context:
        intro.append(f"Context from the main agent:\n{context}")
    intro.append("The view image follows.")
    content = [{"type": "text", "text": "\n\n".join(intro)}, image_part(view_image(view))]
    result = run_agent(sub, Ctx(run, view), content, label=f"{agent}[{view.id}]")
    # one small JSON file per view summary; the feature tree is built from these
    path = run.dir / "summaries" / f"{view.id}.{agent}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    run.summaries.append({"view": view.id, "label": view.label, "agent": agent,
                          "file": str(path.relative_to(run.dir))})
    expected = {v.id for v in run.views.values() if v.suggested_subagent}
    if run.cross_check is None and expected and expected <= {s["view"] for s in run.summaries}:
        run.cross_check = cross_check(run)  # every view is analysed: check them against each other once
        return {"analysis": result, "cross_check": run.cross_check}
    return result


def drawing_texts(run: Run) -> list[str]:
    with pymupdf.open(run.drawing) as doc:
        return [s["text"] for page in doc for s in text_spans(page)]


def cross_check(run: Run) -> dict:
    """Find where the view analyses disagree or leave the drawing unexplained (src/conflicts.py), then let the
    arbiter settle each item from the drawing. Its rulings override the summaries when the part is planned."""
    clog = log.getChild("cross_check")
    summaries = {s["view"]: json.loads((run.dir / s["file"]).read_text(encoding="utf-8")) for s in run.summaries}
    view_texts = [a["text"] for v in run.views.values() for a in v.summary.get("annotations", [])]
    items = conflicts.find(summaries, view_texts)
    (run.dir / "cross_check.json").write_text(json.dumps({"items": items}, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    run.trace("cross_check", items=items)
    if not items:
        clog.info("the view analyses agree; nothing to settle")
        return {"items": [], "rulings": [], "note": "The view analyses agree; nothing needed settling."}
    clog.info("%d item(s) to settle: %s", len(items), ", ".join(f"{i['id']} {i['kind']}" for i in items))
    views = [{"id": v.id, "label": v.label, "axes": v.frame, "sheet_box": v.sheet_box}
             for v in run.views.values() if v.pdf]
    compact = {view: {k: s.get(k) for k in ("dimensions", "hypotheses", "cannot_tell", "notes")}
               for view, s in summaries.items()}
    text = ("# Items to settle\n```json\n" + json.dumps(items, indent=1, ensure_ascii=False) + "\n```\n\n"
            "# Views (sheet_box: x0, y0, x1, y1 as fractions of the page; extend_view takes regions as fractions of "
            "the view itself)\n```json\n"
            + json.dumps(views, indent=1, ensure_ascii=False) + "\n```\n\n"
            "# View summaries\n```json\n" + json.dumps(compact, indent=1, ensure_ascii=False) + "\n```\n\n"
            "The full sheet follows.")
    with pymupdf.open(run.drawing) as doc:
        sheet = png_of(doc[0])
    try:
        report = run_agent(load_agent("analysis/arbiter"), Ctx(run), [{"type": "text", "text": text},
                                                                        image_part(sheet)], label="arbiter")
    except Exception as error:
        clog.warning("the arbiter failed (%s); planning from the summaries alone", error)
        return {"items": items, "rulings": [], "note": f"The arbiter failed: {error}"}
    (run.dir / "cross_check.json").write_text(json.dumps({"items": items, **report}, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    for r in report["rulings"]:
        clog.info("  %s %s: %s", r["item"], r["decision"], r["fact"][:140])
    return {"items": items, **report,
            "note": "Rulings marked `resolved` were settled from the drawing and override the view summaries. "
                    "`already_covered` items need no change. Treat `unresolved` items as assumptions."}


def read_view_text(ctx: Ctx) -> dict:
    """Read every text item inside this view with its position (fractions of the view image)."""
    view = ctx.view
    if view.pdf is None:
        return {"texts": [], "note": "This view is the whole sheet; use read_drawing_text data from the main "
                                     "agent's context or read the image."}
    with pymupdf.open(view.pdf) as doc:
        return {"texts": text_spans(doc[0])}


def zoom_view(ctx: Ctx, x0: Fraction, y0: Fraction, x1: Fraction, y1: Fraction) -> ToolOutput:
    """Zoom into a region of this view to read small text or details."""
    view = ctx.view
    if view.pdf is None and view.sheet_box:  # a scanned view: zoom within its box on the sheet
        b = view.sheet_box
        x0, x1 = b[0] + x0 * (b[2] - b[0]), b[0] + x1 * (b[2] - b[0])
        y0, y1 = b[1] + y0 * (b[3] - b[1]), b[1] + y1 * (b[3] - b[1])
    with pymupdf.open(view.pdf or ctx.run.drawing) as doc:
        image = png_of(doc[0], region(doc[0], x0, y0, x1, y1))
    caption = f"View {view.id}, region x {x0:.2f}-{x1:.2f}, y {y0:.2f}-{y1:.2f}"
    return ToolOutput({"shown": caption}, [(caption, image)])


def run_cadquery(ctx: Ctx, code: Annotated[str, "The complete CadQuery script; it must assign `result`"]) -> dict:
    """Run a CadQuery script in its own process. On success the STEP is exported and the part is measured;
    otherwise the error and the failing line are returned."""
    run = ctx.run
    run.build_attempts += 1
    folder = run.dir / "build"
    folder.mkdir(exist_ok=True)
    name = f"attempt_{run.build_attempts:02d}"
    script, step, report = folder / f"{name}.py", folder / f"{name}.step", folder / f"{name}.json"
    script.write_text(code, encoding="utf-8")
    try:
        process = subprocess.run([sys.executable, str(ROOT / "src" / "cad_runner.py"), str(script), str(step),
                                  str(report)], cwd=folder, timeout=CAD_TIMEOUT_S, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {"ok": False, "attempt": run.build_attempts,
                "error": f"The script took more than {CAD_TIMEOUT_S} s; simplify it (fewer booleans or fillets)"}
    if not report.is_file():
        log.getChild("cad").warning("attempt %d crashed: %s", run.build_attempts, process.stderr[-500:])
        return {"ok": False, "attempt": run.build_attempts,
                "error": f"The CAD process crashed (exit {process.returncode}): {process.stderr[-800:]}"}
    result = json.loads(report.read_text(encoding="utf-8"))
    if result.get("ok"):
        shutil.copyfile(script, run.dir / "model.py")
        shutil.copyfile(step, run.dir / "model.step")
        log.getChild("cad").info("attempt %d ok: size %s, volume %s", run.build_attempts,
                                 result["bbox"]["size"], result["volume"])
    else:
        log.getChild("cad").info("attempt %d failed: %s", run.build_attempts,
                                 result.get("error") or "; ".join(result.get("problems", [])))
    return {"attempt": run.build_attempts, **result}


ViewFraction = Annotated[float, "Fraction of the view's box on the sheet, 0 to 1 from its top-left corner"]


def extend_view(ctx: Ctx, view_id: Annotated[str, "A view id from the views list"],
                x0: ViewFraction, y0: ViewFraction, x1: ViewFraction, y1: ViewFraction) -> ToolOutput:
    """Zoom into a region of one view and follow it along the projection lines: the same band in every view
    drawn beside it (same height on the sheet) and above or below it (same width). Use it where features meet,
    to see the same place in every view that shows it, with the text found there."""
    run = ctx.run
    view = run.views.get(view_id)
    if view is None or not view.sheet_box or not view.outline_pt:
        raise ValueError(f"Unknown or unplaced view {view_id!r}; use one of "
                         f"{sorted(k for k, v in run.views.items() if v.sheet_box)}")
    with pymupdf.open(run.drawing) as doc:
        page = doc[0]
        w, h = page.rect.width, page.rect.height
        vb = [view.sheet_box[0] * w, view.sheet_box[1] * h, view.sheet_box[2] * w, view.sheet_box[3] * h]
        fx0, fx1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
        fy0, fy1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
        if fx1 - fx0 < 0.03 or fy1 - fy0 < 0.03:
            raise ValueError("Region too small: give fractions of the view, e.g. x0=0.6 y0=0.4 x1=0.9 y1=0.8")
        r = pymupdf.Rect(vb[0] + fx0 * (vb[2] - vb[0]), vb[1] + fy0 * (vb[3] - vb[1]),
                         vb[0] + fx1 * (vb[2] - vb[0]), vb[1] + fy1 * (vb[3] - vb[1]))
        images = [(f"{view.label} ({view.id}), region x {fx0:.2f}-{fx1:.2f}, y {fy0:.2f}-{fy1:.2f}",
                   png_of(page, r, long_side=1400))]
        aligned = []
        own = view.outline_pt
        for other in run.views.values():
            ob = other.outline_pt
            if other is view or not ob or not other.sheet_box or other.suggested_subagent in ("isometric", "detail"):
                continue
            box = pymupdf.Rect(other.sheet_box[0] * w, other.sheet_box[1] * h,
                               other.sheet_box[2] * w, other.sheet_box[3] * h)
            y_shared = min(own[3], ob[3]) - max(own[1], ob[1])
            x_shared = min(own[2], ob[2]) - max(own[0], ob[0])
            if y_shared > 0.3 * min(own[3] - own[1], ob[3] - ob[1]):    # drawn beside: same heights
                band, how = pymupdf.Rect(box.x0, r.y0, box.x1, r.y1), "same height"
            elif x_shared > 0.3 * min(own[2] - own[0], ob[2] - ob[0]):  # drawn above or below: same widths
                band, how = pymupdf.Rect(r.x0, box.y0, r.x1, box.y1), "same width"
            else:
                continue
            band = band & box
            if band.is_empty or band.width < 2 or band.height < 2:
                continue
            images.append((f"Aligned band in {other.label} ({other.id}): {how} as the region above", 
                           png_of(page, band, long_side=1400)))
            aligned.append({"view": other.id, "label": other.label, "shares": how, "axes": other.frame})
        texts = [s for s in text_spans(page)
                 if s["bbox"][2] * w >= r.x0 and s["bbox"][0] * w <= r.x1
                 and s["bbox"][3] * h >= r.y0 and s["bbox"][1] * h <= r.y1]
    return ToolOutput({"region": images[0][0], "axes": view.frame, "texts_in_region": [s["text"] for s in texts],
                       "aligned_views": aligned}, images)


SheetFraction = Annotated[float, "Fraction of the whole sheet, 0 to 1 from its top-left corner"]


ScanKind = Literal["front", "top", "side", "section", "detail", "isometric", "flat_pattern"]
ScanDirection = Literal["FRONT", "REAR", "TOP", "BOTTOM", "LEFT", "RIGHT", "none"]


def scan_view(ctx: Ctx, x0: SheetFraction, y0: SheetFraction, x1: SheetFraction, y1: SheetFraction,
              kind: Annotated[ScanKind, "What this view is; flat_pattern = the unfolded blank of a sheet-metal part"],
              direction: Annotated[ScanDirection, "The standard direction it is seen from (orthographic views and "
                                                  "sections); none for isometric, detail and flat_pattern"] = "none",
              label: Annotated[str, "The view's label as written on the sheet, if any"] = "") -> ToolOutput:
    """Scanned sheets only: read one view's geometry with image processing. Give a box around the view and its
    dimensions, what kind of view it is and the direction it is seen from. Returns every dimension found (two facing
    arrowheads: id, end points, length in pixels, direction), the leaders with where their arrows point, holes and
    the part outline, an overlay with the ids, and enlarged crops of each dimension's value text to read. The scan
    becomes a view of its own (its id): analyse it with call_subagent, and the built model is compared with it.
    Then call set_dimension_values with the values you read."""
    import raster_lines
    import raster_vector

    run = ctx.run
    box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    if box[2] - box[0] < 0.03 or box[3] - box[1] < 0.03:
        raise ValueError("Box too small: give fractions of the sheet around one whole view and its dimensions")
    gray = raster_lines.render(str(run.drawing), box)
    with pymupdf.open(run.drawing) as doc:
        page = doc[0]
        zoom = raster_lines.SHEET_PX / max(page.rect.width, page.rect.height)
        sheet_w, sheet_h = page.rect.width * zoom, page.rect.height * zoom
    result = raster_vector.vectorise(gray, flat=kind == "flat_pattern", pictorial=kind == "isometric")
    scan_id = f"S{len(run.scans) + 1}"
    folder = run.dir / "scans" / scan_id
    raster_vector.save(result, gray, str(folder))
    frac = lambda p: [round(box[0] + p[0] / sheet_w, 4), round(box[1] + p[1] / sheet_h, 4)]
    dims = []
    for n, d in enumerate(result["dimensions"], 1):
        dims.append({"id": f"D{n}", "length_px": d["length_px"], "orientation": d["orientation"],
                     "from_px": d["from"], "to_px": d["to"], "from_sheet": frac(d["from"]), "to_sheet": frac(d["to"]),
                     "text_box_px": d["text_box"]})
    shape = result["shape"]
    leaders = [{"id": f"L{n}", "tip_px": l["at"], "tip_sheet": frac(l["at"]), "direction": l["direction"]}
               | ({"bend": l["bend"]} if l.get("bend") else {}) for n, l in enumerate(result["leaders"], 1)]
    flat_blank = result.get("blank") or {}
    if flat_blank.get("corners"):
        check_corners(run, scan_id, gray, flat_blank)
    run.scans[scan_id] = {"box": box, "dimensions": dims, "shape": shape, "leaders": leaders, "blank": flat_blank,
                          "folder": str(folder), "kind": kind, "direction": direction, "label": label}
    # the scan is a view of its own: its crop goes to its subagent, and the model is compared with it
    crop_png = folder / "view.png"
    cv2.imwrite(str(crop_png), gray)
    view = View(scan_id, label or kind.replace("_", " ").upper(), None, crop_png,
                {"flat_pattern": "top"}.get(kind, kind))
    view.sheet_box = box
    view.scan = {"kind": kind, "direction": direction, "box": box}
    if direction != "none":
        import validate as checks
        d = view_reader.ViewDirection[direction]
        b = checks.ps.view_basis(d.look, d.up)
        view.frame = {"horizontal": checks.axis_name(b[0]), "vertical": checks.axis_name(b[1]),
                      "looking_along": checks.axis_name(np.array(d.look, float)), "source": "scan_view direction"}
    run.views[scan_id] = view
    # overlay with the dimension ids, and one labelled crop per dimension value
    over = cv2.imread(str(folder / "vector_overlay.png"))
    for d in dims:
        mid = ((d["from_px"][0] + d["to_px"][0]) / 2, (d["from_px"][1] + d["to_px"][1]) / 2)
        cv2.putText(over, d["id"], (int(mid[0]) + 3, int(mid[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
    for l in leaders:
        tip = tuple(int(v) for v in l["tip_px"])
        cv2.circle(over, tip, 4, (0, 150, 0), 2)
        cv2.putText(over, l["id"], (tip[0] + 5, tip[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 0), 1)
    crops = []
    for d in dims:
        x0b, y0b, x1b, y1b = d["text_box_px"] or [int(min(d["from_px"][0], d["to_px"][0])),
                                                  int(min(d["from_px"][1], d["to_px"][1])) - 20,
                                                  int(max(d["from_px"][0], d["to_px"][0])),
                                                  int(max(d["from_px"][1], d["to_px"][1])) + 20]
        pad = 6
        crop = gray[max(0, y0b - pad):y1b + pad, max(0, x0b - pad):x1b + pad]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        tile = np.full((crop.shape[0] + 28, max(crop.shape[1], 120)), 255, np.uint8)
        tile[28:, :crop.shape[1]] = crop
        cv2.putText(tile, d["id"], (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
        crops.append(tile)
    images = [(f"Scan {scan_id}: part black, dimension lines red with their ids, leader arrow tips green (L1, ...), "
               f"holes orange, outline blue" + (", the blank's bend lines magenta (B1, ...)" if flat_blank else ""),
               cv2.imencode(".png", over)[1].tobytes())]
    if flat_blank.get("_image") is not None:
        x, y, w, h = flat_blank["bbox"]
        img = flat_blank["_image"][max(0, y - 30):y + h + 30, max(0, x - 40):x + w + 40]
        images.append((f"Scan {scan_id}: the blank's panels, each in its own colour with its id (P1 = the largest), "
                       "and the bend lines between them in red, labelled with the two panels each joins",
                       cv2.imencode(".png", img)[1].tobytes()))
    view_faces = result.get("faces") or {}
    if view_faces.get("_image") is not None:
        images.append((f"Scan {scan_id}: the faces of this view, each in its own colour with its id (F1 = the "
                       "largest)", cv2.imencode(".png", view_faces["_image"])[1].tobytes()))
    if flat_blank.get("corners"):
        x, y, w, h = flat_blank["bbox"]
        img = raster_vector.corner_image(gray, flat_blank["contour"], flat_blank["corners"])
        img = img[max(0, 2 * y - 40):2 * (y + h) + 40, max(0, 2 * x - 60):2 * (x + w) + 60]
        images.append((f"Scan {scan_id}: the blank's traced outline (blue) with its corners V1..V"
                       f"{len(flat_blank['corners'])}, in order round the outline, for trace_outline",
                       cv2.imencode(".png", img)[1].tobytes()))
    if crops:
        images.append((f"Scan {scan_id}: the value text of each dimension, enlarged (rotated text stays rotated)",
                       cv2.imencode(".png", crop_grid(crops))[1].tobytes()))
    log.info("scan %s: %d dimension(s), %d leader(s), %d hole(s), outline closed: %s", scan_id, len(dims),
             len(result["leaders"]), len(shape.get("holes", [])), shape.get("closed"))
    return ToolOutput({"scan": scan_id, "box": box, "dimensions": [
        {k: d[k] for k in ("id", "length_px", "orientation", "from_sheet", "to_sheet")} for d in dims],
        "leaders": [{k: l[k] for k in ("id", "tip_sheet", "bend") if k in l} for l in leaders],
        "holes_px": shape.get("holes", []),
        "outline": {"closed": shape.get("closed"), "bbox_px": shape.get("outline_bbox")}}
        | ({"blank": {"panels": flat_blank.get("panels"), "bend_lines": [
            {"id": b["id"], "from_sheet": frac(b["from"]), "to_sheet": frac(b["to"])}
            for b in flat_blank.get("bend_lines", [])],
            "panels": [{k: p[k] for k in ("id", "bends")} for p in flat_blank.get("panels", [])],
            "bend_lines_join": {b["id"]: b["panels"] for b in flat_blank.get("bend_lines", [])},
            "corners": len(flat_blank.get("corners", [])),
            "corners_dropped": (flat_blank.get("corner_check") or {}).get("dropped", []),
            "note": "Panels are the regions the outline and the bend lines enclose; a leader with 'bend' points at "
                    "that bend line. Reliable on a copy of the blank without dimensions (extension lines enclose "
                    "false panels). set_dimension_values gives all of it in mm; then trace_outline turns the traced "
                    "outline into exact lines and arcs, using the corner ids V1.. on the outline image."}}
           if flat_blank else {})
        | ({"faces": view_faces["faces"],
            "faces_note": "Each region the view's edges enclose is a face of the part (F1 = the largest); 'touches' "
                          "lists the faces it meets. On a formed part, match the blank's panels to these faces: which "
                          "lies flat, which rise, which drop."} if view_faces.get("faces") else {})
        | {"next": ("Read each dimension's value from its crop and call set_dimension_values with them" if dims else
                    "No dimensions in this scan: call set_dimension_values with scale_from=<the scan of the same "
                    "view, or another view at the same drawing scale, whose values you set>")
                   + f". Then analyse this view with call_subagent(view_id='{scan_id}')."}, images)


def check_corners(run: Run, scan_id: str, gray: np.ndarray, flat_blank: dict):
    """Have a vision model judge every traced corner from a close-up, in one call, and drop the ones that are not
    on the part (a trace that followed an arrowhead or a leader). The remaining corners are renumbered."""
    import raster_vector

    corners = flat_blank["corners"]
    ids = [f"V{n}" for n in range(1, len(corners) + 1)]
    x, y, w, h = flat_blank["bbox"]
    overview = raster_vector.corner_image(gray, flat_blank["contour"], corners)
    overview = overview[max(0, 2 * y - 40):2 * (y + h) + 40, max(0, 2 * x - 60):2 * (x + w) + 60]
    grid = raster_vector.corner_patches(gray, flat_blank["contour"], corners)
    content = [{"type": "text", "text": f"Scan {scan_id}: {len(corners)} traced corners, {', '.join(ids)}. "
                                        "Judge every one."},
               {"type": "text", "text": "Overview:"}, image_part(cv2.imencode(".png", overview)[1].tobytes()),
               {"type": "text", "text": "Close-ups, one per corner:"},
               image_part(cv2.imencode(".png", grid)[1].tobytes())]
    try:
        answer = run_agent(load_agent("corner_check"), Ctx(run, None, {"corners": ids}), content,
                           label=f"corner_check[{scan_id}]")
    except Exception as error:
        log.warning("corner check of %s failed (%s): corners kept as traced", scan_id, error)
        return
    verdict = {c["corner"]: c for c in answer["corners"]}
    keep = [i for i, v in zip(corners, ids) if verdict.get(v, {}).get("verdict") != "artifact"]
    dropped = [f"{v} ({verdict[v]['reason']})" for v in ids if verdict.get(v, {}).get("verdict") == "artifact"]
    if len(keep) >= 3:
        flat_blank["corners"] = keep
    flat_blank["corner_check"] = {"dropped": dropped, "on_edge": [v for v in ids
                                                                   if verdict.get(v, {}).get("verdict") == "on_edge"]}
    log.info("corner check %s: %d of %d corners dropped as artifacts: %s", scan_id, len(dropped), len(ids),
             "; ".join(dropped) or "none")
    run.trace("corner_check", scan=scan_id, verdicts=answer["corners"], dropped=dropped)


def crop_grid(crops: list, max_width: int = 2400) -> np.ndarray:
    """Labelled crops in rows, as many per row as fit in max_width, so the sheet stays a sensible size however
    many dimensions a scan finds (one column of 50 crops would be taller than a model accepts)."""
    rows, row, width = [], [], 0
    for c in crops:
        if row and width + c.shape[1] + 12 > max_width:
            rows.append(row)
            row, width = [], 0
        row.append(c)
        width += c.shape[1] + 12
    if row:
        rows.append(row)
    lines = []
    for r in rows:
        height = max(c.shape[0] for c in r)
        cells = [np.pad(c, ((0, height - c.shape[0]), (0, 12)), constant_values=255) for c in r]
        lines.append(np.hstack(cells))
    full = max(line.shape[1] for line in lines)
    return np.vstack([np.pad(line, ((0, 8), (0, full - line.shape[1])), constant_values=255) for line in lines])


def scan_frame(s: dict, px_per_mm: float):
    """Pixels to mm in the view's own frame: origin at the lower-left corner of the blank's (or else the outline's)
    box, x to the right, y up on the sheet."""
    bbox = (s.get("blank") or {}).get("bbox") or s["shape"].get("outline_bbox")
    ox, oy = (bbox[0], bbox[1] + bbox[3]) if bbox else (0, 0)
    return lambda p: [round((p[0] - ox) / px_per_mm, 2), round((oy - p[1]) / px_per_mm, 2)]


def leaders_in_mm(s: dict, px_per_mm: float) -> list[dict]:
    """Leader arrow tips in mm, in the view's frame (scan_frame). On a flat pattern a BEND leader's tip lies on its
    bend line: 'bend' names the bend line it points at."""
    to_mm = scan_frame(s, px_per_mm)
    return [{"id": l["id"], "tip_mm": to_mm(l["tip_px"])} | ({"bend": l["bend"]} if l.get("bend") else {})
            for l in s["leaders"]]


def blank_in_mm(s: dict, px_per_mm: float) -> dict:
    """The flat blank in mm, in the view's frame (scan_frame): outline as lines and arcs, bend lines, holes."""
    b = s.get("blank") or {}
    if not b.get("segments"):
        return {}
    to_mm = scan_frame(s, px_per_mm)
    mm = lambda v: round(v / px_per_mm, 2)
    outline = []
    for seg in b["segments"]:
        if "line" in seg:
            outline.append({"line": [to_mm(p) for p in seg["line"]]})
        else:
            a = seg["arc"]
            outline.append({"arc": {"from": to_mm(a["from"]), "mid": to_mm(a["mid"]), "to": to_mm(a["to"]),
                                    "centre": to_mm(a["centre"]), "radius": mm(a["radius"])}})
    profile = []
    for seg in outline:
        if "line" in seg:
            profile.append(seg["line"][0])
            continue
        a = seg["arc"]
        c = np.array(a["centre"])
        ang = [math.atan2(*(np.array(a[k]) - c)[::-1]) for k in ("from", "mid", "to")]
        sweep = (ang[2] - ang[0]) % (2 * math.pi)
        if (ang[1] - ang[0]) % (2 * math.pi) > sweep:      # the arc runs the other way round
            sweep -= 2 * math.pi
        steps = max(2, int(abs(math.degrees(sweep)) / 10))
        profile += [[round(float(c[0] + a["radius"] * math.cos(ang[0] + sweep * k / steps)), 2),
                     round(float(c[1] + a["radius"] * math.sin(ang[0] + sweep * k / steps)), 2)] for k in range(steps)]
    panels = [{"id": p["id"], "area_mm2": round(p["area_px"] / px_per_mm ** 2, 1), "inside": to_mm(p["inside"]),
               "bends": p["bends"]} for p in b.get("panels", [])]
    return {"size_mm": [mm(b["bbox"][2]), mm(b["bbox"][3])], "profile": profile, "outline": outline,
            "panels": panels,
            "bend_lines": [{"id": bl["id"], "from": to_mm(bl["from"]), "to": to_mm(bl["to"]), "joins": bl["panels"]}
                           for bl in b["bend_lines"]],
            "holes": [{"centre": to_mm(h["centre"]), "d_mm": mm(h["d_px"])} for h in b["holes"]],
            "note": "Frame: origin at the lower-left corner of the blank's box, x right, y up, as drawn. 'panels': "
                    "the regions between the bend lines, as on the panel image, each with a point inside it (use "
                    "that point as base_flange.fixed for the panel that stays flat) and the bends on its edges; "
                    "each bend line 'joins' two panels. 'profile' "
                    "is the outline as points (arcs sampled), ready for base_flange.profile. The outline "
                    "is traced from the scan (about 0.3 mm; small steps where a leader meets the edge are tracing "
                    "noise): use the labelled radii, diameters and sizes where they differ. Bend lines are the "
                    "drawn ones, clipped to the blank: use them as the bends' lines."}


def set_dimension_values(ctx: Ctx, scan: Annotated[str, "The scan id returned by scan_view, e.g. S1"],
                         values: Annotated[str, "JSON object of the values you read, by dimension id, in mm, e.g. "
                                                "'{\"D1\": 61, \"D2\": 52.2}'. Leave out any you cannot read"] = "{}",
                         scale_from: Annotated[str, "For a scan with too few dimensions: the id of a scan at the same "
                                                    "drawing scale whose values are set (e.g. the dimensioned copy "
                                                    "of the same blank); its scale is used"] = "") -> dict:
    """Fit one mm-per-pixel scale to the dimension values you read, check every dimension against it (a misread
    value or a wrong arrow pairing shows as a large error), and convert the scan's geometry to mm. A scan without
    dimensions takes the scale of another scan (scale_from), checked against that scan's overall sizes."""
    run = ctx.run
    if scan not in run.scans:
        raise ValueError(f"Unknown scan {scan!r}; use one of {sorted(run.scans)}")
    s = run.scans[scan]
    read = {k: float(v) for k, v in json.loads(values or "{}").items()}
    pairs = [(d, read[d["id"]]) for d in s["dimensions"] if d["id"] in read and read[d["id"]] > 0]
    scale_check = None
    if len(pairs) < 2 and scale_from:
        ref = run.scans.get(scale_from)
        if not ref or not ref.get("scale_px_per_mm"):
            raise ValueError(f"Scan {scale_from!r} has no scale yet: set its dimension values first")
        px_per_mm = ref["scale_px_per_mm"]
        # the same part at the same drawing scale: its size here should match the overall sizes read there
        bbox = (s.get("blank") or {}).get("bbox") or s["shape"].get("outline_bbox")
        largest = {o: max([v for d, v in ref.get("values", []) if d["orientation"] == o], default=None)
                   for o in ("horizontal", "vertical")}
        if bbox:
            # a second copy is often drawn at a slightly different scale: when its size matches an overall value
            # read on the other scan (within 5%), take the scale from that size instead
            overall = [bbox[2 if o == "horizontal" else 3] / largest[o] for o in ("horizontal", "vertical")
                       if largest[o] and abs(bbox[2 if o == "horizontal" else 3] / px_per_mm - largest[o])
                       <= 0.05 * largest[o]]
            if overall:
                px_per_mm = float(np.mean(overall))
            size = {"horizontal": bbox[2] / px_per_mm, "vertical": bbox[3] / px_per_mm}
            # no part is smaller than its largest dimension; larger is fine when its overall size is not dimensioned
            scale_check = {o: {"size_here_mm": round(size[o], 1), "largest_value_there_mm": largest[o],
                               "ok": largest[o] is None or size[o] >= largest[o] - max(2.0, 0.05 * largest[o])}
                           for o in size}
    elif len(pairs) < 2:
        raise ValueError("Give at least two dimension values to fit the scale, or scale_from=<a scan at the same "
                         "drawing scale whose values are set>")
    else:
        px_per_mm = float(np.median([d["length_px"] / v for d, v in pairs]))
        s["values"] = pairs
    checks = []
    for d, v in pairs:
        measured = d["length_px"] / px_per_mm
        error = (measured - v) / v * 100
        checks.append({"id": d["id"], "value": v, "measured_mm": round(measured, 2), "error_pct": round(error, 1),
                       "ok": abs(error) <= 3})
    shape = s["shape"]
    mm = lambda px: round(px / px_per_mm, 2)
    s["scale_px_per_mm"] = px_per_mm
    result = {"scan": scan, "px_per_mm": round(px_per_mm, 4),
              "dimensions_checked": checks,
              "suspect": [c["id"] for c in checks if not c["ok"]],
              "holes_mm": [{"centre_px": h["centre"], "d_mm": mm(h["d_px"])} for h in shape.get("holes", [])],
              "outline": {"closed": shape.get("closed"),
                          "size_mm": [mm(v) for v in shape.get("outline_bbox", [0, 0, 0, 0])[2:]]},
              "leaders_mm": leaders_in_mm(s, px_per_mm),
              "note": "A suspect dimension was misread or paired wrongly: re-read its crop, or leave it out. Hole "
                      "and outline sizes come from image processing and are approximate; dimension values win."}
    if scale_from and scale_check is not None:
        result["scale_from"] = {"scan": scale_from, "check": scale_check,
                                "scale": ("fitted to the overall size read there" if px_per_mm !=
                                          run.scans[scale_from]["scale_px_per_mm"] else
                                          "copied: no overall value read there matches this scan's size; read the "
                                          "overall width and height on that scan if they are dimensioned"),
                                "note": "At the same scale the part here is at least as large as the largest "
                                        "value read on that scan (equal when that value is the overall size). Not "
                                        "ok: the two are not at the same scale; do not use this scan's mm."}
    if s.get("blank"):
        result["blank_mm"] = blank_in_mm(s, px_per_mm)
    run.trace("scan_values", scan=scan, values=read, result=result)
    log.info("scan %s: %.3f px/mm from %d value(s); suspect: %s", scan, px_per_mm, len(pairs),
             result["suspect"] or "none")
    return result


def trace_outline(ctx: Ctx, scan: Annotated[str, "The scan id of a flat blank (scan_view kind flat_pattern)"],
                  spans: Annotated[str, "JSON list, in order round the outline, of what each stretch between two "
                                        "corners is: {\"from\": \"V3\", \"to\": \"V7\", \"kind\": \"line\" | "
                                        "\"arc\", \"radius\": 25 (mm, the labelled radius; leave out to fit it), "
                                        "\"corner_radius\": 2 (mm, rounds the corner where this line meets the "
                                        "next line)}. Stretches you leave out stay as traced"]) -> ToolOutput:
    """Turn a scanned blank's traced outline into exact lines and arcs. You say what each stretch between two
    numbered corners is (from the outline image of scan_view and the drawing's radius callouts); each is fitted to
    the scanned edge with a robust fit (leader lines crossing the edge are ignored), a labelled radius is kept
    exactly, and the pieces are joined where they meet. Returns the outline in mm (same frame as blank_mm), a
    report per stretch (a stretch not ok was labelled wrongly: split it at another corner, or change its kind or
    radius, and call again) and an image of the fit."""
    import outline_fit
    import raster_vector

    run = ctx.run
    s = run.scans.get(scan)
    if not s or not (s.get("blank") or {}).get("corners"):
        raise ValueError(f"{scan!r} has no traced blank: trace_outline works on a scan_view of kind flat_pattern")
    px_per_mm = s.get("scale_px_per_mm")
    if not px_per_mm:
        raise ValueError(f"Set the scale of {scan} first: set_dimension_values (with scale_from if it has no dimensions)")
    b = s["blank"]
    fit = outline_fit.fit_outline(b["contour"], b["corners"], json.loads(spans), px_per_mm)
    to_mm = scan_frame(s, px_per_mm)
    outline = []
    for p in fit["pieces"]:
        if p["kind"] == "arc":
            arc = outline_fit.arcs([p])[0]
            outline.append({"arc": {k: to_mm(arc[k]) for k in ("from", "mid", "to", "centre")}
                            | {"radius": round(p["r"] / px_per_mm, 2)}})
        else:
            outline.append({"line": [to_mm(p["start"]), to_mm(p["end"])]})
    profile = [to_mm(p) for p in outline_fit.sample(fit["pieces"])]
    xs, ys = np.array(profile).T
    area = 0.5 * abs(float(xs @ np.roll(ys, 1) - ys @ np.roll(xs, 1)))
    s["outline_fit"] = {"spans": json.loads(spans), "profile": profile}
    x, y, w, h = b["bbox"]
    gray = cv2.imread(str(Path(s["folder"]) / "view.png"), cv2.IMREAD_GRAYSCALE)
    img = raster_vector.corner_image(gray, b["contour"], b["corners"], fit["pieces"])
    img = img[max(0, 2 * y - 40):2 * (y + h) + 40, max(0, 2 * x - 60):2 * (x + w) + 60]
    bad = [f"{e.get('from', e.get('corner'))}->{e.get('to', '')}" for e in fit["report"] if not e["ok"]]
    log.info("outline %s: %d piece(s), area %.0f mm2, not ok: %s", scan, len(fit["pieces"]), area, bad or "none")
    run.trace("outline_fit", scan=scan, spans=json.loads(spans), report=fit["report"])
    return ToolOutput({"scan": scan, "fits": fit["report"], "not_ok": bad, "outline": outline, "profile": profile,
                       "area_mm2": round(area, 1),
                       "note": "Frame as blank_mm (origin at the blank's lower-left corner, x right, y up). on_edge: "
                               "share of the traced edge within 1.5 px of the fitted piece; ok needs 0.6. 'profile' "
                               "is the outline as points with the arcs sampled: use it as base_flange.profile."},
                      [(f"Scan {scan}: fitted outline (red) over the traced one (blue), corners V1..", 
                        cv2.imencode(".png", img)[1].tobytes())])


Millimetres = Annotated[float, "mm along the overlay's axis, as labelled on the overlay image"]


def zoom_comparison(ctx: Ctx, h_from: Millimetres, h_to: Millimetres,
                    v_from: Millimetres, v_to: Millimetres) -> ToolOutput:
    """Show one region of the overlay larger: drawn edges in black, model edges in red, same axes as the overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    check = ctx.check or {}
    if "points" not in check:
        raise ValueError("This view has no measured overlay to zoom into")
    data = np.load(check["points"])
    (h0, h1), (v0, v1) = sorted((h_from, h_to)), sorted((v_from, v_to))
    if h1 - h0 < 0.5 or v1 - v0 < 0.5:
        raise ValueError("Region too small: give at least 0.5 mm in each direction")
    fig, ax = plt.subplots(figsize=(6, 6 * min(2.0, max(0.4, (v1 - v0) / (h1 - h0)))))
    for key, color, size in (("drawing", "black", 2.5), ("model", "red", 1.5)):
        P = data[key]
        P = P[(P[:, 0] >= h0) & (P[:, 0] <= h1) & (P[:, 1] >= v0) & (P[:, 1] <= v1)]
        ax.plot(P[:, 0], P[:, 1], ".", color=color, ms=size, label=key)
    ax.set_xlim(h0, h1)
    ax.set_ylim(v0, v1)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)
    ax.set_xlabel(f"{check.get('horizontal', '?')} (mm)")
    ax.set_ylabel(f"{check.get('vertical', '?')} (mm)")
    ax.legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    caption = f"Zoom: {check.get('horizontal')} {h0:g}..{h1:g} mm, {check.get('vertical')} {v0:g}..{v1:g} mm"
    return ToolOutput({"shown": caption}, [(caption, buffer.getvalue())])


TOOLS = {f.__name__: f for f in (scan_view, set_dimension_values, trace_outline,
                                 view_drawing, read_drawing_text, split_views, call_subagent,
                                 read_view_text, zoom_view, run_cadquery, zoom_comparison, extend_view)}
JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool_schema(fn) -> dict:
    """OpenAI function schema from a tool's type hints, Annotated descriptions, and docstring."""
    hints = get_type_hints(fn, include_extras=True)
    props, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        if name == "ctx":
            continue
        hint, desc = hints[name], ""
        if get_origin(hint) is Annotated:
            hint, desc = get_args(hint)[0], get_args(hint)[1]
        if get_origin(hint) in (Union, types.UnionType):
            hint = next(a for a in get_args(hint) if a is not type(None))
        if get_origin(hint) is Literal:
            prop = {"type": "string", "enum": list(get_args(hint))}
        else:
            prop = {"type": JSON_TYPES[hint]}
        if desc:
            prop["description"] = desc
        if param.default is inspect.Parameter.empty:
            required.append(name)
        props[name] = prop
    return {"type": "function", "function": {
        "name": fn.__name__, "description": inspect.getdoc(fn) or "",
        "parameters": {"type": "object", "properties": props, "required": required},
    }}


def view_image(view: View) -> bytes:
    if view.pdf is None:
        return view.png.read_bytes()
    with pymupdf.open(view.pdf) as doc:
        return png_of(doc[0], long_side=1400)


MAX_IMAGE_SIDE = 4000  # providers refuse images over 8000 px on a side; stay well under it


def image_part(png: bytes) -> dict:
    """An image for a model message, scaled down if a side is longer than MAX_IMAGE_SIDE."""
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is not None and max(image.shape[:2]) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(image.shape[:2])
        image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
        png = cv2.imencode(".png", image)[1].tobytes()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


# --------------------------------------------------------------------------- agent loop
def run_agent(agent: Agent, ctx: Ctx, content: list[dict], label: str | None = None) -> dict:
    """Run one agent until it replies with JSON; return the parsed JSON."""
    run, label = ctx.run, label or agent.name
    alog = log.getChild(label)
    tools = [TOOLS[name] for name in agent.tools]
    schemas = [tool_schema(f) for f in tools]
    messages: list[dict] = [{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": content}]
    max_turns = int(agent.settings.get("max_turns", 10))
    started, agent_cost = time.monotonic(), 0.0
    alog.info("started (%d tools, max %d turns)", len(tools), max_turns)
    run.trace("agent_start", agent=label, tools=agent.tools)

    used: set[str] = set()
    stopped = False
    for turn in range(1, max_turns + REPAIR_TURNS + 1):  # extra turns only for fixing a rejected answer
        last = turn >= max_turns or run.over_budget
        if last and not stopped:
            stopped = True
            messages.append({"role": "user", "content": "Stop using tools now and reply with the final JSON object."})
        kwargs = {"tools": schemas, "tool_choice": "none" if last else "auto"} if schemas else {}
        if "temperature" in agent.settings:
            kwargs["temperature"] = agent.settings["temperature"]
        t0 = time.monotonic()
        reply = run.client.chat.completions.create(model=MODEL, messages=messages,
                                                   extra_body={"usage": {"include": True}}, **kwargs)
        usage = reply.usage.model_dump() if reply.usage else {}
        cost = usage.get("cost")
        if cost is None:
            run.cost_unknown_calls += 1
        else:
            run.cost_usd += cost
            agent_cost += cost
        message = reply.choices[0].message
        calls = message.tool_calls or []
        alog.info("turn %d: %s in %.1fs, cost %s (run total $%.4f)", turn,
                  f"{len(calls)} tool call(s)" if calls else "answer", time.monotonic() - t0,
                  f"${cost:.4f}" if cost is not None else "unavailable", run.cost_usd)
        if message.content:
            alog.debug("says: %s", message.content[:2000])
        run.trace("llm", agent=label, turn=turn, id=reply.id, usage=usage, content=message.content,
                  tool_calls=[{"name": c.function.name, "arguments": c.function.arguments} for c in calls])

        if not calls:
            missing = missing_steps(agent, run, used)
            if missing and not stopped:
                # a reply in the middle of the work (an overview written out, a thought) is not the final answer
                alog.info("reply before the workflow is done; still to call: %s", ", ".join(missing))
                run.trace("not_done", agent=label, turn=turn, missing=missing)
                messages += [{"role": "assistant", "content": message.content or ""},
                             {"role": "user", "content": "Not finished yet: the workflow still needs "
                              + ", ".join(f"`{m}`" for m in missing) + ". Continue with the next step, using "
                              "your tools. Reply with the final JSON only when every step is done."}]
                continue
            try:
                result = parse_json(message.content or "")
                if agent.output_model:
                    context = {"views": list(run.views)}
                    if agent.output_model is schema.ArbiterReport:
                        context["texts"] = drawing_texts(run)
                    if ctx.check and "corners" in ctx.check:
                        context["corners"] = ctx.check["corners"]
                    result = agent.output_model.model_validate(result, context=context).model_dump(mode="json")
            except (ValueError, ValidationError) as error:
                problem = validation_problems(error)
                alog.warning("reply rejected, asking for a fix: %s", problem.replace("\n", " | ")[:400])
                run.trace("rejected", agent=label, turn=turn, problems=problem)
                messages += [{"role": "assistant", "content": message.content or ""},
                             {"role": "user", "content": f"Your reply was rejected:\n{problem}\n\n"
                                                         "Fix these problems and reply with only the corrected "
                                                         "JSON object."}]
                continue
            alog.info("done in %.1fs, cost $%.4f", time.monotonic() - started, agent_cost)
            run.trace("agent_end", agent=label, result=result, cost_usd=agent_cost)
            return result

        used |= {c.function.name for c in calls}
        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in calls]})
        images = []
        for call in calls:
            output = execute(call.function.name, call.function.arguments, ctx, tools, alog)
            if isinstance(output, ToolOutput):
                images += output.images
                output = output.data
            text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        if images:  # tool messages carry text only, so images follow as a user message
            parts = []
            for caption, png in images:
                parts += [{"type": "text", "text": caption}, image_part(png)]
            messages.append({"role": "user", "content": parts})

    raise RuntimeError(f"{label} gave no valid answer in {max_turns} turns (+{REPAIR_TURNS} repair turns)")


def missing_steps(agent: Agent, run: Run, used: set[str]) -> list[str]:
    """Tools the agent's YAML requires before it may give its final answer (`required_tools`, plus
    `required_tools_scanned` when the sheet is a scan), that it has and has not used yet."""
    required = list(agent.settings.get("required_tools") or [])
    if run.scanned:
        required += agent.settings.get("required_tools_scanned") or []
    return [t for t in required if t in agent.tools and t not in used]


def execute(name: str, arguments: str, ctx: Ctx, tools: list, alog: logging.Logger):
    fn = next((f for f in tools if f.__name__ == name), None)
    t0 = time.monotonic()
    try:
        if fn is None:
            raise ValueError(f"Unknown tool {name!r}")
        args = json.loads(arguments or "{}")
        alog.info("→ %s(%s)", name, ", ".join(f"{k}={short(v)}" for k, v in args.items()))
        output = fn(ctx, **args)
        ok = True
    except Exception as error:
        alog.warning("✗ %s failed: %s: %s", name, type(error).__name__, error)
        output, ok = {"error": f"{type(error).__name__}: {error}"}, False
    data = output.data if isinstance(output, ToolOutput) else output
    ctx.run.trace("tool", agent=alog.name.removeprefix("draw2cad."), tool=name, arguments=arguments, ok=ok,
                  seconds=round(time.monotonic() - t0, 2), result=data,
                  images=len(output.images) if isinstance(output, ToolOutput) else 0)
    return output


def short(value, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return repr(text[:limit] + "…") if len(text) > limit else repr(text) if isinstance(value, str) else text


def validation_problems(error: Exception) -> str:
    if not isinstance(error, ValidationError):
        return f"- not valid JSON: {error}"
    lines = []
    for e in error.errors()[:25]:
        where = ".".join(str(p) for p in e["loc"]) or "(top level)"
        lines.append(f"- {where}: {e['msg']}")
    return "\n".join(lines)


def parse_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{"):text.rfind("}") + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError(str(error)) from None
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


# --------------------------------------------------------------------------- entry point
def open_run(drawing: Path, run_dir: Path, max_cost_usd: float) -> Run:
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Add OPENROUTER_API_KEY to .env or your environment")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=180, max_retries=2)
    return Run(drawing.resolve(), run_dir.resolve(), client, max_cost_usd=max_cost_usd)


AXIS_VECTORS = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}


def apply_extensions(tree: dict) -> list[str]:
    """Features that merge into a round body (`extends_into`) run up to that body's axis: compute their height
    here instead of letting the model guess it. The body's axis passes through its position along its direction;
    the height is the distance from the feature's start, along the feature's direction, to that axis."""
    import numpy as np

    by_id = {f["id"]: f for f in tree["features"]}
    notes = []
    for f in tree["features"]:
        target_id = f.get("extends_into")
        if not target_id:
            continue
        target = by_id.get(target_id)
        d = AXIS_VECTORS.get(f.get("direction") or "")
        if target is None or f.get("position") is None or d is None:
            notes.append(f"{f['id']}: extends_into {target_id} needs this feature's position and direction")
            continue
        axis = AXIS_VECTORS.get(target.get("direction") or "")
        if "d" not in target["params"] or target.get("position") is None or axis is None:
            notes.append(f"{f['id']}: extends_into {target_id} needs a round body (d, position on its axis, direction)")
            continue
        d, axis = np.array(d, float), np.array(axis, float)
        if abs(float(np.dot(d, axis))) > 1e-6:
            notes.append(f"{f['id']}: it runs along {target_id}'s axis, so it cannot rise to it")
            continue
        start, centre = np.array(f["position"], float), np.array(target["position"], float)
        height = round(float(np.dot(centre - start, d)), 3)
        if height <= 0:
            notes.append(f"{f['id']}: {target_id}'s axis is behind its start face ({height:g} mm)")
            continue
        old = f["params"].get("height")
        f["params"]["height"] = height
        i = int(np.argmax(np.abs(d)))
        f["evidence"] = [e for e in f["evidence"] if e["role"] != "height"] + [{
            "view": "notes", "text": f"extends into {target_id}", "role": "height",
            "derived": f"up to the axis of {target_id} ({target['name']}): {'XYZ'[i]} {centre[i]:g} - {start[i]:g}"}]
        notes.append(f"{f['id']}.height = {height:g} mm, up to the axis of {target_id}" +
                     (f" (the plan said {old:g})" if isinstance(old, (int, float)) and old != height else ""))
    return notes


def save_plan(run: Run, result: dict) -> dict:
    for note in apply_extensions(result["feature_tree"]):
        log.info("  extension: %s", note)
    (run.dir / "feature_tree.json").write_text(json.dumps(result["feature_tree"], indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    (run.dir / "plan.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    tree = result["feature_tree"]
    for f in tree["features"]:
        log.info("  %s %-18s %-30s %s", f["id"], f["op"], f["name"][:30],
                 ", ".join(f"{k}={v}" for k, v in f["params"].items() if not isinstance(v, list)))
    log.info("plan: %d feature(s), %d assumption(s), %d unsupported", len(tree["features"]),
             len(tree["assumptions"]), len(tree["unsupported"]))
    return result


def plan(run: Run) -> dict:
    """Main agent + view subagents -> feature tree (feature_tree.json, plan.json)."""
    content = [{"type": "text", "text": f"Drawing file: {run.drawing.name}. Start with step 1."}]
    result = run_agent(load_agent("main"), Ctx(run), content)
    return save_plan(run, {**result, "summaries": run.summaries, "cross_check": run.cross_check})


def revise_plan(run: Run, plan_result: dict, issues: list[dict], pose: dict | None, round_no: int) -> dict:
    """Main agent again, without splitting or re-analysing: fix the plan issues found by validation."""
    agent = load_agent("main")
    agent.settings = {**agent.settings, "tools": ["view_drawing", "read_drawing_text"]}
    summaries = {}
    for s in plan_result.get("summaries", []):
        path = run.dir / s["file"]
        if path.is_file():
            summaries[s["view"]] = json.loads(path.read_text(encoding="utf-8"))
    current = {k: plan_result[k] for k in ("overview", "feature_tree", "uncertainties") if k in plan_result}
    text = (f"# Revision {round_no}\n\nThe part was built from your plan and checked against every view of the "
            "drawing. The checks found the problems below in the **plan itself**. This is a revision: do not "
            "split the views or call subagents again; the view summaries below are still valid. Re-read the "
            "drawing with view_drawing and read_drawing_text where the issues point. Fix these problems, keep "
            "everything else and every feature id unchanged, and reply with the complete corrected plan JSON.\n\n"
            "Each issue has a `basis`. A **measured** issue comes from projecting the model onto a drawn view and "
            "checking its dimensions. A **visual only** issue is a judgement by eye on a view that could not be "
            "measured (often a small pictorial view, where hidden steps cannot be seen): treat it as a hint. "
            "If an issue contradicts a dimension you read from a section or an orthographic view, keep your value "
            "and say why in `uncertainties`; do not remove a feature the drawing dimensions.\n\n"
            "## Issues to fix\n```json\n" + json.dumps(issues, indent=1, ensure_ascii=False) + "\n```\n\n"
            + ("## Orientation check\n```json\n" + json.dumps(pose, indent=1) + "\n```\n\n" if pose else "")
            + "## Your current plan\n```json\n" + json.dumps(current, indent=1, ensure_ascii=False) + "\n```\n\n"
            "## View summaries\n```json\n" + json.dumps(summaries, indent=1, ensure_ascii=False) + "\n```"
            + ("\n\n## Cross-check rulings (settled from the drawing; they override the summaries)\n```json\n"
               + json.dumps(plan_result["cross_check"].get("rulings", []), indent=1, ensure_ascii=False) + "\n```"
               if (plan_result.get("cross_check") or {}).get("rulings") else ""))
    result = run_agent(agent, Ctx(run), [{"type": "text", "text": text}], label=f"main[revision {round_no}]")
    return save_plan(run, {**result, "summaries": plan_result.get("summaries", []),
                           "cross_check": plan_result.get("cross_check")})


def build(run: Run, plan_result: dict, corrections: dict | None = None) -> dict:
    """Builder agent: feature tree -> model.py + model.step, then viewer.html."""
    brief = {"overview": plan_result.get("overview"), "feature_tree": plan_result["feature_tree"]}
    text = "Build this part.\n\n```json\n" + json.dumps(brief, indent=1, ensure_ascii=False) + "\n```"
    if corrections:
        for name in ("model.step", "model.py"):  # archived by validate(); a failed rebuild must not reuse them
            (run.dir / name).unlink(missing_ok=True)
        text += (f"\n\n# Corrections, round {corrections['round']}\n\nYour previous model was checked against the "
                 "drawing. " + ("The feature tree above has been corrected by the planner since then. "
                                if corrections["plan_revised"] else "")
                 + "Start from your previous script below, apply the tree changes and fix the build issues, "
                 "and change nothing else.\n\n## Build issues\n```json\n"
                 + json.dumps(corrections["issues"], indent=1, ensure_ascii=False)
                 + "\n```\n\n## Previous script\n```python\n" + corrections["previous_script"] + "\n```")
    report = run_agent(load_agent("builder"), Ctx(run), [{"type": "text", "text": text}])
    step = run.dir / "model.step"
    if not step.is_file():
        report = {**report, "status": "failed", "notes": report.get("notes", []) + [
            "No script produced a valid single solid, so no STEP was written."]}
    elif report["status"] == "failed":
        report["status"] = "partial"  # a valid solid exists even if the builder gave up on part of it
    (run.dir / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    states = {f["id"]: f["state"] for f in report["features"]}  # only `state` changes after planning
    tree = plan_result["feature_tree"]
    for f in tree["features"]:
        f["state"] = "built" if states.get(f["id"]) in ("built", "approximated") and step.is_file() else "not_started"
    (run.dir / "feature_tree.json").write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")

    for f in report["features"]:
        if f["state"] != "built":
            log.warning("  %s %s: %s", f["id"], f["state"], f["note"])
    log.info("build: %s, STEP %s", report["status"], step if step.is_file() else "not written")
    if step.is_file():
        write_viewer(run)
    return report


def write_viewer(run: Run):
    models = [("model", run.dir / "model.step")]
    reference = run.drawing.with_name("reference.step")
    if reference.is_file():
        models.append(("reference", reference))
    viewer = visualization.write_viewer(models, run.dir / "viewer.html", title=run.dir.name)
    log.info("viewer: %s", viewer)


def validate(run: Run, plan_result: dict, round_no: int) -> dict:
    """Project the model into every view (src/validate.py), then one validation agent per view."""
    out_dir = run.dir / "validation" / f"round_{round_no}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.step", "model.py"):  # keep every round's model
        shutil.copyfile(run.dir / name, out_dir / name)
    vlog = log.getChild("validate")
    result = {"round": round_no, "reports": [], "pose": None, "measured": []}
    # a scan: the views the planner scanned (kind, direction and measured scale known), else the views cropped
    # from the sheet automatically (each tried from every direction)
    labelled = [v for v in run.views.values() if v.scan]
    scanned = [] if any(v.pdf for v in run.views.values()) else (labelled or scan_views(run))
    if not any(v.pdf for v in run.views.values()) and not scanned:
        vlog.warning("no vector views, and no views found on the scanned sheet; model not validated")
        return result
    t0 = time.monotonic()
    if scanned:
        scans_json = out_dir / "scans.json"
        entries = ([{"id": v.id, "label": v.label, **v.scan, "px_per_mm": run.scans[v.id].get("scale_px_per_mm"),
                     "folder": run.scans[v.id].get("folder")} for v in labelled] if labelled else
                   [{"id": v.id, "label": v.label, "kind": "auto", "direction": "none", "box": v.sheet_box,
                     "image": str(v.png), "folder": str(v.png.parent)} for v in scanned])
        scans_json.write_text(json.dumps(entries, indent=1), encoding="utf-8")
        command = [sys.executable, str(ROOT / "src" / "raster_check.py"), str(run.drawing),
                   str(run.dir / "model.step"), str(scans_json), str(out_dir)]
    else:
        command = [sys.executable, str(ROOT / "src" / "validate.py"), str(run.dir / "views"),
                   str(run.dir / "model.step"), str(out_dir), str(run.drawing)]
    try:
        subprocess.run(command, cwd=ROOT / "src", timeout=VALIDATE_TIMEOUT_S, capture_output=True, text=True)
        data = json.loads((out_dir / "validation.json").read_text(encoding="utf-8"))
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as error:
        vlog.warning("projection check failed: %s; model not validated", error)
        return result
    result["pose"] = data.get("pose")
    result["measured"] = [{"view": e["view_id"], **e["metrics"]} for e in data["views"] if e.get("metrics")]
    vlog.info("projections done in %.1fs; pose: %s", time.monotonic() - t0, (data.get("pose") or {}).get("verdict"))
    run.trace("projections", round=round_no, data=data)

    by_pdf = {v.pdf.stem: v for v in run.views.values() if v.pdf} | {v.id: v for v in scanned}
    mark_unknown_cameras(data["views"], vlog)
    tree = plan_result["feature_tree"]
    compact = [{k: f.get(k) for k in ("id", "name", "op", "params", "on", "position", "direction", "pattern")}
               for f in tree["features"]]
    for entry in data["views"]:
        view = by_pdf.get(entry["view_id"])
        # a cropped scan's kind is the one the check found it fits best
        kind = (entry.get("class") if view.pdf is None and not view.scan else view.suggested_subagent) if view else None
        if not kind or "images" not in entry:
            if view and kind:
                vlog.info("%s not validated: %s", view.id, entry.get("unresolved"))
            continue
        frame = (f"In this view, horizontal = {entry['horizontal']}, vertical = {entry['vertical']}, looking along "
                 f"{entry['looking_along']}" + (f", cut plane normal {entry['cut_plane_normal']}"
                                                if entry.get("cut_plane_normal") else "") + ".")
        measured = {k: entry.get(k) for k in ("metrics", "missing_regions", "extra_regions", "callouts", "note",
                                              "class", "camera", "scale_mm_per_px", "orientation", "camera_doubt",
                                              "parent_view", "cut_offset_mm", "side_note", "orientation_note",
                                              "direction_note")
                    if entry.get(k) is not None}
        text = (f"View: {view.id} (label on the sheet: {view.label}). {frame}\n\n"
                "## Projection metrics, regions (in this view's axes, mm) and callout checks\n```json\n"
                + json.dumps(measured, indent=1, ensure_ascii=False) + "\n```\n\n"
                "## Pose check (all views together)\n```json\n" + json.dumps(data.get("pose"), indent=1) + "\n```\n\n"
                "## Feature tree the model was built from\n```json\n"
                + json.dumps({"frame": tree["frame"], "features": compact,
                              "assumptions": tree.get("assumptions", [])}, indent=1, ensure_ascii=False) + "\n```")
        content = [{"type": "text", "text": text}]
        for key, caption in (("drawing", "The drawing view:"), ("model", "The model projection, same direction:"),
                             ("overlay", "Overlay (drawing black, model red; orange = drawn but missing, "
                                         "purple = in model but not drawn):")):
            if key in entry["images"]:
                content += [{"type": "text", "text": caption},
                            image_part(Path(entry["images"][key]).read_bytes())]
        try:
            report = run_agent(load_agent(f"validation/{kind}"), Ctx(run, view, entry), content,
                               label=f"validation/{kind}[{view.id}]")
        except Exception as error:
            vlog.warning("%s: validation agent failed: %s", view.id, error)
            continue
        report = screen_orientation_issues(report, entry, data.get("pose") or {}, vlog)
        report = weigh_by_evidence(report, entry, vlog)
        (out_dir / f"{view.id}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        result["reports"].append(report)
        vlog.info("%s: %s (%s), %d issue(s)", view.id, report["verdict"], report["orientation"], len(report["issues"]))
        for issue in report["issues"]:
            vlog.info("    [%s/%s] %s %s: %s", issue["severity"], issue["owner"], issue.get("feature") or "-",
                      issue["kind"], issue["fix"])
    return result


def scan_views(run: Run) -> list[View]:
    """The views of a scanned sheet, cropped from its pixels (src/regions.py), once per run: each saved with only
    its own ink, for raster_check.py to compare the model with."""
    cached = getattr(run, "_scan_views", None)
    if cached is not None:
        return cached
    import cv2
    import regions
    folder = run.dir / "views" / "scans"
    folder.mkdir(parents=True, exist_ok=True)
    sheet = regions.render(str(run.drawing))
    h, w = sheet.shape
    views = []
    for crop in regions.view_crops(sheet):
        view_dir = folder / crop["id"]
        view_dir.mkdir(exist_ok=True)
        png = view_dir / "view.png"
        cv2.imwrite(str(png), crop["image"])
        # its regions, filled and labelled P1, P2, ... (P1 = the largest), for checking by eye
        found = regions.fill_regions(crop["image"])
        cv2.imwrite(str(view_dir / "regions.png"), regions.label_image(crop["image"], found))
        (view_dir / "regions.json").write_text(json.dumps(
            [{k: r[k] for k in ("id", "area_px", "box", "inside", "touches", "shared_lines")} for r in found],
            indent=1), encoding="utf-8")
        x0, y0, x1, y1 = crop["box"]
        view = View(f"scan_{crop['id']}", f"{crop['id']} (cropped from the scanned sheet)", None, png, None)
        view.sheet_box = [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]
        views.append(view)
    log.info("scanned sheet: %d view(s) cropped for validation", len(views))
    run._scan_views = views
    return views


def mark_unknown_cameras(entries: list[dict], vlog: logging.Logger, good: float = 0.6, poor: float = 0.4):
    """When one view fits the model well and another fits poorly from every camera tried, the model is the same in
    both: the poor view most likely is not one the check can project (an isometric from a non-standard camera, or
    on an unlabelled scan a flat blank or a detail), so its mismatch is not evidence against the model. When no view
    fits, nothing is marked: then the model is what is wrong."""
    fits = lambda e, v: e["metrics"]["drawn_covered"] >= v and e["metrics"]["model_covered"] >= v
    scanned = [e for e in entries if e.get("metrics") and str(e.get("note", "")).startswith("scanned view")]
    pool = scanned or [e for e in entries if e.get("class") == "isometric" and e.get("metrics")]
    if not any(fits(e, good) for e in pool):
        return
    for e in pool:
        if e["metrics"]["drawn_covered"] < poor or e["metrics"]["model_covered"] < poor:
            e["camera_doubt"] = ("Another view of this model fits it well, this one fits poorly from every camera "
                                 "tried: it is probably not a view the check can project (an isometric from a "
                                 "non-standard camera, a flat blank of a bent part, a detail), so the mismatch may "
                                 "be the view, not the model.")
            vlog.info("%s: poor fit while another view fits well; camera in doubt", e["view_id"])


def screen_orientation_issues(report: dict, entry: dict, pose: dict, vlog: logging.Logger) -> dict:
    """One view's left and right is not enough to call the part mirrored or turned: a wrong_orientation issue
    stands only if the pose check, which scores every view together, finds a better orientation. In a section
    whose viewing direction is only a convention, it never stands. Screened issues stay visible in the notes."""
    trusted = pose.get("better_orientation_found", False) and entry.get("side_known", True)
    kept, dropped = [], []
    for issue in report["issues"]:
        (kept if issue["kind"] != "wrong_orientation" or trusted else dropped).append(issue)
    if not dropped:
        return report
    for issue in dropped:
        vlog.info("    screened out (not confirmed by the pose check): %s %s", issue.get("feature") or "-",
                  issue["description"][:120])
    notes = report["notes"] + [f"Screened out, not confirmed by the pose check: {i['description']}" for i in dropped]
    high = any(i["severity"] == "high" for i in kept)
    verdict = "fail" if high else ("minor" if kept else "pass")
    orientation = report["orientation"] if trusted else "matches"
    return {**report, "issues": kept, "notes": notes, "verdict": verdict, "orientation": orientation}


def weigh_by_evidence(report: dict, entry: dict, vlog: logging.Logger) -> dict:
    """An issue is only as strong as what backs it. In a measured view (projection metrics and callout checks)
    issues keep their severity. In a view that could not be measured (a pictorial view, judged by eye), they
    become advisory: low severity, so they never start a correction round on their own."""
    measured = "metrics" in entry
    doubt = entry.get("camera_doubt")
    issues = []
    for issue in report["issues"]:
        issue = {**issue, "basis": "measured view" if measured and not doubt else
                 "camera in doubt: " + doubt if doubt else "visual only: this view was not measured"}
        if issue["severity"] == "high" and (not measured or doubt):
            vlog.info("    advisory only (%s): %s %s", "camera in doubt" if doubt else "view not measured",
                      issue.get("feature") or "-", issue["description"][:120])
            issue["severity"] = "low"
        issues.append(issue)
    verdict = report["verdict"]
    if (not measured or doubt) and verdict == "fail":
        verdict = "minor"
    return {**report, "issues": issues, "verdict": verdict}


def issue_key(issue: dict) -> tuple:
    return issue.get("feature"), issue["kind"]


def escalate(issues: list[dict], sent_to: dict):
    """Issues that come back after a correction round. A build issue the builder did not fix is usually a plan
    problem it cannot solve (the tree or an assumption says otherwise): send it to the planner this time."""
    for issue in issues:
        key = issue_key(issue)
        if issue["severity"] != "high" or key not in sent_to:
            continue
        if sent_to[key] == "build" and issue["owner"] == "build":
            issue["owner"] = "plan"
            issue["description"] += " (Escalated: this survived a build correction, so the plan must change.)"
            log.info("    escalated to the planner: %s %s", issue.get("feature") or "-", issue["kind"])


def round_rank(r: dict):
    """Best round first: fewest high-severity issues, then the best measured fit, then the latest."""
    return r["high_issues"], -r["drawn_covered"], -r["round"]


def solve(drawing: Path, rebuild: Path | None = None, paths: bool = False) -> Path:
    """Plan, build, validate and correct a drawing in a new run folder, or rebuild an existing run's plan."""
    max_cost = float(load_agent("main").settings.get("max_cost_usd", 1.0))
    if rebuild:
        plan_result = (json.loads((rebuild / "plan.json").read_text(encoding="utf-8"))
                       if (rebuild / "plan.json").is_file() else {})
        plan_result["feature_tree"] = json.loads((rebuild / "feature_tree.json").read_text(encoding="utf-8"))
        drawing = Path(json.loads((rebuild / "trace.jsonl").open().readline())["drawing"])
        run = open_run(drawing, rebuild, max_cost)
        if (run.dir / "views" / f"{run.drawing.stem}_views.json").is_file():
            index_views(run)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS_DIR / f"{drawing.parent.parent.name}-{drawing.parent.name}-{stamp}"
        run_dir.mkdir(parents=True)
        run = open_run(drawing, run_dir, max_cost)
        run.trace("run_start", drawing=str(drawing), model=MODEL)
    log.info("run %s: %s with %s", run.dir.name, drawing, MODEL)
    if paths and not rebuild:  # the region-by-region analysis (paths.py), saved in this run's paths/ folder
        import paths as path_analysis
        try:
            path_analysis.analyse_sheet(sys.modules[__name__], run, run.dir / "paths")
        except Exception as error:
            log.warning("path analysis failed: %s", error)
    started = time.monotonic()
    rounds, report, corrections = [], {"status": "failed"}, None
    sent_to = {}  # issue key -> owner it was sent to in the last correction round
    try:
        if rebuild:
            run.trace("rebuild_start")
        else:
            plan_result = plan(run)
        for round_no in range(1, MAX_CORRECTIONS + 2):
            report = build(run, plan_result, corrections)
            if not (run.dir / "model.step").is_file():
                break
            check = validate(run, plan_result, round_no)
            issues = [i for r in check["reports"] for i in r["issues"]]
            escalate(issues, sent_to)
            high = [i for i in issues if i["severity"] == "high"]
            covered = [m["drawn_covered"] for m in check["measured"]]
            rounds.append({"round": round_no, "build": report["status"], "high_issues": len(high),
                           "validated_views": len(check["reports"]), "measured_views": len(covered),
                           "drawn_covered": round(sum(covered) / len(covered), 3) if covered else 0.0,
                           "verdicts": {r["view"]: r["verdict"] for r in check["reports"]}})
            mark_features(run, plan_result, check, high)
            if not check["reports"]:
                break
            if not high:
                log.info("validation: no high-severity issue in round %d", round_no)
                break
            if round_no > MAX_CORRECTIONS or run.over_budget:
                log.warning("validation: %d high-severity issue(s) left after %d correction round(s)",
                            len(high), round_no - 1)
                break
            plan_issues = [i for i in issues if i["owner"] == "plan"]
            sent_to = {issue_key(i): i["owner"] for i in issues if i["severity"] == "high"}
            log.info("correction round %d: %d plan issue(s), %d build issue(s)", round_no, len(plan_issues),
                     len(issues) - len(plan_issues))
            if plan_issues:
                plan_result = revise_plan(run, plan_result, plan_issues, check["pose"], round_no)
            corrections = {"round": round_no, "plan_revised": bool(plan_issues),
                           "issues": [i for i in issues if i["owner"] == "build"],
                           "previous_script": (run.dir / "model.py").read_text(encoding="utf-8")}
        keep_best_round(run, rounds)
    except Exception as error:
        run.trace("run_end", ok=False, error=f"{type(error).__name__}: {error}", cost_usd=run.cost_usd)
        log.error("run failed after %.1fs, cost $%.4f: %s", time.monotonic() - started, run.cost_usd, error)
        raise
    seconds = round(time.monotonic() - started, 1)
    final = min(rounds, key=round_rank) if rounds else None
    out = run.dir / "result.json"
    out.write_text(json.dumps({
        "drawing": str(drawing), "model": MODEL, "status": final["build"] if final else report["status"],
        "validated": bool(final and final["measured_views"]),  # judged by eye only does not count
        "measured_views": final["measured_views"] if final else 0,
        "drawn_covered": final["drawn_covered"] if final else None,
        "high_issues_left": final["high_issues"] if final else None,
        "kept_round": final["round"] if final else None, "rounds": rounds,
        "step": "model.step" if (run.dir / "model.step").is_file() else None,
        "script": "model.py" if (run.dir / "model.py").is_file() else None,
        "plan": "plan.json", "feature_tree": "feature_tree.json", "build_report": "build_report.json",
        "viewer": "viewer.html" if (run.dir / "viewer.html").is_file() else None,
        "build_attempts": run.build_attempts, "seconds": seconds,
        "cost_usd": run.cost_usd, "calls_without_cost": run.cost_unknown_calls,
    }, indent=2), encoding="utf-8")
    run.trace("run_end", ok=True, status=report["status"], rounds=rounds, cost_usd=run.cost_usd, seconds=seconds)
    unknown = f" (+{run.cost_unknown_calls} call(s) with no reported cost)" if run.cost_unknown_calls else ""
    log.info("finished in %.1fs, total cost $%.4f%s; result: %s", seconds, run.cost_usd, unknown, out)
    return out


def mark_features(run: Run, plan_result: dict, check: dict, high: list[dict]):
    """Feature states after validation: failing if a high-severity issue names it, passing otherwise."""
    if not check["reports"]:
        return
    failing = {i["feature"] for i in high if i.get("feature")}
    tree = plan_result["feature_tree"]
    for f in tree["features"]:
        if f["state"] == "built":
            f["state"] = "failing" if f["id"] in failing else "passing"
    (run.dir / "feature_tree.json").write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")


def keep_best_round(run: Run, rounds: list[dict]):
    """A correction can make things worse: keep the round with the fewest high-severity issues, and between
    equals the one whose projections cover the drawing best."""
    if not rounds:
        return
    best = min(rounds, key=round_rank)
    if best["round"] == rounds[-1]["round"] and (run.dir / "model.step").is_file():
        return  # the last validated model is the best one and is still in place
    source = run.dir / "validation" / f"round_{best['round']}"
    for name in ("model.step", "model.py"):
        shutil.copyfile(source / name, run.dir / name)
    log.warning("keeping round %d (%d high issue(s)): the later rounds were not better", best["round"],
                best["high_issues"])
    write_viewer(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("drawing", type=Path, nargs="?", help="PDF drawing")
    parser.add_argument("--build", type=Path, metavar="RUN_DIR",
                        help="rebuild the STEP of an existing run from its plan, without planning again")
    parser.add_argument("--view", action="store_true", help="open the 3D viewer in the browser at the end")
    parser.add_argument("-v", "--verbose", action="store_true", help="also log what the models say")
    parser.add_argument("--paths", action="store_true",
                        help="first analyse every region of every view with the path agent (saved in paths/)")
    args = parser.parse_args()
    if not args.drawing and not args.build:
        parser.error("give a drawing, or --build RUN_DIR")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        out = solve(args.drawing, args.build, args.paths)
    except (ValueError, OSError, RuntimeError) as error:
        sys.exit(str(error))
    viewer = out.with_name("viewer.html")
    if args.view and viewer.is_file():
        webbrowser.open(viewer.resolve().as_uri())


if __name__ == "__main__":
    main()
