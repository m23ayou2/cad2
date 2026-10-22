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
import validate as view_checks  # noqa: E402
import visualization  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402

MODEL = os.getenv("D2C_MODEL") or "anthropic/claude-haiku-5.5"  # the web interface sets D2C_MODEL per job
# Scanned drawings (no vector geometry, no text layer) take another path: VISION_MODEL reads the sheet (views,
# parts outlined directly, parts matched across views; src/part_views.py); then the planner and the validation
# agents run on SCAN_REASON_MODEL, and the builder on the cheap SCAN_BUILD_MODEL.
VISION_MODEL = os.getenv("D2C_VISION_MODEL") or "anthropic/claude-haiku-5.5"
SCAN_REASON_MODEL = os.getenv("D2C_SCAN_REASON_MODEL") or "anthropic/claude-haiku-5.5"
SCAN_BUILD_MODEL = os.getenv("D2C_SCAN_BUILD_MODEL") or "anthropic/claude-haiku-5.5"
# A scan's budget for planning, building and validation (USD), on top of what the vision model costs: the reasoning
# model costs far more per call than the vector path's, whose budget is agents/main.md's max_cost_usd.
SCAN_MAX_COST_USD = float(os.getenv("D2C_SCAN_MAX_COST") or 3.0)
# How a scan's parts are found: "local" = image processing segments and matches them and one MODEL call reviews
# the groups (src/local_llm_match.py); "llm" = VISION_MODEL outlines and matches them (src/part_views.py).
SCAN_PARTS = os.getenv("D2C_SCAN_PARTS") or "local"
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
    scanned: bool = False  # no vector geometry: the vision model reads the sheet (plan_from_parts)

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
        run.views = {"sheet": View("sheet", "WHOLE SHEET", None, png, None)}
        run.trace("split", ok=False, error=str(error))
        return ToolOutput({
            "views": [{"id": "sheet", "label": "WHOLE SHEET", "suggested_subagent": None}],
            "note": ("The sheet could not be split automatically (probably a scan). Call call_subagent once per "
                     "view you saw, always with view_id 'sheet', and say in `context` which view to analyse "
                     "and where it is on the sheet."),
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


TOOLS = {f.__name__: f for f in (view_drawing, read_drawing_text, split_views, call_subagent,
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


def image_part(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


# --------------------------------------------------------------------------- agent loop
def agent_model(agent: Agent, run: Run) -> str:
    """The model an agent runs on: MODEL for a vector drawing; on a scan, SCAN_BUILD_MODEL for the builder and
    SCAN_REASON_MODEL for the planner and the validation agents."""
    if not run.scanned:
        return MODEL
    return SCAN_BUILD_MODEL if agent.name == "builder" else SCAN_REASON_MODEL


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
        reply = run.client.chat.completions.create(model=agent_model(agent, run), messages=messages,
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
            try:
                result = parse_json(message.content or "")
                if agent.output_model:
                    context = {"views": list(run.views)}
                    if agent.output_model is schema.ArbiterReport:
                        context["texts"] = drawing_texts(run)
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
    if run.scanned:  # a sheet-metal blank: outline, bend lines and hole centres from the scan's own flat pattern
        import flat_blank
        notes = flat_blank.snap_blank(result["feature_tree"], run.dir / "parts")
        for note in notes:
            log.info("  %s", note)
        if notes:
            run.trace("blank_snapped", notes=notes)
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


def is_scanned(drawing: Path) -> bool:
    """A scan: no page has vector drawing commands or text, only images."""
    with pymupdf.open(drawing) as doc:
        return not any(page.get_drawings() or page.get_text("words") for page in doc)


PARTS_BRIEF = """# The views are already read

This drawing is a scan. Its views were read by a vision model ({vision}): each view was found on the sheet,
split into coloured parts (P1, P2, ... numbered across all views), and the parts were matched across views into
groups (G1, G2, ...). One group = one physical part of the object (a panel, a tab, a bend, a hole ...) with
every label it has in every view, and the dimensions printed on the drawing that belong to it.

The groups replace split_views and the view subagents, which are not available. Do step 1 (`view_drawing`,
`read_drawing_text`; the drawing has no text layer, so read every value from the image), then plan the part:
- Every group must end up in the feature tree (one or more features), or in `uncertainties` with the reason.
  Use the groups to decide the operations and their order: for a sheet-metal part, which flat panel becomes
  which formed face, which panel stays flat, where each hole is, which way each bend goes.
- **Sizes come from the dimensions printed on the drawing.** Zoom with `view_drawing` (regions as fractions of
  the sheet; each view's `sheet_box_fraction` below) to read them. The outlines below are in PDF points of each
  cropped view (1 pt = 1/72 inch, the view's top-left origin, y down) and are approximate: use them for
  positions, proportions and which edge is which, and to turn distances into mm only through a printed
  dimension of the same view.
- The matching is automatic and can be wrong. When the drawing contradicts a group, follow the drawing and
  write it in `uncertainties`.
- `frame.views` lists the views by the ids below (v1, v2, ...).

## Sheet metal (the sheet shows a flat pattern)

Build the part as a `base_flange` followed by one `bend` per bend line (see the sheet-metal rules in step 4);
the builder folds it with `fold_sheet`. Use the groups to fill it in:
- **The blank** (`base_flange.profile`, in mm): origin at the lower-left corner of the blank, x right, y up,
  as drawn. Take every coordinate from the ordinate dimensions of the dimensioned flat pattern, using the flat
  view's part outlines to see which ordinate locates which edge, and the radius callouts (R25, R3.5, R2 ...)
  on the copy with the callouts for the arcs and rounded corners. Follow the outline all the way round.
- **The panels**: each flat-pattern group that is a panel or a tab is one panel of the blank; its group says
  which face of the formed part it becomes in the isometric views.
- **`fixed`**: a point `[x, y]` inside the panel whose group is the face lying flat in the isometrics (the
  base the part rests on).
- **Each bend line**: the straight boundary between two matched flat panels (a bend group's outline in the
  flat view straddles it; the line is its middle), running right across the strip that joins the two panels.
  Its `direction` and `radius` come from the callout next to it (`BEND UP R1` -> up, radius 1; `BEND DOWN` ->
  down; no radius given -> the notes' radius). Check each one against the isometric: the panel beyond the line,
  seen from the fixed panel, must rise (up) or drop (down) as drawn there.
- **Holes**: a `hole` at each hole's flat position on the blank `[x, y, 0]`, diameter from its callout; the
  hole's group says which panel it is in, and its ordinates on the dimensioned flat pattern give the centre.

## Views
```json
{views}
```

## Matched groups
```json
{groups}
```

## Parts per view (outlines in the view's points)
```json
{parts}
```

The images follow: all views coloured by group (same colour = same group in every view), then each view alone."""


def plan_from_parts(run: Run) -> dict:
    """Scanned drawing: the vision model reads it (src/part_views.py, saved in parts/), then the main agent plans
    the feature tree from the matched groups, with only the sheet as a tool."""
    import src.llm as llm
    import src.part_views as part_views
    llm.VISION_MODEL, llm.MODEL, part_views.VISION_MODEL = VISION_MODEL, MODEL, VISION_MODEL
    part_views.PARTS_METHOD = "direct"
    parts_dir = run.dir / "parts"
    t0 = time.monotonic()
    if SCAN_PARTS == "local":
        # the model only finds the views; their parts are segmented and matched locally (src/local_parts.py),
        # and one call reviews the groups (src/local_llm_match.py)
        from src.local_llm_match import scan_parts
        log.info("scanned drawing: views found by %s, parts segmented and matched locally, groups reviewed by %s",
                 VISION_MODEL, MODEL)
        (parts_dir / "views").mkdir(parents=True, exist_ok=True)
        with pymupdf.open(run.drawing) as doc:
            found = part_views.find_views(doc, parts_dir, parts_dir / "trace.jsonl")
        (parts_dir / "views_found.json").write_text(json.dumps(found, indent=2))
        scan_parts(run.drawing, parts_dir, parts_dir / "views_found.json", model=MODEL)
    else:
        log.info("scanned drawing: reading the views with %s (views, parts, matching)", VISION_MODEL)
        part_views.run(run.drawing, run_dir=parts_dir)
    calls = [json.loads(line) for line in (parts_dir / "trace.jsonl").read_text().splitlines()]
    cost = sum(c.get("cost") or 0 for c in calls)
    run.cost_usd += cost
    run.max_cost_usd += cost  # the planner's budget is its own, not shared with the vision model
    run.cost_unknown_calls += sum(1 for c in calls if c.get("cost") is None)
    views = json.loads((parts_dir / "views.json").read_text())
    groups = json.loads((parts_dir / "matches.json").read_text())["groups"]
    run.trace("parts_pipeline", model=VISION_MODEL, calls=len(calls), cost_usd=cost,
              seconds=round(time.monotonic() - t0, 1), views=[v["name"] for v in views],
              parts=sum(len(v["parts"]) for v in views), groups=[g.get("name") for g in groups])
    log.info("vision model: %d view(s), %d group(s), cost $%.4f", len(views), len(groups), cost)

    with pymupdf.open(run.drawing) as doc:
        sizes = {i + 1: (p.rect.width, p.rect.height) for i, p in enumerate(doc)}
    listing, parts = [], {}
    for v in views:
        w, h = sizes[v["sheet"]]
        x0, y0, x1, y1 = v["rect_pt"]
        # register the view, so the plan can cite it as evidence and view_drawing can show it
        run.views[v["id"]] = View(v["id"], v["name"] or v["id"], None, parts_dir / "views" / f"{v['id']}_matched.png",
                                  None, sheet_box=[x0 / w, y0 / h, x1 / w, y1 / h])
        listing.append({"id": v["id"], "name": v["name"], "kind": v["kind"], "sheet": v["sheet"],
                        "sheet_box_fraction": [round(x0 / w, 3), round(y0 / h, 3), round(x1 / w, 3), round(y1 / h, 3)],
                        "size_pt": [round(x1 - x0, 1), round(y1 - y0, 1)]})
        parts[v["id"]] = [{"label": p["id"], "name": p.get("name"), "feature": p.get("feature"),
                           "dimensions": p.get("dimensions", []), "hidden": p.get("hidden", False),
                           "outline_pt": [[round(x, 1), round(y, 1)] for x, y in p["polygon_pt"]],
                           "holes_pt": [[[round(x, 1), round(y, 1)] for x, y in hole] for hole in p.get("holes_pt", [])],
                           "notes": p.get("notes", "")} for p in v["parts"]]
    text = PARTS_BRIEF.format(vision=VISION_MODEL, views=json.dumps(listing, indent=1, ensure_ascii=False),
                              groups=json.dumps(groups, indent=1, ensure_ascii=False),
                              parts=json.dumps(parts, ensure_ascii=False))
    content = [{"type": "text", "text": f"Drawing file: {run.drawing.name}.\n\n{text}"},
               {"type": "text", "text": "All views, coloured by group:"},
               image_part((parts_dir / "matched_all.png").read_bytes())]
    for v in views:
        content += [{"type": "text", "text": f"View {v['id']} ({v['name']}), coloured by group:"},
                    image_part((parts_dir / "views" / f"{v['id']}_matched.png").read_bytes())]
    agent = load_agent("main")
    agent.settings = {**agent.settings, "tools": ["view_drawing", "read_drawing_text"]}
    result = run_agent(agent, Ctx(run), content, label="main[scan]")
    return save_plan(run, {**result, "summaries": [], "cross_check": None, "parts": "parts/matches.json"})


def plan(run: Run) -> dict:
    """Main agent + view subagents -> feature tree (feature_tree.json, plan.json); a scan goes through
    plan_from_parts instead."""
    if run.scanned:
        return plan_from_parts(run)
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
    if not step.is_file():  # one more try, told why the last one failed
        log.info("build: no valid solid; asking the builder once more")
        failed = {"notes": report.get("notes", []),
                  "features": [{k: f.get(k) for k in ("id", "state", "note")} for f in report.get("features", [])]}
        report = run_agent(load_agent("builder"), Ctx(run), [{"type": "text", "text": text + (
            "\n\n# Your previous attempt failed\n\nIt produced no valid single solid. Its report:\n```json\n"
            + json.dumps(failed, indent=1, ensure_ascii=False) + "\n```\nFind the cause and fix it; do not give up "
            "while a different construction could still give one connected solid.")}], label="builder[retry]")
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
    scans = scan_entries(run, out_dir) if run.scanned else []
    if not scans and not any(v.pdf for v in run.views.values()):
        vlog.warning("no views to compare with; model not validated")
        return result
    t0 = time.monotonic()
    # a scan is compared on its pixels (src/raster_check.py), a vector drawing on its geometry (src/validate.py);
    # both write validation.json in the same format
    command = ([sys.executable, str(ROOT / "src" / "raster_check.py"), str(run.drawing), str(run.dir / "model.step"),
                str(out_dir / "scans.json"), str(out_dir)] if scans else
               [sys.executable, str(ROOT / "src" / "validate.py"), str(run.dir / "views"),
                str(run.dir / "model.step"), str(out_dir), str(run.drawing)])
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

    by_pdf = {v.pdf.stem: v for v in run.views.values() if v.pdf}
    for entry in data["views"] if scans else []:  # a scanned view: its agent is the kind it fitted best
        s = next(x for x in scans if x["id"] == entry["view_id"])
        kind = entry.get("class") if entry.get("class") in VALIDATORS else None
        by_pdf[s["id"]] = View(s["id"], s["label"], None, out_dir / f"{s['id']}.drawing.png", kind)
    tree = plan_result["feature_tree"]
    compact = [{k: f.get(k) for k in ("id", "name", "op", "params", "on", "position", "direction", "pattern")}
               for f in tree["features"]]
    for entry in data["views"]:
        view = by_pdf.get(entry["view_id"])
        kind = view.suggested_subagent if view else None
        if not kind or "images" not in entry:
            if view and kind:
                vlog.info("%s not validated: %s", view.id, entry.get("unresolved"))
            continue
        frame = (f"In this view, horizontal = {entry['horizontal']}, vertical = {entry['vertical']}, looking along "
                 f"{entry['looking_along']}" + (f", cut plane normal {entry['cut_plane_normal']}"
                                                if entry.get("cut_plane_normal") else "") + ".")
        measured = {k: entry.get(k) for k in ("metrics", "missing_regions", "extra_regions", "callouts", "note",
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
        filled = str(entry["images"].get("drawing", "")).endswith("_filled.png")
        captions = ((("drawing", "The scanned view, every region it encloses filled in its own colour:"),
                     ("model", "The model, projected from the camera that fits the view best, drawn in the view's "
                               "own pixel frame, every face filled in its own colour:"),
                     ("overlay", "Silhouettes over the scan: green = both, orange = drawn but not in the model, "
                                 "purple = in the model but not drawn; the model's edges in purple lines:"),
                     ("matches", "Match sheet: scan (left) and model (right), one line per hole matched between "
                                 "them; red = a suspect match:"))
                    if filled else
                    (("drawing", "The drawing view:"), ("model", "The model projection, same direction:"),
                     ("overlay", "Overlay (drawing black, model red; orange = drawn but missing, "
                                 "purple = in model but not drawn):")))
        if scans:
            content[0]["text"] += ("\n\nThis drawing is a scan: the camera and scale were fitted to the scan, not "
                                   "measured, and no dimension callouts are checked. Judge the shape: what is drawn "
                                   "but missing in the model, what the model has that is not drawn.")
        for key, caption in captions:
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


VALIDATORS = ("front", "top", "side", "section", "detail", "isometric")


def scan_entries(run: Run, out_dir: Path) -> list[dict]:
    """The views the vision model found on a scan (parts/views.json), as src/raster_check.py entries in
    scans.json: isometrics as isometric views; orthographic views and sections as "auto" (a scan gives no reliable
    direction, so every standard direction is tried and the best fit kept); flat patterns are left out, since the
    model is the formed part."""
    views_json = run.dir / "parts" / "views.json"
    if not views_json.is_file():
        return []
    with pymupdf.open(run.drawing) as doc:
        sizes = {i + 1: (p.rect.width, p.rect.height) for i, p in enumerate(doc)}
    entries = []
    for v in json.loads(views_json.read_text()):
        if v.get("kind") == "flat_pattern":
            continue
        w, h = sizes[v["sheet"]]
        x0, y0, x1, y1 = v["rect_pt"]
        entries.append({"id": v["id"], "label": v.get("name") or v["id"],
                        "kind": "isometric" if v.get("kind") == "isometric" else "auto", "direction": "none",
                        "box": [x0 / w, y0 / h, x1 / w, y1 / h], "folder": str(out_dir / "scans" / v["id"])})
        Path(entries[-1]["folder"]).mkdir(parents=True, exist_ok=True)
    (out_dir / "scans.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return entries


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
    issues = []
    for issue in report["issues"]:
        issue = {**issue, "basis": "measured view" if measured else "visual only: this view was not measured"}
        if not measured and issue["severity"] == "high":
            vlog.info("    advisory only (view not measured): %s %s", issue.get("feature") or "-",
                      issue["description"][:120])
            issue["severity"] = "low"
        issues.append(issue)
    verdict = report["verdict"]
    if not measured and verdict == "fail":
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


def solve(drawing: Path, rebuild: Path | None = None) -> Path:
    """Plan, build, validate and correct a drawing in a new run folder, or rebuild an existing run's plan."""
    max_cost = float(load_agent("main").settings.get("max_cost_usd", 1.0))
    if rebuild:
        drawing = Path(json.loads((rebuild / "trace.jsonl").open().readline())["drawing"])
    scanned = is_scanned(drawing)
    if scanned:
        max_cost = SCAN_MAX_COST_USD
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
        run.trace("run_start", drawing=str(drawing), scanned=scanned,
                  models={"vision": VISION_MODEL, "reasoning": SCAN_REASON_MODEL, "builder": SCAN_BUILD_MODEL}
                  if scanned else {"all": MODEL})
    run.scanned = scanned
    if scanned:
        log.info("run %s: %s is a scan: views read by %s, planning and validation on %s, building on %s",
                 run.dir.name, drawing, VISION_MODEL, SCAN_REASON_MODEL, SCAN_BUILD_MODEL)
    else:
        log.info("run %s: %s with %s", run.dir.name, drawing, MODEL)
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
        "drawing": str(drawing), "model": SCAN_REASON_MODEL if run.scanned else MODEL,
        "models": ({"vision": VISION_MODEL, "reasoning": SCAN_REASON_MODEL, "builder": SCAN_BUILD_MODEL}
                   if run.scanned else {"all": MODEL}), "status": final["build"] if final else report["status"],
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
    args = parser.parse_args()
    if not args.drawing and not args.build:
        parser.error("give a drawing, or --build RUN_DIR")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        out = solve(args.drawing, args.build)
    except (ValueError, OSError, RuntimeError) as error:
        sys.exit(str(error))
    viewer = out.with_name("viewer.html")
    if args.view and viewer.is_file():
        webbrowser.open(viewer.resolve().as_uri())


if __name__ == "__main__":
    main()
