"""Local segmentation (local_parts) plus one cheap LLM call that matches the coloured regions across views.

The LLM no longer traces outlines: it sees every view with the local parts drawn in colour and numbered (P1, P2,
...), the whole sheet, and a list of facts per part (kind, holes and their sizes, neighbouring panels), and only
says which numbered parts are the same physical part and which are not parts at all (dimension strips, letters).

  mode "blind":  the model groups the parts by itself;
  mode "review": it is also shown the local matcher's groups and corrects them.

Writes matches_llm_<mode>.json and matched_llm_<mode>_all.png next to local_parts' outputs; the call is traced
(with its cost) in trace.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from src import local_parts
from src.llm import ask_json, image_part

MODEL = "anthropic/claude-haiku-5.5"
MAX_PX = 1400

PROMPT = """These images are from one engineering drawing (a scan). The LAST image is the whole sheet, for context.
Each other image is one view of the part, named {names}. A local image-processing step has already split every
view into regions and drawn them in colour, each with its label (P1, P2, ...). Labels are unique across views.

Facts per region (from the image processing; sizes in PDF points, comparable within and across views):
{facts}

Your task: say which labels show the SAME physical part of the object in different views. A flat pattern shows the
sheet before bending; an isometric shows the formed part; an orthographic view shows it from one side. A group
may have one member per view at most, and a part need not be visible in every view. Use the drawing itself (bend
callouts, dimensions, which panel is joined to which through a bend, which holes are larger) to decide. Holes
belong in their own groups, matched hole to hole; bends to bends. Regions that are not part of the object at all
(strips between dimension lines, letters, paper) go in "not_parts".
{proposal}
Reply with JSON only:
{{"groups": [{{"name": "upright tab", "members": [{{"view": "v1", "parts": ["P3"]}}, {{"view": "v3", "parts": ["P27"]}}],
   "reason": "short"}}],
 "not_parts": ["P40"]}}"""

REVIEW = """
A geometric matcher (hole counts, neighbours through bends, hole size ratios) proposed these groups. Check each one
against the images; keep what is right, correct what is wrong, and add what is missing:
{groups}
"""


def facts(views: list[dict]) -> str:
    lines = []
    for v in views:
        lines.append(f"{v['id']} ({v['kind']}{', ' + v['name'] if v['name'] else ''}):")
        for p in v["parts"]:
            s = f"  {p['id']}: {p['feature']}"
            if p["feature"] == "panel":
                s += f", {len(p['holes_pt'])} holes"
                if p.get("neighbours"):
                    s += f", joined to {', '.join(p['neighbours'])}"
            if p["feature"] in ("hole", "panel") and p.get("notes"):
                s += f" ({p['notes']})"
            lines.append(s)
    return "\n".join(lines)


def label_image(run_dir: Path, view: dict, zoom: float) -> bytes:
    """The view with every part coloured and labelled (holes on top), at most MAX_PX on its long side."""
    gray = cv2.imread(str(run_dir / "views" / f"{view['id']}.png"), 0)
    parts = sorted(view["parts"], key=lambda p: p["feature"] == "hole")
    img = local_parts.draw(gray, [(p["id"], p["polygon_pt"], p["holes_pt"] if p["feature"] != "hole" else [],
                                   local_parts.PALETTE[i % len(local_parts.PALETTE)]) for i, p in enumerate(parts)], zoom)
    s = min(1.0, MAX_PX / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(run_dir / "views" / f"{view['id']}_labels.png"), img)
    return cv2.imencode(".png", img)[1].tobytes()


def llm_match(run_dir: Path, out: dict, zoom: float, mode: str = "blind", model: str = MODEL) -> dict:
    views = out["views"]
    known = {p["id"] for v in views for p in v["parts"]}
    view_of = {p["id"]: v["id"] for v in views for p in v["parts"]}
    proposal = ""
    if mode == "review":
        proposal = REVIEW.format(groups="\n".join(
            f"  {g['id']} {g['name']}: " + "; ".join(f"{m['view']} {','.join(m['parts'])}" for m in g["members"])
            for g in out["matches"]["groups"]))
    prompt = PROMPT.format(names=", ".join(v["id"] for v in views), facts=facts(views), proposal=proposal)
    content = [{"type": "text", "text": prompt}]
    for v in views:
        content += [{"type": "text", "text": f"View {v['id']}:"}, image_part(label_image(run_dir, v, zoom))]
    sheet = cv2.imread(str(run_dir / "sheet.png"))
    content += [{"type": "text", "text": "Whole sheet:"}, image_part(cv2.imencode(".png", sheet)[1].tobytes())]
    data, _ = ask_json(content, f"llm_match_{mode}", run_dir / "trace.jsonl", model=model)

    groups, dropped = [], []
    for g in data.get("groups", []):
        members = {}
        for m in g.get("members", []):
            for label in m.get("parts", []):
                if label in known:          # trust the label's own view, not the one the model wrote
                    members.setdefault(view_of[label], []).append(label)
                else:
                    dropped.append(label)
        if members:
            groups.append({"id": f"G{len(groups) + 1}", "name": g.get("name", ""), "reason": g.get("reason", ""),
                           "members": [{"view": vid, "parts": ps} for vid, ps in members.items()]})
    matches = {"groups": groups, "not_parts": [p for p in data.get("not_parts", []) if p in known],
               "unknown_labels": dropped, "model": model, "mode": mode}
    (run_dir / f"matches_llm_{mode}.json").write_text(json.dumps(matches, indent=2))
    draw_matched(run_dir, views, groups, zoom, f"_llm_{mode}")
    return matches


def draw_matched(run_dir: Path, views: list[dict], groups: list[dict], zoom: float, suffix: str = "") -> None:
    """Every view coloured by group (views/vK_matched<suffix>.png) and all of them side by side
    (matched<suffix>_all.png)."""
    owner = {}
    for i, g in enumerate(groups):
        for m in g["members"]:
            for label in m["parts"]:
                owner[label] = (g["id"], local_parts.PALETTE[i % len(local_parts.PALETTE)])
    images = []
    for v in views:
        gray = cv2.imread(str(run_dir / "views" / f"{v['id']}.png"), 0)
        img = local_parts.draw(gray, [(owner[p["id"]][0], p["polygon_pt"], p["holes_pt"], owner[p["id"]][1])
                                      for p in sorted(v["parts"], key=lambda p: p["feature"] == "hole")
                                      if p["id"] in owner], zoom)
        cv2.imwrite(str(run_dir / "views" / f"{v['id']}_matched{suffix}.png"), img)
        images.append(img)
    cv2.imwrite(str(run_dir / f"matched{suffix}_all.png"), local_parts.side_by_side(images, [v["id"] for v in views]))


def scan_parts(drawing: Path, run_dir: Path, views_json: Path, model: str = MODEL) -> dict:
    """The scan path's parts step: local segmentation and matching, the corrector agent (one call per view), then
    one LLM call reviewing the groups.
    Writes the same files as part_views (views.json, matches.json, views/vK_matched.png, matched_all.png); the
    local matcher's own groups stay in matches_local.json. Returns {"views", "groups", "seconds"}."""
    import time
    from src import corrector
    t0 = time.monotonic()
    out = local_parts.run(drawing, run_dir, views_json)
    (run_dir / "matches_local.json").write_text((run_dir / "matches.json").read_text())
    # the corrector agent fixes the segmentation's artifacts (views.json and matches.json are rewritten)
    corrector.correct(run_dir, model)
    out["views"] = json.loads((run_dir / "views.json").read_text())
    out["matches"] = local = json.loads((run_dir / "matches.json").read_text())
    try:
        reviewed = llm_match(run_dir, out, out["zoom"], "review", model)
        groups = reviewed["groups"] or local["groups"]
    except ValueError as exc:            # no readable answer: keep the local groups
        print(f"[llm_match_review] failed ({exc}); keeping the local groups", flush=True)
        groups = local["groups"]
    (run_dir / "matches.json").write_text(json.dumps({**local, "groups": groups}, indent=2, default=str))
    draw_matched(run_dir, out["views"], groups, out["zoom"])
    return {"views": out["views"], "groups": groups, "seconds": round(time.monotonic() - t0, 1)}
