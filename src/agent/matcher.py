"""Matching agent: decide which coloured part labels in different views are the same physical part.

The agent sees every view with its parts drawn and labelled (P1, P2, ...), the parts' JSON, and the
whole sheet. It returns groups of labels. Its answer is checked in code (every label used exactly once,
no unknown labels, no view repeated inside a group); if the check fails, the errors are sent back and
the agent corrects itself, up to MAX_ROUNDS times.
"""
import json
from pathlib import Path

from src.llm import ask_json, image_part

MAX_ROUNDS = 3

PROMPT = """You are matching parts of one mechanical object across the views of its engineering drawing.
Each view below has its parts drawn as coloured polygons labelled P1, P2, ... (labels are unique
across all views; colours mean nothing yet). The same physical part of the object appears in several
views, with a different label in each.

Group the labels that are the same physical part. Rules:
- A group lists, per view, the label(s) of that part in that view. One part can need several labels in
  one view (split by a section, a bend or a break), and it can be missing from some views (hidden,
  edge-on, or too small to see). A part seen in only one view is a group with one member.
- Every label must appear in exactly one group.
- Use projection alignment between orthographic views (same width/height, same position along the
  shared axis), section and view labels, matching dimensions, and the isometric for overall shape.
- From all its labels together, collect the part's dimensions as printed on the drawing.

Parts per view (JSON):
{parts}

Reply with JSON only:
{{"groups": [{{"id": "G1", "name": "flange", "members": [{{"view": "v1", "parts": ["P1"]}}, {{"view": "v2", "parts": ["P5", "P6"]}}],
  "dimensions": ["Ø36", "4 thick"], "confidence": 0.0-1.0, "reason": "why these labels are the same part"}}]}}"""


def summarize(views: list[dict]) -> list[dict]:
    out = []
    for view in views:
        parts = []
        for p in view["parts"]:
            xs, ys = [q[0] for q in p["polygon_pt"]], [q[1] for q in p["polygon_pt"]]
            parts.append({"label": p["id"], "name": p.get("name"), "feature": p.get("feature"),
                          "dimensions": p.get("dimensions", []), "hidden": p.get("hidden", False),
                          "bbox_in_view_pt": [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))],
                          "notes": p.get("notes", "")})
        out.append({"view": view["id"], "name": view["name"], "kind": view["kind"], "sheet": view["sheet"],
                    "view_box_on_sheet_pt": view["rect_pt"], "parts": parts})
    return out


def check(groups: list[dict], views: list[dict]) -> list[str]:
    labels = {p["id"]: v["id"] for v in views for p in v["parts"]}
    seen: dict[str, str] = {}
    errors = []
    for group in groups:
        gid = group.get("id", "?")
        group_views = [m.get("view") for m in group.get("members", [])]
        if len(group_views) != len(set(group_views)):
            errors.append(f"{gid}: lists the same view twice; put all its labels for a view in one member")
        for member in group.get("members", []):
            for label in member.get("parts", []):
                if label not in labels:
                    errors.append(f"{gid}: unknown label {label}")
                elif labels[label] != member.get("view"):
                    errors.append(f"{gid}: {label} belongs to view {labels[label]}, not {member.get('view')}")
                elif label in seen:
                    errors.append(f"{label} is in both {seen[label]} and {gid}")
                else:
                    seen[label] = gid
    missing = sorted(set(labels) - set(seen), key=lambda s: int(s[1:]))
    if missing:
        errors.append("labels in no group: " + ", ".join(missing))
    return errors


def match_parts(views: list[dict], run: Path, trace: Path) -> dict:
    content = [{"type": "text", "text": PROMPT.format(parts=json.dumps(summarize(views), indent=1))}]
    for view in views:
        content += [{"type": "text", "text": f"View {view['id']}: {view['name']} ({view['kind']})"},
                    image_part((run / "views" / f"{view['id']}_colored.png").read_bytes())]
    for sheet in sorted({v["sheet"] for v in views}):
        content += [{"type": "text", "text": f"Whole sheet {sheet}, for context:"},
                    image_part((run / f"sheet_{sheet}.png").read_bytes())]

    history = None
    for round_ in range(1, MAX_ROUNDS + 1):
        data, history = ask_json(content, f"match_round{round_}", trace, history)
        groups = data.get("groups", [])
        errors = check(groups, views)
        if not errors:
            return {"groups": groups, "rounds": round_, "errors": []}
        print(f"[match_round{round_}] {len(errors)} problems, asking for a fix", flush=True)
        content = [{"type": "text", "text": "Your grouping has these problems:\n- " + "\n- ".join(errors)
                    + "\nReply with the complete corrected JSON."}]

    # Out of rounds: keep valid assignments, give each leftover label its own group.
    labels = {p["id"]: v["id"] for v in views for p in v["parts"]}
    used, clean = set(), []
    for group in groups:
        members = []
        for m in group.get("members", []):
            parts = [l for l in m.get("parts", []) if labels.get(l) == m.get("view") and l not in used]
            used.update(parts)
            if parts:
                members.append({"view": m["view"], "parts": parts})
        if members:
            clean.append({**group, "members": members})
    for label in sorted(set(labels) - used, key=lambda s: int(s[1:])):
        clean.append({"id": f"G{len(clean) + 1}", "name": "unmatched", "members": [
            {"view": labels[label], "parts": [label]}], "dimensions": [], "confidence": 0.0,
            "reason": "left over after the agent ran out of correction rounds"})
    return {"groups": clean, "rounds": MAX_ROUNDS, "errors": errors}
