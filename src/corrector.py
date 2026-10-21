"""The corrector agent (agents/corrector.md): finds the artifacts in the local segmentation of each view and the
harness applies its corrections, before the parts are matched and planned from.

Per view, one call. The agent sees the view with the parts coloured and labelled, the plain view, and per-part
facts measured on the pixels (area, width, the area and place of the notches in its outline). It answers with
operations; this module applies them exactly:

  fill_notches  close the part's outline with a small disc: notches and gaps narrower than the disc are filled
                (leader lines and arrowheads cut them), without growing into another part;
  merge         one region from several (pieces of one face split by a centre line or a leader);
  remove        drop the part (paper between dimension lines, a letter's counter);
  kind          change its kind (panel, hole, bend, other).

The corrected parts replace views.json (the raw one is kept as views_raw.json); matches.json follows (removed
labels dropped, merged labels replaced by the one kept).

    uv run python src/corrector.py runs/<run>/parts        # correct an existing segmentation in place
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import local_parts  # noqa: E402
from src.llm import ask_json, image_part  # noqa: E402

MODEL = "anthropic/claude-haiku-5.5"
NOTCH_PX = 7          # radius of the disc that closes notches (px of the native scan: about 3.4 pt)
KINDS = {"panel", "hole", "bend", "other"}


def agent_text() -> str:
    text = (ROOT / "agents" / "corrector.md").read_text(encoding="utf-8")
    return re.match(r"^---\n.*?\n---\n(.*)$", text, re.S).group(1).strip()


def masks(view: dict, shape: tuple, zoom: float) -> dict[str, np.ndarray]:
    """Each part's region as a mask in the view's pixels (holes cut out of the parts that hold them)."""
    out = {}
    for p in view["parts"]:
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [(np.array(p["polygon_pt"]) * zoom).round().astype(np.int32)], 1)
        for h in p.get("holes_pt", []):
            cv2.fillPoly(m, [(np.array(h) * zoom).round().astype(np.int32)], 0)
        out[p["id"]] = m > 0
    return out


def disc(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def facts(view: dict, gray: np.ndarray, zoom: float, ms: dict) -> list[dict]:
    strokes = cv2.morphologyEx((gray < 200).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    paper = local_parts._paper(strokes)
    grown = {k: cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0 for k, m in ms.items()}
    out = []
    for p in view["parts"]:
        m = ms[p["id"]]
        if not m.any():
            continue
        filled = m.copy()
        for h in p.get("holes_pt", []):
            cv2.fillPoly(filled.view(np.uint8), [(np.array(h) * zoom).round().astype(np.int32)], 1)
        dent = (cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, disc(NOTCH_PX)) > 0) & ~filled
        n, lab, st, cen = cv2.connectedComponentsWithStats(dent.astype(np.uint8), 8)
        spots = sorted(range(1, n), key=lambda i: -st[i, 4])[:4]
        width = 2 * float(cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3).max())
        out.append({"part": p["id"], "kind": p["feature"], "area_pt2": round(float(m.sum()) / zoom ** 2, 1),
                    "width_pt": round(width / zoom, 1), "dent_pt2": round(float(dent.sum()) / zoom ** 2, 1),
                    "dents_at": [[round(cen[i][0] / zoom, 1), round(cen[i][1] / zoom, 1)] for i in spots
                                 if st[i, 4] >= 4],
                    "touches_paper": bool((grown[p["id"]] & paper).any()),
                    "holes": len(p.get("holes_pt", [])),
                    "touches": sorted(k for k in ms if k != p["id"] and (grown[p["id"]] & ms[k]).any())})
    return out


def apply(view: dict, corrections: list[dict], ms: dict, zoom: float) -> tuple[list[dict], dict, list[str]]:
    """The view's parts after the corrections; the label each label became (None = removed); what was done."""
    parts = {p["id"]: dict(p) for p in view["parts"]}
    became, done = {}, []
    for c in corrections:
        op = c.get("op")
        if op == "remove" and c.get("part") in parts:
            parts.pop(c["part"])
            became[c["part"]] = None
            done.append(f"removed {c['part']}")
        elif op == "kind" and c.get("part") in parts and c.get("kind") in KINDS:
            parts[c["part"]]["feature"] = c["kind"]
            done.append(f"{c['part']} -> {c['kind']}")
        elif op == "merge":
            ids = [i for i in c.get("parts", []) if i in parts]
            if len(ids) < 2:
                continue
            m = np.zeros_like(next(iter(ms.values())))
            for i in ids:
                m |= ms[i]
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, disc(3)) > 0
            keep = parts[ids[0]]
            keep["polygon_pt"] = local_parts.mask_polygon(m, zoom)
            keep["holes_pt"] = [h for i in ids for h in parts[i].get("holes_pt", [])]
            ms[ids[0]] = m
            for i in ids[1:]:
                parts.pop(i)
                became[i] = ids[0]
            done.append(f"merged {', '.join(ids)} into {ids[0]}")
        elif op == "fill_notches" and c.get("part") in parts:
            p = parts[c["part"]]
            filled = ms[p["id"]].copy()
            for h in p.get("holes_pt", []):
                cv2.fillPoly(filled.view(np.uint8), [(np.array(h) * zoom).round().astype(np.int32)], 1)
            closed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, disc(NOTCH_PX)) > 0
            others = np.zeros_like(closed)
            for k, om in ms.items():     # never grow into another part (its own holes stay holes)
                if k != p["id"] and k in parts and parts[k]["feature"] != "hole":
                    others |= om
            closed &= ~others
            added = float((closed & ~filled).sum()) / zoom ** 2
            p["polygon_pt"] = local_parts.mask_polygon(closed, zoom)
            done.append(f"filled {added:.1f} pt2 of notches in {p['id']}")
    return list(parts.values()), became, done


def correct(parts_dir: Path, model: str = MODEL) -> dict:
    """Correct every view of a local_parts run folder in place. Returns {view id: {"corrections", "notes",
    "done"}}."""
    views_path = parts_dir / "views.json"
    raw = parts_dir / "views_raw.json"
    if not raw.exists():
        raw.write_text(views_path.read_text())
    views = json.loads(raw.read_text())
    trace = parts_dir / "trace.jsonl"
    rules = agent_text()
    report, became_all = {}, {}
    for v in views:
        gray = cv2.imread(str(parts_dir / "views" / f"{v['id']}.png"), 0)
        zoom = gray.shape[1] / (v["rect_pt"][2] - v["rect_pt"][0])
        ms = masks(v, gray.shape, zoom)
        listing = facts(v, gray, zoom, ms)
        labelled = draw_labels(gray, v, zoom)
        cv2.imwrite(str(parts_dir / "views" / f"{v['id']}_to_correct.png"), labelled)
        content = [{"type": "text", "text": rules + f"\n\n# This view\n\n{v['id']}: {v.get('name') or ''} "
                                            f"({v['kind']}), {gray.shape[1] / zoom:.0f} x {gray.shape[0] / zoom:.0f} pt "
                                            f"(the images are {gray.shape[1]} x {gray.shape[0]} px).\n\n"
                                            "## Parts\n```json\n" + json.dumps(listing, indent=1) + "\n```"},
                   {"type": "text", "text": "The view, parts coloured and labelled:"},
                   image_part(cv2.imencode(".png", labelled)[1].tobytes()),
                   {"type": "text", "text": "The same view without colours:"},
                   image_part(cv2.imencode(".png", gray)[1].tobytes())]
        data, _ = ask_json(content, f"corrector_{v['id']}", trace, model=model)
        corrections = [c for c in data.get("corrections", []) if isinstance(c, dict)]
        v["parts"], became, done = apply(v, corrections, ms, zoom)
        became_all.update(became)
        report[v["id"]] = {"corrections": corrections, "notes": data.get("notes", ""), "done": done}
        cv2.imwrite(str(parts_dir / "views" / f"{v['id']}_corrected.png"), draw_labels(gray, v, zoom))
        print(f"[corrector {v['id']}] " + ("; ".join(done) or "no change"), flush=True)
    views_path.write_text(json.dumps(views, indent=2))
    (parts_dir / "corrections.json").write_text(json.dumps(report, indent=2))
    matches_path = parts_dir / "matches.json"
    if matches_path.exists():
        matches = json.loads(matches_path.read_text())
        for g in matches["groups"]:
            for m in g["members"]:
                m["parts"] = list(dict.fromkeys(became_all.get(p, p) for p in m["parts"]
                                                if became_all.get(p, p) is not None))
            g["members"] = [m for m in g["members"] if m["parts"]]
        matches["groups"] = [g for g in matches["groups"] if g["members"]]
        matches_path.write_text(json.dumps(matches, indent=2))
    return report


def draw_labels(gray: np.ndarray, view: dict, zoom: float) -> np.ndarray:
    parts = sorted(view["parts"], key=lambda p: p["feature"] == "hole")
    return local_parts.draw(gray, [(p["id"], p["polygon_pt"], p["holes_pt"] if p["feature"] != "hole" else [],
                                    local_parts.PALETTE[i % len(local_parts.PALETTE)])
                                   for i, p in enumerate(parts)], zoom)


if __name__ == "__main__":
    for vid, r in correct(Path(sys.argv[1])).items():
        print(vid, json.dumps(r, indent=1)[:1500])
