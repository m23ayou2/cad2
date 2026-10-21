"""Run the local (no LLM) segmentation and matching of src/local_parts.py on one drawing.

    uv run python local_parts_demo.py [cases/tier4/01/drawing.pdf] [--views runs/<run>/views.json]
                                      [--truth runs/<run>]

--llm blind|review|both adds one LLM call (Claude Haiku) that matches the locally segmented, numbered parts
across views (src/local_llm_match.py): blind on its own, review correcting the local matcher's groups.
--views reuses the view boxes of a part_views run (so only segmentation and matching are tested); without it the
views are found by regions.view_crops. --truth scores the result against a part_views run taken as ground truth:
each of its parts is found when one of ours overlaps it (IoU >= 0.5), and each of its groups is right when all
its found members land in one of our groups that holds no member of another group.
Outputs go to runs/<tier>-<case>-local-<timestamp>/.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.local_parts import run


def iou(a: list, b: list, holes_a=(), holes_b=(), s: float = 4.0) -> float:
    pts = np.array(a + b) * s
    w, h = (pts.max(0) + 4).astype(int)
    ma, mb = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    for m, poly, holes in ((ma, a, holes_a), (mb, b, holes_b)):
        cv2.fillPoly(m, [(np.array(poly) * s).round().astype(np.int32)], 1)
        for hl in holes:
            cv2.fillPoly(m, [(np.array(hl) * s).round().astype(np.int32)], 0)
    union = (ma | mb).sum()
    return float((ma & mb).sum() / union) if union else 0.0


def score(ours: dict, truth_dir: Path) -> str:
    truth_views = json.loads((truth_dir / "views.json").read_text())
    truth_groups = json.loads((truth_dir / "matches.json").read_text())["groups"]
    mine = {v["id"]: v for v in ours["views"]}
    found, lines = {}, ["## Parts found (IoU with the truth part >= 0.5)"]
    for tv in truth_views:
        for tp in tv["parts"]:
            best = max(((iou(tp["polygon_pt"], p["polygon_pt"], tp["holes_pt"], p["holes_pt"]), p["id"])
                        for p in mine[tv["id"]]["parts"]), default=(0, None))
            found[tp["id"]] = best[1] if best[0] >= 0.5 else None
            lines.append(f"- {tv['id']} {tp['id']} {tp['name']} ({tp['feature']}): "
                         + (f"{best[1]} (IoU {best[0]:.2f})" if found[tp["id"]] else f"missed (best IoU {best[0]:.2f})"))
    group_of = {p: g["id"] for g in ours["matches"]["groups"] for m in g["members"] for p in m["parts"]}
    truth_of = {found[p]: g["id"] for g in truth_groups for m in g["members"] for p in m["parts"] if found.get(p)}
    lines.append("\n## Groups")
    right = 0
    for g in truth_groups:
        members = [(m["view"], p, found.get(p)) for m in g["members"] for p in m["parts"]]
        ids = {group_of.get(o) for _, _, o in members if o}
        ok = len(ids) == 1 and None not in ids and all(truth_of.get(p, g["id"]) == g["id"]
                                                       for p, gg in group_of.items() if gg in ids)
        missing = [f"{v}:{p}" for v, p, o in members if not o]
        right += ok
        lines.append(f"- {g['id']} {g['name']}: {'RIGHT' if ok else 'WRONG'}; ours "
                     + ", ".join(f"{v}:{p}->{o}->{group_of.get(o)}" for v, p, o in members if o)
                     + (f"; not segmented: {', '.join(missing)}" if missing else ""))
    lines.insert(0, f"# Score: {right}/{len(truth_groups)} groups right, "
                    f"{sum(1 for v in found.values() if v)}/{len(found)} parts found\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drawing", nargs="?", default="cases/tier4/01/drawing.pdf")
    ap.add_argument("--views", type=Path)
    ap.add_argument("--truth", type=Path)
    ap.add_argument("--llm", choices=["blind", "review", "both"])
    args = ap.parse_args()
    drawing = Path(args.drawing)
    run_dir = Path(__file__).resolve().parent / "runs" / \
        f"{drawing.parent.parent.name}-{drawing.parent.name}-local-{datetime.now():%Y%m%d-%H%M%S}"
    print(f"Run folder: {run_dir}", flush=True)
    out = run(drawing, run_dir, args.views)
    for v in out["views"]:
        kinds = {}
        for p in v["parts"]:
            kinds[p["feature"]] = kinds.get(p["feature"], 0) + 1
        print(f"{v['id']} ({v['kind']}): {kinds}" + (f", panels moved from {v['transferred_from']}"
                                                     if v["transferred_from"] else ""))
    print(f"{len(out['matches']['groups'])} groups; reference view {out['matches']['reference_view']}")
    print("timing:", out["matches"]["timing"])
    if args.truth:
        report = score(out, args.truth)
        (run_dir / "score.md").write_text(report)
        print(report)
    for mode in (["blind", "review"] if args.llm == "both" else [args.llm] if args.llm else []):
        from src.local_llm_match import llm_match
        m = llm_match(run_dir, out, out["zoom"], mode)
        cost = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()][-1]
        print(f"\nLLM {mode}: {len(m['groups'])} groups, not parts: {m['not_parts']}, unknown labels: "
              f"{m['unknown_labels']}; {cost['model']} {cost['seconds']} s, ${cost.get('cost') or 0:.4f}")
        if args.truth:
            report = score({"views": out["views"], "matches": m}, args.truth)
            (run_dir / f"score_llm_{mode}.md").write_text(report)
            print(report.splitlines()[0])
            print("\n".join(line for line in report.splitlines() if "WRONG" in line))


if __name__ == "__main__":
    main()
