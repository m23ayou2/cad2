"""Experiment: read a drawing region by region (no vectors).

Every view is cropped from the sheet's pixels, every region its lines enclose is filled and labelled P1, P2, ...,
and a path agent analyses each region on its own: what it is, which dimensions size it, and how to create it.

    uv run python paths.py cases/tier4/01/drawing.pdf [--max-regions 15] [--views V2,V3] [--workers 4]

Writes runs/paths-<case>-<time>/: views/ (each view, and its regions coloured and labelled), paths/ (each
region's close-up and the agent's JSON), paths.json (everything) and summary.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

import regions as rg

log = logging.getLogger("draw2cad.paths")


def png(img) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def analyse(m, run, agent, sheet_png: bytes, view: dict, labelled_png: bytes, region: dict,
            all_regions: list[dict], out: Path) -> dict:
    focus = rg.focus_image(view["image"], region)
    cv2.imwrite(str(out / f"{view['id']}_{region['id']}.png"), focus)
    lines = [{"with": s["with"], "from_px": s["from"], "to_px": s["to"]} for s in region["shared_lines"]]
    info = {"view": view["id"], "region": region["id"],
            "area_share_of_view": round(region["area_px"] / max(1, sum(r["area_px"] for r in all_regions)), 3),
            "touches": region["touches"], "shared_straight_lines": lines,
            "regions_in_this_view": [r["id"] for r in all_regions]}
    content = [{"type": "text", "text": "Region to analyse:\n```json\n" + json.dumps(info, indent=1) + "\n```"},
               {"type": "text", "text": "The whole sheet:"}, m.image_part(sheet_png),
               {"type": "text", "text": f"View {view['id']}, every region coloured with its id:"},
               m.image_part(labelled_png),
               {"type": "text", "text": f"{region['id']} close up (tinted orange, outlined):"}, m.image_part(png(focus))]
    label = f"path[{view['id']}.{region['id']}]"
    try:
        result = m.run_agent(agent, m.Ctx(run), content, label=label)
    except Exception as error:
        log.warning("%s failed: %s", label, error)
        return {"view": view["id"], "region": region["id"], "error": str(error)}
    (out / f"{view['id']}_{region['id']}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    log.info("%s: %s, %s, %d dimension(s), create %s", label, result["role"], result["shape"],
             len(result["dimensions"]), (result.get("create") or {}).get("op", "-"))
    return result


def summary(results: list[dict], views: list[dict], run, seconds: float) -> str:
    lines = [f"# Paths: {run.drawing.parent.name}/{run.drawing.name}", "",
             f"{len(views)} view(s), {len(results)} region(s) analysed, {seconds:.0f} s, ${run.cost_usd:.3f}", ""]
    for v in views:
        rows = [r for r in results if r.get("view") == v["id"]]
        lines += [f"## {v['id']}", "", f"![{v['id']}](views/{v['id']}_regions.png)", "",
                  "| region | role | shape | dimensions | create | conf. |", "|---|---|---|---|---|---|"]
        for r in rows:
            if r.get("skipped"):
                lines.append(f"| {r['region']} | background (skipped) | | annotation share {r['annotation_share']} | | |")
                continue
            if "error" in r:
                lines.append(f"| {r['region']} | error | | {r['error'][:80]} | | |")
                continue
            dims = ", ".join(d["text"] for d in r["dimensions"]) or "-"
            c = r.get("create") or {}
            create = f"{c.get('op')} {json.dumps(c.get('params', {}))}" if c else "-"
            lines.append(f"| {r['region']} | {r['role']} | {r['shape']} | {dims} | {create} | {r['confidence']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def analyse_sheet(m, run, out_dir: Path, max_regions: int = 15, min_share: float = 0.01, only_views: str = "",
                  workers: int = 4) -> list[dict]:
    """Crop the views, fill and label their regions, and run the path agent on each region. `m` is the main module
    (its run_agent, Ctx, image_part, load_agent). Writes out_dir/views (the views and their labelled regions), out_dir/regions (each region's close-up and
    the agent's JSON), paths.json and summary.md."""
    started = time.monotonic()
    (out_dir / "views").mkdir(parents=True, exist_ok=True)
    (out_dir / "regions").mkdir(exist_ok=True)
    sheet = rg.render(str(run.drawing))
    small = cv2.resize(sheet, None, fx=1400 / max(sheet.shape), fy=1400 / max(sheet.shape), interpolation=cv2.INTER_AREA)
    sheet_png = png(small)
    all_views = rg.view_crops(sheet)
    views = [v for v in all_views if not only_views or v["id"] in only_views.split(",")]
    overview = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    for v in all_views:
        x0, y0, x1, y1 = v["box"]
        cv2.rectangle(overview, (x0, y0), (x1, y1), (0, 0, 255), 4)
        cv2.putText(overview, v["id"], (x0 + 6, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    cv2.imwrite(str(out_dir / "views" / "sheet_views.png"), overview)
    agent = m.load_agent("path")
    jobs = []
    import raster_lines
    skipped = []
    for v in views:
        regions = rg.fill_regions(v["image"])
        # paper boxed in by dimension lines is background: no need to ask the agent (only on a dimensioned view,
        # where annotations make a good share of the ink; on an isometric a long edge can pass for one)
        sep = raster_lines.separate_structure(v["image"])
        annot = sep["dimension"] | sep["extension"] | sep["text"]
        ink = v["image"] < rg.INK
        import raster_vector
        dimensioned = len(raster_vector.dimensions(sep, sep["text"])[0]) >= 2      # it carries real dimensions
        rg.annotation_share(v["image"], regions, annot)
        background = [r for r in regions if dimensioned and r["annotation_share"] >= 0.6]
        for r in background:
            skipped.append({"view": v["id"], "region": r["id"], "role": "background", "skipped": True,
                            "annotation_share": r["annotation_share"],
                            "description": "bounded mostly by dimension, extension or leader lines: not analysed"})
        largest = regions[0]["area_px"] if regions else 0
        bg = {r["id"] for r in background}
        picked = [r for r in regions if r["area_px"] >= min_share * largest and r["id"] not in bg][:max_regions]
        labelled = rg.label_image(v["image"], regions)
        cv2.imwrite(str(out_dir / "views" / f"{v['id']}.png"), v["image"])
        cv2.imwrite(str(out_dir / "views" / f"{v['id']}_regions.png"), labelled)
        log.info("%s: %d region(s), analysing %d", v["id"], len(regions), len(picked))
        run.trace("path_view", view=v["id"], box=v["box"], regions=[
            {k: r[k] for k in ("id", "area_px", "box", "touches")} for r in regions], analysed=[r["id"] for r in picked])
        jobs += [(v, png(labelled), r, regions) for r in picked]

    def job(item):
        if run.over_budget:
            return {"view": item[0]["id"], "region": item[2]["id"], "error": "cost limit reached"}
        return analyse(m, run, agent, sheet_png, item[0], item[1], item[2], item[3], out_dir / "regions")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(job, jobs)) + skipped
    log.info("paths: %d region(s) analysed, %d skipped as background", len(jobs), len(skipped))
    seconds = time.monotonic() - started
    (out_dir / "paths.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary(results, views, run, seconds), encoding="utf-8")
    run.trace("paths_end", regions=len(results), seconds=round(seconds, 1), cost_usd=run.cost_usd)
    log.info("paths: %d region(s) in %.0f s, $%.3f -> %s", len(results), seconds, run.cost_usd, out_dir / "summary.md")
    return results


def main():
    import main as m
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("drawing", type=Path)
    parser.add_argument("--max-regions", type=int, default=15, help="largest regions analysed per view")
    parser.add_argument("--min-share", type=float, default=0.01,
                        help="skip regions smaller than this share of the view's largest region")
    parser.add_argument("--views", default="", help="only these views, e.g. V2,V3")
    parser.add_argument("--workers", type=int, default=4, help="regions analysed at the same time")
    parser.add_argument("--max-cost", type=float, default=2.0, help="stop starting agents past this cost (USD)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    drawing = args.drawing.resolve()
    case = "-".join(drawing.parent.parts[-2:])
    run_dir = m.RUNS_DIR / f"paths-{case}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    run = m.open_run(drawing, run_dir, args.max_cost)
    run.trace("paths_start", drawing=str(drawing), model=m.MODEL)
    analyse_sheet(m, run, run_dir, args.max_regions, args.min_share, args.views, args.workers)


if __name__ == "__main__":
    sys.exit(main())
