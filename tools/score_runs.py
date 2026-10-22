"""Score run folders against their case's reference STEP (volume IoU, tools/voxel_iou.py), several at a time.

Each run's drawing is read from its trace.jsonl; the reference is reference.step next to it. Prints one line per
run (case, IoU, mirrored IoU, model and reference sizes, the run's route, cost and time from result.json) and
writes them to --out as JSON.

    uv run python tools/score_runs.py runs/tier4-0*-2026102* [--res 160] [--jobs 3] [--out scores.json]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import voxel_iou  # noqa: E402


def case_of(run: Path) -> tuple[Path | None, Path | None]:
    first = json.loads((run / "trace.jsonl").open(encoding="utf-8").readline())
    drawing = Path(first["drawing"])
    drawing = drawing if drawing.is_absolute() else ROOT / drawing
    reference = drawing.with_name("reference.step")
    return drawing, reference if reference.is_file() else None


def score_run(run: Path, res: int) -> dict:
    row = {"run": run.name}
    try:
        drawing, reference = case_of(run)
        row["case"] = f"{drawing.parent.parent.name}/{drawing.parent.name}"
        result = json.loads((run / "result.json").read_text()) if (run / "result.json").is_file() else {}
        row.update({k: result.get(k) for k in ("scan_route", "status", "cost_usd", "seconds")})
        if reference is None:
            row["error"] = "no reference.step next to the drawing"
        elif not (run / "model.step").is_file():
            row["error"] = "no model.step"
        else:
            row.update(voxel_iou.score(run / "model.step", reference, res))
    except Exception as error:  # one broken run must not stop the batch
        row["error"] = f"{type(error).__name__}: {error}"
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--res", type=int, default=160)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    runs = [r for r in args.runs if (r / "trace.jsonl").is_file()]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(score_run, runs, [args.res] * len(runs)))
    for r in sorted(rows, key=lambda r: (r.get("case", ""), r["run"])):
        if "error" in r:
            print(f"{r.get('case', '?'):10s} {r['run']:34s} {r['error']}")
            continue
        print(f"{r['case']:10s} {r['run']:34s} IoU {r['iou']:.3f} (mirrored {r['iou_mirrored']:.3f})  "
              f"model {r['model_bbox']} ref {r['ref_bbox']}  route {r.get('scan_route') or '-'}  "
              f"${r.get('cost_usd') or 0:.3f}  {r.get('seconds') or '-'} s")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
