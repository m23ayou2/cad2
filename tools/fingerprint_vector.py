"""Fingerprint everything the vector path sends or decides, with no API calls, to check that a change leaves
vector drawings untouched: run it before and after the change and compare the two files.

    uv run python tools/fingerprint_vector.py before.json      # on the old code
    uv run python tools/fingerprint_vector.py after.json       # on the new code
    uv run python tools/fingerprint_vector.py --compare before.json after.json

It hashes every agent's system prompt and settings (agents/**/*.md), every tool schema, every output schema and
the main constants; split_views on two vector drawings (listing and overview image); and the exact first request
(model, messages, tools, settings) of the vector planner, a view subagent, the builder, a plan revision and a
validation agent, captured by a fake client that records the request and stops. The builder, revision and
validation requests use a built vector run as fixture (--fixture, default: the newest one under runs/).
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ("tier1/01", "tier2/03")
CONSTANTS = ("MODEL", "VISION_MODEL", "SCAN_REASON_MODEL", "SCAN_BUILD_MODEL", "SCAN_MAX_COST_USD", "SCAN_PARTS",
             "REPAIR_TURNS", "CAD_TIMEOUT_S", "VALIDATE_TIMEOUT_S", "MAX_CORRECTIONS", "PARTS_BRIEF")


def h(obj) -> str:
    data = obj if isinstance(obj, (bytes, str)) else json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()[:16]


class Stop(Exception):
    pass


class Recorder:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise Stop()


def newest_fixture() -> Path | None:
    """The newest run of a vector drawing with a plan and a model."""
    import main
    for run in sorted((ROOT / "runs").glob("tier[123]-*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if (run / "plan.json").is_file() and (run / "model.step").is_file() and (run / "trace.jsonl").is_file():
            drawing = json.loads((run / "trace.jsonl").open().readline()).get("drawing", "")
            if drawing and not main.is_scanned(ROOT / drawing):
                return run
    return None


def fingerprint(fixture: Path | None) -> dict:
    sys.path.insert(0, str(ROOT))
    import main
    import schema
    work = Path(tempfile.mkdtemp(prefix="fingerprint_"))

    def fake_run(drawing: Path, name: str) -> "main.Run":
        folder = work / name  # the same names on every machine: the requests must not depend on the folder
        folder.mkdir(parents=True)
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Recorder()))
        run = main.Run(drawing.resolve(), folder, client, max_cost_usd=1.0)
        run.scanned = main.is_scanned(drawing)
        return run

    def first_request(run, fn, *args):
        try:
            fn(*args)
        except Stop:
            pass
        calls = run.client.chat.completions.calls
        return calls[0] if calls else None

    out = {}
    for md in sorted((ROOT / "agents").rglob("*.md")):
        name = str(md.relative_to(ROOT / "agents"))[:-3]
        agent = main.load_agent(name)
        out[f"prompt:{name}"] = h(agent.system_prompt)
        out[f"settings:{name}"] = h(agent.settings)
    out.update({f"tool:{n}": h(main.tool_schema(f)) for n, f in sorted(main.TOOLS.items())})
    out.update({f"schema:{n}": h(m.model_json_schema()) for n, m in sorted(schema.MODELS.items())})
    out["constants"] = h({k: getattr(main, k) for k in CONSTANTS})
    plan = json.loads((fixture / "plan.json").read_text()) if fixture else None
    for case in CASES:
        drawing, tag = ROOT / "cases" / case / "drawing.pdf", case.replace("/", "-")
        run = fake_run(drawing, f"{tag}-split")
        out[f"model_per_agent:{tag}"] = h({n: main.agent_model(main.load_agent(n), run) for n in ("main", "builder")})
        split = main.split_views(main.Ctx(run))
        out[f"split_views:{tag}"] = h(split.data)
        out[f"split_views_image:{tag}"] = h(split.images[0][1])
        run = fake_run(drawing, f"{tag}-plan")
        out[f"first_request:planner:{tag}"] = h(first_request(run, main.plan, run))
        run = fake_run(drawing, f"{tag}-sub")
        main.split_views(main.Ctx(run))
        view = next(v for v in run.views.values() if v.suggested_subagent)
        out[f"first_request:subagent:{tag}"] = h(
            first_request(run, main.call_subagent, main.Ctx(run), view.suggested_subagent, view.id, "context"))
        if plan is None:
            continue
        run = fake_run(drawing, f"{tag}-build")
        out[f"first_request:builder:{tag}"] = h(first_request(run, main.build, run, json.loads(json.dumps(plan))))
        run = fake_run(drawing, f"{tag}-revise")
        main.split_views(main.Ctx(run))
        issue = {"kind": "wrong_size", "severity": "high", "owner": "plan", "feature": "F01", "description": "d",
                 "drawing": "d", "model": "m", "fix": "f", "evidence": []}
        out[f"first_request:revision:{tag}"] = h(
            first_request(run, main.revise_plan, run, json.loads(json.dumps(plan)), [issue], None, 1))
    if fixture:  # a validation agent's first request on a built vector model (projections run locally, ~30 s)
        drawing = ROOT / json.loads((fixture / "trace.jsonl").open().readline())["drawing"]
        run = fake_run(drawing, "validate")
        main.split_views(main.Ctx(run))
        for name in ("model.step", "model.py"):
            shutil.copyfile(fixture / name, run.dir / name)
        out["first_request:validation"] = h(first_request(run, main.validate, run, json.loads(json.dumps(plan)), 1))
    shutil.rmtree(work, ignore_errors=True)
    return out


def compare(before: Path, after: Path) -> int:
    b, a = json.loads(before.read_text()), json.loads(after.read_text())
    changed = [k for k in b if a.get(k) != b[k]]
    print(f"identical: {len(b) - len(changed)} / {len(b)}")
    print("changed:", changed or "none")
    print("added (new entries, not used by the vector path):", [k for k in a if k not in b] or "none")
    return 1 if changed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", nargs="?", type=Path)
    ap.add_argument("--fixture", type=Path, help="a built vector run (default: the newest under runs/)")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    if not args.out:
        ap.error("give an output file, or --compare BEFORE AFTER")
    sys.path.insert(0, str(ROOT))
    fixture = args.fixture or newest_fixture()
    print(f"fixture: {fixture or 'none (builder, revision and validation requests skipped)'}")
    result = fingerprint(fixture)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"{len(result)} fingerprints -> {args.out}")
