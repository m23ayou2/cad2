"""Draw2CAD web interface: drop in a drawing, watch the agents work, get the STEP.

    uv run python solve.py [--port 8000] [--parallel 3] [--no-open]

Opens http://127.0.0.1:8000 in the browser. Each drawing you add becomes a job that runs the full pipeline
(main.py) in its own process, up to --parallel at a time; the others wait in the queue. While a job runs, the
left side shows its log as it is written and the right side shows the newest STEP the builder has produced
(every build attempt, then the final model), updated as soon as a new one appears. Finished models download as
STEP. Jobs are saved under runs/ui/<job>/ (the drawing, job.json, log.txt) next to their run folder, so the list
survives a restart; a job that was running when the server stopped is marked interrupted.

Set OPENROUTER_API_KEY in .env. Model scripts run without a sandbox, as in main.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
JOBS_DIR = ROOT / "runs" / "ui"
MAX_UPLOAD = 50 * 1024 * 1024
# the models the interface offers; the first is the default (main.py's MODEL)
MODELS = [
    {"id": "anthropic/claude-haiku-5.5", "name": "Claude Haiku 5.5", "note": "Fast and cheap. Recommended"},
    {"id": "anthropic/claude-sonnet-5.5", "name": "Claude Sonnet 5.5", "note": "Stronger reasoning, about 20x the cost"},
    {"id": "anthropic/claude-opus-5.5", "name": "Claude Opus 5.5", "note": "Strongest, about 40x the cost"},
    {"id": "openai/gpt-6-luna", "name": "GPT-6 Luna", "note": "OpenAI, cheap"},
]
MODEL_IDS = {m["id"] for m in MODELS}
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".woff2": "font/woff2"}

sys.path.insert(0, str(ROOT / "src"))


# --------------------------------------------------------------------------- jobs
class Jobs:
    """Jobs on disk (runs/ui/<id>/job.json), run by a pool of `parallel` worker threads, one process each."""

    def __init__(self, parallel: int):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.queue: list[str] = []
        self.wake = threading.Condition(self.lock)
        for path in sorted(JOBS_DIR.glob("*/job.json")):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] == "running":       # the server stopped under it
                job.update(status="interrupted", ended=now())
                self._save(job)
            elif job["status"] == "queued":
                self.queue.append(job["id"])
            self.jobs[job["id"]] = job
        for _ in range(parallel):
            threading.Thread(target=self._worker, daemon=True).start()

    def _save(self, job: dict):
        path = JOBS_DIR / job["id"] / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        tmp.replace(path)  # never a half-written job.json

    def add(self, name: str, pdf: bytes, prompt: str = "", model: str = MODELS[0]["id"]) -> dict:
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        folder = JOBS_DIR / job_id
        folder.mkdir(parents=True)
        (folder / "drawing.pdf").write_bytes(pdf)
        # the prompt is kept and shown with the job; for now only the drawing drives the pipeline
        job = {"id": job_id, "name": name, "prompt": prompt, "model": model, "status": "queued", "created": now(), "started": None,
               "ended": None, "returncode": None, "run_dir": None}
        with self.lock:
            self.jobs[job_id] = job
            self._save(job)
            self.queue.append(job_id)
            self.wake.notify()
        return job

    def stop(self, job_id: str) -> dict:
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] == "queued":
                self.queue.remove(job_id)
                job.update(status="stopped", ended=now())
                self._save(job)
            proc = self.procs.get(job_id)
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)  # main.py and the CAD processes it started
        return job

    def _worker(self):
        while True:
            with self.lock:
                while not self.queue:
                    self.wake.wait()
                job_id = self.queue.pop(0)
                job = self.jobs[job_id]
                job.update(status="running", started=now())
                self._save(job)
            folder = JOBS_DIR / job_id
            with (folder / "log.txt").open("ab") as log:
                proc = subprocess.Popen([sys.executable, str(ROOT / "main.py"), str(folder / "drawing.pdf")],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                        env={**os.environ, "PYTHONUNBUFFERED": "1",
                                             "D2C_MODEL": job.get("model") or MODELS[0]["id"]})
                with self.lock:
                    self.procs[job_id] = proc
                code = proc.wait()
            with self.lock:
                self.procs.pop(job_id, None)
                stopped = code < 0 or job["status"] == "stopped"
                job.update(status="stopped" if stopped else ("done" if code == 0 else "failed"),
                           ended=now(), returncode=code, run_dir=job["run_dir"] or find_run_dir(job_id))
                self._save(job)

    def view(self, job_id: str) -> dict:
        """The job as the page shows it: status, its run folder, cost so far, final result if any."""
        with self.lock:
            job = dict(self.jobs[job_id])
        if not job["run_dir"]:
            job["run_dir"] = find_run_dir(job_id)
            if job["run_dir"]:
                with self.lock:
                    self.jobs[job_id]["run_dir"] = job["run_dir"]
        run = ROOT / job["run_dir"] if job["run_dir"] else None
        job["cost_usd"], job["result"] = None, None
        if run and (run / "trace.jsonl").is_file():
            cost = 0.0
            for line in (run / "trace.jsonl").read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record.get("event") == "llm":
                    cost += (record.get("usage") or {}).get("cost") or 0
                elif record.get("event") in ("parts_pipeline", "scan_views"):
                    cost += record.get("cost_usd") or 0
            job["cost_usd"] = round(cost, 4)
        if run and (run / "result.json").is_file():
            r = json.loads((run / "result.json").read_text(encoding="utf-8"))
            job["result"] = {k: r.get(k) for k in ("status", "validated", "drawn_covered", "high_issues_left",
                                                   "seconds", "cost_usd", "step")}
            job["cost_usd"] = round(r.get("cost_usd") or job["cost_usd"] or 0, 4)
        job["has_step"] = bool(run and (run / "model.step").is_file())
        return job


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def find_run_dir(job_id: str) -> str | None:
    """main.py logs `run <name>: ...` once it has made its run folder."""
    log = JOBS_DIR / job_id / "log.txt"
    if log.is_file():
        match = re.search(r" run (\S+): ", log.read_text(encoding="utf-8", errors="replace"))
        if match and (ROOT / "runs" / match.group(1)).is_dir():
            return str(Path("runs") / match.group(1))
    return None


# --------------------------------------------------------------------------- models
_mesh_lock = threading.Lock()
_mesh_cache: dict[str, dict] = {}


def latest_step(run: Path) -> tuple[Path, str] | None:
    """The newest STEP of a run: a build attempt while the builder works, model.step once kept."""
    candidates = list((run / "build").glob("attempt_*.step")) + [run / "model.step"]
    candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        return None
    step = max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name == "model.step"))
    if step.name == "model.step":
        label = "final model" if (run / "result.json").is_file() else "current model (kept attempt)"
    else:
        report = step.with_suffix(".json")
        ok = report.is_file() and json.loads(report.read_text(encoding="utf-8")).get("ok")
        label = f"build {step.stem.replace('_', ' ')} ({'valid solid' if ok else 'not accepted'})"
    return step, label


def model_payload(run: Path, since: str | None) -> dict:
    found = latest_step(run)
    if not found:
        return {"key": None}
    step, label = found
    key = f"{step.name}:{step.stat().st_mtime_ns}"
    if key == since:
        return {"key": key, "unchanged": True}
    with _mesh_lock:  # OCCT tessellation, one at a time
        if key not in _mesh_cache:
            import visualization
            _mesh_cache[key] = visualization.mesh(step)
    return {"key": key, "label": label, "file": step.name, "mesh": _mesh_cache[key]}


# --------------------------------------------------------------------------- projections
# The model projected into every view of the drawing, next to the drawn view (src/validate.py, no model calls).
# The pipeline does it once per validation round (validation/round_N/); the user can run it again on the current
# model from the interface (projections/<time>/).
_projecting: dict[str, dict] = {}   # job id -> {"running": bool, "error": the last failed run's error or None}
_project_lock = threading.Lock()


def projection_sets(run: Path) -> list[Path]:
    folders = list((run / "validation").glob("round_*")) + list((run / "projections").glob("*"))
    return sorted((f for f in folders if (f / "validation.json").is_file()),
                  key=lambda f: (f / "validation.json").stat().st_mtime)


def same_file(a: Path, b: Path) -> bool:
    return a.is_file() and b.is_file() and a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()


def projections_payload(job: dict, chosen: str | None) -> dict:
    run = ROOT / job["run_dir"]
    sets = projection_sets(run)
    names = [str(s.relative_to(run)) for s in sets]
    out = {"available": (run / "views").is_dir() and any((run / "views").glob("*.pdf")),
           "running": _projecting.get(job["id"], {}).get("running", False),
           "error": _projecting.get(job["id"], {}).get("error"), "sets": names, "set": None, "views": []}
    if not sets:
        return out
    folder = run / chosen if chosen in names else sets[-1]
    data = json.loads((folder / "validation.json").read_text(encoding="utf-8"))
    out.update(set=str(folder.relative_to(run)), matches_model=same_file(folder / "model.step", run / "model.step"),
               pose=(data.get("pose") or {}).get("verdict"))
    for v in data.get("views", []):
        images = {k: f"/api/jobs/{job['id']}/file?p=" + str((folder / Path(p).name).relative_to(run))
                  for k, p in (v.get("images") or {}).items() if (folder / Path(p).name).is_file()}
        out["views"].append({
            "id": v.get("view_id"), "label": v.get("label") or v.get("view_id"), "class": v.get("class"),
            "axes": {k: v.get(k) for k in ("horizontal", "vertical", "looking_along")},
            "metrics": v.get("metrics"), "callouts": v.get("callouts") or [],
            "missing": len(v.get("missing_regions") or []), "extra": len(v.get("extra_regions") or []),
            "note": v.get("unresolved") or v.get("note"), "images": images})
    return out


def run_projections(job: dict):
    """Project the current model.step into every view, in its own process (OpenCascade can crash)."""
    run = ROOT / job["run_dir"]
    out = run / "projections" / datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        out.mkdir(parents=True)
        (out / "model.step").write_bytes((run / "model.step").read_bytes())
        proc = subprocess.run([sys.executable, str(ROOT / "src" / "validate.py"), str(run / "views"),
                               str(out / "model.step"), str(out), str(JOBS_DIR / job["id"] / "drawing.pdf")],
                              cwd=ROOT / "src", capture_output=True, text=True, timeout=600)
        error = None if (out / "validation.json").is_file() else \
            f"the projection failed: {(proc.stderr or proc.stdout).strip()[-400:] or 'no output'}"
    except Exception as exc:
        error = f"the projection failed: {exc}"
    with _project_lock:
        _projecting[job["id"]] = {"running": False, "error": error}


# --------------------------------------------------------------------------- steps
VIEW_KINDS = ("front", "top", "side", "section", "detail", "isometric")


def step_title(label: str, seen: dict) -> str:
    """A sentence for one agent run, from its trace label (main, front[view1], validation/top[view4], ...)."""
    base, _, inner = label.partition("[")
    inner = inner.rstrip("]")
    seen[base] = seen.get(base, 0) + 1
    if base == "main":
        if inner == "scan":
            return "Plan the part from the matched views"
        return f"Revise the plan ({inner})" if inner else "Read the drawing and plan the part"
    if base in VIEW_KINDS:
        return f"Read the {base} view ({inner})"
    if base.startswith("validation/"):
        return f"Check the model against the {base.split('/')[1]} view ({inner})"
    return {"arbiter": "Settle where the views disagree",
            "builder": "Build the part" if seen[base] == 1 else "Rebuild the part with the corrections",
            "red_team": "Final review of the result",
            "corner_check": f"Check the traced outline corners ({inner})",
            "path": f"Read one region of the sheet ({inner})"}.get(base, label)


def mm(size) -> str:
    return " × ".join(f"{v:g}" for v in size) + " mm"


def tool_line(r: dict) -> str:
    """One line for one tool call, from its result."""
    tool, res = r.get("tool"), r.get("result") if isinstance(r.get("result"), dict) else {}
    if not r.get("ok"):
        return f"{tool} failed: {str(r.get('result') or r.get('error') or '')[:200]}"
    if tool == "view_drawing":
        return f"Looked at the drawing: {res.get('shown', 'sheet')}"
    if tool == "read_drawing_text":
        n = sum(len(s.get("texts", [])) for s in res.get("sheets", []))
        return f"Read {n} text items on the drawing" if n else "No text layer: the drawing is a scan"
    if tool == "split_views":
        views = res.get("views", [])
        return f"Found {len(views)} view(s): " + ", ".join(v.get("label") or v.get("id") for v in views)
    if tool == "call_subagent":
        return f"Got the analysis of the {res.get('label') or res.get('view') or 'view'} view"
    if tool == "run_cadquery":
        if res.get("ok"):
            return f"Build attempt {res.get('attempt')}: valid solid, {mm(res['bbox']['size'])}, volume {res.get('volume'):,.0f} mm³"
        why = res.get("error") or "; ".join(res.get("problems", [])) or "not accepted"
        return f"Build attempt {res.get('attempt')}: {why[:220]}"
    if tool == "scan_view":
        return (f"Scanned view {res.get('scan')}: {len(res.get('dimensions', []))} dimension(s), "
                f"{len(res.get('leaders', []))} leader(s)")
    if tool == "set_dimension_values":
        checked = res.get("dimensions_checked", [])
        bad = sum(1 for d in checked if not d.get("ok"))
        return (f"Scale of {res.get('scan')}: {res.get('px_per_mm')} px/mm, {len(checked)} dimension(s) checked"
                + (f", {bad} suspect" if bad else ", all agree"))
    if tool == "read_view_text":
        return f"Read {len(res.get('texts', []))} text items in the view"
    if tool in ("zoom_view", "zoom_comparison", "extend_view"):
        return f"Zoomed in: {res.get('shown') or res.get('region') or 'detail'}"
    return f"{tool.replace('_', ' ').capitalize()}: {res.get('shown') or res.get('region') or 'done'}"


def end_line(result) -> str | None:
    if not isinstance(result, dict):
        return None
    if "verdict" in result:
        return f"Verdict: {result['verdict']}, {len(result.get('issues', []))} issue(s)"
    if "feature_tree" in result:
        return f"Plan: {len(result['feature_tree'].get('features', []))} feature(s)"
    if "rulings" in result:
        return f"{len(result['rulings'])} ruling(s)"
    if "status" in result:
        return f"Result: {result['status']}"
    return None


def steps(run: Path, running: bool) -> dict:
    """The run as a list of steps (one per agent run, plus the projection checks), from trace.jsonl."""
    out, open_steps, seen = [], {}, {}
    start = end = None
    trace = run / "trace.jsonl"
    lines = trace.read_text(encoding="utf-8").splitlines() if trace.is_file() else []
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:   # a line being written
            continue
        t = datetime.fromisoformat(r["t"]) if r.get("t") else None
        start = start or t
        end = t or end
        event, agent = r.get("event"), r.get("agent")
        if event == "agent_start":
            step = {"title": step_title(agent, seen), "agent": agent, "status": "running", "t0": t, "t1": None,
                    "cost": None, "details": []}
            out.append(step)
            open_steps[agent] = step
        elif event == "tool" and agent in open_steps:
            open_steps[agent]["details"].append(tool_line(r))
        elif event == "rejected" and agent in open_steps:
            open_steps[agent]["details"].append("Answer sent back for a fix: " + str(r.get("problems", ""))[:200])
        elif event == "agent_end" and agent in open_steps:
            step = open_steps.pop(agent)
            step.update(status="done", t1=t, cost=r.get("cost_usd"))
            if (line_ := end_line(r.get("result"))):
                step["details"].append(line_)
        elif event == "parts_pipeline":
            out.append({"title": f"Read the scanned drawing with {str(r.get('model', '')).split('/')[-1]}",
                        "status": "done", "t0": t - timedelta(seconds=r.get("seconds") or 0) if t else None, "t1": t,
                        "cost": r.get("cost_usd"),
                        "details": [f"{len(r.get('views', []))} view(s): " + ", ".join(r.get("views", [])),
                                    f"{r.get('parts')} part(s) outlined, matched into {len(r.get('groups', []))} group(s): "
                                    + ", ".join(str(g) for g in r.get("groups", []))]})
        elif event == "scan_views":
            axes = r.get("axes") or {}
            out.append({"title": f"Read the scanned drawing with {str(r.get('model', '')).split('/')[-1]}",
                        "status": "done", "t0": t - timedelta(seconds=r.get("seconds") or 0) if t else None, "t1": t,
                        "cost": r.get("cost_usd"),
                        "details": [f"{len(r.get('views', []))} view(s): " + ", ".join(r.get("views", [])),
                                    "No flat pattern: planned from the views. Axes worked out for "
                                    + (", ".join(f"{k} (right {a.get('horizontal')}, up {a.get('vertical')})"
                                                 for k, a in axes.items()) or "no view")]})
        elif event == "projections":
            views = ((r.get("data") or {}).get("views")) or []
            out.append({"title": f"Project the model into every view (round {r.get('round')})", "status": "done",
                        "t0": t, "t1": t, "cost": None,
                        "details": [f"{len(views)} view(s) compared with the model"]})
        elif event == "run_end" and not r.get("ok"):
            out.append({"title": "The run stopped with an error", "status": "failed", "t0": t, "t1": t, "cost": None,
                        "details": [str(r.get("error", ""))[:400]]})
    for step in out:
        if step["status"] == "running" and not running:
            step["status"] = "stopped"
        t0, t1 = step.pop("t0"), step.pop("t1")
        step["seconds"] = round(((t1 or (datetime.now() if running else end)) - t0).total_seconds()) if t0 else None
        step.pop("agent", None)
    return {"steps": out, "seconds": round(((datetime.now() if running else end) - start).total_seconds())
            if start else 0}


# --------------------------------------------------------------------------- chat
CHAT_SYSTEM = """You are the assistant of Draw2CAD, a tool that turns an engineering drawing into a 3D STEP part.
The part below was made from the user's drawing by a team of agents (they read the views, planned a feature tree,
built it in CadQuery and checked it against the drawing). Answer the user's questions about this part and how
it was made, from the run data below only: sizes, features, what was checked, what is uncertain, cost and time.
Be short and concrete, in plain words for a machinist, not a programmer. If something is not in the data, say
so. You cannot change the part from this chat yet: if asked for a change, say which features it would touch
and suggest running the drawing again."""


def run_context(run: Path) -> str:
    def load(name):
        path = run / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    plan, result, build = load("plan.json") or {}, load("result.json") or {}, load("build_report.json") or {}
    tree = plan.get("feature_tree") or {}
    data = {
        "result": {k: result.get(k) for k in ("status", "validated", "measured_views", "drawn_covered",
                                              "high_issues_left", "rounds", "seconds", "cost_usd")},
        "overview": plan.get("overview"),
        "features": [{k: f.get(k) for k in ("id", "name", "op", "params", "state")} for f in tree.get("features", [])],
        "assumptions": tree.get("assumptions"), "unsupported": tree.get("unsupported"),
        "uncertainties": plan.get("uncertainties"),
        "build": {"status": build.get("status"), "notes": build.get("notes")},
    }
    return json.dumps(data, ensure_ascii=False, default=str)[:60000]


def chat(job: dict, message: str, model: str | None = None) -> list[dict]:
    from dotenv import load_dotenv
    from openai import OpenAI

    path = JOBS_DIR / job["id"] / "chat.json"
    history = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    load_dotenv(ROOT / ".env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"), timeout=120)
    messages = [{"role": "system", "content": CHAT_SYSTEM + "\n\n# Run data\n" + run_context(ROOT / job["run_dir"])}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})
    reply = client.chat.completions.create(model=model or job.get("model") or MODELS[0]["id"], messages=messages,
                                           max_tokens=1500,
                                           extra_body={"usage": {"include": True}})
    usage = reply.usage.model_dump() if reply.usage else {}
    history += [{"role": "user", "content": message, "t": now()},
                {"role": "assistant", "content": reply.choices[0].message.content or "", "t": now(),
                 "model": reply.model, "cost_usd": usage.get("cost")}]
    path.write_text(json.dumps(history, indent=1, ensure_ascii=False), encoding="utf-8")
    return history


# --------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    jobs: Jobs

    def log_message(self, fmt, *args):  # keep the terminal for errors
        pass

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str, download: str | None = None):
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        # always ask the server again: an updated page must never be served from the browser's cache
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def job_or_404(self, job_id: str) -> dict | None:
        if job_id not in self.jobs.jobs:
            self.send_json({"error": "no such job"}, HTTPStatus.NOT_FOUND)
            return None
        return self.jobs.view(job_id)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
        try:
            if not parts:
                return self.send_file(WEB / "index.html", STATIC_TYPES[".html"])
            if parts[0] == "vendor":
                vendor = (WEB / "vendor").resolve()
                path = vendor.joinpath(*parts[1:]).resolve()
                if vendor in path.parents and path.is_file():
                    return self.send_file(path, STATIC_TYPES.get(path.suffix, "application/octet-stream"))
            if parts == ["api", "jobs"]:
                return self.send_json([self.jobs.view(j) for j in sorted(self.jobs.jobs, reverse=True)])
            if parts == ["api", "info"]:
                return self.send_json({"models": MODELS, "default": MODELS[0]["id"]})
            if len(parts) == 4 and parts[:2] == ["api", "jobs"]:
                job = self.job_or_404(parts[2])
                if job is None:
                    return
                if parts[3] == "log":
                    offset = int(query.get("offset", ["0"])[0])
                    log = JOBS_DIR / job["id"] / "log.txt"
                    data = log.read_bytes()[offset:] if log.is_file() else b""
                    return self.send_json({"text": data.decode("utf-8", errors="replace"),
                                           "offset": offset + len(data), "job": job})
                if parts[3] == "steps":
                    run = ROOT / job["run_dir"] if job["run_dir"] else None
                    data = steps(run, job["status"] == "running") if run else {"steps": [], "seconds": 0}
                    return self.send_json({**data, "job": job})
                if parts[3] == "chat":
                    path = JOBS_DIR / job["id"] / "chat.json"
                    return self.send_json(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else [])
                if parts[3] == "projections":
                    if not job["run_dir"]:
                        return self.send_json({"available": False, "sets": [], "views": []})
                    return self.send_json(projections_payload(job, query.get("set", [None])[0]))
                if parts[3] == "file":  # an image of the run (projections), nothing outside its folder
                    run = (ROOT / job["run_dir"]).resolve() if job["run_dir"] else None
                    path = (run / query.get("p", [""])[0]).resolve() if run else None
                    if not path or run not in path.parents or path.suffix != ".png" or not path.is_file():
                        return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return self.send_file(path, "image/png")
                if parts[3] == "drawing":
                    return self.send_file(JOBS_DIR / job["id"] / "drawing.pdf", "application/pdf")
                if parts[3] == "model":
                    if not job["run_dir"]:
                        return self.send_json({"key": None})
                    return self.send_json(model_payload(ROOT / job["run_dir"], query.get("since", [None])[0]))
                if parts[3] == "step":
                    run = ROOT / job["run_dir"] if job["run_dir"] else None
                    found = latest_step(run) if run else None
                    if not found:
                        return self.send_json({"error": "no STEP yet"}, HTTPStatus.NOT_FOUND)
                    stem = Path(job["name"]).stem or "model"
                    return self.send_file(found[0], "application/step", f"{stem}.step")
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as error:  # one bad request must not take the server down
            self.send_json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        parts = [p for p in urlparse(self.path).path.strip("/").split("/") if p]
        try:
            if parts == ["api", "jobs"]:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= MAX_UPLOAD:
                    return self.send_json({"error": "the drawing must be a PDF under 50 MB"},
                                          HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                data = self.rfile.read(size)
                if not data.startswith(b"%PDF"):
                    return self.send_json({"error": "that file is not a PDF"}, HTTPStatus.BAD_REQUEST)
                name = re.sub(r"[^\w.\- ]", "_", unquote(self.headers.get("X-Filename", "drawing.pdf")))[:120]
                prompt = unquote(self.headers.get("X-Prompt", ""))[:4000]
                model = self.headers.get("X-Model") or MODELS[0]["id"]
                if model not in MODEL_IDS:
                    return self.send_json({"error": f"unknown model {model}"}, HTTPStatus.BAD_REQUEST)
                job = self.jobs.add(name, data, prompt, model)
                return self.send_json(self.jobs.view(job["id"]), HTTPStatus.CREATED)
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "projections":
                job = self.job_or_404(parts[2])
                if job is None:
                    return
                run = ROOT / job["run_dir"] if job["run_dir"] else None
                if not run or not (run / "model.step").is_file():
                    return self.send_json({"error": "there is no model to project yet"}, HTTPStatus.CONFLICT)
                if not ((run / "views").is_dir() and any((run / "views").glob("*.pdf"))):
                    return self.send_json({"error": "projections need the drawing's views as vector PDF; this "
                                                    "drawing is a scan, which is not supported yet"},
                                          HTTPStatus.CONFLICT)
                with _project_lock:
                    if _projecting.get(job["id"], {}).get("running"):
                        return self.send_json(projections_payload(job, None))
                    _projecting[job["id"]] = {"running": True, "error": None}
                threading.Thread(target=run_projections, args=(job,), daemon=True).start()
                return self.send_json(projections_payload(job, None), HTTPStatus.ACCEPTED)
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "chat":
                job = self.job_or_404(parts[2])
                if job is None:
                    return
                if not job["run_dir"] or job["status"] in ("queued", "running"):
                    return self.send_json({"error": "the part is not finished yet"}, HTTPStatus.CONFLICT)
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                message = str(body.get("message", "")).strip()[:4000]
                if not message:
                    return self.send_json({"error": "empty message"}, HTTPStatus.BAD_REQUEST)
                model = body.get("model") if body.get("model") in MODEL_IDS else None
                return self.send_json(chat(job, message, model))
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop":
                if self.job_or_404(parts[2]) is None:
                    return
                self.jobs.stop(parts[2])
                return self.send_json(self.jobs.view(parts[2]))
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as error:
            self.send_json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--parallel", type=int, default=3, help="jobs that run at the same time")
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = parser.parse_args()
    Handler.jobs = Jobs(args.parallel)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Draw2CAD is running at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_open:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
