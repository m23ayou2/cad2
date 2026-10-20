# AI Engineer: take-home challenge

**Start with [CHALLENGE.md](CHALLENGE.md)**: it explains the client's need, what we ask, and how we review it.

This README covers how to run what is in the repository.

Deliver a functional product with a complete workflow from drawing submission to
rendered results and STEP files, an improved agent, evidence from experiments,
concurrent execution with isolated runs, and saved traces.
The product must have a UI usable without technical knowledge. Include a short technical write-up with an
answer to the client. The starter only provides a single-call reconstruction:
the agent loop, evaluation, and concurrency are yours to design. See
CHALLENGE.md for expectations.

## Getting started

You need [uv](https://docs.astral.sh/uv/). Python 3.12 and every dependency are installed by the first command.

```bash
uv sync
uv run python solve.py            # web interface at http://127.0.0.1:8000
```

Before running, copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` to the
key we sent you. An existing environment variable takes precedence.

`solve.py` opens the web interface in your browser:

1. **Start**: one prompt box. Attach the drawing with the paperclip (or drop it anywhere), add a
   note if you like, pick the model (Claude Haiku 5.5 by default; Sonnet 5.5, Opus 5.5 or GPT-6 Luna), press
   Enter. For now only the PDF drives the pipeline; the note is kept with the job.
2. **Working**: the full pipeline (`main.py`, every agent on the chosen model) runs in its own process. A live
   line shows what it is doing now; below it, each step (reading each view, settling disagreements, each build
   attempt, each check against the drawing) appears as it starts, with its time and cost. Open a step to see
   what it did, or **Raw log** for everything.
3. **Done**: the page splits. Left, a summary (built or not, checked against the drawing or not, time,
   cost) and a chat to ask about the part (sizes, features, what was checked, what is uncertain); right,
   the part in 3D with **Download STEP**, and a **Projections** tab: the model projected into every view of
   the drawing, side by side with the drawn view or overlaid on it (drawing black, model red; orange =
   drawn but missing in the model, purple = in the model but not drawn), with how much of each view matches
   and every dimension callout checked. The pipeline's own checks are listed there; **Run projections**
   projects the current model again (no model calls, about 30 s). Vector PDFs only for now: a scanned
   drawing cannot be projected yet.

Several drawings can run at once: three at a time by default (`--parallel N`), the rest wait in the queue.
**New drawing** goes back to the start page, which lists recent jobs to reopen. A job can be stopped; its
run folder is kept. Jobs are saved under `runs/ui/<job>/`
(the drawing, `job.json`, `log.txt`), and each job's run folder (plan, build
attempts, validation, `model.step`, `trace.jsonl`, `result.json`) is
`runs/ui-<job>-<time>/`, so nothing is lost if the server stops: the list comes
back on restart, and a job that was running is marked interrupted. The interface
works offline (three.js is served from `web/vendor/`); only OpenRouter is called.

One drawing from the command line, without the interface:
`uv run python main.py cases/tier1/01/drawing.pdf`.

The pipeline runs the scripts the model writes on your machine, in a child process, with no sandbox.

## Cases

`cases/` contains **four tiers of increasing difficulty**, from `tier1` to `tier4`,
with **eight cases per tier**, numbered `01` to `08`. Each case contains a drawing
(`drawing.pdf`) and its reference STEP part (`reference.step`).

Everything here is a starting point. Choose your own interface and document how to run it (see CHALLENGE.md).
