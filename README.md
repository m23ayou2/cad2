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

1. **Jobs**: add one or more PDF drawings (or drop them on the page), pick the model (Claude Haiku 5.5 by
   default; Sonnet 5.5, Opus 5.5 or GPT-6 Luna), optionally add a note, and start. The table lists every job
   with its state, result, how much of the drawing the part matches, open issues, time, cost and model.
2. **A job**: the left column is its title block (drawing, state, result, model, time, cost), **Download
   STEP**, an assessment once it is finished, and its steps (each agent run and check, with time and cost;
   open one to see what it did, or the full log). The right side shows the **Drawing** while the part is
   built, then the **3D model** and the **Projections**: the model projected into every view of the drawing,
   side by side with the drawn view or overlaid on it (drawing black, model red; orange = drawn but missing
   in the model, purple = in the model but not drawn), how much of each view matches, and every dimension
   callout checked. **Run projections** projects the current model again (no model calls, about 30 s; vector
   PDFs only for now). Under the steps, ask questions about the finished part.

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
