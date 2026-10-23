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
   open one to see what it did, or the full log). The right side shows the **Drawing**, the **3D model** as
   soon as a first solid is built (it updates by itself after every build attempt, correction round or
   rebuild, and the camera stays where you left it; a dot on the tab marks a new model while you are on
   another tab), and, once the job is finished, the **Projections**: the model projected into every view of the drawing,
   side by side with the drawn view or overlaid on it (drawing black, model red; orange = drawn but missing
   in the model, purple = in the model but not drawn), how much of each view matches, and every dimension
   callout checked. **Run projections** projects the current model again (no model calls, about 30 s; vector
   PDFs only for now). Under the steps, ask questions about the finished part.

Several drawings can run at once: three at a time by default (`--parallel N`), the rest wait in the queue.
Drawings added together form a **batch**: starting them opens the batch page, which shows every part side by side
in one 3D view (each tinted like its row in the list), each part appearing as soon as its first solid is built and
updated after every build. Click a row to fly to that part, **Show all** to see them all again, **Open** for
that job's page. The start page lists recent batches under **See all in 3D**.
**New drawing** goes back to the start page, which lists recent jobs to reopen. A job can be stopped; its
run folder is kept. Jobs are saved under `runs/ui/<job>/`
(the drawing, `job.json`, `log.txt`), and each job's run folder (plan, build
attempts, validation, `model.step`, `trace.jsonl`, `result.json`) is
`runs/ui-<job>-<time>/`, so nothing is lost if the server stops: the list comes
back on restart, and a job that was running is marked interrupted. The interface
works offline (three.js is served from `web/vendor/`); only OpenRouter is called.

One drawing from the command line, without the interface:
`uv run python main.py cases/tier1/01/drawing.pdf`.

Several drawings from the command line, each in its own process, three at a time by default:
`uv run python main.py cases/tier1/*/drawing.pdf --parallel 4 --view`. Each drawing gets its own run folder as
above; the batch folder `runs/batch-<time>/` keeps one log per drawing, `batch.json` (result, match, issues,
time, cost and run folder of each) and `viewer.html`, every built part side by side in 3D, which `--view` opens
at the end. A summary table is printed when all are done. Ctrl+C stops them all.

The pipeline runs the scripts the model writes on your machine, in a child process, with no sandbox.

## Cases

`cases/` contains **four tiers of increasing difficulty**, from `tier1` to `tier4`,
with **eight cases per tier**, numbered `01` to `08`. Each case contains a drawing
(`drawing.pdf`) and its reference STEP part (`reference.step`).

Everything here is a starting point. Choose your own interface and document how to run it (see CHALLENGE.md).
