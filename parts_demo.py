"""Run only the parts pipeline (views -> points -> edges -> parts -> matching) on one drawing, no CAD.

    uv run python parts_demo.py [cases/tier4/01/drawing.pdf]

Outputs go to runs/<tier>-<case>-parts-<timestamp>/. The full drawing-to-STEP run is main.py.
"""
import sys
from pathlib import Path

from src.part_views import run


def main():
    drawing = Path(sys.argv[1] if len(sys.argv) > 1 else "cases/tier4/01/drawing.pdf")
    run(drawing, Path(__file__).resolve().parent / "runs")


if __name__ == "__main__":
    main()
