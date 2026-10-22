"""Orthographic views of a STEP (looking along -Z, +Y and -X: filled silhouette and edges), to compare a model with
its drawing or reference by eye.

    uv run python tools/render_ortho.py <part.step> <out.png> [title]
"""
import sys

import cadquery as cq
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

VIEWS = [("looking along -Z (top): X right, Y up", 0, 1), ("looking along +Y (front): X right, Z up", 0, 2),
         ("looking along -X (side): Y right, Z up", 1, 2)]


def render(path: str, out: str, title: str = "") -> None:
    comp = cq.Compound.makeCompound(cq.importers.importStep(path).solids().vals())
    bb = comp.BoundingBox()
    verts, tris = comp.tessellate(max(0.05, max(bb.xlen, bb.ylen, bb.zlen) / 400), 0.2)
    V, T = np.array([[v.x, v.y, v.z] for v in verts]), np.array(tris)
    edges = [np.array([e.positionAt(t).toTuple() for t in np.linspace(0, 1, 24)]) for e in comp.Edges()]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (name, i, j) in zip(axes, VIEWS):
        ax.add_collection(PolyCollection(V[T][:, :, [i, j]], facecolors="#c9d6ea", edgecolors="none"))
        ax.add_collection(LineCollection([E[:, [i, j]] for E in edges], colors="k", linewidths=0.5))
        ax.set_xlim(V[:, i].min() - 3, V[:, i].max() + 3)
        ax.set_ylim(V[:, j].min() - 3, V[:, j].max() + 3)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3)
        ax.set_title(name, fontsize=10)
    fig.suptitle(f"{title or path}  bbox x {bb.xmin:.1f}..{bb.xmax:.1f}  y {bb.ymin:.1f}..{bb.ymax:.1f}  "
                 f"z {bb.zmin:.1f}..{bb.zmax:.1f}")
    fig.tight_layout()
    fig.savefig(out, dpi=85)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
