"""Volume IoU between a model STEP and a reference STEP, independent of the frame each was built in.

Both solids are voxelised at the same pitch by casting rays through the B-rep (exact, no tessellation). The model's
grid is tried in each of the 24 axis-aligned rotations (and the 24 mirrored ones, reported separately), and for
each the best shift is found by FFT cross-correlation with the reference grid. IoU = overlap / union, in voxels.
Takes 1 to 10 minutes; thin sheet metal needs a fine pitch (--res 200 or more).

    uv run python tools/voxel_iou.py <model.step> <reference.step> [--res 160]
"""
import argparse
import itertools
import json

import cadquery as cq
import numpy as np
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector


def load(path):
    solids = cq.importers.importStep(str(path)).solids().vals()
    return cq.Compound.makeCompound(solids) if len(solids) > 1 else solids[0]


def voxelise(shape, pitch):
    """Occupancy grid (nx, ny, nz) of `shape`, rays cast along +Z through cell centres."""
    bb = shape.BoundingBox()
    lo = np.array([bb.xmin, bb.ymin, bb.zmin]) - pitch
    hi = np.array([bb.xmax, bb.ymax, bb.zmax]) + pitch
    n = np.ceil((hi - lo) / pitch).astype(int)
    grid = np.zeros(n, dtype=bool)
    hit = IntCurvesFace_ShapeIntersector()
    hit.Load(shape.wrapped, 1e-6)
    zc = lo[2] + (np.arange(n[2]) + 0.5) * pitch
    jitter = pitch * np.array([0.0137, 0.0291])  # keeps rays off exact edges
    for i in range(n[0]):
        x = lo[0] + (i + 0.5) * pitch + jitter[0]
        for j in range(n[1]):
            y = lo[1] + (j + 0.5) * pitch + jitter[1]
            hit.Perform(gp_Lin(gp_Pnt(x, y, lo[2] - pitch), gp_Dir(0, 0, 1)), -1e9, 1e9)
            k = hit.NbPnt()
            if k < 2:
                continue
            ts = np.sort([hit.Pnt(m + 1).Z() for m in range(k)])
            ts = ts[np.concatenate(([True], np.diff(ts) > 1e-7))]  # a ray through a shared edge hits both faces
            for a, b in zip(ts[0::2], ts[1::2]):
                grid[i, j, (zc >= a) & (zc <= b)] = True
    return grid


def orientations(grid, mirror):
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product((False, True), repeat=3):
            parity = 1 if perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1
            if (parity * (-1) ** sum(flips) == -1) != mirror:
                continue
            g = np.transpose(grid, perm)
            for axis, flip in enumerate(flips):
                if flip:
                    g = np.flip(g, axis)
            yield g


def best_overlap(a, b):
    """Max over integer shifts of |a ∩ shift(b)|, by FFT cross-correlation."""
    shape = [sa + sb for sa, sb in zip(a.shape, b.shape)]
    fa = np.fft.rfftn(a.astype(np.float32), shape, axes=(0, 1, 2))
    fb = np.fft.rfftn(b[::-1, ::-1, ::-1].astype(np.float32), shape, axes=(0, 1, 2))
    return float(np.fft.irfftn(fa * fb, shape, axes=(0, 1, 2)).max())


def score(model_path, ref_path, res=160) -> dict:
    m, r = load(model_path), load(ref_path)
    rb, mb = r.BoundingBox(), m.BoundingBox()
    pitch = max(rb.xlen, rb.ylen, rb.zlen) / res
    R, M = voxelise(r, pitch), voxelise(m, pitch)
    nr, nm = R.sum(), M.sum()
    out = {"pitch_mm": round(pitch, 3), "model_volume": round(m.Volume(), 1), "ref_volume": round(r.Volume(), 1),
           "model_bbox": sorted(round(v, 1) for v in (mb.xlen, mb.ylen, mb.zlen)),
           "ref_bbox": sorted(round(v, 1) for v in (rb.xlen, rb.ylen, rb.zlen))}
    for mirror in (False, True):
        best = 0.0
        for g in orientations(M, mirror):
            ov = best_overlap(R, g)
            best = max(best, ov / (nr + nm - ov))
        out["iou_mirrored" if mirror else "iou"] = round(best, 3)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("reference")
    ap.add_argument("--res", type=int, default=160, help="voxels along the reference's longest side")
    args = ap.parse_args()
    print(json.dumps(score(args.model, args.reference, args.res)))
