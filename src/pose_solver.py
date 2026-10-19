"""
pose_solver.py
==============
Find how a STEP model must be oriented so that its projections reproduce the views of a drawing.

Search space: the 24 orientations of a cube (every way to turn the model so its axes stay axis-aligned).
For each orientation the model is projected (hidden-line removal) / cut for every drawing view, and the
projected edges are compared with the edges drawn in that view (symmetric mean nearest-point distance, mm).
The orientation must satisfy ALL views at once.

Needs: cadquery-ocp, numpy, scipy (optional), views.py
"""
import itertools
import math
import sys

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.Bnd import Bnd_Box
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax2, gp_Ax2 as _Ax2, gp_Dir, gp_Pnt, gp_Trsf

from views import (PT_PER_MM, OrthographicView, PictorialView, Role, SectionView, SheetDrawing,
                   View)

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


# ------------------------------------------------------------------ model side
def load_step(path):
    r = STEPControl_Reader()
    if r.ReadFile(path) != 1:
        raise IOError(f"cannot read {path}")
    r.TransferRoots()
    return r.OneShape()          # OCCT converts to mm


def bbox(shape):
    b = Bnd_Box()
    BRepBndLib.Add_s(shape, b)
    lo, hi = b.CornerMin(), b.CornerMax()
    return np.array([lo.X(), lo.Y(), lo.Z()]), np.array([hi.X(), hi.Y(), hi.Z()])


def volume(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def transform(shape, M, t=(0, 0, 0)):
    tr = gp_Trsf()
    tr.SetValues(M[0][0], M[0][1], M[0][2], t[0], M[1][0], M[1][1], M[1][2], t[1], M[2][0], M[2][1], M[2][2], t[2])
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def cube_rotations():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[i, p] = s
            if round(np.linalg.det(M)) == 1:
                out.append(M)
    return out            # 24 matrices


def view_basis(look, up):
    look, up = np.array(look, float), np.array(up, float)
    look /= np.linalg.norm(look)
    up = up - np.dot(up, look) * look      # an isometric "up" (0, 0, 1) is not perpendicular to the look
    up /= np.linalg.norm(up)
    right = np.cross(look, up)
    return np.array([right, up, -look])       # rows: model -> (x right, y up, z towards viewer)


def project_edges(shape, step=0.1, polylines=False):
    """Visible edges of `shape` seen from +Z (camera at +Z looking -Z), as 2D points (mm), or with `polylines` as
    one polyline per edge (to draw them as continuous lines)."""
    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))))
    algo.Update()
    algo.Hide()
    hs = HLRBRep_HLRToShape(algo)
    pts = []
    for comp in (hs.VCompound(), hs.Rg1LineVCompound(), hs.OutLineVCompound()):
        if comp is None or comp.IsNull():
            continue
        ex = TopExp_Explorer(comp, TopAbs_EDGE)
        while ex.More():
            c = BRepAdaptor_Curve(TopoDS.Edge_s(ex.Current()))
            u0, u1 = c.FirstParameter(), c.LastParameter()
            n = max(2, int(GCPnts_AbscissaPoint.Length_s(c) / step) + 1)
            edge = [(c.Value(u0 + (u1 - u0) * k / n).X(), c.Value(u0 + (u1 - u0) * k / n).Y()) for k in range(n + 1)]
            if polylines:
                pts.append(np.array(edge))
            else:
                pts += edge
            ex.Next()
    if polylines:
        return pts
    return np.array(pts) if pts else np.zeros((0, 2))


def model_view_points(shape, spec, polylines=False):
    """spec = dict(look, up, cut_normal=None, cut_offset=0) applied in the *drawing* frame.
    cut_offset: mm from the model's centre along the cut normal, where the cutting plane lies."""
    lo, hi = bbox(shape)
    if spec.get("cut_normal") is not None:
        n = np.array(spec["cut_normal"], float)
        c = (lo + hi) / 2 + n * float(spec.get("cut_offset") or 0.0)
        big = float(np.max(hi - lo)) * 4 + 10
        # remove the half on the camera side of the plane (camera looks along -n)
        axis = int(np.argmax(np.abs(n)))
        size = np.array([big, big, big])
        whole = volume(shape)
        # a plane exactly on the symmetry plane meets cylinder seams and coincident faces, and the
        # boolean then silently removes nothing: shift it by a negligible amount until the cut works
        for eps in (1e-3, 1e-2, 1e-1):
            origin = c.copy()
            origin[axis] += eps if n[axis] > 0 else -eps
            corner = origin - size / 2
            corner[axis] = origin[axis] if n[axis] > 0 else origin[axis] - big
            box = BRepPrimAPI_MakeBox(gp_Pnt(*corner), *size).Shape()
            cut = BRepAlgoAPI_Cut(shape, box).Shape()
            if volume(cut) < 0.999 * whole:
                shape = cut
                break
        else:
            raise ValueError("the section cut failed: the cutting plane removed no material")
    shape = transform(shape, view_basis(spec["look"], spec["up"]))
    return project_edges(shape, polylines=polylines)


# ------------------------------------------------------------------ drawing side
def drawing_points(view: View, step=0.1):
    """Visible edges of a drawn view, in mm real-size, y up, detached marks (cut-line stubs) removed."""
    s = view.scale.value
    if not s:
        raise ValueError(f"{view.view_id}: scale unknown")
    x0, y0, x1, y1 = view.outline_pt
    k = 1.0 / PT_PER_MM / s
    inside = lambda x, y: x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1
    pts = []
    for sg in view.geometry.segments:
        if sg.role != Role.VISIBLE or not (inside(sg.x0, sg.y0) and inside(sg.x1, sg.y1)):
            continue
        n = max(1, int(sg.length * k / step))
        for i in range(n + 1):
            t = i / n
            pts.append((sg.x0 + (sg.x1 - sg.x0) * t, sg.y0 + (sg.y1 - sg.y0) * t))
    for c in view.geometry.curves:
        if c.role != Role.VISIBLE or not inside(c.cx, c.cy):
            continue
        if c.kind == "circle":
            n = max(8, int(2 * math.pi * c.r * k / step))
            for i in range(n):
                a = 2 * math.pi * i / n
                pts.append((c.cx + c.r * math.cos(a), c.cy + c.r * math.sin(a)))
        elif c.pts is not None:                      # arcs, ellipses, splines: resample the polyline
            Q = np.asarray(c.pts)
            for a, b in zip(Q[:-1], Q[1:]):
                n = max(1, int(math.hypot(*(b - a)) * k / step))
                for i in range(n):
                    pts.append(tuple(a + (b - a) * i / n))
    P = np.array(pts, float) * k
    P[:, 1] *= -1                 # PDF y is down
    return P


def chamfer(A, B):
    """A = drawing points, B = model points (aligned on bounding-box centres).
    Returns (mean symmetric distance, max drawn->model, max model->drawn).
    drawn->model large : something is drawn that the model does not have (a real mismatch).
    model->drawn large : the model has an edge the drawing omits (often just drafting convention)."""
    if len(A) == 0 or len(B) == 0:
        return float("inf"), float("inf"), float("inf")
    A = A - (A.min(0) + A.max(0)) / 2
    B = B - (B.min(0) + B.max(0)) / 2
    if cKDTree:
        da = cKDTree(B).query(A)[0]
        db = cKDTree(A).query(B)[0]
    else:
        da = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1)).min(1)
        db = np.sqrt(((B[:, None] - A[None]) ** 2).sum(-1)).min(1)
    return float((da.mean() + db.mean()) / 2), float(da.max()), float(db.max())


# ------------------------------------------------------------------ drawing view specs (drawing frame)
def view_specs(sheet: SheetDrawing):
    """Projection of each linked view *in the drawing's own frame* (not the model's)."""
    out = []
    for v in sheet.views:
        if isinstance(v, PictorialView) or not v.scale:
            continue
        try:
            sp = v.projection_spec()
        except Exception:
            continue
        out.append((v, {"look": sp.look, "up": sp.up,
                        "cut_normal": sp.cut_plane.normal if sp.cut_plane else None}))
    return out


def solve(step_path, sheet: SheetDrawing, shape=None, tol_mm=0.15):
    shape = shape if shape is not None else load_step(step_path)
    specs = view_specs(sheet)
    drawn = {v.view_id: drawing_points(v) for v, _ in specs}
    results = []
    for R in cube_rotations():
        cand = transform(shape, R)
        per = {}
        for v, sp in specs:
            per[v.view_id] = chamfer(drawn[v.view_id], model_view_points(cand, sp))   # (drawing, model)
        total = sum(m for m, _, _ in per.values())
        results.append({"R": R, "total": total, "per_view": per})
    results.sort(key=lambda r: r["total"])
    best = results[0]["total"]
    tied = [r for r in results if r["total"] <= best + tol_mm]   # poses the views cannot tell apart
    return results, tied, tol_mm


def describe_rotation(R):
    """Say where each model axis ends up in the drawing frame."""
    names = ["X", "Y", "Z"]
    axes = ["x", "y", "z"]
    parts = []
    for i in range(3):
        col = R[:, i]
        j = int(np.argmax(np.abs(col)))
        parts.append(f"model {names[i]} -> drawing {'+' if col[j] > 0 else '-'}{axes[j].upper()}")
    return ", ".join(parts)


if __name__ == "__main__":
    step_path, folder = sys.argv[1], sys.argv[2]
    sheet = SheetDrawing.from_folder(folder).link_views()
    results, tied, tol = solve(step_path, sheet)
    print(f"{len(results)} orientations tried; best total residual {results[0]['total']:.3f} mm")
    for r in results[:8]:
        per = "  ".join(f"{k.split('_view')[-1][:14]}={m:.3f}" for k, (m, _, _) in r["per_view"].items())
        print(f"  {r['total']:7.3f}  {per}   {describe_rotation(r['R'])}")
