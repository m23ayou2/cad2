"""Direct checks on a STEP/BRep solid: no projection needed (cylinders, radii, depths, extents)."""
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box


def cylinder_inventory(shape):
    """-> list of dict(radius, axis(3), key, depth): one entry per distinct cylinder axis."""
    groups = {}
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current())
        s = BRepAdaptor_Surface(f)
        if s.GetType() == GeomAbs_Cylinder:
            cyl = s.Cylinder()
            ax = cyl.Axis()
            d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
            p = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
            if d[np.argmax(np.abs(d))] < 0:
                d = -d
            perp = p - np.dot(p, d) * d
            key = (round(cyl.Radius(), 3), tuple(np.round(np.abs(d), 3)), tuple(np.round(perp, 2)))
            b = Bnd_Box()
            BRepBndLib.Add_s(f, b)
            lo, hi = b.CornerMin(), b.CornerMax()
            a = int(np.argmax(np.abs(d)))
            rng = (min(lo.X(), lo.Y(), lo.Z()) if False else [lo.X(), lo.Y(), lo.Z()][a], [hi.X(), hi.Y(), hi.Z()][a])
            g = groups.setdefault(key, {"radius": cyl.Radius(), "axis": d, "lo": rng[0], "hi": rng[1]})
            g["lo"], g["hi"] = min(g["lo"], rng[0]), max(g["hi"], rng[1])
        ex.Next()
    return [{"radius": g["radius"], "axis": g["axis"], "depth": g["hi"] - g["lo"], "key": k} for k, g in groups.items()]


def check_annotations(shape, annotations, tol=0.05):
    """Match every Ø / R callout of the drawing to cylinders of the solid."""
    inv = cylinder_inventory(shape)
    rows, seen = [], set()
    for a in annotations:
        if a.kind not in ("diameter", "radius") or a.text in seen:
            continue
        seen.add(a.text)
        r = a.value / 2 if a.kind == "diameter" else a.value
        hits = [c for c in inv if abs(c["radius"] - r) <= tol]
        ok, why = True, []
        want = a.count
        if a.extra.get("explicit_count", False):
            if len(hits) != want:
                ok = False
                why.append(f"{len(hits)} found, drawing says {want}x")
        elif not hits:
            ok = False
            why.append("not found")
        if a.extra.get("depth") is not None and hits:
            depths = sorted({round(c["depth"], 2) for c in hits})
            if not any(abs(d - a.extra["depth"]) <= tol for d in depths):
                ok = False
                why.append(f"depth {depths} vs {a.extra['depth']}")
        if a.extra.get("thru") and hits and a.kind == "diameter":
            if any(has_bottom_face(shape, c) for c in hits):
                ok = False
                why.append("marked THRU but the hole has a bottom")
        if not hits and "not found" not in why:
            ok = False
            why.append("no cylinder of this size")
        rows.append((a.text, ok, "; ".join(why) or f"{len(hits)} cylinder(s) r={r:g}"))
    return rows


def _bbox(face):
    b = Bnd_Box()
    BRepBndLib.Add_s(face, b)
    lo, hi = b.CornerMin(), b.CornerMax()
    return np.array([lo.X(), lo.Y(), lo.Z()]), np.array([hi.X(), hi.Y(), hi.Z()])


def plane_faces(shape):
    """Axis-aligned planar faces: dict(axis, coord, lo, hi)."""
    out = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current())
        s = BRepAdaptor_Surface(f)
        if s.GetType() == GeomAbs_Plane:
            n = s.Plane().Axis().Direction()
            nv = np.array([n.X(), n.Y(), n.Z()])
            a = int(np.argmax(np.abs(nv)))
            if abs(nv[a]) > 0.999:
                lo, hi = _bbox(f)
                out.append({"axis": a, "coord": float(lo[a]), "lo": lo, "hi": hi})
        ex.Next()
    return out


def has_bottom_face(shape, cyl, tol=0.06):
    """A flat disc closing a cylinder (blind hole bottom) = plane normal to the axis, same centre, same size."""
    d = cyl["axis"]
    a = int(np.argmax(np.abs(d)))
    perp = np.array(cyl["key"][2])
    others = [i for i in range(3) if i != a]
    for p in plane_faces(shape):
        if p["axis"] != a:
            continue
        size = [p["hi"][i] - p["lo"][i] for i in others]
        centre = [(p["hi"][i] + p["lo"][i]) / 2 for i in others]
        if all(abs(sz - 2 * cyl["radius"]) <= tol for sz in size) and \
                all(abs(c - perp[i]) <= tol for c, i in zip(centre, others)):
            return True
    return False


def distance_candidates(shape):
    """Every distance a drafter could dimension in an axis-aligned part (weak evidence by itself)."""
    coords = [set(), set(), set()]
    lo, hi = _bbox(shape)
    for i in range(3):
        coords[i] |= {round(float(lo[i]), 3), round(float(hi[i]), 3)}
    for p in plane_faces(shape):
        coords[p["axis"]].add(round(p["coord"], 3))
    for c in cylinder_inventory(shape):
        a = int(np.argmax(np.abs(c["axis"])))
        perp = c["key"][2]
        for i in range(3):
            if i != a:
                coords[i].add(round(float(perp[i]), 3))
    cand = set()
    for cs in coords:
        cs = sorted(cs)
        cand |= {round(abs(x - y), 3) for x in cs for y in cs if x != y}
        cand |= {round(abs(x - y) / 2, 3) for x in cs for y in cs if x != y}
    cand |= {round(c["depth"], 3) for c in cylinder_inventory(shape)}
    return cand


def check_linear(shape, annotations, tol=0.05):
    cand = distance_candidates(shape)
    rows, seen = [], set()
    for a in annotations:
        if a.kind != "linear" or a.text in seen:
            continue
        seen.add(a.text)
        ok = any(abs(c - a.value) <= tol for c in cand)
        rows.append((a.text, ok, "a matching distance exists (value only, weak)" if ok else "no such distance in the model"))
    return rows
