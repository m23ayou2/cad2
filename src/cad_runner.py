"""Run one CadQuery script in its own process: export STEP and report measurements.

    python src/cad_runner.py script.py out.step report.json

The script must assign the final part to `result`. The report says whether the
part is one valid solid, and gives its bounding box, volume, face count, and the
cylindrical faces grouped by diameter, so the builder can check its own work.
"""
import json
import sys
import traceback
from collections import Counter


def measure(shape) -> dict:
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    bb = shape.BoundingBox()
    cylinders = Counter()
    for face in shape.Faces():
        if face.geomType() == "CYLINDER":
            # through the face adaptor: a face's own surface can be a trimmed surface with no .Cylinder()
            cylinders[round(2 * BRepAdaptor_Surface(face.wrapped).Cylinder().Radius(), 3)] += 1
    return {
        "solids": len(shape.Solids()),
        "valid": shape.isValid(),
        "bbox": {"min": [round(bb.xmin, 3), round(bb.ymin, 3), round(bb.zmin, 3)],
                 "max": [round(bb.xmax, 3), round(bb.ymax, 3), round(bb.zmax, 3)],
                 "size": [round(bb.xlen, 3), round(bb.ylen, 3), round(bb.zlen, 3)]},
        "volume": round(shape.Volume(), 3),
        "faces": len(shape.Faces()),
        "cylinder_faces_by_diameter": {str(d): n for d, n in sorted(cylinders.items())},
    }


def make_union_order_safe(cq):
    """CadQuery's Workplane.union silently drops the part being added when the body is already in several
    pieces (e.g. a barrel unioned before the support that joins it to the plate): the volume does not change and
    no error is raised. When the body has more than one solid, fuse all of them and the new part in one
    operation instead, so a bridging part joins every piece it touches whatever order the script uses."""
    original = cq.Workplane.union
    if getattr(original, "order_safe", False):
        return

    def union(self, toUnion=None, clean=True, glue=False, tol=None):
        mine = [s for v in self.vals() if isinstance(v, cq.Shape) for s in v.Solids()]
        if len(mine) <= 1 or toUnion is None:
            return original(self, toUnion, clean=clean, glue=glue, tol=tol)
        others = toUnion.vals() if isinstance(toUnion, cq.Workplane) else [toUnion]
        theirs = [s for v in others if isinstance(v, cq.Shape) for s in v.Solids()]
        # copies: pieces that come out of an earlier boolean together make a new fuse ignore the added part
        mine = [s.copy() for s in mine]
        fused = mine[0].fuse(*mine[1:], *theirs, glue=glue, tol=tol)
        return self.newObject([fused.clean() if clean else fused])

    union.order_safe = True
    cq.Workplane.union = union


def run(script: str, step: str) -> dict:
    import cadquery as cq

    make_union_order_safe(cq)
    namespace = {"__name__": "__main__", "__file__": script}
    with open(script, encoding="utf-8") as f:
        exec(compile(f.read(), script, "exec"), namespace)
    result = namespace.get("result")
    if result is None:
        raise ValueError("The script must assign the final part to `result`")
    if isinstance(result, cq.Workplane):
        objects = [o for o in result.vals() if isinstance(o, cq.Shape)]
        if not objects:
            raise ValueError("`result` contains no shape")
        shape = objects[0] if len(objects) == 1 else cq.Compound.makeCompound(objects)
    elif isinstance(result, cq.Shape):
        shape = result
    else:
        raise ValueError(f"`result` must be a cq.Workplane or cq.Shape, not {type(result).__name__}")

    notes = []
    solids = shape.Solids()
    if len(solids) > 1:
        # A union done in the wrong order leaves touching pieces apart (a part fused into a compound of two
        # disjoint solids joins only one of them): one fuse of all the solids merges every piece that touches.
        copies = [s.copy() for s in solids]
        fused = copies[0].fuse(*copies[1:]).clean()
        if len(fused.Solids()) < len(solids):
            notes.append(f"{len(solids)} touching pieces were fused into {len(fused.Solids())} solid(s)")
            shape = fused.Solids()[0] if len(fused.Solids()) == 1 else fused
    report = measure(shape)
    report["notes"] = notes
    problems = []
    if report["solids"] == 0:
        problems.append("the result contains no solid")
    elif report["solids"] > 1:
        pieces = []
        for s in shape.Solids():
            b = s.BoundingBox()
            pieces.append(f"{s.Volume():.0f} mm3 at x {b.xmin:.1f}..{b.xmax:.1f}, y {b.ymin:.1f}..{b.ymax:.1f}, "
                          f"z {b.zmin:.1f}..{b.zmax:.1f}")
        problems.append(f"the result is {report['solids']} separate solids that do not touch: "
                        + "; ".join(pieces) + ". Check the position and size of the feature that should join them")
    if not report["valid"]:
        problems.append("the shape is not valid (self-intersection or a failed boolean)")
    report["problems"] = problems
    report["ok"] = not problems
    if report["ok"]:
        cq.exporters.export(shape, step)
    return report


if __name__ == "__main__":
    script_path, step_path, report_path = sys.argv[1:4]
    try:
        out = run(script_path, step_path)
    except Exception as error:
        tb = traceback.extract_tb(error.__traceback__)
        lines = [f"line {fr.lineno}: {fr.line}" for fr in tb if fr.filename == script_path]
        out = {"ok": False, "error": f"{type(error).__name__}: {error}", "script_lines": lines[-3:]}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
