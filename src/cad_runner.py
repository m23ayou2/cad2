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


def run(script: str, step: str) -> dict:
    import cadquery as cq

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

    report = measure(shape)
    problems = []
    if report["solids"] == 0:
        problems.append("the result contains no solid")
    elif report["solids"] > 1:
        problems.append(f"the result is {report['solids']} separate solids; features that should be joined "
                        f"do not touch or overlap (check positions and heights)")
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
