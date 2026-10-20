"""Score a STEP part against a reference STEP: volume, bounding box and best-aligned volume IoU.

    uv run python src/score_step.py model.step cases/tier4/01/reference.step

The model may sit anywhere and face any way, so both solids are moved to their centre of mass and the IoU is the
best over the 24 axis-aligned rotations of the model. Prints one JSON object.
"""
import itertools
import json
import sys

import cadquery as cq
import numpy as np


def load(path: str) -> cq.Shape:
    shape = cq.importers.importStep(path).val()
    return shape.translate(-shape.Center()) if hasattr(shape, "Center") else shape


def rotations() -> list[np.ndarray]:
    """The 24 proper rotations that map the axes onto the axes."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                m[row, col] = s
            if np.isclose(np.linalg.det(m), 1):
                out.append(m)
    return out


def rotate(shape: cq.Shape, m: np.ndarray) -> cq.Shape:
    from OCP.gp import gp_GTrsf, gp_Mat, gp_XYZ, gp_Trsf
    trsf = gp_Trsf()
    trsf.SetValues(*m[0], 0, *m[1], 0, *m[2], 0)
    return shape.transformShape(cq.Matrix(trsf))


def score(model_path: str, ref_path: str) -> dict:
    model, ref = load(model_path), load(ref_path)
    vm, vr = model.Volume(), ref.Volume()
    bm = sorted(model.BoundingBox().__getattribute__(a) for a in ("xlen", "ylen", "zlen"))
    br = sorted(ref.BoundingBox().__getattribute__(a) for a in ("xlen", "ylen", "zlen"))
    best = (0.0, None)
    for m in rotations():
        turned = rotate(model, m)
        turned = turned.translate(-turned.Center())
        try:
            inter = turned.intersect(ref).Volume()
        except Exception:
            continue
        iou = inter / (vm + vr - inter) if vm + vr - inter > 0 else 0.0
        if iou > best[0]:
            best = (iou, m.astype(int).tolist())
    return {"iou": round(best[0], 4), "rotation": best[1], "volume_mm3": round(vm, 1),
            "ref_volume_mm3": round(vr, 1), "volume_ratio": round(vm / vr, 3) if vr else None,
            "bbox_mm": [round(x, 1) for x in bm], "ref_bbox_mm": [round(x, 1) for x in br]}


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1], sys.argv[2])))
