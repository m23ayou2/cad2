"""Tier 4 case 3 rebuilt by hand as flat plates, to test the plate recipe (IoU 0.944 against the reference).

A demonstration, not part of the pipeline: every number is from the drawing, but the part's structure (base plate,
two gussets, a back wall between them) was confirmed by cutting the reference STEP, so the score is what the
plates approach reaches once the drawing is read correctly.

Frame: X along the front view (0 = the hook tips' left end, base plate 29.5..138.5), Y along the top view's
vertical (0 = bottom edge, 150 = top edge), Z up (0 = underside of the base). t = 5 (right view: "5"; gussets
33..38 and 112..117).

    uv run python experiments/hand_builds/case3_plates.py [out.step]
"""
import sys
from pathlib import Path

import cadquery as cq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from plates import plate  # noqa: E402

T = 5.0
BIG = 400.0

# base plate: top view outline (29.5..138.5 x 0..150), window, notches, holes
base = cq.Workplane("XY").box(138.5 - 29.5, 150, T, centered=False).translate((29.5, 0, 0))
# window: 15.9 from the base's left edge, 89.1 to its right edge; 48.9..101.1; R6 corners
window = (cq.Workplane("XY").center((45.4 + 118.6) / 2, (48.9 + 101.1) / 2)
          .rect(118.6 - 45.4, 101.1 - 48.9).extrude(BIG).translate((0, 0, -BIG / 2))
          .edges("|Z").fillet(6))
base = base.cut(window)
# scallops: one notch in each long edge, 26.3 deep (bottom: 26.3; top: 150 - 123.7), R11.1 corners
for y0 in (0.0, 150.0):
    notch = (cq.Workplane("XY").center((58.0 + 108.0) / 2, y0)
             .rect(108.0 - 58.0, 2 * 26.3).extrude(BIG).translate((0, 0, -BIG / 2))
             .edges("|Z").fillet(11.1))
    base = base.cut(notch)
# holes: 2x Ø6.3 at x 40.2, 2x Ø5 at x 118.8; y 18.7 and 150 - 18.7
for x, d in ((40.2, 6.3), (118.8, 5.0)):
    for y in (18.7, 150 - 18.7):
        base = base.cut(cq.Workplane("XY").center(x, y).circle(d / 2).extrude(BIG).translate((0, 0, -BIG / 2)))

# gussets: the front view's outline (full size), 5 thick at y 33..38 and 112..117
GUSSET = [(0, 47.5), (0, 56.5), (1.2, 57.6), (5.6, 51.5), (9.0, 48.0), (13.0, 47.0), (17.0, 48.0), (20.7, 52.0),
          (20.7, 63.0), (21.7, 64.0),                     # right prong tip, R1 (64 = top, from the right view)
          (108.9, 24.8),                                   # 155.2 deg slope down to 64 - 39.2
          (128.4, 24.8), (135.5, 21.9), (138.4, 14.8),     # R10 corner
          (138.4, 0.0), (29.5, 0.0),
          (25.0, 15.0), (20.7, 29.0), (16.0, 36.0), (13.0, 38.0), (9.0, 39.0), (2.0, 45.5)]  # 160.3 deg, R8.3
gussets = [plate(GUSSET, "+Y", y, T) for y in (33.0, 112.0)]

# back wall: the right view's rectangle between the gussets (y 38..112), 5 thick at the base's left edge,
# its top cut along the gussets' slope (58.2 inside, 60.6 outside in the right view)
prism = plate(GUSSET, "+Y", -BIG / 2, BIG)
wall = plate([(38, 0), (112, 0), (112, 64), (38, 64)], "+X", 29.5, T).intersect(prism)

result = base
for g in gussets:
    result = result.union(g)
result = result.union(wall)

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "case3_plates.step"
    cq.exporters.export(result, out)
    print(f"{out}: volume {result.val().Volume():.0f} mm3")
