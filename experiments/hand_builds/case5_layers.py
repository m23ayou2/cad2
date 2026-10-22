"""Tier 4 case 5 rebuilt by hand as Z layers of plan regions (src/layers.py), to test the layers recipe (IoU 0.865
against the reference).

A demonstration, not part of the pipeline: the dimensions are the drawing's, but the layers (feet, floor, walls)
were confirmed by slicing the reference STEP, so the score is what the layers approach reaches once the drawing is
read correctly. Not on the drawing: the slot length and the channel running past the arm end.

Frame: origin at the bore centre, Z = 0 on the underside, X along the bottom view's 181.47, Y along its 108.96.
  Z 0..10   feet: the two arms, minus the Ø100 recess from below ("2.00": the centre floor is 12 - 2 = 10 up)
  Z 10..12  floor: the whole outline, minus the Ø76 bore
  Z 12..40  walls: the outline, minus each arm's channel (61.41 wide, open at the arm end) and the bore
  slots 8 wide (R4) at 62.43 along each arm, from Z 4 up; Ø5 4X through the lugs

    uv run python experiments/hand_builds/case5_layers.py [out.step]
"""
import math
import sys
from pathlib import Path

import cadquery as cq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from layers import circle, frame_point, frame_rect, minus, mirror_x, round_inside, slot, stack, union  # noqa: E402

ALPHA = 90 - 74.63                     # arms: A-A at 74.63 deg from the vertical
W_ARM, W_CHANNEL = 85.41, 61.41        # walls (85.41 - 61.41) / 2 = 12, as in section B-B
L_ARM, S_SLOT = 81.43, 62.43           # arm end and slots, from the centre along A-A
# arm offset across A-A, so the part measures 108.96 deep (the Ø100 ring's bottom at -50)
OFFSET = (108.96 - 50 - L_ARM * math.sin(math.radians(ALPHA))) / math.cos(math.radians(ALPHA)) - W_ARM / 2

right_arm = frame_rect((0, L_ARM), (OFFSET - W_ARM / 2, OFFSET + W_ARM / 2), ALPHA)
right_channel = frame_rect((0, L_ARM + 5), (OFFSET - W_CHANNEL / 2, OFFSET + W_CHANNEL / 2), ALPHA)
right_slot = slot(frame_point(S_SLOT, OFFSET - W_CHANNEL / 2 + 8, ALPHA),
                  frame_point(S_SLOT, OFFSET + W_CHANNEL / 2 - 8, ALPHA), 8)
arms = union(right_arm, mirror_x(right_arm))
channels = union(right_channel, mirror_x(right_channel))
slots = union(right_slot, mirror_x(right_slot))
lug_xy = [(s * 52.34 * math.tan(math.radians(20.91)), 52.34) for s in (-1, 1)] + \
         [(s * 44.33 * math.tan(math.radians(38.29)), -44.33) for s in (-1, 1)]
lug_holes = union(*[circle(x, y, 2.5) for x, y in lug_xy])
outline = round_inside(union(arms, circle(0, 0, 50), *[circle(x, y, 5.5) for x, y in lug_xy]), 5)  # R5.00 TYP
bore = circle(0, 0, 38)

result = stack([
    (0, 4, minus(arms, circle(0, 0, 50), lug_holes)),                # feet, below the slots
    (4, 10, minus(arms, circle(0, 0, 50), slots, lug_holes)),        # feet, slots from Z 4
    (10, 12, minus(outline, bore, slots, lug_holes)),                # floor
    (12, 40, minus(outline, channels, bore, slots, lug_holes)),      # walls
])

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "case5_layers.step"
    cq.exporters.export(result, out)
    print(f"{out}: volume {result.val().Volume():.0f} mm3")
