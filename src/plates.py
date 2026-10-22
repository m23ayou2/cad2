"""Flat plates drawn at full size in one view, for build scripts (`from plates import plate`).

A thin-walled part drawn formed, with no flat pattern to fold, is built from its plates: each one is extruded from
the outline of the view that shows it at full size. The profile is in that view's axes, so a script never builds
the workplane itself (CadQuery's "XZ" workplane extrudes towards -Y, an easy sign to get wrong).
"""
import cadquery as cq


def plate(profile, direction, start, thickness):
    """`profile` [[u, v], ...] in the axes of the view that shows the plate at full size (direction along Z:
    (x, y); along Y: (x, z); along X: (y, z)), extruded `thickness` from `start`, the feature's position along
    that axis, in `direction` ("+Y", "-X", ...)."""
    axis = direction.strip()[-1].upper()
    low = start - thickness if direction.strip().startswith("-") else start
    planes = {"Z": cq.Plane((0, 0, low), (1, 0, 0), (0, 0, 1)),
              "Y": cq.Plane((0, low + thickness, 0), (1, 0, 0), (0, -1, 0)),  # local v = +Z needs normal -Y
              "X": cq.Plane((low, 0, 0), (0, 1, 0), (1, 0, 0))}
    return cq.Workplane(planes[axis]).polyline([tuple(p) for p in profile]).close().extrude(thickness)
