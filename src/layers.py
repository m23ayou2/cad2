"""Plan regions for 2.5D machined parts, for build scripts (`from layers import ...`).

A machined part is a stack of Z layers; in each layer, the material is one plan region (the plan outline minus
the channels, pockets, bores and slots open in that layer). Regions are built here in 2D (shapely, XY plane, mm):
rectangles in a turned frame for arms drawn at an angle, circles, slots, unions and differences, and R... fillets
between features. Each layer is then extruded once (stack), so the CAD does almost no booleans.
"""
import math

import cadquery as cq
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


def rect(x0, y0, x1, y1):
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def circle(cx, cy, r):
    return Point(cx, cy).buffer(r, 48)


def polygon(points):
    return Polygon(points)


def slot(p0, p1, width):
    """An obround: the line p0-p1 widened to `width`, round ends (a slot 8 wide with R4 ends: width=8)."""
    return LineString([tuple(p0), tuple(p1)]).buffer(width / 2, 24)


def frame_point(s, t, angle=0.0, origin=(0.0, 0.0)):
    """The point at `s` along an axis through `origin` turned `angle` degrees counterclockwise from +X, and `t`
    square to it (+ to the left of the axis)."""
    a = math.radians(angle)
    return origin[0] + s * math.cos(a) - t * math.sin(a), origin[1] + s * math.sin(a) + t * math.cos(a)


def frame_rect(along, across, angle=0.0, origin=(0.0, 0.0)):
    """A rectangle in a turned frame: `along` = (s0, s1) on the axis, `across` = (t0, t1) square to it."""
    (s0, s1), (t0, t1) = along, across
    return Polygon([frame_point(s, t, angle, origin) for s, t in ((s0, t0), (s1, t0), (s1, t1), (s0, t1))])


def union(*regions):
    return unary_union(regions)


def minus(region, *cuts):
    return region.difference(unary_union(cuts)) if cuts else region


def mirror_x(region):
    """Mirrored about the Y axis (the other half of a part symmetric left-right)."""
    return affinity.scale(region, xfact=-1, yfact=1, origin=(0, 0))


def round_inside(region, r):
    """Fillets of radius r in the region's inside corners, e.g. 'R5 TYP' joining a lug to the body."""
    return region.buffer(r, 24).buffer(-r, 24)


def _wire(coords, z):
    return cq.Wire.makePolygon([cq.Vector(x, y, z) for x, y in list(coords)[:-1]], close=True)


def extrude(region, z0, z1) -> cq.Workplane:
    """The region (its holes kept) as a solid from z0 to z1."""
    solids = []
    for poly in getattr(region, "geoms", [region]):
        if poly.is_empty or poly.area < 1e-6:
            continue
        poly = orient(poly)
        face = cq.Face.makeFromWires(_wire(poly.exterior.coords, z0), [_wire(i.coords, z0) for i in poly.interiors])
        solids.append(cq.Solid.extrudeLinear(face, cq.Vector(0, 0, z1 - z0)))
    if not solids:
        raise ValueError(f"empty region between z {z0} and {z1}")
    shape = solids[0] if len(solids) == 1 else solids[0].fuse(*solids[1:]).clean()
    return cq.Workplane("XY").add(shape)


def stack(layers) -> cq.Workplane:
    """Extrude [(z0, z1, region), ...] and fuse the layers into one part."""
    parts = [extrude(region, z0, z1).val() for z0, z1, region in layers]
    shape = parts[0] if len(parts) == 1 else parts[0].fuse(*parts[1:]).clean()
    return cq.Workplane("XY").add(shape)
