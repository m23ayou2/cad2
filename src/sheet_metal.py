"""Fold a flat sheet-metal blank into the formed part, so a build script does not do the folding arithmetic itself.

    from sheet_metal import fold_sheet
    result = fold_sheet(profile, thickness=3, k_factor=0.33, radius=3,
                        bends=[{"line": [[45.7, 50.2], [63.2, 50.2]], "angle": 90, "direction": "up"},
                               {"line": [[50.2, 86.7], [50.2, 99.3]], "angle": 90, "direction": "up", "radius": 1}],
                        holes=[{"at": [59.7, 86.7], "d": 7}])

The blank lies in z = 0..thickness, in the frame of its profile. Each bend line is the middle of its bend zone
(as drawn on a flat pattern): the zone, radians(angle) * (radius + k_factor * thickness) long, is centred on it. The
blank is split into panels at the bend zones; the panel that stays flat is the one containing `fixed` (default: the
largest), and every other panel is folded through each bend between it and the fixed one, so folds chain correctly
whatever their order. 'up' folds towards +z (the side the blank is drawn from), 'down' towards -z. Holes are cut in
the flat blank, so they fold with their panel.
"""
from __future__ import annotations

import math

import cadquery as cq
import numpy as np

BIG = 4000.0


def _box(origin, xdir, x0, x1, y0, y1, z0=-BIG, z1=BIG) -> cq.Solid:
    """A box in the frame (xdir, z x xdir, z) at origin: x in [x0, x1], y in [y0, y1], z in [z0, z1]."""
    plane = cq.Plane(origin=cq.Vector(*origin, 0), xDir=cq.Vector(*xdir, 0), normal=(0, 0, 1))
    return (cq.Workplane(plane).center((x0 + x1) / 2, (y0 + y1) / 2).rect(x1 - x0, y1 - y0)
            .extrude(z1 - z0).translate((0, 0, z0)).val())


def _inside(poly: np.ndarray, p: np.ndarray) -> bool:
    """Point in polygon (ray casting)."""
    x, y = p
    xs, ys = poly[:, 0], poly[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    cross = (ys > y) != (ys2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = xs + (y - ys) * (xs2 - xs) / (ys2 - ys)
    return bool(np.sum(cross & (x < xi)) % 2)


class _Bend:
    def __init__(self, spec: dict, thickness: float, k_factor: float, radius: float):
        (x0, y0), (x1, y1) = spec["line"]
        self.p0, p1 = np.array([x0, y0], float), np.array([x1, y1], float)
        length = float(np.linalg.norm(p1 - self.p0))
        if length < 1e-6:
            raise ValueError(f"bend line {spec['line']} has no length")
        self.u = (p1 - self.p0) / length
        self.n = np.array([-self.u[1], self.u[0]])           # one side; flipped towards the moving panel later
        self.length = length
        self.angle = float(spec.get("angle", 90))
        self.up = str(spec.get("direction", "up")).lower() == "up"
        self.r = float(spec.get("radius", radius))
        self.t = thickness
        self.ba = math.radians(self.angle) * (self.r + float(spec.get("k_factor", k_factor)) * thickness)
        self.spec = spec

    def reach(self, poly: np.ndarray):
        """Stretch the line along itself to the blank's edges: a traced or rounded bend line often stops a little
        short of them, and the strip must cut the blank right across to separate the panels."""
        # A bend has material on both sides of its line: stretch only while both sides are inside the blank. A
        # point on the line alone is not enough, since a bend line often continues straight into an edge of the
        # outline (a tab beside a longer panel), and running along that edge would cut off other panels.
        side = self.n * 0.05
        both = lambda p: _inside(poly, p + side) and _inside(poly, p - side)
        ends = []
        for start, sign in ((self.p0, -1), (self.p0 + self.u * self.length, 1)):
            d = 0.0
            while d < 10 * self.t + 10 and both(start + self.u * sign * (d + 0.1)):
                d += 0.1
            ends.append(d + 0.05)
        self.p0 = self.p0 - self.u * ends[0]
        self.length += ends[0] + ends[1]

    def zone_box(self) -> cq.Solid:
        """The strip the bend zone takes on the flat blank: along the line, BA wide, centred on it."""
        return _box(self.p0, self.u, 0.0, self.length, -self.ba / 2, self.ba / 2)

    def probe(self, side: int) -> cq.Vector:
        """A point just beyond the zone on one side of the line (side = +1 / -1 along n), mid-sheet."""
        p = self.p0 + self.u * self.length / 2 + side * self.n * (self.ba / 2 + min(1.0, 0.25 * self.t + 0.25))
        return cq.Vector(p[0], p[1], self.t / 2)

    def moves(self, child_side: int) -> list:
        """The fold of the panel on `child_side`, in the parent's flat frame: slide it back by BA so its edge meets
        the parent's, then rotate it about the bend axis. Returns (ops, zone solid)."""
        n = self.n * child_side
        e = self.p0 - n * self.ba / 2                         # the parent's edge line
        z_axis = self.t + self.r if self.up else -self.r
        a0, a1 = cq.Vector(*e, z_axis), cq.Vector(*(e + self.u), z_axis)
        # the rotation sense that lifts the panel (up) or drops it (down)
        test = cq.Vertex.makeVertex(*(e + n * 5), self.t / 2).rotate(a0, a1, 10)
        sign = 1 if (test.Z > self.t / 2) == self.up else -1
        ops = [("translate", cq.Vector(*(-n * self.ba), 0)), ("rotate", a0, a1, sign * self.angle)]
        # the zone: a ring radius .. radius + thickness about the axis, swept by the angle from the sheet, as long
        # as the line. Built in the plane across the line (x along n, y along +z), then extruded along the line.
        t0, t1 = self.extent
        plane = cq.Plane(origin=cq.Vector(*(self.p0 + self.u * t0), z_axis) - cq.Vector(*(n * self.ba / 2), 0),
                         xDir=cq.Vector(*n, 0), normal=cq.Vector(*self.u, 0))
        zs = 1 if plane.yDir.z > 0 else -1
        start = math.radians(-90 if self.up else 90)          # from the axis towards the sheet
        sweep = math.radians(self.angle) * (1 if self.up else -1)
        at = lambda rad, a: (rad * math.cos(a), zs * rad * math.sin(a))
        ro, ri = self.r + self.t, self.r
        # true arcs, so the bend is one smooth face, not facets
        zone = (cq.Workplane(plane).moveTo(*at(ro, start))
                .threePointArc(at(ro, start + sweep / 2), at(ro, start + sweep))
                .lineTo(*at(ri, start + sweep))
                .threePointArc(at(ri, start + sweep / 2), at(ri, start))
                .close().extrude(t1 - t0).val())
        return ops, zone


def _apply(shape, ops):
    for op in ops:
        shape = shape.translate(op[1]) if op[0] == "translate" else shape.rotate(op[1], op[2], op[3])
    return shape


def fold_sheet(profile, thickness: float, bends: list[dict], k_factor: float = 0.33, radius: float | None = None,
               fixed=None, holes=(), report: list | None = None) -> cq.Workplane:
    """Fold the blank `profile` ([[x, y], ...], mm) of `thickness` along `bends` (each: line [[x, y], [x, y]],
    angle (default 90), direction up/down, optional radius and k_factor). `radius` is the default inside bend
    radius (default: the thickness). `fixed`: a point [x, y] in the panel that stays flat. `holes`: [{"at": [x, y],
    "d": d}] cut through the flat blank. `report`, if given a list, receives one line per decision made."""
    log = report if report is not None else []
    radius = thickness if radius is None else radius
    blank = cq.Workplane("XY").polyline([tuple(map(float, p)) for p in profile]).close().extrude(thickness).val()
    for h in holes:
        x, y = h["at"]
        blank = blank.cut(cq.Workplane("XY").center(x, y).circle(h["d"] / 2).extrude(thickness + 2)
                          .translate((0, 0, -1)).val())
    folds = [_Bend(b, thickness, k_factor, radius) for b in bends]
    poly = np.array(profile, float)
    for f in folds:
        f.reach(poly)
    panels_shape = blank
    for f in folds:
        strip = f.zone_box()
        # the zone is as long as the blank it takes, not as the (stretched) strip
        taken = np.array([[v.X, v.Y] for v in blank.intersect(strip).Vertices()])
        along = (taken - f.p0) @ f.u if len(taken) else np.array([0.0, f.length])
        f.extent = (float(along.min()), float(along.max()))
        panels_shape = panels_shape.cut(strip)
    panels = sorted(panels_shape.Solids(), key=lambda s: -s.Volume())
    if not panels:
        raise ValueError("the bend zones remove the whole blank: check the bend lines and units")

    def panel_at(point: cq.Vector):
        for i, p in enumerate(panels):
            if p.isInside(point, 1e-3):
                return i
        return None

    # which panels each bend joins
    edges = []
    for k, f in enumerate(folds):
        a, b = panel_at(f.probe(-1)), panel_at(f.probe(1))
        if a is None or b is None or a == b:
            log.append(f"bend {k + 1} ({f.spec['line']}): does not separate two panels "
                       f"(found {a} and {b}); not folded")
            continue
        edges.append((k, a, b))
    root = 0
    if fixed is not None:
        point = cq.Vector(float(fixed[0]), float(fixed[1]), thickness / 2)
        found = panel_at(point)
        if found is None:      # e.g. a hole centre: the panel nearest to it
            near = cq.Vertex.makeVertex(*point.toTuple())
            found = min(range(len(panels)), key=lambda i: near.distance(panels[i]))
            log.append(f"fixed point {fixed} is not on the material (a hole?); the nearest panel stays flat")
        root = found
    # walk outwards from the fixed panel: each panel's fold is its own bend, then its parent's
    ops = {root: []}
    parts = [panels[root]]
    queue = [root]
    while queue:
        parent = queue.pop(0)
        for k, a, b in edges:
            if parent not in (a, b):
                continue
            child, side = (b, 1) if parent == a else (a, -1)
            if child in ops:
                continue
            move, zone = folds[k].moves(side)
            ops[child] = move + ops[parent]
            parts.append(_apply(panels[child], ops[child]))
            parts.append(_apply(zone, ops[parent]))
            log.append(f"bend {k + 1}: panel {child} folded {'up' if folds[k].up else 'down'} "
                       f"{folds[k].angle:g} deg, R{folds[k].r:g}, zone {folds[k].ba:.2f} mm")
            queue.append(child)
    for i, p in enumerate(panels):
        if i not in ops:
            parts.append(p)
            log.append(f"panel {i} is not joined to the fixed panel by any bend: left flat")
    result = parts[0]
    for p in parts[1:]:
        result = result.fuse(p)
    result = result.clean()
    if len(result.Solids()) > 1:
        log.append(f"the folded part has {len(result.Solids())} separate solids: a bend line does not reach "
                   "the panel edges, or a panel was left unfolded")
    return cq.Workplane("XY").add(result)
