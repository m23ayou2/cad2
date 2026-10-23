"""Render STEP files in 3D in the browser.

    uv run python src/visualization.py runs/<run>/model.step [other.step ...] [--out viewer.html] [--grid] [--no-open]

Each STEP is tessellated with CadQuery and embedded in a single HTML page that
draws it with three.js (loaded from jsDelivr, so viewing needs internet). Drag to
rotate, scroll to zoom, right-drag to pan; the checkboxes show or hide each model.
Two layouts: "overlay" (the default) draws every model in place, the first opaque and the others see-through, to
compare a model with its reference; "grid" sets different parts side by side, each in its own cell (a batch of
drawings), and a click on a part's name moves the camera to it.
"""
from __future__ import annotations

import argparse
import json
import math
import webbrowser
from pathlib import Path

COLORS = ["#4f86c6", "#e07b39", "#5bb26f", "#b45fc0", "#c9a227", "#3fa7a3", "#d0587e", "#7a7f87"]


def mesh(step: Path) -> dict:
    """Triangles and edge polylines of one STEP file, in mm."""
    import cadquery as cq

    objects = cq.importers.importStep(str(step)).vals()
    shape = objects[0] if len(objects) == 1 else cq.Compound.makeCompound(objects)
    bb = shape.BoundingBox()
    tolerance = max(math.dist((bb.xmin, bb.ymin, bb.zmin), (bb.xmax, bb.ymax, bb.zmax)) * 1e-3, 0.01)
    vertices, triangles = shape.tessellate(tolerance, 0.2)
    edges = []
    for edge in shape.Edges():
        n = 1 if edge.geomType() == "LINE" else 32
        points = [edge.positionAt(i / n) for i in range(n + 1)]
        edges.append([round(c, 4) for p in points for c in (p.x, p.y, p.z)])
    return {
        "positions": [round(c, 4) for v in vertices for c in (v.x, v.y, v.z)],
        "indices": [i for t in triangles for i in t],
        "edges": edges,
        "size": [round(bb.xlen, 2), round(bb.ylen, 2), round(bb.zlen, 2)],
        "volume": round(shape.Volume(), 1),
    }


def write_viewer(models: list[tuple[str, Path]], out: Path, title: str = "Draw2CAD viewer",
                 layout: str = "overlay") -> Path:
    """models: (label, step path) pairs. layout "overlay": the first is opaque, the others see-through, all in
    place; "grid": every model opaque, each in its own cell."""
    data = []
    for i, (label, step) in enumerate(models):
        data.append({"label": label, "file": str(step), "color": COLORS[i % len(COLORS)],
                     "opacity": 1.0 if i == 0 or layout == "grid" else 0.35, **mesh(step)})
    page = PAGE.replace("__TITLE__", title).replace("__LAYOUT__", json.dumps(layout))
    out.write_text(page.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    return out


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; font: 14px system-ui, sans-serif; background: #f4f5f7; color: #222; }
  #panel { position: fixed; top: 12px; left: 12px; background: #fff; padding: 10px 14px; border-radius: 8px;
           box-shadow: 0 1px 4px rgba(0,0,0,.15); max-width: 420px; }
  #panel h1 { font-size: 15px; margin: 0 0 6px; }
  #panel label { display: block; margin: 4px 0; }
  #panel b.focus { cursor: pointer; text-decoration: underline dotted; }
  #panel small { color: #666; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }
</style>
<script type="importmap">
{ "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
               "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/" } }
</script>
</head><body>
<div id="panel"><h1>__TITLE__</h1><div id="models"></div>
<small>Drag: rotate · Scroll: zoom · Right-drag: pan · Z is up</small></div>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const models = __DATA__;
const layout = __LAYOUT__;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color("#f4f5f7");
scene.add(new THREE.HemisphereLight(0xffffff, 0x888888, 1.6));
const camera = new THREE.PerspectiveCamera(35, innerWidth / innerHeight, 0.1, 1e6);
camera.up.set(0, 0, 1);
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
camera.add(sun); sun.position.set(1, 2, 3); scene.add(camera);
const controls = new OrbitControls(camera, renderer.domElement);

const box = new THREE.Box3();
const list = document.getElementById("models");
const groups = [];
function frame(target, distance) {
  const center = target.getCenter(new THREE.Vector3());
  const radius = target.getSize(new THREE.Vector3()).length() / 2 || 1;
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(1, -1.4, 0.9).normalize().multiplyScalar(radius * distance));
  camera.near = radius / 100; camera.far = radius * 100; camera.updateProjectionMatrix();
}
for (const m of models) {
  const group = new THREE.Group();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(m.positions, 3));
  geometry.setIndex(m.indices);
  geometry.computeVertexNormals();
  // see-through overlays are pulled slightly towards the camera so coincident faces do not flicker
  const overlay = m.opacity < 1;
  const material = new THREE.MeshStandardMaterial({ color: m.color, metalness: 0.1, roughness: 0.6,
    side: THREE.DoubleSide, transparent: overlay, opacity: m.opacity, depthWrite: !overlay,
    polygonOffset: true, polygonOffsetFactor: overlay ? -2 : 1, polygonOffsetUnits: overlay ? -2 : 1 });
  const surface = new THREE.Mesh(geometry, material);
  surface.renderOrder = overlay ? 1 : 0;
  group.add(surface);
  const segments = [];
  for (const e of m.edges) for (let i = 0; i + 5 < e.length; i += 3) segments.push(...e.slice(i, i + 6));
  const lines = new THREE.BufferGeometry();
  lines.setAttribute("position", new THREE.Float32BufferAttribute(segments, 3));
  group.add(new THREE.LineSegments(lines, new THREE.LineBasicMaterial({ color: 0x222222,
    transparent: m.opacity < 1, opacity: Math.max(m.opacity, 0.5) })));
  scene.add(group);
  groups.push(group);
  const row = document.createElement("label");
  row.innerHTML = `<input type="checkbox" checked> <span class="swatch" style="background:${m.color}"></span>` +
    `<b>${m.label}</b> <small>${m.size.join(" × ")} mm · ${m.volume} mm³</small>`;
  row.querySelector("input").onchange = (ev) => { group.visible = ev.target.checked; };
  if (layout === "grid") {  // the name moves the camera to that part
    const name = row.querySelector("b");
    name.className = "focus";
    name.onclick = (ev) => { ev.preventDefault(); frame(new THREE.Box3().setFromObject(group), 2.6); };
  }
  list.appendChild(row);
}
if (layout === "grid") {  // one cell per part, rows of ceil(sqrt(n)), each part centred in its cell on z = 0
  const boxes = groups.map((g) => new THREE.Box3().setFromObject(g));
  const cell = 1.3 * Math.max(...boxes.map((b) => { const s = b.getSize(new THREE.Vector3()); return Math.max(s.x, s.y); }), 1);
  const cols = Math.ceil(Math.sqrt(groups.length));
  groups.forEach((g, i) => {
    const c = boxes[i].getCenter(new THREE.Vector3());
    g.position.set((i % cols) * cell - c.x, -Math.floor(i / cols) * cell - c.y, -boxes[i].min.z);
  });
}
for (const g of groups) box.expandByObject(g);
frame(box, 3.2);
const axes = new THREE.AxesHelper(box.getSize(new THREE.Vector3()).length() / 2 * 0.4 || 1); scene.add(axes);

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description="Render STEP files in 3D in the browser.")
    parser.add_argument("steps", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="HTML file to write (default: next to the first STEP)")
    parser.add_argument("--grid", action="store_true", help="set the parts side by side instead of overlaying them")
    parser.add_argument("--no-open", action="store_true", help="write the page without opening it")
    args = parser.parse_args()
    out = args.out or args.steps[0].with_name("viewer.html")
    write_viewer([(p.stem if p.stem != "model" else p.parent.name, p) for p in args.steps], out,
                 layout="grid" if args.grid else "overlay")
    print(out)
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
