"""Find the views on a drawing sheet and save each one as its own image."""
import sys, os, json, math
import pymupdf
from shapely.geometry import LineString, box
from shapely.ops import unary_union

GAP_OUTLINE = 6      # pt: visible-layer entities closer than this belong to the same view outline
GAP_ANNOT = 3        # pt: annotation entities closer than this form one annotation group
TEXT_ATTACH = 8      # pt: text this close to an annotation group follows that group
SHEET_INFO_Y = 705   # pt: text below this y is title block / notes (heuristic)
MARGIN = 12


def bez(p1, p2, p3, p4, n=10):
    pts = []
    for i in range(n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3
        pts.append((a * p1.x + b * p2.x + c * p3.x + d * p4.x,
                    a * p1.y + b * p2.y + c * p3.y + d * p4.y))
    return LineString(pts)


def geom_of(d):
    gs = []
    for it in d["items"]:
        k = it[0]
        if k == "l":
            if (it[1].x, it[1].y) != (it[2].x, it[2].y):
                gs.append(LineString([(it[1].x, it[1].y), (it[2].x, it[2].y)]))
        elif k == "c":
            gs.append(bez(it[1], it[2], it[3], it[4]))
        elif k == "re":
            gs.append(box(*it[1]).exterior)
        elif k == "qu":
            q = it[1]
            gs.append(LineString([(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y), (q.ul.x, q.ul.y)]))
    return unary_union(gs) if gs else None


def components(geoms, gap):
    n = len(geoms)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if geoms[i].distance(geoms[j]) <= gap:
                parent[find(i)] = find(j)
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


import re
_LABEL_RE = re.compile(r"\b(VIEW|SECTION|ISOMETRIC|DIMETRIC|TRIMETRIC|PERSPECTIVE)\b|^DETAIL\s+[A-Z]\b|"
                       r"^(TOP|FRONT|BACK|REAR|LEFT|RIGHT|BOTTOM|PLAN|ELEVATION)$")


def is_label(text):
    return bool(_LABEL_RE.search(" ".join(text.upper().split())))


def rect_dist(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


def split(pdf_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    W, H = page.rect.width, page.rect.height

    # ---- collect entities (drop page background + frame layer)
    ents = []
    for d in page.get_drawings():
        layer = d.get("layer") or ""
        r = d["rect"]
        if layer == "frame":
            continue
        if layer == "" and r.width >= W - 1 and r.height >= H - 1:
            continue
        g = geom_of(d)
        if g is None:
            continue
        ents.append({"d": d, "layer": layer, "g": g, "b": g.bounds})

    outline = [e for e in ents if e["layer"] == "visible"]
    others = [e for e in ents if e["layer"] != "visible"]

    # ---- 1) view outlines = connected components of the visible layer
    comps = components([e["g"] for e in outline], GAP_OUTLINE)
    pieces = []
    for c in comps:
        members = [outline[i] for i in c]
        g = unary_union([m["g"] for m in members])
        pieces.append({"ents": list(members), "g": g})

    def size(p):
        b = p["g"].bounds
        return max(b[2] - b[0], b[3] - b[1])

    def merge(a, b):
        a["ents"] += b["ents"]
        a["g"] = unary_union([a["g"], b["g"]])

    pieces.sort(key=size, reverse=True)
    # (a) nested pieces (holes, bores, concentric circles) join the piece that encloses them
    changed = True
    while changed:
        changed = False
        for i, b in enumerate(pieces):
            bb = b["g"].bounds
            for a in pieces[:i]:
                ab = a["g"].bounds
                if ab[0] - 1 <= bb[0] and ab[1] - 1 <= bb[1] and bb[2] <= ab[2] + 1 and bb[3] <= ab[3] + 1:
                    merge(a, b); pieces.pop(i); changed = True; break
            if changed:
                break
    # (b) tiny pieces (e.g. cut-line stubs) join a much larger piece that is close by
    big = max(size(p) for p in pieces)
    for p in [p for p in pieces if size(p) < 0.2 * big]:
        cand = [q for q in pieces if q is not p and size(q) >= 0.2 * big]
        q = min(cand, key=lambda q: q["g"].distance(p["g"]))
        if q["g"].distance(p["g"]) <= 20:
            merge(q, p); pieces.remove(p)
    views = [{"ents": p["ents"], "texts": [], "outline": p["g"], "ob": p["g"].bounds} for p in pieces]
    views.sort(key=lambda v: (round(v["ob"][1] / 100), v["ob"][0]))

    def nearest_view_geom(g):
        return min(range(len(views)), key=lambda i: views[i]["outline"].distance(g))

    # ---- 2) hatch / centre lines: nearest outline (bbox first, geometry as tie-break)
    annot = []
    for e in others:
        if e["layer"] in ("hatch", "center"):
            i = min(range(len(views)), key=lambda k: (rect_dist(e["b"], views[k]["ob"]), views[k]["outline"].distance(e["g"])))
            views[i]["ents"].append(e)
        else:
            annot.append(e)

    # ---- 3) dimension / leader graphics -> annotation groups -> view whose outline they touch
    groups = []
    for c in components([e["g"] for e in annot], GAP_ANNOT):
        members = [annot[i] for i in c]
        g = unary_union([m["g"] for m in members])
        vi = nearest_view_geom(g)
        views[vi]["ents"].extend(members)
        groups.append({"g": g, "view": vi})

    # ---- 4) text spans
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                if s["text"].strip():
                    s["dir"] = ln["dir"]
                    spans.append(s)
    # title block = the biggest rectangle of the frame layer that is not the page border;
    # notes block = the 'NOTES' line and everything below it, left of the title block
    title_bb = None
    rects = [d["rect"] for d in page.get_drawings() if (d.get("layer") or "") == "frame"
             and d["rect"].width > 100 and d["rect"].height > 40]
    rects.sort(key=lambda r: r.width * r.height, reverse=True)
    if len(rects) >= 2:
        title_bb = tuple(rects[1])
    notes_y = min([s["bbox"][1] for s in spans if s["text"].strip().upper().startswith("NOTES")], default=None)
    sheet_info = []
    label_spans, pending, attached = [], [], []
    for s in spans:
        bb = tuple(s["bbox"])
        in_title = title_bb and bb[0] >= title_bb[0] - 1 and bb[2] <= title_bb[2] + 1 and bb[1] >= title_bb[1] - 1
        in_notes = notes_y is not None and bb[1] >= notes_y - 1 and (not title_bb or bb[2] <= title_bb[0])
        legacy = (title_bb is None and notes_y is None and bb[1] >= SHEET_INFO_Y)
        if in_title or in_notes or legacy:
            sheet_info.append(s)
        elif is_label(s["text"]):
            label_spans.append(s)
        else:
            sb = box(*bb)
            gd = sorted((grp["g"].distance(sb), grp["view"]) for grp in groups)
            if gd and gd[0][0] <= TEXT_ATTACH:
                views[gd[0][1]]["texts"].append(s)
                attached.append((gd[0][1], bb))
            else:
                pending.append(s)
    # stacked callout lines: text touching an already attached line joins the same view
    changed = True
    while changed and pending:
        changed = False
        for s in list(pending):
            bb = s["bbox"]
            for vi, ab in attached:
                gap_y = max(bb[1] - ab[3], ab[1] - bb[3], 0)
                overlap_x = min(bb[2], ab[2]) - max(bb[0], ab[0])
                if gap_y <= 3 and overlap_x > 0:
                    views[vi]["texts"].append(s)
                    attached.append((vi, bb))
                    pending.remove(s)
                    changed = True
                    break
    for s in pending:   # free text: nearest outline
        bb = s["bbox"]
        vi = min(range(len(views)), key=lambda k: rect_dist(bb, views[k]["ob"]))
        views[vi]["texts"].append(s)
    # view labels sit under their view: pick the nearest outline ABOVE the label that is horizontally
    # aligned with it; only if there is none, fall back to the nearest outline
    for s in label_spans:
        bb = s["bbox"]
        cx = (bb[0] + bb[2]) / 2
        above = [k for k in range(len(views))
                 if views[k]["ob"][0] - 5 <= cx <= views[k]["ob"][2] + 5 and views[k]["ob"][3] <= bb[1] + 2]
        if above:
            vi = max(above, key=lambda k: views[k]["ob"][3])
        else:
            vi = min(range(len(views)), key=lambda k: rect_dist(bb, views[k]["ob"]))
        views[vi]["texts"].append(s)

    # ---- 5) name + extent of each view
    for v in views:
        label = [s["text"].strip() for s in v["texts"] if is_label(s["text"])]
        v["label"] = label[0] if label else "UNLABELED"
        xs0 = [e["b"][0] for e in v["ents"]] + [s["bbox"][0] for s in v["texts"]]
        ys0 = [e["b"][1] for e in v["ents"]] + [s["bbox"][1] for s in v["texts"]]
        xs1 = [e["b"][2] for e in v["ents"]] + [s["bbox"][2] for s in v["texts"]]
        ys1 = [e["b"][3] for e in v["ents"]] + [s["bbox"][3] for s in v["texts"]]
        v["extent"] = (min(xs0), min(ys0), max(xs1), max(ys1))

    # ---- fonts (reuse the original embedded font when possible)
    fontbuf = {}
    for f in page.get_fonts(full=True):
        try:
            nm, ext, typ, buf = doc.extract_font(f[0])
            if buf:
                fontbuf[f[3]] = buf
        except Exception:
            pass

    # ---- 6) write one vector PDF per view
    meta = []
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    for n, v in enumerate(views, 1):
        x0, y0, x1, y1 = v["extent"]
        dx, dy = MARGIN - x0, MARGIN - y0
        nd = pymupdf.open()
        np_ = nd.new_page(width=(x1 - x0) + 2 * MARGIN, height=(y1 - y0) + 2 * MARGIN)
        ocg = {}

        def T(p):
            return pymupdf.Point(p.x + dx, p.y + dy)
        for e in v["ents"]:
            d = e["d"]
            L = e["layer"]
            if L and L not in ocg:
                ocg[L] = nd.add_ocg(L, on=True)
            sh = np_.new_shape()
            for it in d["items"]:
                k = it[0]
                if k == "l":
                    sh.draw_line(T(it[1]), T(it[2]))
                elif k == "c":
                    sh.draw_bezier(T(it[1]), T(it[2]), T(it[3]), T(it[4]))
                elif k == "re":
                    r = it[1]
                    sh.draw_rect(pymupdf.Rect(r.x0 + dx, r.y0 + dy, r.x1 + dx, r.y1 + dy))
                elif k == "qu":
                    q = it[1]
                    sh.draw_quad(pymupdf.Quad(T(q.ul), T(q.ur), T(q.ll), T(q.lr)))
            lc = d.get("lineCap", 0)
            lc = lc[0] if isinstance(lc, (tuple, list)) else (lc or 0)
            sh.finish(fill=d.get("fill"), color=d.get("color"), width=d.get("width") or 0,
                      dashes=d.get("dashes"), closePath=bool(d.get("closePath")),
                      even_odd=bool(d.get("even_odd")), lineCap=lc, lineJoin=d.get("lineJoin") or 0, oc=ocg.get(L, 0))
            sh.commit()
        for s in v["texts"]:
            dirx, diry = s["dir"]
            rot = 0 if abs(dirx) > .7 and dirx > 0 else 180 if abs(dirx) > .7 else 90 if diry < 0 else 270
            fname = "helv"
            txt = s["text"]
            if s["font"] in fontbuf:
                fname = "F_" + str(abs(hash(s["font"])) % 9999)
                np_.insert_font(fontname=fname, fontbuffer=fontbuf[s["font"]])
            else:
                txt = txt.replace("\u2300", "\u00d8")
            fs = s["size"]
            orig_len = max(s["bbox"][2] - s["bbox"][0], s["bbox"][3] - s["bbox"][1])
            try:
                got = pymupdf.get_text_length(txt, fontname=fname if fname == "helv" else "helv", fontsize=fs)
                if got > orig_len > 0:
                    fs *= orig_len / got      # keep the original text footprint
            except Exception:
                pass
            np_.insert_text(T(pymupdf.Point(s["origin"])), txt, fontsize=fs, fontname=fname, rotate=rot)
        slug = "".join(c if c.isalnum() else "_" for c in v["label"]).strip("_").lower()
        fn = os.path.join(outdir, f"{stem}_view{n}_{slug}.pdf")
        nd.save(fn)
        np_.get_pixmap(dpi=90).save(fn.replace(".pdf", ".png"))
        meta.append({
            "file": os.path.basename(fn), "label": v["label"],
            "outline_bbox_sheet_pt": [round(c, 1) for c in v["ob"]],
            "extent_bbox_sheet_pt": [round(c, 1) for c in v["extent"]],
            "offset_to_view_pt": [round(dx, 2), round(dy, 2)],
            "entities": {L or "-": sum(1 for e in v["ents"] if e["layer"] == L) for L in sorted({e["layer"] for e in v["ents"]})},
            "texts": [s["text"] for s in v["texts"]],
        })
    json.dump({"source": os.path.basename(pdf_path), "sheet_info_texts": [s["text"] for s in sheet_info], "views": meta},
              open(os.path.join(outdir, f"{stem}_views.json"), "w"), indent=1, ensure_ascii=False)

    # overview with boxes
    ov = pymupdf.open(pdf_path)[0]
    sh = ov.new_shape()
    cols = [(1, 0, 0), (0, .5, 0), (0, 0, 1), (.8, .4, 0)]
    for i, v in enumerate(views):
        sh.draw_rect(pymupdf.Rect(*v["extent"]))
        sh.finish(color=cols[i % 4], width=2.5, dashes="[6 3] 0")
        sh.commit()
        ov.insert_text((v["extent"][0] + 4, v["extent"][1] - 4), f"view {i+1}: {v['label']}", fontsize=14, color=cols[i % 4])
    ov.get_pixmap(dpi=80).save(os.path.join(outdir, f"{stem}_overview.png"))
    return meta


if __name__ == "__main__":
    for m in split(sys.argv[1], sys.argv[2]):
        print(m["file"], "|", m["label"], "|", m["entities"], "|", m["texts"])