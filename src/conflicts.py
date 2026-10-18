"""Find where the view analyses disagree or leave the drawing unexplained, before planning.

The analysis agents each see one view. Their summaries can contradict each other, guess, or leave dimensions
unattached; the planner then has to sort it out, and a weaker model often does not. This module finds those
spots with plain rules, so an arbiter agent can settle each one against the drawing:

- count_mismatch:       the same kind of hole counted differently in two views (one count from an `Nx` callout);
- unsupported_value:    a value in a hypothesis that is not on the drawing, nor half, a sum or a difference of
                        two drawing numbers (often "inferred from scale");
- unmeasured_feature:   a feature one view sees (often the isometric) that no view gives a size for;
- uncertain:            a hypothesis the agent itself is unsure of (confidence below 0.6) that carries dimensions;
- unexplained_dimension: a dimension on the drawing that no hypothesis in any view uses.
"""
from __future__ import annotations

import itertools
import re

HOLE_OPS = {"hole", "counterbore_hole", "countersink_hole", "tapped_hole", "slot"}
GENERIC = {"main", "feature", "features", "body", "part", "outer", "inner", "central", "centre", "center", "the",
           "of", "and", "four", "two", "pattern", "holes", "hole", "face", "faces", "end", "front", "top", "side",
           "left", "right", "upper", "lower", "lengthwise", "axial", "visible", "drawn", "shown"}
NUMBER = re.compile(r"\d+(?:\.\d+)?")
COUNT = re.compile(r"^\s*(\d+)\s*[xX×]")
MAX_ITEMS = 10


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text.upper().replace("⌀", "Ø").replace("∅", "Ø"))


def drawing_numbers(texts: list[str]) -> set[float]:
    values = {float(n) for t in texts for n in NUMBER.findall(t)}
    derived = set(values)
    for a, b in itertools.combinations(values, 2):
        derived |= {a + b, abs(a - b)}
    derived |= {v / 2 for v in list(derived)}
    return {round(v, 3) for v in derived}


def tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[_\W]+", name.lower()) if len(t) > 2 and t not in GENERIC}


def find(summaries: dict[str, dict], texts: list[str]) -> list[dict]:
    """summaries: view id -> analysis summary; texts: the dimension texts of the views (not the title block,
    whose dates and scales would excuse any number; empty for scans)."""
    items = []
    numbers = drawing_numbers(texts) if texts else set()

    # count_mismatch: hole-like features counted differently, at least one count from an Nx callout
    counted = {}
    for view, s in summaries.items():
        for h in s.get("hypotheses", []):
            if h["op"] in HOLE_OPS and h.get("count", 1) > 1 or any(COUNT.match(e) for e in h.get("evidence", [])):
                explicit = next((int(COUNT.match(e).group(1)) for e in h.get("evidence", []) if COUNT.match(e)), None)
                counted.setdefault(h["op"], []).append((view, h["name"], h.get("count", 1), explicit,
                                                         h.get("params", {}).get("d")))
    for op, entries in counted.items():
        # same diameter, or no diameter given (an isometric view sizes nothing)
        for d in {e[4] for e in entries if e[4]} or {None}:
            seen = [e[:4] for e in entries if e[4] in (d, None)]
            counts = {c for _, _, c, _ in seen}
            if len(counts) > 1 and any(e for *_, e in seen):
                items.append({"kind": "count_mismatch", "description": f"{op}{f' Ø{d:g}' if d else ''} is counted "
                              f"differently: " + ", ".join(f"{v}: {n} x{c}" for v, n, c, _ in seen),
                              "claims": [{"view": v, "claim": f"{n}: {c}" + (f" (callout {e}x)" if e else "")}
                                         for v, n, c, e in seen], "views": sorted({v for v, *_ in seen})})

    # unsupported_value: numbers that do not come from the drawing
    if numbers:
        for view, s in summaries.items():
            for h in s.get("hypotheses", []):
                for key, value in h.get("params", {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 \
                            and round(float(value), 3) not in numbers:
                        items.append({"kind": "unsupported_value",
                                      "description": f"{view}: {h['name']}.{key} = {value:g} is not a drawing "
                                                     f"dimension nor simple arithmetic on one",
                                      "claims": [{"view": view, "claim": f"{h['name']}: {key} = {value:g} "
                                                  f"({h.get('description', '')[:120]})"}], "views": [view]})

    # unmeasured_feature: a feature seen somewhere, sized nowhere
    sized = [tokens(h["name"]) for s in summaries.values() for h in s.get("hypotheses", [])
             if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in h.get("params", {}).values())]
    for view, s in summaries.items():
        for h in s.get("hypotheses", []):
            if h.get("params") and any(isinstance(v, (int, float)) and not isinstance(v, bool)
                                       for v in h["params"].values()):
                continue
            words = tokens(h["name"])
            if words and not any(words & other for other in sized):
                items.append({"kind": "unmeasured_feature",
                              "description": f"{view} sees '{h['name']}' ({h.get('op')}), but no view gives its "
                                             f"size or position",
                              "claims": [{"view": view, "claim": f"{h['name']}: {h.get('description', '')[:160]}"}],
                              "views": [view]})

    # uncertain: the agent itself is unsure of a hypothesis that carries dimensions
    for view, s in summaries.items():
        for h in s.get("hypotheses", []):
            if h.get("confidence", 1) < 0.6 and (h.get("evidence") or h.get("params")):
                items.append({"kind": "uncertain",
                              "description": f"{view}: '{h['name']}' ({h['op']}) at confidence {h['confidence']:.2f}, "
                                             f"based on {h.get('evidence') or 'no quoted text'}",
                              "claims": [{"view": view, "claim": f"{h['name']}: {h.get('description', '')[:160]}"}],
                              "views": [view]})

    # unexplained_dimension: on the drawing, used by no hypothesis
    cited = {norm(e) for s in summaries.values() for h in s.get("hypotheses", []) for e in h.get("evidence", [])}
    cited |= {norm(str(v)) for s in summaries.values() for h in s.get("hypotheses", [])
              for v in h.get("params", {}).values() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for view, s in summaries.items():
        for d in s.get("dimensions", []):
            if norm(d["text"]) not in cited and norm(f"{d['value']:g}") not in cited:
                items.append({"kind": "unexplained_dimension",
                              "description": f"{view}: dimension '{d['text']}' ({d['refers_to'][:100]}) is not used "
                                             f"by any feature in any view",
                              "claims": [{"view": view, "claim": f"'{d['text']}' = {d['refers_to'][:160]}"}],
                              "views": [view]})

    items = sorted(items, key=lambda i: ["count_mismatch", "unsupported_value", "unmeasured_feature", "uncertain",
                                         "unexplained_dimension"].index(i["kind"]))
    seen_claims, unique = set(), []
    for item in items:  # one item per hypothesis or dimension, the most important kind first
        key = tuple(c["claim"].split(":")[0] for c in item["claims"])
        if item["kind"] != "count_mismatch" and key in seen_claims:
            continue
        seen_claims.add(key)
        unique.append(item)
    items = unique
    order = ["count_mismatch", "unsupported_value", "unmeasured_feature", "uncertain", "unexplained_dimension"]
    items = sorted(items, key=lambda i: order.index(i["kind"]))
    items = items[:MAX_ITEMS]
    for n, item in enumerate(items, 1):
        item["id"] = f"C{n}"
    return items
