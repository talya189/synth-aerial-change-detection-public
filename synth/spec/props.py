"""Street furniture (pure Python): streetlights along the roads.

Each prop is one entity {"id": "prop_NNN", "type", "x", "y", "yaw_deg",
"height", "road"}; types and sizes in configs/props.json.

  streetlight         one side of a two-lane road, arm over the road
  streetlight_double  on the median of a divided road, an arm over each carriageway
  streetlight_small   one side of a street or lane, shorter

A light stands just off the carriageway edge, never within `junction_m` of a
junction, never on a building, wall, path, tree or another road, and never in
front of a door (a house's way to its road).
"""
import math

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from synth.common.config import load_json
from synth.common.rng import rng
from synth.spec.access import _half
from synth.spec.buildings import footprint
from synth.spec.scene_spec import next_id

DEFAULTS = {"spacing_m": {"main": 34.0, "street": 32.0, "lane": 38.0}, "junction_m": 9.0, "p_skip": 0.05}


def _obstacles(spec, rc):
    ent = spec["entities"]
    obs = [footprint(b).buffer(0.8) for b in ent.get("buildings", [])]
    obs += [LineString(sg).buffer(w.get("thickness", 0.3) / 2 + 0.3) for w in ent.get("walls", []) for sg in w["segments"]]
    obs += [LineString(np.asarray(p["centerline"])[:, :2]).buffer(p.get("width", 1.0) / 2 + 0.4) for p in ent.get("paths", [])]
    obs += [Point(t["x"], t["y"]).buffer(1.2) for t in ent.get("trees", [])]
    from synth.spec.yards import _ways
    obs += [L.buffer(1.2) for own, L, hw in _ways(spec, rc, {}) if len(own) == 1]   # in front of a door
    return obs


def plan_streetlights(spec, site, seed, rules=None, log=print):
    R = dict(DEFAULTS, **(rules or {}))
    rc = load_json("configs/roads.json")["classes"]
    cat = load_json("configs/props.json")["types"]
    roads = [rd for rd in spec["entities"].get("roads", []) if rd["class"] in R["spacing_m"]]
    if not roads:
        return []
    allroads = spec["entities"]["roads"]
    lines = {rd["id"]: LineString(np.asarray(rd["centerline"])[:, :2]) for rd in allroads}
    bodies = {rd["id"]: lines[rd["id"]].buffer(_half(rd, rc)) for rd in allroads}
    junctions = []   # points where roads meet
    for rd in allroads:
        c = np.asarray(rd["centerline"])[:, :2]
        if rd.get("connects"):
            junctions.append(Point(c[0]))
        if rd.get("connects_end"):
            junctions.append(Point(c[-1]))
    J = unary_union([p.buffer(R["junction_m"]) for p in junctions]) if junctions else None
    obs = _obstacles(spec, rc)
    r = rng(seed, "streetlights")
    out, taken = [], []
    lim = site.half - 10
    for rd in roads:
        L = lines[rd["id"]]
        hw = _half(rd, rc)
        divided = rd.get("section") == "divided"
        typ = "streetlight_double" if divided else ("streetlight" if rd["class"] == "main" else "streetlight_small")
        sp = R["spacing_m"][rd["class"]]
        side = 1 if r.random() < 0.5 else -1
        d = float(r.uniform(0.2, 0.8)) * sp
        while d < L.length - 2:
            p = L.interpolate(d)
            q = L.interpolate(min(d + 1.0, L.length))
            t = np.array([q.x - p.x, q.y - p.y])
            t /= np.linalg.norm(t) + 1e-9
            n = np.array([-t[1], t[0]])
            d += sp
            if r.random() < R["p_skip"] or (J is not None and J.distance(p) <= (4.0 if divided else 0.0)):   # medians open ~7 m beyond
                continue
            if divided:
                cands = [(np.array([p.x, p.y]), math.degrees(math.atan2(n[1], n[0])))]
            else:
                off = hw + cat[typ]["clear_m"] + 0.15
                cands = [(np.array([p.x, p.y]) + s * n * off, math.degrees(math.atan2(-s * n[1], -s * n[0]))) for s in (side, -side)]
            for xy, yaw in cands:
                P = Point(xy)
                if abs(xy[0]) > lim or abs(xy[1]) > lim:
                    continue
                if not divided and any(bodies[o].distance(P) < 0.3 for o in bodies):
                    continue
                if any(g.intersects(P) for g in obs) or any(math.hypot(xy[0] - a, xy[1] - b) < 8 for a, b in taken):
                    continue
                pr = {"id": next_id(spec, "props"), "type": typ, "x": round(float(xy[0]), 2), "y": round(float(xy[1]), 2),
                      "yaw_deg": round(yaw, 1), "height": round(float(r.uniform(*cat[typ]["height_m"])), 2), "road": rd["id"]}
                spec["entities"].setdefault("props", []).append(pr)
                out.append(pr)
                taken.append((xy[0], xy[1]))
                break
    log(f"streetlights: {len(out)}")
    return out


def prop_report(spec):
    """Props standing on a building, wall, path or carriageway (must be empty);
    median lights of a divided road are on their own road by design."""
    rc = load_json("configs/roads.json")["classes"]
    ent = spec["entities"]
    fps = [footprint(b) for b in ent.get("buildings", [])]
    walls = [LineString(sg).buffer(w.get("thickness", 0.3) / 2) for w in ent.get("walls", []) for sg in w["segments"]]
    paths = [LineString(np.asarray(p["centerline"])[:, :2]).buffer(p.get("width", 1.0) / 2) for p in ent.get("paths", [])]
    roads = {rd["id"]: LineString(np.asarray(rd["centerline"])[:, :2]).buffer(_half(rd, rc)) for rd in ent.get("roads", [])}
    bad = []
    for pr in ent.get("props", []):
        P = Point(pr["x"], pr["y"]).buffer(0.15)
        on_road = [rid for rid, B in roads.items() if B.intersects(P) and not (pr["type"] == "streetlight_double" and rid == pr.get("road"))]
        if on_road or any(g.intersects(P) for g in fps + walls + paths):
            bad.append(pr["id"])
    return bad
