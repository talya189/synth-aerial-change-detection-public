"""Tree and grove planting for a settlement (pure Python).

Writes `trees` (single trees with ids) and `groves` (orchards / planted pine
patches: one entity, one instance id, trees laid out inside a polygon) into the
spec. Rules per settlement style (configs/vegetation.json "planting"):

  yard     1-4 trees around each house, a few metres from its walls, never on a
           road, path or another house; species mix of the style (citrus and
           olive in a moshav, olive in an Arab village, pines and palms in town)
  street   rows along the given road classes, offset from the edge, skipping
           places where a house or junction is in the way
  groves   rectangular orchards in rows along the contour, on gentle dry ground
           around the village (olive groves ring Arab villages)
  forest   planted pine patches (irregular spacing) on the slopes further out

Wild shrubs and ground cover are not entities: they stay in the ground-cover
preset (scatter systems), which keeps them out of buildings, roads and paths.
"""
import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from synth.common.config import load_json
from synth.common.rng import rng
from synth.spec.buildings import footprint, pick
from synth.spec.scene_spec import next_id


def _cfg():
    return load_json("configs/vegetation.json")


def _blocked(spec, extra_m=0.0):
    """Union of everything a tree may not stand on: buildings (+1.5 m), roads (+edge),
    paths, groves."""
    rc = load_json("configs/roads.json")["classes"]
    shapes = [footprint(b).buffer(1.5 + extra_m) for b in spec["entities"].get("buildings", [])]
    for rd in spec["entities"].get("roads", []):
        c = rc.get(rd.get("class"), {})
        hw = rd.get("width", c.get("width", 4.0)) / 2 + rd.get("shoulder", c.get("shoulder", 0.0))
        shapes.append(LineString(rd["centerline"]).buffer(hw + 1.0 + extra_m))
        if rd.get("end_cap"):
            shapes.append(Point(rd["centerline"][-1]).buffer(rd["end_cap"] + 1.0))
    for pa in spec["entities"].get("paths", []):
        shapes.append(LineString(pa["centerline"]).buffer(pa.get("width", 1.0) / 2 + 0.8))
    for g in spec["entities"].get("groves", []):
        shapes.append(Polygon(g["polygon"]).buffer(2.0))
    for f in spec["entities"].get("fields", []):
        shapes.append(Polygon(f["polygon"]).buffer(1.0))
    for w in spec["entities"].get("walls", []):   # no trunk on a wall; groves and fields stay out of yards
        shapes += [LineString(s).buffer(w.get("thickness", 0.3) / 2 + 0.6) for s in w["segments"]]
        if extra_m > 0 and len(w.get("yard", [])) >= 3:
            shapes.append(Polygon(w["yard"]).buffer(extra_m))
    return unary_union(shapes) if shapes else Polygon()


def _tree(spec, species, x, y, r, sp):
    t = {"id": next_id(spec, "trees"), "species": species, "x": round(float(x), 2), "y": round(float(y), 2),
         "yaw_deg": round(float(r.uniform(0, 360)), 1)}
    if "height_m" in sp:
        t["height"] = round(float(r.uniform(*sp["height_m"])), 2)
    if "crown_m" in sp:
        t["crown"] = round(float(r.uniform(*sp["crown_m"])), 2)
    spec["entities"].setdefault("trees", []).append(t)
    return t


def _ok(site, x, y, blocked, margin=25.0):
    lim = site.half - margin
    if abs(x) > lim or abs(y) > lim:
        return False
    return not blocked.contains(Point(x, y))


def plant_yards(spec, site, rules, species, seed):
    r = rng(seed, "plant", "yards")
    blocked = _blocked(spec)
    taken = []
    out = []
    for b in spec["entities"].get("buildings", []):
        if b.get("type") in ("coop", "greenhouse", "water_tower", "shed"):
            continue
        n = int(r.integers(rules["per_house"][0], rules["per_house"][1] + 1))
        fp = footprint(b)
        for _ in range(n * 8):
            if n <= 0:
                break
            ang = r.uniform(0, 2 * math.pi)
            d = float(r.uniform(*rules["dist_m"]))
            c = fp.centroid
            # a point d metres outside the walls, in a random direction
            ray = LineString([(c.x, c.y), (c.x + 200 * math.cos(ang), c.y + 200 * math.sin(ang))])
            edge = ray.intersection(fp.exterior)
            if edge.is_empty:
                continue
            e = edge if edge.geom_type == "Point" else edge.geoms[0]
            x, y = e.x + d * math.cos(ang), e.y + d * math.sin(ang)
            if not _ok(site, x, y, blocked) or any(math.hypot(x - tx, y - ty) < 3.0 for tx, ty in taken):
                continue
            s_name = pick(r, rules["species"])
            out.append(_tree(spec, s_name, x, y, r, species[s_name]))
            taken.append((x, y))
            n -= 1
    return out


def plant_streets(spec, site, rules, species, seed):
    if not rules["classes"]:
        return []
    r = rng(seed, "plant", "streets")
    rc = load_json("configs/roads.json")["classes"]
    blocked = _blocked(spec)
    out = []
    s_name = pick(r, rules["species"])   # one species per settlement's streets, like real avenues
    for rd in spec["entities"].get("roads", []):
        if rd.get("class") not in rules["classes"] or rd.get("end_cap") or \
                LineString(rd["centerline"]).length < rules.get("min_length_m", 0):
            continue
        line = LineString(rd["centerline"])
        hw = rc[rd["class"]]["width"] / 2 + rc[rd["class"]]["shoulder"]
        for side in (1, -1):
            if side == -1 and r.random() > rules["p_side"]:
                continue
            off = line.parallel_offset(hw + rules["offset_m"], "left" if side > 0 else "right")
            parts = getattr(off, "geoms", [off])
            for part in parts:
                u = float(r.uniform(4, 8))
                while u < part.length - 4:
                    p = part.interpolate(u)
                    if _ok(site, p.x, p.y, blocked):
                        out.append(_tree(spec, s_name, p.x, p.y, r, species[s_name]))
                    u += float(r.uniform(*rules["spacing_m"]))
    return out


def _grove_polygon(site, cx, cy, w, h, ang):
    c, s = math.cos(ang), math.sin(ang)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return Polygon([(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts])


def plant_groves(spec, site, rules, species_w, seed, centre, forest=False, tries=400):
    """Orchards (rows along the contour) or planted forest patches around the village."""
    if rules["count"] <= 0:
        return []
    r = rng(seed, "plant", "forest" if forest else "groves")
    out = []
    village = unary_union([footprint(b) for b in spec["entities"].get("buildings", [])] or [Point(centre)])
    for _ in range(tries):
        if len(out) >= rules["count"]:
            break
        blocked = _blocked(spec, extra_m=2.0)
        # a point on a ring `dist` metres outside the village's outline
        ring = village.convex_hull.buffer(float(r.uniform(*rules["dist_m"]))).exterior
        p = ring.interpolate(float(r.uniform(0, 1)), normalized=True)
        cx, cy = p.x, p.y
        w, h = float(r.uniform(*rules["size_m"][0])), float(r.uniform(*rules["size_m"][1]))
        # rows along the contour: the long side follows it
        dx, dy, sl = site.downhill(cx, cy)
        row = math.atan2(dy, dx) + math.pi / 2
        poly = _grove_polygon(site, cx, cy, w, h, row)
        x0, y0, x1, y1 = poly.bounds
        lim = site.half - 30
        if min(x0, y0) < -lim or max(x1, y1) > lim:
            continue
        if poly.intersects(blocked):
            poly = poly.difference(blocked)
            if poly.geom_type != "Polygon" or poly.area < 0.6 * w * h:
                continue
            poly = poly.simplify(1.0)
        xs, ys = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y0, y1, 9))
        inside = [(x, y) for x, y in zip(xs.ravel(), ys.ravel()) if poly.contains(Point(x, y))]
        if len(inside) < 10:
            continue
        slope = float(np.mean([site.sample(site.slope, x, y)[0] for x, y in inside]))
        wet = float(np.max([site.sample(site.wet, x, y)[0] for x, y in inside]))
        if slope > rules["max_slope_deg"] or wet > 0.5:
            continue
        g = {"id": next_id(spec, "groves"), "species": "pine" if forest else pick(r, species_w),
             "polygon": [[round(x, 1), round(y, 1)] for x, y in poly.exterior.coords[:-1]],
             "spacing_m": round(float(r.uniform(*rules["spacing_m"])), 1),
             "row_deg": round(math.degrees(row), 1), "forest": bool(forest)}
        spec["entities"].setdefault("groves", []).append(g)
        out.append(g)
    return out


def plant(spec, site, style, seed, centre, log=print):
    """All planting for a settlement: groves and forest first (they need open
    land), then street trees, then yard trees."""
    cfg = _cfg()
    rules = style if isinstance(style, dict) else cfg["planting"][style]   # a style definition, or a style name
    sp = cfg["species"]
    groves = plant_groves(spec, site, rules["groves"], rules["groves"]["species"], seed, centre)
    forest = plant_groves(spec, site, rules["forest"], {"pine": 1}, seed, centre, forest=True)
    street = plant_streets(spec, site, rules["street"], sp, seed)
    yards = plant_yards(spec, site, rules["yard"], sp, seed)
    log(f"planted {len(groves)} groves, {len(forest)} forest patches, {len(street)} street trees, {len(yards)} yard trees")
    return {"groves": groves, "forest": forest, "street": street, "yards": yards}


def grove_trees(g, seed):
    """Tree positions inside a grove: rows along `row_deg` at `spacing_m` with a
    little jitter (orchard), or a jittered irregular lattice with gaps (forest)."""
    r = rng(seed, "grove", g["id"])
    poly = Polygon(g["polygon"])
    s = g["spacing_m"]
    a = math.radians(g.get("row_deg", 0.0))
    c, sn = math.cos(a), math.sin(a)
    cx, cy = poly.centroid.x, poly.centroid.y
    R = math.hypot(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) / 2 + s
    out = []
    jit = 0.35 if g.get("forest") else 0.08
    for u in np.arange(-R, R, s):
        for v in np.arange(-R, R, s * (1.0 if g.get("forest") else 1.15)):
            if g.get("forest") and r.random() < 0.18:
                continue   # natural gaps
            uu, vv = u + r.normal(0, jit * s), v + r.normal(0, jit * s)
            x, y = cx + uu * c - vv * sn, cy + uu * sn + vv * c
            if poly.buffer(-1.0).contains(Point(x, y)):
                out.append((float(x), float(y)))
    return out


def plant_fields(spec, site, rules, seed, tries=600):
    """Farm fields on gentle, dry, open ground at a distance from the settlement:
    rectangles along the contour, often split into strips with different crops
    (summer: ploughed / harvested stubble / fallow), like the valley floors
    around Galilee villages. rules: count, size_m [[w0, w1], [h0, h1]],
    max_slope_deg, dist_m [d0, d1], strips [n0, n1], crops {crop: weight}."""
    if not rules or rules.get("count", 0) <= 0:
        return []
    r = rng(seed, "plant", "fields")
    blds = spec["entities"].get("buildings", [])
    village = unary_union([footprint(b) for b in blds]) if blds else Point(0, 0).buffer(50)
    out = []
    for _ in range(tries):
        if len(out) >= rules["count"]:
            break
        blocked = _blocked(spec, extra_m=3.0)
        ring = village.convex_hull.buffer(float(r.uniform(*rules["dist_m"]))).exterior
        p = ring.interpolate(float(r.uniform(0, 1)), normalized=True)
        w, h = float(r.uniform(*rules["size_m"][0])), float(r.uniform(*rules["size_m"][1]))
        dx, dy, _ = site.downhill(p.x, p.y)
        ang = math.atan2(dy, dx) + math.pi / 2          # long side along the contour
        poly = _grove_polygon(site, p.x, p.y, w, h, ang)
        x0, y0, x1, y1 = poly.bounds
        lim = site.half - 25
        if min(x0, y0) < -lim or max(x1, y1) > lim or poly.intersects(blocked):
            continue
        xs, ys = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y0, y1, 9))
        inside = [(x, y) for x, y in zip(xs.ravel(), ys.ravel()) if poly.contains(Point(x, y))]
        if len(inside) < 10:
            continue
        if float(np.mean([site.sample(site.slope, x, y)[0] for x, y in inside])) > rules["max_slope_deg"]:
            continue
        if float(np.max([site.sample(site.wet, x, y)[0] for x, y in inside])) > 0.6:
            continue
        n_strips = int(r.integers(rules["strips"][0], rules["strips"][1] + 1))
        c, s_ = math.cos(ang), math.sin(ang)
        for k in range(n_strips):   # strips across the short side, each its own crop
            a0, a1 = -h / 2 + k * h / n_strips + 0.6, -h / 2 + (k + 1) * h / n_strips - 0.6
            pts = [(-w / 2, a0), (w / 2, a0), (w / 2, a1), (-w / 2, a1)]
            f = {"id": next_id(spec, "fields"), "crop": pick(r, rules["crops"]),
                 "polygon": [[round(p.x + u * c - v * s_, 1), round(p.y + u * s_ + v * c, 1)] for u, v in pts],
                 "row_deg": round(math.degrees(ang) % 180, 1)}
            spec["entities"].setdefault("fields", []).append(f)
            out.append(f)
    return out
