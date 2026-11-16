"""Yard walls and fences around houses (pure Python).

Most houses in Galilee villages and towns stand in a walled plot: a stone,
block or plastered wall (or a metal fence on a low plinth) around the yard,
with a gate where the house meets its road or access alley.

  plot   the house footprint grown by a yard depth, shared fairly with the
         neighbours (Voronoi cells of the building centres), kept clear of
         roads, other buildings, every alley and every other house's way to
         its road
  wall   the plot boundary, minus a gate where the house's way out crosses it
         (the straight line from the front door to its road, or its access
         alley), minus where it would touch the house itself; a wall shared
         with a neighbour is built once

Each wall is one entity: {"id": "wall_NNN", "type": stone | block | plaster |
fence, "encloses": building id, "height", "thickness", "segments": [polyline,
...], "yard": polygon, "gate": [x, y]}. `wall_report` checks that no wall
crosses a building, road or path and that every walled house can get out.
"""
import math

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import linemerge, unary_union, voronoi_diagram

from synth.common.config import load_json
from synth.common.rng import rng
from synth.spec.access import _half, door
from synth.spec.buildings import _weighted, footprint, pick
from synth.spec.scene_spec import next_id

WALL_TYPES = ("stone", "block", "plaster", "fence")
DEFAULTS = {
    "share": {"arab_house": 0.8, "house": 0.75, "villa": 0.9, "moshav_house": 0.7, "old_basalt": 0.85,
              "unfinished": 0.35, "public": 0.9, "mosque": 0.6, "apartment": 0.3, "midrise": 0.2},
    "types": {"stone": 0.35, "block": 0.3, "plaster": 0.25, "fence": 0.1},
    "height_m": {"stone": [1.0, 1.8], "block": [1.5, 2.3], "plaster": [1.3, 2.1], "fence": [1.2, 1.8]},
    "thickness_m": {"stone": 0.4, "block": 0.2, "plaster": 0.25, "fence": 0.2},
    "depth_m": [3.0, 7.0],
    "gate_m": [1.4, 4.0],
    "min_yard_m2": 12.0,
}
# plaster walls take the house's facade colour; stone and block their own
STONE_TINTS = [[[1.0, 0.95, 0.86], 3], [[0.96, 0.9, 0.8], 2], [[1.02, 0.98, 0.92], 1]]
BLOCK_TINTS = [[[1.0, 1.0, 1.0], 3], [[0.92, 0.92, 0.9], 2]]

CLEAR_M = 0.6    # a plot stays this far from a road edge, alley or other house's way out
GAP_M = 0.35     # a wall stops this far from a building, road edge or alley edge


def _ways(spec, rc, fps):
    """Each building's way out: (building id, line, half width). Frontage
    houses: the straight line from the front door to the nearest point of its
    road. Access-path houses: their alley (one way per alley, owned by every
    house it serves)."""
    roads = {rd["id"]: rd for rd in spec["entities"].get("roads", [])}
    out = []
    for b in spec["entities"].get("buildings", []):
        rid = b.get("access", {}).get("road")
        if rid in roads:
            d, _ = door(b)
            L = LineString(np.asarray(roads[rid]["centerline"])[:, :2])
            q = L.interpolate(L.project(Point(d)))
            out.append(({b["id"]}, LineString([d, (q.x, q.y)]), 0.8))
    for pa in spec["entities"].get("paths", []):
        out.append((set(pa.get("serves", [])), LineString(np.asarray(pa["centerline"])[:, :2]), pa.get("width", 1.0) / 2))
    return out


def plan_yards(spec, site, seed, rules=None, log=print):
    """Walls around the plots of the houses (see module doc). rules (place file
    `yards`): share per building type, types (weights), height_m / thickness_m
    per type, depth_m, gate_m, min_yard_m2."""
    R = dict(DEFAULTS, **(rules or {}))
    rc = load_json("configs/roads.json")["classes"]
    cat = load_json("configs/buildings.json")
    blds = spec["entities"].get("buildings", [])
    if not blds:
        return []
    fps = {b["id"]: footprint(b) for b in blds}
    lim = site.half - 2.0
    world = box(-lim, -lim, lim, lim)
    cells = {}
    pts = MultiPoint([fps[b["id"]].centroid for b in blds]) if len(blds) > 1 else None
    if pts is not None:
        for cell in voronoi_diagram(pts, envelope=world).geoms:
            for b in blds:
                if b["id"] not in cells and cell.contains(fps[b["id"]].centroid):
                    cells[b["id"]] = cell.intersection(world)
                    break
    road_body = []
    for rd in spec["entities"].get("roads", []):
        L = LineString(np.asarray(rd["centerline"])[:, :2])
        hw = _half(rd, rc) + rc.get(rd.get("class"), {}).get("shoulder", 0.0)
        road_body.append(L.buffer(hw))
        if rd.get("end_cap"):
            road_body.append(Point(rd["centerline"][-1][:2]).buffer(rd["end_cap"]))
    road_u = unary_union(road_body)
    road_clear, road_gap = road_u.buffer(CLEAR_M), road_u.buffer(GAP_M)
    ways = _ways(spec, rc, fps)
    others_clear = {}   # cached per building: other buildings' footprints + ways
    built = []          # buffered walls already laid (shared boundaries are built once)
    r = rng(seed, "yards")
    out = []
    for b in blds:
        t = b.get("type", "house")
        if r.random() >= R["share"].get(t, 0.0) * R.get("share_scale", 1.0):
            continue
        fp = fps[b["id"]]
        depth = float(r.uniform(*R["depth_m"]))
        yard = fp.buffer(depth, join_style="mitre").intersection(cells.get(b["id"], world))
        near = fp.buffer(depth + 3.0)
        clear = [road_clear]
        clear += [f.buffer(0.7) for bid, f in fps.items() if bid != b["id"] and f.intersects(near)]
        clear += [L.buffer(hw + CLEAR_M) for own, L, hw in ways if b["id"] not in own and L.intersects(near)]
        yard = yard.difference(unary_union(clear))
        if yard.is_empty:
            continue
        parts = getattr(yard, "geoms", [yard])
        yard = max(parts, key=lambda g: g.intersection(fp.buffer(1.5)).area)
        if not isinstance(yard, Polygon) or yard.difference(fp).area < R["min_yard_m2"]:
            continue
        yard = yard.simplify(0.2)
        ring = yard.exterior
        # gate: where this house's way out crosses its plot boundary
        own = [(L, hw) for o, L, hw in ways if b["id"] in o]
        gate_pt, gaps = None, [fp.buffer(GAP_M), road_gap]
        gw = float(r.uniform(*R["gate_m"]))
        for L, hw in own:
            x = L.intersection(ring)
            if not x.is_empty:
                p = x if x.geom_type == "Point" else min(getattr(x, "geoms", [x]), key=lambda g: g.distance(Point(L.coords[0])))
                gate_pt = p.centroid
                gaps.append(gate_pt.buffer(max(gw, 2 * hw + 0.4) / 2))
        if gate_pt is None:   # no way out recorded: a gate in front of the door
            gate_pt = ring.interpolate(ring.project(Point(door(b)[0])))
            gaps.append(gate_pt.buffer(gw / 2))
        gaps += [L.buffer(hw + GAP_M) for o, L, hw in ways if L.intersects(near)]
        line = ring.difference(unary_union(gaps))
        mine = [w for w in built if w.intersects(near)]
        if mine:
            line = line.difference(unary_union(mine))
        if line.is_empty:
            continue
        merged = linemerge(line) if line.geom_type == "MultiLineString" else line
        segs = [g for g in getattr(merged, "geoms", [merged]) if g.geom_type == "LineString" and g.length >= 1.2]
        if not segs:
            continue
        wt = pick(r, R["types"])
        if t == "old_basalt":
            wt = "stone"
        tint = (list(b["tint"]) if wt == "plaster" and "tint" in b else
                _weighted(r, STONE_TINTS) if wt == "stone" else
                _weighted(r, BLOCK_TINTS) if wt == "block" else
                _weighted(r, cat["facades"].get("wall.stucco", [[[1, 1, 1], 1]])))
        w = {"id": next_id(spec, "walls"), "type": wt, "encloses": b["id"],
             "height": round(float(r.uniform(*R["height_m"][wt])), 2), "thickness": R["thickness_m"][wt],
             "tint": [round(float(v), 3) for v in tint],
             "gate": [round(gate_pt.x, 2), round(gate_pt.y, 2)],
             "segments": [[[round(x, 2), round(y, 2)] for x, y in s.coords] for s in segs],
             "yard": [[round(x, 2), round(y, 2)] for x, y in list(yard.exterior.coords)[:-1]]}
        spec["entities"].setdefault("walls", []).append(w)
        built += [s.buffer(w["thickness"] / 2 + 0.4) for s in segs]
        out.append(w)
    log(f"yards: {len(out)} walled plots ({dict((k, sum(w['type'] == k for w in out)) for k in WALL_TYPES)})")
    return out


def open_gaps(spec, paths):
    """Cut a gap in every wall a later path (footpath) passes through; walls
    left with nothing longer than 1.2 m are removed."""
    keep = []
    for w in spec["entities"].get("walls", []):
        lines = wall_lines(w)
        body = unary_union(lines).buffer(w["thickness"] / 2)
        cut = [LineString(np.asarray(p["centerline"])[:, :2]).buffer(p.get("width", 1.0) / 2 + GAP_M)
               for p in paths if LineString(np.asarray(p["centerline"])[:, :2]).buffer(p.get("width", 1.0) / 2 + GAP_M).intersects(body)]
        if cut:
            rest = unary_union(lines).difference(unary_union(cut))
            rest = linemerge(rest) if rest.geom_type == "MultiLineString" else rest
            segs = [g for g in getattr(rest, "geoms", [rest]) if g.geom_type == "LineString" and g.length >= 1.2]
            w["segments"] = [[[round(x, 2), round(y, 2)] for x, y in s.coords] for s in segs]
        if w["segments"]:
            keep.append(w)
    spec["entities"]["walls"] = keep


def wall_lines(w):
    return [LineString(s) for s in w["segments"] if len(s) >= 2]


def wall_report(spec):
    """Rule checks for yard walls (each must be empty):
      walls_through_buildings    a wall overlapping a building
      walls_across_roads_paths   a wall on a road or a path
      walled_in_buildings        a house whose way out (door to road, or its
                                 alley) crosses a wall"""
    rc = load_json("configs/roads.json")["classes"]
    walls = spec["entities"].get("walls", [])
    fps = {b["id"]: footprint(b) for b in spec["entities"].get("buildings", [])}
    bodies = {w["id"]: unary_union([L.buffer(w["thickness"] / 2) for L in wall_lines(w)]) for w in walls}
    through = sorted({wid for wid, B in bodies.items() for f in fps.values() if B.intersects(f)})
    across = set()
    for wid, B in bodies.items():
        for rd in spec["entities"].get("roads", []):
            if LineString(np.asarray(rd["centerline"])[:, :2]).buffer(_half(rd, rc)).intersects(B):
                across.add(wid)
        for pa in spec["entities"].get("paths", []):
            if LineString(np.asarray(pa["centerline"])[:, :2]).buffer(pa.get("width", 1.0) / 2 * 0.9).intersects(B):
                across.add(wid)
    allw = unary_union(list(bodies.values())) if bodies else Polygon()
    blds = {b["id"]: b for b in spec["entities"].get("buildings", [])}
    walled_in = sorted(list(own)[0] for own, L, hw in _ways(spec, rc, fps)
                       if len(own) == 1 and blds.get(next(iter(own)), {}).get("access", {}).get("road")
                       and L.intersects(allw))
    return {"walls_through_buildings": through, "walls_across_roads_paths": sorted(across),
            "walled_in_buildings": walled_in}
