"""Access for every building: a road at its front door, or a path from its door
that joins the road network (pure Python).

  frontage   the front door is within `frontage_m` of a road edge with nothing in
             between: the house opens onto that road
  path       otherwise an access path (alley / driveway) from the door, following
             one shortest-path tree grown from all roads over the land between
             the buildings (buildings are blocked, slopes cost more). A path ends
             where it reaches a road or an access path laid before it, so paths
             merge into shared alleys like in real dense villages.

Each building gets `access` = {"road": id} or {"path": id}; each access path
records the buildings it `serves` and what it `joins` (a road or a path).
`access_report` checks that every building reaches the road network.
"""
import math

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import contains_xy
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from synth.common.config import load_json
from synth.common.rng import rng
from synth.planner import roads as rplan
from synth.spec.buildings import footprint
from synth.spec.scene_spec import next_id

OFFS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def _half(rd, rc):
    c = rc.get(rd.get("class"), {})
    return rd.get("width", c.get("width", 4.0)) / 2


def door(b):
    """Point just outside the middle of the front wall."""
    yaw = math.radians(b.get("yaw_deg", 0.0))
    front = np.array([math.sin(yaw), -math.cos(yaw)])
    return np.asarray(b["center"], float) + front * (b["depth"] / 2 + 0.8), front


def frontage_road(b, spec, rc, fps, frontage_m=9.0):
    """Road the building opens onto directly (door within frontage_m of its edge,
    no building in between), or None."""
    d, _ = door(b)
    P = Point(d)
    best = None
    for rd in spec["entities"].get("roads", []):
        L = LineString(np.asarray(rd["centerline"])[:, :2])
        gap = L.distance(P) - _half(rd, rc)
        if gap <= frontage_m and (best is None or gap < best[0]):
            best = (gap, rd["id"], L)
    if best is None:
        return None
    q = best[2].interpolate(best[2].project(P))
    seg = LineString([d, (q.x, q.y)])
    if any(seg.intersects(f) for bid, f in fps.items() if bid != b["id"]):
        return None
    return best[1]


def plan_access(spec, site, seed, rules=None, log=print):
    """Give every building access (see module doc). rules: frontage_m, surface
    (paved | dirt), width [min, max], res_m (grid)."""
    rules = dict({"frontage_m": 9.0, "surface": "paved", "width": [1.6, 2.6], "res_m": 1.0}, **(rules or {}))
    rc = load_json("configs/roads.json")["classes"]
    blds = spec["entities"].get("buildings", [])
    if not blds:
        return []
    fps = {b["id"]: footprint(b) for b in blds}
    need = []
    for b in blds:
        rid = frontage_road(b, spec, rc, fps, rules["frontage_m"])
        if rid:
            b["access"] = {"road": rid}
        else:
            need.append(b)
    if not need:
        return []
    # --- cost grid over the core: buildings blocked, roads are the sources
    res = rules["res_m"]
    half = site.half
    n = int(2 * half / res) + 1
    c = np.arange(n) * res - half
    X, Y = np.meshgrid(c, c)
    blocked = contains_xy(unary_union([f.buffer(1.0) for f in fps.values()]), X, Y)
    roads_u = unary_union([LineString(np.asarray(rd["centerline"])[:, :2]).buffer(_half(rd, rc))
                           for rd in spec["entities"].get("roads", [])])
    source = contains_xy(roads_u, X, Y) & ~blocked
    slope = ndimage.zoom(site.slope, n / site.slope.shape[0], order=1)[:n, :n]
    cost = 1.0 + (np.clip(slope, 0, 60) / 15.0) ** 2
    ok = ~blocked
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, w = [], [], []
    for di, dj in OFFS:
        a = idx[max(0, -di):n - max(0, di), max(0, -dj):n - max(0, dj)]
        b_ = idx[max(0, di):n - max(0, -di), max(0, dj):n - max(0, -dj)]
        oka = ok.ravel()[a.ravel()] & ok.ravel()[b_.ravel()]
        L = res * math.hypot(di, dj)
        ww = L * 0.5 * (cost.ravel()[a.ravel()] + cost.ravel()[b_.ravel()])
        rows.append(a.ravel()[oka])
        cols.append(b_.ravel()[oka])
        w.append(ww[oka])
    G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n)).tocsr()
    src = idx[source]
    if not len(src):
        return []
    dist, pred, _ = dijkstra(G, indices=src, min_only=True, return_predecessors=True)
    cell = lambda p: (int(np.clip(round((p[1] + half) / res), 0, n - 1)), int(np.clip(round((p[0] + half) / res), 0, n - 1)))
    xy_of = lambda k: (float(k % n) * res - half, float(k // n) * res - half)
    owner = {}          # grid cell -> access path id already passing there
    r = rng(seed, "access")
    out = []
    # nearest-to-road first, so far houses join the alleys of nearer ones
    need.sort(key=lambda b: dist[idx[cell(door(b)[0])]] if np.isfinite(dist[idx[cell(door(b)[0])]]) else 1e9)
    for b in need:
        d, front = door(b)
        # start: the door cell, else the reachable cell closest to the door around the house
        starts = [d] + [np.asarray(b["center"]) + rr * np.array([math.cos(t), math.sin(t)])
                        for rr in (max(b["width"], b["depth"]) / 2 + 1.5, max(b["width"], b["depth"]) / 2 + 3.5)
                        for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
        k0 = None
        for p in starts:
            i, j = cell(p)
            k = idx[i, j]
            if ok[i, j] and np.isfinite(dist[k]):
                k0 = k
                break
        if k0 is None:
            continue
        chain, k, joins = [k0], k0, None
        while not source.ravel()[k]:
            if k in owner:
                joins = ("path", owner[k])
                break
            k = pred[k]
            if k < 0:
                break
            chain.append(k)
        if k < 0:
            continue
        if joins is None:   # reached a road: which one
            P = Point(xy_of(k))
            near = min(spec["entities"]["roads"], key=lambda rd: LineString(np.asarray(rd["centerline"])[:, :2]).distance(P))
            joins = ("road", near["id"])
        pts = np.array([d] + [xy_of(kk) for kk in chain])
        if len(pts) < 2 or LineString(pts).length < 1.0:
            b["access"] = {joins[0]: joins[1]}
            continue
        pts = rplan.resample_polyline(pts, 0.5)[0]
        wdt = round(float(r.uniform(*rules["width"])), 2)
        if len(pts) > 6:   # soften the grid steps, keep both ends
            sm = ndimage.gaussian_filter1d(pts, 2.0, axis=0, mode="nearest")
            sm[0], sm[-1] = pts[0], pts[-1]
            pts = sm
        from synth.spec.network import _keep_clear   # never cut into a wall (other houses, and its own beyond the door)
        others = [f for bid, f in fps.items() if bid != b["id"]]
        pts = _keep_clear(pts, others, min(wdt / 2 + 0.15, 1.0))
        line = LineString(pts)
        near = [f for f in others if f.distance(line) < 3.0]
        if near:   # narrow the alley to its tightest gap (at least 1 m)
            gap = min(f.distance(line) for f in near)
            wdt = round(max(1.0, min(wdt, 2 * (gap - 0.1))), 2)
        pa = {"id": next_id(spec, "paths"), "kind": "access", "surface": rules["surface"],
              "width": wdt,
              "centerline": [[round(float(x), 2), round(float(y), 2)] for x, y in LineString(pts).simplify(0.1).coords],
              "serves": [b["id"]], joins[0] + "_join": joins[1]}
        spec["entities"].setdefault("paths", []).append(pa)
        for kk in chain:
            owner.setdefault(kk, pa["id"])
        b["access"] = {"path": pa["id"]}
        out.append(pa)
    log(f"access: {len(blds) - len(need)} buildings on a road frontage, {len(out)} access paths for the others")
    return out


def access_report(spec):
    """Building ids without access to the road network (must be empty): no road
    at the front door and no chain of access paths ending on a road."""
    paths = {p["id"]: p for p in spec["entities"].get("paths", [])}
    roads = {rd["id"] for rd in spec["entities"].get("roads", [])}

    def reaches_road(pid, seen=()):
        p = paths.get(pid)
        if p is None or pid in seen:
            return False
        if p.get("road_join") in roads:
            return True
        return reaches_road(p.get("path_join"), seen + (pid,))

    bad = []
    for b in spec["entities"].get("buildings", []):
        a = b.get("access", {})
        if not ((a.get("road") in roads) or (a.get("path") and reaches_road(a["path"]))):
            bad.append(b["id"])
    return bad


def paths_through_buildings(spec):
    """Access paths whose surface overlaps a building (must be empty)."""
    from shapely import STRtree
    fps = [footprint(b) for b in spec["entities"].get("buildings", [])]
    ids = [b["id"] for b in spec["entities"].get("buildings", [])]
    if not fps:
        return []
    tree = STRtree(fps)
    bad = []
    for p in spec["entities"].get("paths", []):
        if p.get("kind") != "access":
            continue
        body = LineString(np.asarray(p["centerline"])[:, :2]).buffer(p["width"] / 2 * 0.9)
        served = set(p.get("serves", []))   # an alley meets its own house at the door
        if any(fps[i].intersects(body) for i in tree.query(body) if ids[i] not in served):
            bad.append(p["id"])
    return bad
