"""Parcels along streets, building programs, yard walls and rooftop equipment."""
import math

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from synth.common.rng import rng
from synth.planner.fields import slope_deg


def _pick(r, table):
    keys = list(table)
    p = np.array([table[k] for k in keys], float)
    return keys[r.choice(len(keys), p=p / p.sum())]


def _rect(cx, cy, w, d, yaw):
    """Rectangle centred at (cx, cy); its local +y axis (depth) points along yaw."""
    return affinity.rotate(affinity.translate(box(-w / 2, -d / 2, w / 2, d / 2), cx, cy), yaw, origin=(cx, cy), use_radians=True)


def plan_buildings(seed, grid, streets, road_polys, occ, cfg):
    """Walk along both sides of each street placing parcels at frontage spacing.
    A building faces the street (front facade parallel to the road), sits
    `setback` behind the road edge, and must avoid roads, other parcels and
    steep ground."""
    bcfg = cfg["buildings"]
    rr = rng(seed, "buildings", "layout")
    target = int(rr.integers(bcfg["target_count"][0], bcfg["target_count"][1] + 1))
    roads_union = unary_union(road_polys)
    slope = slope_deg(grid.z, grid.res)
    parcels, buildings, walls = [], [], []
    taken = []
    candidates = []
    for s_idx, st in enumerate(streets):
        p = np.asarray(st["centerline"])[:, :2]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        for side in (-1, 1):
            pos = float(rr.uniform(8, 16))
            while pos < s[-1] - 8:
                k = int(np.searchsorted(s, pos))
                k = min(max(k, 1), len(p) - 1)
                t = p[k] - p[k - 1]
                t /= np.linalg.norm(t) + 1e-9
                nrm = np.array([-t[1], t[0]]) * side
                candidates.append((s_idx, side, pos, p[k], t, nrm))
                pos += float(rr.uniform(*bcfg["frontage_spacing_m"]))
    order = rr.permutation(len(candidates))
    for ci in order:
        if len(buildings) >= target:
            break
        s_idx, side, pos, c, t, nrm = candidates[ci]
        bid = f"b{len(buildings):03d}"
        r = rng(seed, "building", s_idx, side, round(pos, 1))
        w = float(r.uniform(*bcfg["width_m"]))
        d = float(r.uniform(*bcfg["depth_m"]))
        setback = float(r.uniform(*bcfg["setback_m"]))
        hw = streets[s_idx]["width"] / 2 + streets[s_idx].get("shoulder", 0)
        yaw = math.atan2(nrm[1], nrm[0]) - math.pi / 2 + float(r.normal(0, math.radians(2.5)))
        centre = c + nrm * (hw + setback + d / 2)
        fp = _rect(centre[0], centre[1], w, d, yaw)
        back = float(r.uniform(3, 7))
        side_gap = float(r.uniform(1.5, 3.5))
        parcel = _rect(*(c + nrm * (hw + 0.6 + (setback + d + back) / 2)), w + 2 * side_gap, setback + d + back - 0.6, yaw)
        annex = None
        if r.random() < bcfg["annex_p"]:
            aw, ad = float(r.uniform(3.5, 6)), float(r.uniform(4, d * 0.8))
            sgn = 1 if r.random() < 0.5 else -1
            local = np.array([sgn * (w / 2 + aw / 2), -d / 2 + ad / 2 + float(r.uniform(0, d - ad))])
            rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            ac = centre + rot @ local
            annex = _rect(ac[0], ac[1], aw, ad, yaw)
            parcel = parcel.union(annex.buffer(side_gap, join_style="mitre"))
        full = fp
        if annex is not None:
            # touching rectangles can leave a float gap; close it with a tiny buffer
            full = fp.buffer(0.02, join_style="mitre").union(annex.buffer(0.02, join_style="mitre"))
            if full.geom_type != "Polygon":
                annex, full = None, fp
        # constraints
        if not grid.inside(np.array(full.exterior.coords)[:, 0], np.array(full.exterior.coords)[:, 1], margin=15).all():
            continue
        if full.buffer(1.0).intersects(roads_union) or parcel.intersects(roads_union.buffer(0.3)):
            continue
        if any(parcel.intersects(o) for o in taken):
            continue
        xs, ys = np.array(full.exterior.coords).T
        mask_pts = np.array([[x, y] for x, y in zip(np.linspace(xs.min(), xs.max(), 6), np.linspace(ys.min(), ys.max(), 6))])
        sl = grid.sample(np.append(xs, mask_pts[:, 0]), np.append(ys, mask_pts[:, 1]), slope)
        if sl.max() > bcfg["max_slope_deg"]:
            continue
        taken.append(parcel)
        # ---- building program
        floors = int(_pick(r, bcfg["floors_p"]))
        fh = float(r.uniform(*bcfg["floor_height_m"]))
        tiled = r.random() < bcfg["roof_tiled_p"]
        wall_role = _pick(r, bcfg["wall_material_p"])
        tint = bcfg["stucco_tints"][int(r.integers(len(bcfg["stucco_tints"])))]
        corner = np.array(full.exterior.coords)[:-1]
        zc = grid.sample(corner[:, 0], corner[:, 1])
        zin = grid.sample(*np.array(full.representative_point().coords).T)
        ground_hi = float(max(zc.max(), zin.max()))
        ground_lo = float(min(zc.min(), zin.min()))
        b = {
            "id": bid, "kind": "building.house" if floors <= 2 else "building.residential",
            "footprint": [list(map(float, xy)) for xy in np.array(fp.exterior.coords)[:-1]],
            "annex": [list(map(float, xy)) for xy in np.array(annex.exterior.coords)[:-1]] if annex is not None else None,
            "annex_floors": max(1, floors - 1) if annex is not None else 0,
            "yaw": yaw, "width": w, "depth": d, "front_normal": [-float(nrm[0]), -float(nrm[1])],
            "floors": floors, "floor_height": fh,
            "floor_z": ground_hi + 0.2, "foundation_z": ground_lo - 0.4,
            "roof": "tiled_hip" if tiled else "flat",
            "parapet": float(r.uniform(*bcfg["parapet_m"])) if not tiled else 0.0,
            "wall_role": wall_role, "tint": tint, "plinth_role": "wall.stone",
            "expandable": bool(r.random() < bcfg["expandable_p"]) and not tiled,
            "window_spacing": float(r.uniform(2.6, 3.8)), "window_w": float(r.uniform(1.0, 1.6)),
            "window_h": float(r.uniform(1.1, 1.5)), "shutters_p": float(r.uniform(0.2, 0.9)),
            "balcony": {"width_frac": float(r.uniform(0.4, 0.75)), "depth": float(r.uniform(1.1, 1.6)),
                        "offset_frac": float(r.uniform(-0.15, 0.15))} if floors > 1 and r.random() < 0.6 else None,
            "street": s_idx, "seed_path": ["building", bid],
        }
        if not tiled:
            b["rooftop"] = _rooftop(r, b, bcfg["rooftop"])
        buildings.append(b)
        parcels.append({"id": f"p{len(parcels):03d}", "building": bid,
                        "polygon": [list(map(float, xy)) for xy in np.array(parcel.exterior.coords)[:-1]]})
        occ.add_polygon(full.buffer(0.3, join_style="mitre"), "building")
        occ.add_polygon(parcel, "yard")
        if r.random() < bcfg["yard_wall_p"]:
            new_walls = _yard_walls(r, bid, parcel, nrm, bcfg, grid)
            walls.extend(new_walls)
            for wl in new_walls:
                occ.add_polygon(LineString(np.asarray(wl["points"])[:, :2]).buffer(0.25), "wall")
    return buildings, parcels, walls


def _rooftop(r, b, rc):
    """Rooftop items in building-local coordinates (x across front, y depth, origin at centre)."""
    w, d = b["width"], b["depth"]
    items, boxes = [], []

    def place(iw, idd, margin=0.7, tries=30):
        for _ in range(tries):
            x = float(r.uniform(-w / 2 + margin + iw / 2, w / 2 - margin - iw / 2))
            y = float(r.uniform(-d / 2 + margin + idd / 2, d / 2 - margin - idd / 2))
            bb = box(x - iw / 2 - 0.3, y - idd / 2 - 0.3, x + iw / 2 + 0.3, y + idd / 2 + 0.3)
            if not any(bb.intersects(o) for o in boxes):
                boxes.append(bb)
                return x, y
        return None

    if r.random() < rc["stairwell_p"]:
        p = place(2.6, 3.2, margin=0.4)
        if p:
            items.append({"type": "stairwell", "x": p[0], "y": p[1], "w": 2.6, "d": 3.2, "h": 2.5, "yaw": 0.0})
    for _ in range(int(r.integers(rc["solar_heaters"][0], rc["solar_heaters"][1] + 1))):
        p = place(1.1, 2.2)
        if p:
            items.append({"type": "solar_heater", "x": p[0], "y": p[1], "yaw": float(r.normal(0, 0.05)) - b["yaw"]})
    for _ in range(int(r.integers(rc["water_tanks"][0], rc["water_tanks"][1] + 1))):
        p = place(1.1, 1.1)
        if p:
            items.append({"type": "water_tank", "x": p[0], "y": p[1], "radius": float(r.uniform(0.45, 0.6)),
                          "h": float(r.uniform(0.9, 1.3)), "color": ["black", "white", "black", "blue"][int(r.integers(4))]})
    for _ in range(int(r.integers(rc["ac_units"][0], rc["ac_units"][1] + 1))):
        p = place(0.9, 0.5, margin=0.4)
        if p:
            items.append({"type": "ac_unit", "x": p[0], "y": p[1], "yaw": float(r.choice([0, math.pi / 2]))})
    if r.random() < rc["dish_p"]:
        p = place(0.8, 0.8)
        if p:
            items.append({"type": "dish", "x": p[0], "y": p[1], "yaw": float(r.uniform(0, 2 * math.pi))})
    for _ in range(int(r.integers(rc.get("clutter", [0, 4])[0], rc.get("clutter", [0, 4])[1] + 1))):
        cw, cd = float(r.uniform(0.5, 1.6)), float(r.uniform(0.5, 1.4))
        p = place(cw, cd, margin=0.4)
        if p:  # stored goods, crates, small sheds
            items.append({"type": "clutter", "x": p[0], "y": p[1], "w": cw, "d": cd, "h": float(r.uniform(0.3, 1.1)),
                          "shade": float(r.uniform(0.35, 0.95))})
    if b["expandable"]:
        items.append({"type": "rebar", "spacing": float(r.uniform(3.5, 5.0))})
    return items


def _yard_walls(r, bid, parcel, nrm, bcfg, grid):
    """Wall along the parcel boundary with a gate gap on the street side."""
    pts = np.array(parcel.exterior.coords)
    h = float(r.uniform(*bcfg["yard_wall_height_m"]))
    role = "wall.stone" if r.random() < 0.45 else "wall.stucco"
    # find the edge facing the street (normal most opposite to nrm)
    best, best_dot = 0, 1e9
    for k in range(len(pts) - 1):
        e = pts[k + 1] - pts[k]
        en = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-9)
        mid = (pts[k] + pts[k + 1]) / 2
        out = en if np.dot(en, mid - np.array(parcel.centroid.coords[0])) > 0 else -en
        dd = float(np.dot(out, nrm))
        if dd < best_dot:
            best, best_dot = k, dd
    out = []
    n = len(pts) - 1
    for k in range(n):
        a, b = pts[k], pts[k + 1]
        if k == best:
            L = np.linalg.norm(b - a)
            gate = min(3.5, L * 0.4)
            g0 = float(r.uniform(0.15, 0.85 - gate / L)) if L > gate + 1 else 0.3
            for seg in ((a, a + (b - a) * g0), (a + (b - a) * (g0 + gate / L), b)):
                if np.linalg.norm(seg[1] - seg[0]) > 0.8:
                    out.append(seg)
        else:
            out.append((a, b))
    walls = []
    for i, (a, b) in enumerate(out):
        L = np.linalg.norm(b - a)
        m = max(2, int(L / 1.0) + 1)
        xs = np.linspace(a[0], b[0], m)
        ys = np.linspace(a[1], b[1], m)
        zs = grid.sample(xs, ys)
        walls.append({"id": f"w_{bid}_{i}", "kind": "wall.yard", "parcel_of": bid, "role": role,
                      "height": h, "thickness": 0.22,
                      "points": [[float(x), float(y), float(z)] for x, y, z in zip(xs, ys, zs)]})
    return walls
