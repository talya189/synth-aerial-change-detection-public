"""Agricultural field plots: the dominant large-scale texture of rural Levant aerials.

Rectangular plots on gentle ground around the village, aligned with the
nearest road, separated by uncultivated balks. Each plot has a state
(harvested stubble with combine tracks, ploughed furrows, or fallow) and a
row direction. Stones cleared from a plot are piled along its edges.
"""
import math

import numpy as np
from shapely import affinity
from shapely.geometry import Point, box
from shapely.ops import unary_union

from synth.common.rng import rng


def plan_fields(seed, grid, occ, roads, village, cfg, slope):
    fc = cfg.get("fields")
    if not fc:
        return []
    r = rng(seed, "fields_layout")
    target = int(r.integers(fc["count"][0], fc["count"][1] + 1))
    road_lines = [np.asarray(rd["centerline"])[:, :2] for rd in roads]
    from shapely.geometry import LineString
    roads_u = unary_union([LineString(p).buffer(rd["width"] / 2 + rd.get("shoulder", 0) + 2.0) for p, rd in zip(road_lines, roads)])
    fields, taken = [], []
    tries = 0
    while len(fields) < target and tries < 400:
        tries += 1
        d = float(r.uniform(*fc["distance_from_village_m"]))
        a = float(r.uniform(0, 2 * math.pi))
        c = np.asarray(village[:2]) + d * np.array([math.cos(a), math.sin(a)])
        if not grid.inside(c[0], c[1], 60):
            continue
        # orientation from the nearest road segment: plots follow the road grid
        best = None
        for p in road_lines:
            k = int(np.argmin(np.linalg.norm(p - c, axis=1)))
            dist = float(np.linalg.norm(p[k] - c))
            if best is None or dist < best[0]:
                t = p[min(k + 2, len(p) - 1)] - p[max(k - 2, 0)]
                best = (dist, math.atan2(t[1], t[0]))
        ang = best[1] + float(r.normal(0, math.radians(3)))
        w, h = float(r.uniform(*fc["length_m"])), float(r.uniform(*fc["width_m"]))
        poly = affinity.rotate(box(c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2), ang, origin=tuple(c), use_radians=True)
        # clip by roads (fields stop at the road shoulder) and keep the main part
        poly = poly.difference(roads_u)
        if poly.geom_type != "Polygon":
            poly = max(getattr(poly, "geoms", [poly]), key=lambda g: g.area)
        if poly.area < 0.6 * w * h:
            continue
        if any(poly.buffer(fc["balk_m"]).intersects(o) for o in taken):
            continue
        xs, ys = np.array(poly.exterior.coords).T
        inner = np.array([poly.representative_point().coords[0]])
        sample = np.vstack([np.column_stack([xs, ys]), inner])
        if grid.sample(sample[:, 0], sample[:, 1], slope).max() > fc["max_slope_deg"]:
            continue
        if occ.blocked(sample[:, 0], sample[:, 1], ["building", "yard", "wall"], r=3).any():
            continue
        state = list(fc["states"])[int(r.choice(len(fc["states"]), p=np.array(list(fc["states"].values())) / sum(fc["states"].values())))]
        fid = f"field_{len(fields):02d}"
        fields.append({"id": fid, "state": state, "row_angle": ang % math.pi,
                       "polygon": [list(map(float, xy)) for xy in np.array(poly.exterior.coords)[:-1]]})
        taken.append(poly)
        occ.add_polygon(poly, "field")
    return fields


def edge_stone_piles(seed, grid, fields, cfg, variants):
    """Cleared stones heaped along plot edges (rows of basalt cobbles)."""
    fc = cfg.get("fields", {})
    if not variants or not fields:
        return []
    from synth.planner.scatter import surface_frame, tilt_from_normal
    out = []
    for f in fields:
        r = rng(seed, "field_stones", f["id"])
        poly = np.asarray(f["polygon"] + [f["polygon"][0]])
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        L = seg.sum()
        n = int(L * fc.get("edge_stones_per_m", 0.6))
        s = np.sort(r.uniform(0, L, n))
        cum = np.concatenate([[0], np.cumsum(seg)])
        k = np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1)
        t = (s - cum[k]) / seg[k]
        p = poly[k] + (poly[k + 1] - poly[k]) * t[:, None]
        p += r.normal(0, 0.6, p.shape)
        z, nrm = surface_frame(grid, p[:, 0], p[:, 1], 0.4)
        tx, ty = tilt_from_normal(nrm, 1.0)
        for i in range(n):
            v = variants[int(r.integers(len(variants)))]
            sc = float(r.uniform(1.3, 3.2))
            out.append({"id": f"{f['id']}_stone{i}", "type": "model", "kind": "rock.small_dark", "role": "rock.small_dark",
                        "variant": v[0], "x": float(p[i, 0]), "y": float(p[i, 1]), "z": float(z[i] - 0.3 * v[1] * sc),
                        "rot": [float(tx[i]), float(ty[i]), float(r.uniform(0, 2 * math.pi))], "scale": sc})
    return out


def fields_raster(grid, fields, res):
    """RGBA uint8, row 0 = north: R = state code, G = row angle / pi, B = inside mask."""
    from PIL import Image, ImageDraw
    n = int(round(2 * grid.half / res)) + 1
    codes = {"stubble": 85, "ploughed": 170, "fallow": 255}
    layers = [Image.new("L", (n, n), 0) for _ in range(3)]
    draws = [ImageDraw.Draw(l) for l in layers]
    px = lambda xy: [((x + grid.half) / res, (grid.half - y) / res) for x, y in xy]
    for f in fields:
        pts = px(f["polygon"])
        draws[0].polygon(pts, fill=codes[f["state"]])
        draws[1].polygon(pts, fill=int(round(f["row_angle"] / math.pi * 255)))
        draws[2].polygon(pts, fill=255)
    arr = np.stack([np.asarray(l) for l in layers] + [np.full((n, n), 255, np.uint8)], -1)
    return arr
