"""Stone terraces and field walls: the man-made structure of Levant hillsides.

Terraces: inside a zone of moderate slope near the village the terrain is
reshaped into gently sloping benches (step height h) while keeping its
micro-relief. A dry-stone retaining wall follows each step edge (a contour
of the original surface at (k + 0.5) h).
Field walls: low dry-stone walls along agricultural plot boundaries, with
gaps for access.
"""
import math

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours

from synth.common.rng import rng
from synth.planner.fields import smoothstep, value_noise


def plan_terraces(seed, grid, occ, village, cfg, slope):
    tc = cfg.get("terraces")
    if not tc:
        return [], None
    r = rng(seed, "terraces")
    X, Y = grid.xy()
    d_vil = np.hypot(X - village[0], Y - village[1])
    lo, hi = tc["distance_from_village_m"]
    blob = value_noise(grid.z.shape, grid.res, tc["zone_noise_m"], r, octaves=2)
    zone = (smoothstep(slope, tc["slope_deg"][0], tc["slope_deg"][0] + 3) * (1 - smoothstep(slope, tc["slope_deg"][1], tc["slope_deg"][1] + 4))
            * smoothstep(d_vil, lo, lo + 30) * (1 - smoothstep(d_vil, hi - 60, hi))
            * smoothstep(blob, 0.5 - tc["coverage"] * 0.3, 0.55 - tc["coverage"] * 0.3))
    # keep clear of roads, houses, yards, fields (rasterised occupancy -> grid)
    bits = 1 | 2 | 4 | 8 | 256
    blocked = ndimage.zoom(((occ.mask & bits) > 0).astype(float), occ.res / grid.res, order=0)[: grid.n, : grid.n]
    blocked = np.pad(blocked, ((0, grid.n - blocked.shape[0]), (0, grid.n - blocked.shape[1])))
    blocked = ndimage.binary_dilation(blocked > 0.5, iterations=int(6 / grid.res))
    zone = ndimage.gaussian_filter(np.where(blocked, 0, zone), 2.0 / grid.res)
    zone = np.clip((zone - 0.25) / 0.5, 0, 1)
    if zone.max() < 0.5:
        return [], zone
    h = float(r.uniform(*tc["step_m"]))
    zs = grid.z_smooth
    level = np.round(zs / h) * h
    bench = level + tc["bench_keep_slope"] * (zs - level)
    grid.z = grid.z + zone * (bench - zs)          # keep micro-relief and other edits
    grid.z_smooth = grid.z_smooth + zone * (bench - zs)
    # walls along step edges: contours of the ORIGINAL surface at (k + 0.5) h
    walls = []
    kmin, kmax = int(np.floor(zs[zone > 0.5].min() / h)), int(np.ceil(zs[zone > 0.5].max() / h))
    gy, gx = np.gradient(zs, grid.res)
    for k in range(kmin, kmax + 1):
        for c in find_contours(zs, (k + 0.5) * h):
            ij = c  # (row, col) -> local metres
            xs = ij[:, 1] * grid.res - grid.half
            ys = ij[:, 0] * grid.res - grid.half
            inside = grid.sample(xs, ys, zone) > 0.6
            # split into runs inside the zone
            idx = np.nonzero(np.diff(np.concatenate([[0], inside.astype(int), [0]])))[0]
            for a, b in zip(idx[::2], idx[1::2]):
                if (b - a) * grid.res < tc["min_wall_m"]:
                    continue
                px, py = xs[a:b], ys[a:b]
                seg = np.hypot(np.diff(px), np.diff(py))
                s = np.concatenate([[0], np.cumsum(seg)])
                t = np.arange(0, s[-1], 1.0)
                px, py = np.interp(t, s, px), np.interp(t, s, py)
                # wall base = lower bench, sampled just downhill of the edge
                gxs, gys = grid.sample(px, py, gx), grid.sample(px, py, gy)
                gn = np.hypot(gxs, gys) + 1e-9
                zlow = grid.sample(px - gxs / gn * 0.8, py - gys / gn * 0.8)
                walls.append({"id": f"terrace_{len(walls)}", "kind": "wall.terrace", "role": "wall.dry_stone",
                              "height": h + 0.35, "thickness": float(r.uniform(0.5, 0.7)),
                              "points": [[float(x), float(y), float(z - 0.15)] for x, y, z in zip(px, py, zlow)]})
    from shapely.geometry import LineString
    for w in walls:
        occ.add_polygon(LineString(np.asarray(w["points"])[:, :2]).buffer(w["thickness"] / 2 + 0.2), "wall")
    return walls, zone


def field_walls(seed, grid, fields, occ, cfg):
    fc = cfg.get("fields", {})
    out = []
    for f in fields:
        r = rng(seed, "field_wall", f["id"])
        if r.random() > fc.get("wall_p", 0.6):
            continue
        poly = np.asarray(f["polygon"])
        n = len(poly)
        for e in range(n):
            if r.random() < 0.25:  # some edges are open
                continue
            a, b = poly[e], poly[(e + 1) % n]
            L = float(np.linalg.norm(b - a))
            if L < 6:
                continue
            g = float(r.uniform(2.5, 4.0))  # access gap
            g0 = float(r.uniform(0.1, 0.9)) * (L - g)
            for s0, s1 in ((0.5, g0), (g0 + g, L - 0.5)):
                if s1 - s0 < 3:
                    continue
                t = np.arange(s0, s1, 1.0) / L
                p = a + (b - a) * t[:, None] + r.normal(0, 0.08, (len(t), 2))
                z = grid.sample(p[:, 0], p[:, 1])
                out.append({"id": f"fwall_{f['id']}_{e}_{len(out)}", "kind": "wall.field", "role": "wall.dry_stone",
                            "height": float(r.uniform(0.55, 0.95)), "thickness": float(r.uniform(0.5, 0.8)),
                            "points": [[float(x), float(y), float(zz - 0.1)] for (x, y), zz in zip(p, z)]})
    return out
