"""Vegetation and rock placement with clustered point processes.

Neyman-Scott / Thomas process: parent points are drawn with intensity
proportional to a terrain suitability map, children scatter normally around
them, then each child is thinned by suitability again, rejected where the
occupancy map is taken, and spaced by a minimum distance. This produces the
clumped, terrain-dependent patterns of real arid vegetation instead of
uniform noise.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

from synth.common.rng import rng
from synth.planner.fields import smoothstep


def thomas_process(r, half, parent_density, children, sigma, suit_fn, margin=4.0):
    area = (2 * (half - margin)) ** 2
    npar = r.poisson(parent_density * area)
    px = r.uniform(-half + margin, half - margin, npar)
    py = r.uniform(-half + margin, half - margin, npar)
    keep = r.random(npar) < suit_fn(px, py)
    px, py = px[keep], py[keep]
    counts = r.poisson(children, len(px))
    cx = np.repeat(px, counts) + r.normal(0, sigma, counts.sum())
    cy = np.repeat(py, counts) + r.normal(0, sigma, counts.sum())
    ok = (np.abs(cx) < half - 1) & (np.abs(cy) < half - 1)
    cx, cy = cx[ok], cy[ok]
    keep = r.random(len(cx)) < np.sqrt(np.clip(suit_fn(cx, cy), 0, 1))
    return cx[keep], cy[keep]


def min_distance_filter(x, y, d):
    if len(x) < 2 or d <= 0:
        return np.ones(len(x), bool)
    tree = cKDTree(np.column_stack([x, y]))
    keep = np.ones(len(x), bool)
    for i, j in sorted(tree.query_pairs(d)):
        if keep[i] and keep[j]:
            keep[j] = False
    return keep


def surface_frame(grid, x, y, r_m=0.6):
    """Terrain height (minimum under the footprint) and unit normal."""
    zs = np.stack([grid.sample(x + dx, y + dy) for dx, dy in ((r_m, 0), (-r_m, 0), (0, r_m), (0, -r_m), (0, 0))])
    gx = (zs[0] - zs[1]) / (2 * r_m)
    gy = (zs[2] - zs[3]) / (2 * r_m)
    n = np.column_stack([-gx, -gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return zs.min(axis=0), n


def tilt_from_normal(n, amount):
    """Euler X/Y tilt aligning +Z partially (amount 0..1) to the normal n."""
    nx, ny, nz = n[:, 0] * amount, n[:, 1] * amount, n[:, 2]
    return np.arctan2(-ny, nz), np.arctan2(nx, nz)


def plan_vegetation(seed, grid, fields, occ, cfg, parcels, streets, assets_variants):
    out = []
    vc = cfg["vegetation"]
    half = grid.half
    slope, twi, north, macro = fields["slope"], fields["twi"], fields["north"], fields["macro"]
    d_road = fields["d_road"]
    d_build = fields["d_build"]
    twi_n = np.clip((twi - np.percentile(twi, 20)) / (np.percentile(twi, 98) - np.percentile(twi, 20) + 1e-9), 0, 1)

    def S(arr, x, y):
        return grid.sample(x, y, arr)

    suit = {
        # dry grass tussocks: gentle-moderate ground, more where wetter and on north faces,
        # sparse on steep rock and next to traffic
        "dry_grass": lambda x, y: np.clip(
            (0.35 + 0.9 * S(twi_n, x, y)) * (1 - smoothstep(S(slope, x, y), 22, 38))
            * (0.75 + 0.35 * S(north, x, y)) * (0.4 + 1.2 * S(macro, x, y) ** 2)
            * smoothstep(S(d_road, x, y), 1.5, 5), 0, 1),
        # dead shrubs (dwarf shrubland, Sarcopoterium-like): rocky slopes and drainage lines
        "dead_shrub": lambda x, y: np.clip(
            (0.25 + 0.8 * smoothstep(S(slope, x, y), 6, 22) + 0.8 * S(twi_n, x, y) ** 2)
            * (0.5 + 0.9 * S(macro, x, y)) * smoothstep(S(d_road, x, y), 2, 6)
            * smoothstep(S(d_build, x, y), 3, 12), 0, 1),
    }
    # shrub lines along drainage (wadi) channels: wetter ground, denser and greener scrub
    suit["wadi_shrub"] = lambda x, y: np.clip(smoothstep(S(twi_n, x, y), 0.72, 0.92) * smoothstep(S(d_road, x, y), 2, 6)
                                              * smoothstep(S(d_build, x, y), 3, 10), 0, 1)
    for name in [n for n in ("dry_grass", "dead_shrub", "wadi_shrub") if n in vc]:
        c = vc[name]
        variants = assets_variants.get(c["role"])
        if not variants:
            continue
        r = rng(seed, "veg", name)
        x, y = thomas_process(r, half, c["parent_density"], c["children"], c["sigma_m"], suit[name])
        ok = ~occ.blocked(x, y, ["road", "building", "wall", "path", "yard", "field"], r=0.4)
        x, y = x[ok], y[ok]
        ok = min_distance_filter(x, y, c["min_dist_m"])
        x, y = x[ok], y[ok]
        items = _items(r, grid, name, c["role"], x, y, variants, c["scale"], align=0.5, sink=0.04, layer="shrub", occ=occ)
        if c.get("live_fraction"):  # part of the scanned dead shrubs read as live grey-green shrubs
            for it in items:
                if r.random() < c["live_fraction"]:
                    j = r.uniform(0.85, 1.15)
                    it["tint"] = [min(1.5, t * j) for t in c["live_tint"]]
        out += items

    # ---- trees from scanned 3-D tree assets (USD): plantations on slopes + yard trees
    for name, c in vc.get("trees", {}).items():
        variants = assets_variants.get(c["role"])
        if not variants:
            continue
        r = rng(seed, "tree", name)
        a0, a1 = c.get("slope_deg", [2, 30])
        suit_t = lambda x, y, a0=a0, a1=a1: np.clip(
            smoothstep(S(slope, x, y), a0, a0 + 4) * (1 - smoothstep(S(slope, x, y), a1, a1 + 8))
            * (0.6 + 0.6 * S(north, x, y) * 0.5 + 0.4 * S(macro, x, y))
            * smoothstep(S(d_road, x, y), 4, 10) * smoothstep(S(d_build, x, y), 6, 15), 0, 1)
        x, y = thomas_process(r, half, c["parent_density"], c["children"], c["sigma_m"], suit_t)
        pts = list(zip(x, y))
        for pc in parcels:  # yard trees
            if r.random() < c.get("yard_p", 0):
                poly = np.array(pc["polygon"])
                t = r.random(2)
                pts.append(tuple(poly[0] + t[0] * (poly[1] - poly[0]) + t[1] * (poly[3] - poly[0])))
        if not pts:
            continue
        x, y = np.array(pts).T
        ok = ~occ.blocked(x, y, ["road", "building", "wall", "path", "tree", "field"], r=1.5)
        x, y = x[ok], y[ok]
        ok = min_distance_filter(x, y, c["min_dist_m"])
        x, y = x[ok], y[ok]
        if "max_count" in c and len(x) > c["max_count"]:  # GPU memory budget
            keep = np.sort(r.choice(len(x), c["max_count"], replace=False))
            x, y = x[keep], y[keep]
        z, _ = surface_frame(grid, x, y, 0.6)
        for k in range(len(x)):
            var = int(r.integers(len(variants)))
            sc = float(r.uniform(*c["scale"]))
            out.append({"id": f"{name}_{k}", "type": "usd_tree", "kind": c["kind"], "role": c["role"], "twig_keep": c.get("twig_keep", 1.0),
                        "variant": variants[var][0], "x": float(x[k]), "y": float(y[k]), "z": float(z[k] - 0.1),
                        "rot": [0.0, 0.0, float(r.uniform(0, 2 * math.pi))], "scale": sc})
            occ.add_disc(x[k], y[k], 1.5 * sc, "tree")

    # ---- planted vegetation (yards, street lines)
    fp = vc["fan_palm"]
    variants = assets_variants.get(fp["role"])
    if variants:
        for p in parcels:
            r = rng(seed, "veg", "fan_palm", p["id"])
            poly = np.array(p["polygon"])
            for _ in range(int(r.integers(fp["per_parcel"][0], fp["per_parcel"][1] + 1))):
                for _try in range(20):
                    t = r.random(2)
                    x = poly[0, 0] + t[0] * (poly[1, 0] - poly[0, 0]) + t[1] * (poly[3, 0] - poly[0, 0])
                    y = poly[0, 1] + t[0] * (poly[1, 1] - poly[0, 1]) + t[1] * (poly[3, 1] - poly[0, 1])
                    if not occ.blocked(x, y, ["building", "wall", "tree", "road"], r=1.2)[0]:
                        out += _items(r, grid, "fan_palm", fp["role"], np.array([x]), np.array([y]), variants, fp["scale"], align=0.0, sink=0.05, layer="tree", occ=occ, radius=1.0)
                        break

    dp = vc["date_palm"]
    if assets_variants.get(dp["role"]) and assets_variants.get(dp["trunk_role"]):
        r = rng(seed, "veg", "date_palm")
        pts = []
        for s_idx, st in enumerate(streets):
            p = np.asarray(st["centerline"])[:, :2]
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(seg)])
            pos = float(r.uniform(5, 15))
            while pos < s[-1] - 5:
                if r.random() < dp["street_p"]:
                    k = min(max(int(np.searchsorted(s, pos)), 1), len(p) - 1)
                    t = p[k] - p[k - 1]
                    t /= np.linalg.norm(t)
                    side = 1 if r.random() < 0.5 else -1
                    off = st["width"] / 2 + st.get("shoulder", 0) + float(r.uniform(1.2, 2.2))
                    pts.append(p[k] + np.array([-t[1], t[0]]) * side * off)
                pos += float(r.uniform(*dp["street_spacing_m"]))
        for pc in parcels:
            if r.random() < dp["yard_p"]:
                poly = np.array(pc["polygon"])
                t = r.random(2)
                pts.append(poly[0] + t[0] * (poly[1] - poly[0]) + t[1] * (poly[3] - poly[0]))
        for x, y in pts:
            if occ.blocked(x, y, ["building", "wall", "road", "tree"], r=1.0)[0]:
                continue
            z, _ = surface_frame(grid, np.array([x]), np.array([y]), 0.4)
            pid = f"date_palm_{len([o for o in out if o['type'] == 'date_palm'])}"
            rr = rng(seed, "date_palm", pid)
            out.append({"id": pid, "type": "date_palm", "kind": "veg.palm.date", "role": dp["role"], "trunk_role": dp["trunk_role"],
                        "x": float(x), "y": float(y), "z": float(z[0] - 0.15), "yaw": float(rr.uniform(0, 2 * math.pi)),
                        "height": float(rr.uniform(*dp["height_m"])), "lean_deg": float(abs(rr.normal(0, 3))),
                        "lean_dir": float(rr.uniform(0, 2 * math.pi)), "fronds": int(rr.integers(18, 30)), "seed": int(rr.integers(1 << 30))})
            occ.add_disc(x, y, 1.5, "tree")
    return out


def plan_rocks(seed, grid, fields, occ, cfg, assets_variants):
    out = []
    slope, macro, d_road = fields["slope"], fields["macro"], fields["d_road"]
    half = grid.half
    for name, c in cfg["rocks"].items():
        variants = assets_variants.get(c["role"])
        if not variants:
            continue
        r = rng(seed, "rocks", name)
        ms = c.get("min_slope_deg", 0)
        gain = c.get("slope_gain", 2.0)

        def suit(x, y, ms=ms, gain=gain):
            s = grid.sample(x, y, slope)
            base = (0.12 + gain * smoothstep(s, 4, 30)) * (0.3 + 1.2 * grid.sample(x, y, macro))
            if ms:
                base *= smoothstep(s, ms, ms + 6)
            return np.clip(base * smoothstep(grid.sample(x, y, d_road), 2.5, 7), 0, 1)

        x, y = thomas_process(r, half, c["parent_density"], c["children"], c["sigma_m"], suit)
        ok = ~occ.blocked(x, y, ["road", "building", "wall", "path", "yard", "rock", "tree", "field"], r=0.6)
        x, y = x[ok], y[ok]
        ok = min_distance_filter(x, y, 0.8)
        x, y = x[ok], y[ok]
        out += _items(r, grid, f"rock_{name}", c["role"], x, y, variants, c["scale"], align=1.0, sink=None,
                      bury=c.get("bury", [0.2, 0.4]), layer="rock", occ=occ, radius=0.4, power_law=True)
    return out


def _items(r, grid, name, role, x, y, variants, scale_rng, align, sink=None, bury=None, layer=None, occ=None, radius=0.3, power_law=False):
    if len(x) == 0:
        return []
    z, n = surface_frame(grid, x, y, 0.5)
    tx, ty = tilt_from_normal(n, align)
    lo, hi = scale_rng
    if power_law:  # many small, few large
        u = r.random(len(x))
        sc = lo * (hi / lo) ** (u ** 2.2)
    else:
        sc = r.uniform(lo, hi, len(x))
    var = r.integers(0, len(variants), len(x))
    yaw = r.uniform(0, 2 * math.pi, len(x))
    tx = tx + r.normal(0, 0.05, len(x)) * (align > 0)
    ty = ty + r.normal(0, 0.05, len(x)) * (align > 0)
    items = []
    for k in range(len(x)):
        h = variants[var[k]][1] * sc[k]  # variant height (m) x scale
        dz = -(bury[0] + (bury[1] - bury[0]) * r.random()) * h if bury else -(sink or 0.0)
        items.append({"id": f"{name}_{k}", "type": "model", "kind": role,
                      "role": role, "variant": variants[var[k]][0],
                      "x": float(x[k]), "y": float(y[k]), "z": float(z[k] + dz),
                      "rot": [float(tx[k]), float(ty[k]), float(yaw[k])], "scale": float(sc[k])})
        if occ is not None and layer:
            occ.add_disc(x[k], y[k], radius * sc[k], layer)
    return items
