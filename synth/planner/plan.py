"""Scene planner: (seed, DTM, configs) -> plan.json + terrain arrays.

Runs in the standard Python venv. All decisions about *what goes where* are
made here; the Blender builder only turns the plan into geometry.
"""
import math
import os

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Point

from synth.common import camera_math as cm
from synth.common.config import load_json, load_scene_config, project_path, rel_path, save_json
from synth.common.rng import rng
from synth.common.sun import sample_sun
from synth.planner import agriculture, terraces
from synth.planner import buildings as bplan
from synth.planner import roads as rplan
from synth.planner import scatter
from synth.planner.fields import Grid, aspect_northness, resample, slope_deg, smoothstep, value_noise, wetness_index
from synth.planner.occupancy import Occupancy
from synth.prep import dtm as dtm_prep

PLAN_VERSION = 1


def _boundary_candidates(grid, step=10.0, inset=3.0):
    h = grid.half - inset
    t = np.arange(-h, h, step)
    pts = np.vstack([np.column_stack([t, np.full_like(t, -h)]), np.column_stack([t, np.full_like(t, h)]),
                     np.column_stack([np.full_like(t, -h), t]), np.column_stack([np.full_like(t, h), t])])
    return pts


def _road_entry(r, grid, slope, max_grade):
    """Main road endpoints on the core boundary, on gentle ground, far apart, and
    with an elevation difference the road can climb without hairpins (real
    regional roads follow plateaus and valleys rather than scaling escarpments)."""
    pts = _boundary_candidates(grid)
    s = grid.sample(pts[:, 0], pts[:, 1], slope)
    good = pts[s < np.percentile(s, 40)]
    z = grid.sample(good[:, 0], good[:, 1], grid.z_smooth)
    D = np.linalg.norm(good[:, None] - good[None], axis=-1)
    G = np.abs(z[:, None] - z[None]) / np.maximum(D, 1)
    for frac in (0.35, 0.6, 1.0, 10.0):
        ok = (G < frac * max_grade) & (D > 1.1 * grid.half)
        if ok.any():
            break
    ia, ib = np.nonzero(ok & (D >= np.percentile(D[ok], 60)))
    k = r.integers(len(ia))
    return good[ia[k]], good[ib[k]]


def _feasible_target(r, grid, start, pref_angle, L, max_grade, spread=1.0, margin=30, roads=()):
    """Target at distance ~L from start, near the preferred direction, whose
    elevation difference the road can climb without switchbacks. Falls back to
    the gentlest candidate."""
    zs = float(grid.sample(start[0], start[1], grid.z_smooth)[0])
    best, best_score = None, 1e9
    for k in range(32):
        ang = pref_angle + r.normal(0, 0.35 * spread) if k < 16 else r.uniform(0, 2 * math.pi)
        tgt = np.clip(start + L * np.array([math.cos(ang), math.sin(ang)]), -grid.half + margin, grid.half - margin)
        Lt = np.linalg.norm(tgt - start)
        if Lt < 0.55 * L:
            continue
        if _crosses(start, tgt, roads):
            continue
        g = abs(float(grid.sample(tgt[0], tgt[1], grid.z_smooth)[0]) - zs) / Lt
        dev = abs((ang - pref_angle + math.pi) % (2 * math.pi) - math.pi)
        score = max(0.0, g / max_grade - 0.45) * 10 + dev * 0.3 + (2 if k >= 16 else 0)
        if score < best_score:
            best, best_score = tgt, score
    return best


def _branch_route(router, roads, parent_xy, parent_hw, j, target, max_grade, stub_m=8.0):
    """Route a branch road that leaves its parent near-perpendicularly: a straight
    stub (parent half-width + stub_m) is laid first, routing continues from its end.
    If the junction is at the parent's end (a continuation), no stub is needed."""
    k = int(np.argmin(np.linalg.norm(parent_xy - j, axis=1)))
    interior = 3 <= k <= len(parent_xy) - 4
    if not interior:
        return router.route(j, target, max_grade, extra_cost=_avoid_cost(router, roads, j)), None
    t = parent_xy[min(k + 3, len(parent_xy) - 1)] - parent_xy[max(k - 3, 0)]
    t /= np.linalg.norm(t) + 1e-9
    n = np.array([-t[1], t[0]])
    if np.dot(n, np.asarray(target) - j) < 0:
        n = -n
    stub_end = j + n * (parent_hw + stub_m)
    raw = router.route(stub_end, target, max_grade, extra_cost=_avoid_cost(router, roads, j))
    if raw is None:
        return None, None
    return np.vstack([j, j + n * (parent_hw + 2.0), stub_end, raw[1:]]), (j, n, parent_hw + stub_m)


def _crosses(start, tgt, roads, keep_free_m=25.0):
    """Would the straight line start->tgt cross an existing road (away from the junction)?"""
    seg = LineString([start, tgt])
    for rd in roads:
        c = np.asarray(rd["centerline"])[:, :2]
        far = c[np.linalg.norm(c - start, axis=1) > keep_free_m]
        if len(far) > 1 and seg.intersects(LineString(far)):
            return True
    return False


def _avoid_cost(router, roads, start, keep_free_m=25.0, penalty=400.0):
    """Extra routing cost on cells occupied by existing roads, except near the
    new road's own junction: stops new roads from crossing or running along
    existing ones."""
    cost = np.zeros((router.n, router.n))
    for rd in roads:
        p = np.asarray(rd["centerline"])[:, :2]
        far = np.linalg.norm(p - start, axis=1) > keep_free_m
        i = np.clip(np.round((p[far, 1] + router.half) / router.res).astype(int), 0, router.n - 1)
        j = np.clip(np.round((p[far, 0] + router.half) / router.res).astype(int), 0, router.n - 1)
        mark = np.zeros_like(cost, bool)
        mark[i, j] = True
        cost[mark] = penalty
    # discourage running alongside an existing road too (corridor ~20 m)
    from scipy import ndimage as ndi
    near = ndi.binary_dilation(cost > 0, iterations=max(1, int(round(20.0 / router.res))))
    return np.maximum(cost, np.where(near, 12.0, 0.0))


def _road_record(rid, kind, cfg, pts_xy, z):
    return {"id": rid, "kind": f"road.{cfg['kind']}" if cfg["kind"] != "footpath" else "path.footpath",
            "surface": cfg["kind"], "width": float(cfg["width"]) if not isinstance(cfg["width"], list) else None,
            "shoulder": float(cfg.get("shoulder", 0.0)), "blend_m": float(cfg.get("blend_m", 3)),
            "centerline": [[float(x), float(y), float(zz)] for (x, y), zz in zip(pts_xy, z)], "role_kind": kind}


class SiteUnsuitable(RuntimeError):
    """The chosen site cannot host the configured village (e.g. too few buildings)."""


def plan_scene(seed, scene_config, overrides=None, out_dir=None, log=print):
    """Plan a scene. Each seed picks one of the top-k ranked sites; if that site
    cannot host a village, the next ranked sites are tried deterministically."""
    cfg = load_scene_config(scene_config, overrides)
    tcfg = cfg["terrain"]
    top_k = int(tcfg.get("site_top_k", 1))
    if "site_rank" in tcfg:
        ranks = [int(tcfg["site_rank"])]
    else:
        first = int(rng(seed, "site").integers(top_k))
        ranks = [(first + i) % max(top_k, 1) for i in range(max(top_k, 1))]
    last = None
    for site_rank in ranks:
        try:
            return _plan_once(seed, scene_config, cfg, site_rank, out_dir, log)
        except SiteUnsuitable as e:
            log(f"site {site_rank} unsuitable ({e}); trying the next ranked site")
            last = e
    raise last


def _plan_once(seed, scene_config, cfg, site_rank, out_dir, log):
    tcfg = cfg["terrain"]
    npz = dtm_prep.prepare(tcfg["source"], tcfg["aoi"], tcfg["core_size_m"], tcfg.get("centre_latlon"),
                           files=tcfg.get("files"), site_rank=site_rank)
    dmeta = load_json(npz.replace(".npz", ".json"))
    d = np.load(npz)
    grid = Grid(d["core_z"].astype(np.float64), dmeta["core_res_m"], dmeta["core_half_m"])
    grid.z_smooth = d["core_z_smooth"].astype(np.float64)
    slope0 = slope_deg(grid.z_smooth, grid.res)
    r = rng(seed, "layout")
    rcfg = cfg["roads"]

    # ------------------------------------------------------------ main road
    router = rplan.Router(grid, rcfg["routing_res_m"])
    a, b = _road_entry(r, grid, slope0, rcfg["main"]["max_grade"])
    raw = router.route(a, b, rcfg["main"]["max_grade"])
    main_xy = rplan.smooth_polyline(raw, rcfg["main"]["smooth_m"], keep_start=False, keep_end=False)
    main_z = rplan.vertical_profile(grid, main_xy, rcfg["main"]["max_grade"], rcfg["main"]["profile_smooth_m"])
    roads = [_road_record("road_main", "main", rcfg["main"], main_xy, main_z)]
    log(f"main road {len(main_xy)} m")

    # ------------------------------------------------- village nucleus + streets
    rs = rcfg["street"]
    X, Y = grid.xy()
    main_line = LineString(main_xy)
    sub = (slice(None, None, 8), slice(None, None, 8))
    cand = np.column_stack([X[sub].ravel(), Y[sub].ravel()])
    from scipy import ndimage as ndi
    area_slope = ndi.gaussian_filter(slope0, 35 / grid.res)  # room for a village, not one flat point
    sl = area_slope[sub].ravel()
    dist = np.array([main_line.distance(Point(p)) for p in cand])
    lo, hi = rs["village_offset_m"]
    ok = (sl < 7) & (dist > lo) & (dist < hi) & grid.inside(cand[:, 0], cand[:, 1], 90)
    if not ok.any():
        ok = (sl < 10) & (dist > 30) & grid.inside(cand[:, 0], cand[:, 1], 60)
    village = cand[ok][r.integers(ok.sum())]
    log(f"village centre {village}")

    streets = []
    junction_from = main_xy
    for s_i in range(rs["count"]):
        rr = rng(seed, "street", s_i)
        L = float(rr.uniform(*rs["length_m"]))
        parent = np.asarray(junction_from)
        if s_i == 0:
            j = parent[np.argmin(np.linalg.norm(parent - village, axis=1))]
            direction = (village - j) / (np.linalg.norm(village - j) + 1e-9)
        else:
            k = int(len(parent) * rr.uniform(0.35, 0.65))
            j = parent[k]
            t = parent[min(k + 3, len(parent) - 1)] - parent[max(k - 3, 0)]
            t /= np.linalg.norm(t) + 1e-9
            direction = np.array([-t[1], t[0]]) * (1 if rr.random() < 0.5 else -1)
        target = _feasible_target(rr, grid, j, math.atan2(direction[1], direction[0]), L, rs["max_grade"], spread=1.0, roads=roads)
        if target is None:
            continue
        parent_rec = roads[0] if s_i == 0 else streets[0]
        raw, stub = _branch_route(router, roads, parent, parent_rec["width"] / 2, j, target, rs["max_grade"])
        if raw is None or len(raw) < 5:
            continue
        raw[0] = j
        xy = rplan.smooth_polyline(raw, rs["smooth_m"], keep_start=stub is None, keep_end=False)
        if stub is not None:
            xy = rplan.straight_start(xy, *stub)
        xy = rplan.trim_hairpins(xy)
        if s_i == 0:
            parent_z = main_z[np.argmin(np.linalg.norm(main_xy - j, axis=1))]
        else:
            pz = np.asarray(streets[0]["centerline"])
            parent_z = pz[np.argmin(np.linalg.norm(pz[:, :2] - j, axis=1)), 2]
        z = rplan.vertical_profile(grid, xy, rs["max_grade"], rs["profile_smooth_m"], z_start=float(np.ravel(parent_z)[0]))
        rec = _road_record(f"road_street_{s_i}", "street", rs, xy, z)
        rec["junction_with"] = "road_main" if s_i == 0 else "road_street_0"
        streets.append(rec)
        roads.append(rec)
        junction_from = xy
    log(f"{len(streets)} streets")

    # ------------------------------------------------------------ dirt tracks
    tc = rcfg["track"]
    for t_i in range(tc["count"]):
        rr = rng(seed, "track", t_i)
        src = np.asarray((streets[-1] if streets and t_i == 0 else roads[0])["centerline"])
        j = src[-1, :2] if t_i == 0 else src[int(len(src) * rr.uniform(0.2, 0.8)), :2]
        jz = src[-1, 2] if t_i == 0 else src[np.argmin(np.linalg.norm(src[:, :2] - j, axis=1)), 2]
        L = float(rr.uniform(*tc["length_m"]))
        # head away from the previous tracks so tracks fan out instead of running parallel
        prev = [np.asarray(rd["centerline"])[-1, :2] - np.asarray(rd["centerline"])[0, :2] for rd in roads if rd["surface"] == "dirt"]
        pref = rr.uniform(0, 2 * math.pi) if not prev else math.atan2(-prev[-1][1], -prev[-1][0])
        best = _feasible_target(rr, grid, j, pref, L, tc["max_grade"], spread=2.0, roads=roads)
        if best is None:
            continue
        src_rec = (streets[-1] if streets and t_i == 0 else roads[0])
        raw, stub = _branch_route(router, roads, src[:, :2], src_rec["width"] / 2, j, best, tc["max_grade"])
        if raw is None or len(raw) < 5:
            continue
        raw[0] = j
        xy = rplan.smooth_polyline(raw, tc["smooth_m"], keep_start=stub is None, keep_end=False)
        if stub is not None:
            xy = rplan.straight_start(xy, *stub)
        xy = rplan.trim_hairpins(xy)
        z = rplan.vertical_profile(grid, xy, tc["max_grade"], tc["profile_smooth_m"], z_start=float(jz))
        rec = _road_record(f"road_track_{t_i}", "track", tc, xy, z)
        rec["junction_with"] = src_rec["id"]
        roads.append(rec)
    log(f"{len(roads)} roads total")

    # --------------------------------------------- terrain cut/fill + occupancy
    road_w = rplan.apply_corridors(grid, roads)
    occ = Occupancy(grid.half, cfg["occupancy_res_m"])
    road_polys = []
    for rd in roads:
        p = np.asarray(rd["centerline"])[:, :2]
        poly = rplan.road_polygon(p, rd["width"] / 2 + rd["shoulder"])
        road_polys.append(poly)
        occ.add_polygon(poly, "road")

    # ---------------------------------------------------------------- buildings
    buildings, parcels, walls = bplan.plan_buildings(seed, grid, streets, road_polys, occ, cfg)
    log(f"{len(buildings)} buildings, {len(walls)} wall segments")
    if len(buildings) < cfg["buildings"].get("min_count", 0):
        raise SiteUnsuitable(f"only {len(buildings)} buildings")

    # ------------------------------------------------------------ fields
    plots = agriculture.plan_fields(seed, grid, occ, roads, village, cfg, slope_deg(grid.z_smooth, grid.res))
    log(f"{len(plots)} fields")
    terrace_walls, _ = terraces.plan_terraces(seed, grid, occ, village, cfg, slope_deg(grid.z_smooth, grid.res))
    walls += terrace_walls + terraces.field_walls(seed, grid, plots, occ, cfg)
    log(f"{len(terrace_walls)} terrace walls, {len(walls)} walls total")

    # ---------------------------------------------------------------- footpaths
    paths = _plan_footpaths(seed, grid, occ, roads, parcels, cfg)
    log(f"{len(paths)} footpaths")

    # -------------------------------------------------------------- env fields
    coarse = 4.0
    zc = resample(grid.z, grid.res, coarse)
    twi = resample(wetness_index(zc, coarse), coarse, grid.res)[: grid.n, : grid.n]
    twi = np.pad(twi, ((0, grid.n - twi.shape[0]), (0, grid.n - twi.shape[1])), mode="edge")
    fr = rng(seed, "fields")
    macro = value_noise(grid.z.shape, grid.res, 60.0, fr)
    d_road_occ = occ.distance_to(["road"])
    d_build_occ = occ.distance_to(["building", "yard"])
    to_grid = lambda a: resample(a, occ.res, grid.res)[: grid.n, : grid.n]
    fields = {"slope": slope_deg(grid.z, grid.res), "twi": twi, "north": aspect_northness(grid.z_smooth, grid.res),
              "macro": macro, "d_road": to_grid(d_road_occ), "d_build": to_grid(d_build_occ)}

    # ----------------------------------------------------- vegetation and rocks
    dims = load_json("cache/asset_dims.json")
    reg = load_json("configs/assets.json")["assets"]

    def variants_for(role):
        out = []
        for stem, a in sorted(reg.items()):
            if a.get("status") == "ok" and role in a.get("roles", []):
                if a.get("kind") in ("atlas", "surface", "decal"):
                    out.append((stem, 1.0))
                    continue
                for m in a["models"]:
                    if m in dims:
                        out.append((m, dims[m][2]))
        return out

    roles = {c["role"] for k, c in cfg["vegetation"].items() if k != "trees"} | {c["role"] for c in cfg["rocks"].values()}
    roles |= {c["role"] for c in cfg.get("scatter", {}).values()} | {"rock.small_dark"}
    roles |= {c["role"] for c in cfg["vegetation"].get("trees", {}).values()}
    roles |= {cfg["vegetation"]["date_palm"]["trunk_role"]}
    av = {ro: variants_for(ro) for ro in roles}
    missing = sorted(ro for ro, v in av.items() if not v)
    if missing:
        log(f"MISSING assets for roles: {missing}")
        if cfg.get("missing_asset_policy", "error") == "error":
            raise RuntimeError(f"missing assets for roles {missing}")
    veg = scatter.plan_vegetation(seed, grid, fields, occ, cfg, parcels, streets, av)
    rocks = scatter.plan_rocks(seed, grid, fields, occ, cfg, av)
    rocks += agriculture.edge_stone_piles(seed, grid, plots, cfg, av.get("rock.small_dark"))
    log(f"{len(veg)} vegetation items, {len(rocks)} rocks")

    # ------------------------------------------- scatter systems (Geometry Nodes)
    free, wet = _scatter_fields(grid, occ, fields)
    scatter_systems = []
    for name, sc in cfg.get("scatter", {}).items():
        params = {k: v for k, v in sc.items() if k not in ("role", "kind")}
        params["Seed"] = int(rng(seed, "scatter", name).integers(1 << 30))
        scatter_systems.append({"name": name, "role": sc["role"], "kind": sc["kind"], "params": params})

    # ------------------------------------------------------------- splat map
    splat = _splat(seed, grid, fields, occ, road_w, roads, cfg)

    # -------------------------------------------------- sun, water, cameras
    lat, lon = dmeta["origin_latlon"]
    sun = sample_sun(rng(seed, "sun"), lat, lon, cfg["sun"])
    vz = float(grid.sample(*village)[0])
    ccfg = dict(cfg["camera"])
    if ccfg.get("view_sampler"):
        ccfg["views"] = _sample_views(seed, ccfg["view_sampler"])
    cams = []
    for v in ccfg["views"]:
        tgt = np.array([village[0], village[1], vz])
        if v.get("target") == "random":
            # outside the village: aim at a point of interest (road, field, trees,
            # terrace) most of the time, at open countryside otherwise
            rt = rng(seed, "view_target", v["name"])
            lim = grid.half * 0.55
            pois = [p for rd in roads for p in np.asarray(rd["centerline"])[::25, :2]]
            pois += [np.mean(np.asarray(f["polygon"]), axis=0) for f in plots]
            pois += [np.array([t["x"], t["y"]]) for t in veg if t["type"] == "usd_tree"][::10]
            pois += [np.asarray(w["points"])[len(w["points"]) // 2][:2] for w in walls if w["kind"] == "wall.terrace"]
            pois = [p for p in pois if abs(p[0]) < lim and abs(p[1]) < lim]
            if pois and rt.random() < 0.75:
                xy = pois[int(rt.integers(len(pois)))] + rt.normal(0, 25, 2)
            else:
                xy = rt.uniform(-lim, lim, 2)
            xy = np.clip(xy, -lim, lim)
            tgt = np.array([xy[0], xy[1], float(grid.sample(*xy)[0])])
        cams += _plan_cameras(seed, grid, d, dmeta, dict(ccfg, views=[v]), tgt)

    # ------------------------------------------------------------------ write
    out_dir = project_path(out_dir or f"outputs/plan_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "terrain.npz"), core_z=grid.z.astype(np.float32),
                        free=free.astype(np.float32), wet=wet.astype(np.float32),
                        mid_z=d["mid_z"], far_z=d["far_z"])
    for k, img in enumerate(splat):
        Image.fromarray(img).save(os.path.join(out_dir, f"splat{k}.png"))
    Image.fromarray(agriculture.fields_raster(grid, plots, cfg["splat_res_m"])).save(os.path.join(out_dir, "fields.png"))
    water_level_abs = cfg.get("water_level_abs", -210.5)
    plan = {
        "version": PLAN_VERSION, "seed": seed, "scene_config": scene_config, "config": cfg,
        "dtm": dmeta, "dtm_npz": rel_path(npz),
        "terrain": {"file": "terrain.npz", "splat": [f"splat{k}.png" for k in range(len(splat))], "fields": "fields.png",
                    "splat_res_m": cfg["splat_res_m"],
                    "water_level": water_level_abs - dmeta["origin_utm"][2]},
        "village_centre": [float(village[0]), float(village[1]), vz],
        "roads": [rd for rd in roads], "paths": paths, "buildings": buildings, "parcels": parcels,
        "walls": walls, "fields": plots, "vegetation": veg, "rocks": rocks, "scatter_systems": scatter_systems, "sun": sun, "cameras": cams,
        "counts": {"roads": len(roads), "paths": len(paths), "buildings": len(buildings), "walls": len(walls), "fields": len(plots),
                   "vegetation": len(veg), "rocks": len(rocks)},
    }
    save_json(plan, os.path.join(out_dir, "plan.json"))
    _plan_map(plan, grid, os.path.join(out_dir, "plan_map.png"))
    log(f"plan written to {out_dir}")
    return os.path.join(out_dir, "plan.json")


def _plan_footpaths(seed, grid, occ, roads, parcels, cfg):
    """Desire lines: least-cost on a fine grid where flat open ground is cheap
    and yards/buildings are impassable, so paths cut corners like real ones."""
    fc = cfg["roads"]["footpath"]
    if not parcels:
        return []
    router = rplan.Router(grid, 2.0)
    from scipy import ndimage
    blocked = (occ.mask & (2 | 4 | 8)) > 0
    blk = ndimage.zoom(blocked.astype(float), occ.res / 2.0, order=0)[: router.n, : router.n] > 0.5
    blk = np.pad(blk, ((0, router.n - blk.shape[0]), (0, router.n - blk.shape[1])))
    extra = np.where(blk, 400.0, 0.0)
    r = rng(seed, "footpaths")
    out = []
    for k in range(fc["count"] * 4):
        if len(out) >= fc["count"]:
            break
        p = parcels[int(r.integers(len(parcels)))]
        poly = np.array(p["polygon"])
        start = poly[int(r.integers(len(poly)))] + r.normal(0, 1.0, 2)
        # nearest point on any road other than the parcel's own frontage, 40-150 m away
        best = None
        for rd in roads:
            c = np.asarray(rd["centerline"])[:, :2]
            dd = np.linalg.norm(c - start, axis=1)
            m = (dd > 40) & (dd < 150)
            if m.any():
                i = int(np.argmin(np.where(m, dd, np.inf)))
                if best is None or dd[i] < best[0]:
                    best = (dd[i], c[i])
        if best is None:
            continue
        goal = best[1]
        raw = router.route(start, goal, 0.25, extra_cost=extra)
        if raw is None or len(raw) < 6:
            continue
        xy = rplan.smooth_polyline(raw, 2.5)
        tan = np.gradient(xy, axis=0)
        nrm = np.column_stack([-tan[:, 1], tan[:, 0]]) / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9)
        wobble = ndimage.gaussian_filter1d(r.normal(0, 0.8, len(xy)), 3)
        wobble[:3] = wobble[-3:] = 0
        xy = xy + nrm * wobble[:, None]
        z = grid.sample(xy[:, 0], xy[:, 1])
        w = float(r.uniform(*fc["width"]))
        out.append({"id": f"path_{k}", "kind": "path.footpath", "surface": "footpath", "width": w,
                    "centerline": [[float(x), float(y), float(zz)] for (x, y), zz in zip(xy, z)]})
        occ.add_polygon(LineString(xy).buffer(w / 2 + 0.2), "path")
    return out


def _scatter_fields(grid, occ, fields):
    """Per-vertex terrain attributes for the GN scatter systems:
    free = 1 where scatter may grow (not on roads, buildings, yards, walls,
    paths, fields), wet = normalised wetness (drainage lines ~1)."""
    from scipy import ndimage as ndi
    bits = 1 | 2 | 4 | 8 | 16 | 256
    blocked = (occ.mask & bits) > 0
    blocked = ndi.binary_dilation(blocked, iterations=2)
    b = ndi.zoom(blocked.astype(np.float32), occ.res / grid.res, order=0)
    b = np.pad(b, ((0, max(0, grid.n - b.shape[0])), (0, max(0, grid.n - b.shape[1]))))[: grid.n, : grid.n]
    free = 1.0 - (b > 0.5)
    twi = fields["twi"]
    wet = np.clip((twi - np.percentile(twi, 20)) / (np.percentile(twi, 98) - np.percentile(twi, 20) + 1e-9), 0, 1)
    return free, wet


def _splat(seed, grid, fields, occ, road_w, roads, cfg):
    """Human-made land-cover weights for the core, at splat_res_m:
    R = road-shoulder gravel/dust, G = trampled ground around houses and paths.
    Natural cover is procedural in the shader (identical on every terrain ring)."""
    res = cfg["splat_res_m"]
    r = rng(seed, "splat")
    n = int(round(2 * grid.half / res)) + 1
    nz = lambda wl: value_noise((n, n), res, wl, r, octaves=4)
    fit = lambda a: a[:n, :n] if a.shape[0] >= n else np.pad(a, ((0, n - a.shape[0]), (0, n - a.shape[1])), mode="edge")
    d_road = fit(resample(occ.distance_to(["road"]), occ.res, res))
    d_bld = fit(resample(occ.distance_to(["building", "yard"]), occ.res, res))
    d_path = fit(resample(occ.distance_to(["path"]), occ.res, res))
    shoulder = 1 - smoothstep(d_road, 0.0, 0.8 + 2.0 * nz(10))
    trampled = np.maximum((1 - smoothstep(d_bld, 0, 4 + 8 * nz(12))) * 0.95, (1 - smoothstep(d_path, 0.2, 1.5)) * 0.8)
    twi = fit(resample(fields["twi"], grid.res, res))
    wet = smoothstep(twi, np.percentile(twi, 88), np.percentile(twi, 99)) * (0.6 + 0.6 * nz(25))
    rgba = (np.stack([shoulder, trampled, np.clip(wet, 0, 1), np.ones_like(shoulder)], -1)[::-1] * 255).clip(0, 255).astype(np.uint8)
    return [rgba]


def _sample_views(seed, vs):
    """Draw this scene's views from the configured distributions (weighted choices)."""
    r = rng(seed, "views")
    pick = lambda opts: opts[int(r.choice(len(opts), p=np.array([o[1] for o in opts]) / sum(o[1] for o in opts)))][0]
    n = int(r.integers(vs["count"][0], vs["count"][1] + 1))
    views = []
    for i in range(n):
        views.append({"name": f"v{i}", "tilt_deg": float(pick(vs["tilt_deg"])) + float(r.uniform(-3, 3)) * (1 if i else 0),
                      "gsd_m": float(pick(vs["gsd_m"])), "heading_deg": None,
                      "target": "village" if r.random() < vs.get("village_fraction", 0.6) else "random"})
        views[-1]["tilt_deg"] = max(0.0, views[-1]["tilt_deg"])
    return views


def _plan_cameras(seed, grid, d, dmeta, ccfg, target):
    """Camera poses aimed at the village. Heading is chosen so the camera stays
    clear of the terrain (oblique views can otherwise end up inside a hill)."""
    cams = []
    fz = d["mid_z"]
    mhalf, mres = dmeta["mid_half_m"], dmeta["mid_res_m"]
    from scipy import ndimage

    def terrain_at(x, y):
        i, j = (y + mhalf) / mres, (x + mhalf) / mres
        return float(ndimage.map_coordinates(fz, [[i], [j]], order=1, mode="nearest")[0])

    resx, resy = ccfg["resolution"]
    for k, v in enumerate(ccfg["views"]):
        r = rng(seed, "camera", v["name"])
        R = cm.slant_range_for_gsd(v["gsd_m"], ccfg["focal_mm"], ccfg["sensor_width_mm"], resx)
        best = None
        for _ in range(24):
            hd = v["heading_deg"] if v["heading_deg"] is not None else float(r.uniform(0, 360))
            pos, eul = cm.camera_pose(target, R, v["tilt_deg"], hd)
            # clearance along the line of sight
            clear = min((target[2] + (pos[2] - target[2]) * f) - terrain_at(*(target[:2] + (pos[:2] - target[:2]) * f))
                        for f in np.linspace(0.2, 1.0, 9))
            if best is None or clear > best[0]:
                best = (clear, hd, pos, eul)
            if clear > 25:
                break
        clear, hd, pos, eul = best
        j = ccfg.get("attitude_jitter_deg", 0.0)
        eul = (eul[0] + math.radians(r.normal(0, j)), math.radians(r.normal(0, j)), eul[2] + math.radians(r.normal(0, j)))
        across, along = cm.gsd_at_centre(R * math.cos(math.radians(v["tilt_deg"])), ccfg["focal_mm"], ccfg["sensor_width_mm"], resx, v["tilt_deg"])
        cams.append({"name": v["name"], "tilt_deg": v["tilt_deg"], "heading_deg": hd, "slant_range_m": R,
                     "location": [float(c) for c in pos], "rotation_euler": [float(e) for e in eul],
                     "focal_mm": ccfg["focal_mm"], "sensor_width_mm": ccfg["sensor_width_mm"],
                     "resolution": [resx, resy], "gsd_centre_m": {"across": across, "along": along},
                     "height_above_target_m": float(pos[2] - target[2]), "terrain_clearance_m": float(clear),
                     "K": cm.intrinsics(ccfg["focal_mm"], ccfg["sensor_width_mm"], resx, resy).tolist()})
    return cams


def _plan_map(plan, grid, path):
    """Top-down debug map of the plan (roads, buildings, vegetation, rocks)."""
    from PIL import ImageDraw
    s = 2  # px per metre
    n = int(2 * grid.half * s)
    z = grid.z
    gy, gx = np.gradient(z, grid.res)
    hs = np.clip(0.7 + 0.5 * (-gx * 0.7 + gy * 0.7) / np.sqrt(1 + gx ** 2 + gy ** 2), 0, 1)
    base = (np.stack([hs * 200, hs * 185, hs * 160], -1)[::-1]).astype(np.uint8)
    im = Image.fromarray(base).resize((n, n))
    dr = ImageDraw.Draw(im)
    P = lambda x, y: ((x + grid.half) * s, (grid.half - y) * s)
    for rd in plan["roads"]:
        col = (60, 60, 60) if rd["surface"] == "asphalt" else (160, 120, 70)
        dr.line([P(x, y) for x, y, _ in rd["centerline"]], fill=col, width=max(1, int(rd["width"] * s)))
    for p in plan["paths"]:
        dr.line([P(x, y) for x, y, _ in p["centerline"]], fill=(210, 180, 120), width=2)
    for f in plan.get("fields", []):
        col = {"stubble": (225, 205, 140), "ploughed": (120, 90, 60), "fallow": (190, 180, 120)}[f["state"]]
        dr.polygon([P(x, y) for x, y in f["polygon"]], outline=col, fill=None)
    for pc in plan["parcels"]:
        dr.polygon([P(x, y) for x, y in pc["polygon"]], outline=(120, 90, 200))
    for w in plan["walls"]:
        dr.line([P(x, y) for x, y, _ in w["points"]], fill=(90, 70, 160), width=1)
    for b in plan["buildings"]:
        dr.polygon([P(x, y) for x, y in b["footprint"]], fill=(220, 60, 60))
        if b["annex"]:
            dr.polygon([P(x, y) for x, y in b["annex"]], fill=(200, 90, 90))
    for v in plan["vegetation"]:
        x, y = P(v["x"], v["y"])
        rr = 3 if v["type"] == "date_palm" else 1
        dr.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(40, 140, 40) if "grass" not in v.get("role", "") else (150, 160, 60))
    for rk in plan["rocks"]:
        x, y = P(rk["x"], rk["y"])
        dr.point((x, y), fill=(30, 30, 30))
    im.save(path)
