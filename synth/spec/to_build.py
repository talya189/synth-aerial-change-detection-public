"""Scene spec → build data for the Blender builders (standard Python venv).

The Blender side (synth/blender/run_build.py) consumes a "build" dict (the
format the planner writes as plan.json). This module produces it from a scene
spec: terrain arrays, entities with derived heights / foundations / rooftop
equipment, cameras solved from GSD, and the sun from date and time.
Everything derived is deterministic: per-entity random streams keyed by the
entity id, so an unchanged entity is identical in every rebuild.
"""
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

from synth.common import camera_math as cm
from synth.common.config import load_json, project_path, save_json
from synth.common.rng import rng
from synth.common.sun import solar_position
from synth.planner.buildings import _rooftop
from synth.spec.buildings import plan_record
from synth.planner import roads as rplan
from synth.planner.fields import Grid, slope_deg, value_noise
from synth.spec.scene_spec import resolved, validate



# spec wall type -> wall material role (synth/blender/building_builder.wall_materials)
WALL_ROLE = {"stone": "wall.stone", "block": "wall.block", "plaster": "wall.stucco", "fence": "wall.concrete"}


def flat_test_terrain(spec):
    reg = spec["region"]
    n = int(reg["size_m"]) + 1
    half = (n - 1) / 2
    r = rng(spec["seed"], "terrain")
    z = (value_noise((n, n), 1.0, 60.0, r, octaves=3) - 0.5) * 2 * reg.get("relief_m", 0.5)
    z += (value_noise((n, n), 1.0, 6.0, r, octaves=2) - 0.5) * 0.15
    return z.astype(np.float64), half


def dem_terrain(region):
    """Real elevation (cached preprocessing in synth.prep.dtm): 1 m core with
    micro-relief, 5 m mid ring, 30 m far field, local metres around the site."""
    import json as _json
    from synth.prep import dtm as dtm_prep
    npz = dtm_prep.prepare(region.get("source", "copernicus_glo30"), region["aoi"], region.get("core_size_m", 1000),
                           region.get("centre_latlon"), site_rank=int(region.get("site_rank", 0)),
                           selection=region.get("selection"))
    meta = _json.load(open(npz.replace(".npz", ".json")))
    d = np.load(npz)
    meta["core_z_smooth"] = d["core_z_smooth"].astype(np.float64)  # for hydrology (no micro-relief)
    return (d["core_z"].astype(np.float64), meta["core_half_m"], d["mid_z"].astype(np.float32),
            d["far_z"].astype(np.float32), meta)


def terrain_wetness(z, coarse=8.0, area_m2=(1.5e3, 4.0e4)):
    """Drainage channels (0..1) from the area draining through each cell
    (multiple-flow-direction routing on the smoothed surface): 0 below
    area_m2[0], 1 above area_m2[1], log-scaled between. Unlike a wetness index
    it does not mark flat ground, only real valleys and wadis."""
    from scipy import ndimage
    from synth.planner.fields import mfd_accumulation, resample
    zs = ndimage.gaussian_filter(z, 4.0)
    acc = mfd_accumulation(resample(zs, 1.0, coarse)) * coarse * coarse
    w = (np.log10(acc) - np.log10(area_m2[0])) / (np.log10(area_m2[1]) - np.log10(area_m2[0]))
    w = ndimage.gaussian_filter(resample(np.clip(w, 0, 1), coarse, 1.0), 2.0)
    n = z.shape[0]
    return np.clip(np.pad(w, ((0, max(0, n - w.shape[0])), (0, max(0, n - w.shape[1]))), mode="edge")[:n, :n], 0, 1)


def road_records(spec_roads, grid, rcfg):
    """Spec roads -> build records with their vertical profile (x, y, z every metre).
    Parents first; a road joined at its start / end starts / ends at the joined
    road's height. Used by the build and by the network planner's grade check,
    so both see exactly the same profiles. `grid.z_smooth` must be set."""
    roads, by_id, todo = [], {}, list(spec_roads)
    while todo:   # parents before branches
        ready = [rd for rd in todo if all(not rd.get(k) or rd[k] in by_id for k in ("connects", "connects_end"))]
        if not ready:   # a cycle of end junctions: build one of them with a free end (its start junction still holds)
            cyc = [rd for rd in todo if not rd.get("connects") or rd["connects"] in by_id]
            if not cyc:
                raise ValueError(f"road junction cycle among {[rd['id'] for rd in todo]}")
            ready = [dict(cyc[0], connects_end=None)]
            todo[todo.index(cyc[0])] = ready[0]
        for rd in ready:
            todo.remove(rd)
            c = rcfg["classes"].get(rd.get("class"), {})
            kind = rd.get("kind", c.get("kind"))
            p = _densify(np.asarray(rd["centerline"], float), 1.0)
            if c:
                def joined_z(key, q):   # height of the joined road where this one meets it
                    if not rd.get(key):
                        return None
                    pc = np.asarray(by_id[rd[key]]["centerline"])
                    return float(pc[np.argmin(np.linalg.norm(pc[:, :2] - q, axis=1)), 2])
                zz = rplan.vertical_profile(grid, p, rd.get("max_grade", c["max_grade"]), c["profile_smooth_m"],
                                            z_start=joined_z("connects", p[0]), z_end=joined_z("connects_end", p[-1]))
            else:
                zz = grid.sample(p[:, 0], p[:, 1]) + 0.06
            rec = {"id": rd["id"], "kind": kind, "surface": kind.split(".")[1], "class": rd.get("class"),
                   "width": float(rd.get("width", c.get("width", 4.0))), "shoulder": float(rd.get("shoulder", c.get("shoulder", 0.0))),
                   "blend_m": float(c.get("blend_m", 3.0)), "max_grade": float(rd.get("max_grade", c.get("max_grade", 0.2))),
                   "role_kind": "main" if (rd.get("class") == "main" or (not c and kind == "road.asphalt")) else
                                ("track" if kind == "road.dirt" else "street"),
                   "junction_with": rd.get("connects"), "junction_end": rd.get("connects_end"),
                   "end_cap": rd.get("end_cap"), "profiled": bool(c),
                   "section": rd.get("section", "two_lane" if c.get("markings") else "plain"),
                   "median_m": float(rd.get("median_m", 0.0)), "lanes": int(rd.get("lanes", 1)),
                   "centerline": [[float(x), float(y), float(h)] for (x, y), h in zip(p, zz)]}
            roads.append(rec)
            by_id[rec["id"]] = rec
    return roads


def junction_fillets(roads, ent, grid, rcfg, lift=0.02):
    """Rounded kerb corners where an asphalt road joins another: the gap between
    the two road edges is filled with a circular fillet of the class's
    `corner_radius_m` (the smaller road's), clipped by buildings, walls and tree
    trunks. Returned as triangles on the ground (just under the road surfaces,
    which overlap them by 0.3 m), labelled as the joining road."""
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import unary_union
    by_id = {r_["id"]: r_ for r_ in roads}
    body = lambda r_: LineString(np.asarray(r_["centerline"])[:, :2]).buffer(r_["width"] / 2, cap_style="flat")
    from synth.spec.buildings import footprint
    obst = [footprint(b).buffer(0.2) for b in ent.get("buildings", [])]
    obst += [LineString(sg).buffer(w.get("thickness", 0.3) / 2 + 0.05) for w in ent.get("walls", []) for sg in w["segments"]]
    obst += [Point(t["x"], t["y"]).buffer(0.5) for t in ent.get("trees", [])]
    out = []
    for r_ in roads:
        if r_["surface"] != "asphalt":
            continue
        for key, end in (("junction_with", 0), ("junction_end", -1)):
            par = by_id.get(r_.get(key))
            if par is None or par["surface"] != "asphalt":
                continue
            c = np.asarray(r_["centerline"])[:, :2]
            j = Point(c[end])
            R = min(rcfg["classes"].get(r_.get("class"), {}).get("corner_radius_m", 4.0),
                    rcfg["classes"].get(par.get("class"), {}).get("corner_radius_m", 4.0))
            win = j.buffer(par["width"] / 2 + r_["width"] / 2 + R + 4.0)
            U = unary_union([body(par), body(r_)]).intersection(win)
            closed = U.buffer(R, join_style="round").buffer(-R, join_style="round")
            fill = closed.difference(U.buffer(-0.3)).intersection(j.buffer(par["width"] / 2 + R + 1.0))
            near = [o for o in obst if o.intersects(win)]
            if near:
                fill = fill.difference(unary_union(near))
            for g in getattr(fill, "geoms", [fill]):
                if g.geom_type != "Polygon" or g.difference(U).area < 0.3:
                    continue
                tris = []
                for t in constrained_delaunay_triangles(g.segmentize(0.75)).geoms:
                    xy = np.asarray(t.exterior.coords)[:3]
                    a_, b_ = xy[1] - xy[0], xy[2] - xy[0]
                    if a_[0] * b_[1] - a_[1] * b_[0] < 0:   # counter-clockwise: faces up
                        xy = xy[::-1]
                    z = grid.sample(xy[:, 0], xy[:, 1]) + lift
                    tris.append([[float(x), float(y), float(zz)] for (x, y), zz in zip(xy, z)])
                out.append({"id": f"{r_['id']}_fillet_{len(out)}", "label_id": r_["id"], "kind": r_["kind"],
                            "role_kind": r_["role_kind"], "triangles": tris})
    return out


def max_grade_10m(centerline):
    """Steepest grade of a road profile over 10 m windows (how the reports measure it)."""
    p = np.asarray(centerline, float)
    seg = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    k = np.searchsorted(s, s + 10.0).clip(0, len(s) - 1)
    g = np.abs(p[k, 2] - p[:, 2]) / np.maximum(s[k] - s, 1e-6)
    g = g[s[k] - s > 5]
    return float(g.max()) if len(g) else 0.0


def grade_violations(records, tol=0.01):
    """{road id: steepest 10 m grade} for profiled roads above their class limit."""
    return {r["id"]: round(max_grade_10m(r["centerline"]), 3) for r in records
            if r.get("profiled") and max_grade_10m(r["centerline"]) > r["max_grade"] + tol}


def build_data(spec, out_dir):
    validate(spec)
    spec = resolved(spec)
    os.makedirs(out_dir, exist_ok=True)
    seed = spec["seed"]
    region = spec["region"]
    if region["type"] == "dem":
        z, half, mid, far, dmeta = dem_terrain(region)
        region["resolved"] = {k: dmeta[k] for k in ("origin_latlon", "origin_utm", "epsg", "site_rank", "source_sha256")}
    elif region["type"] == "flat_test":
        z, half = flat_test_terrain(spec)
        mid = np.zeros((int(2 * 1600.0 / 5.0) + 1,) * 2, np.float32)
        far = np.zeros((int(2 * 9000.0 / 30.0) + 1,) * 2, np.float32)
        dmeta = None
    else:
        raise ValueError(f"unknown region type {region['type']}")
    grid = Grid(z, 1.0, half)
    n = z.shape[0]
    ent = spec["entities"]

    # ---------------------------------------------------------- roads / paths
    # Roads with a `class` (configs/roads.json) are physical: a grade-limited
    # vertical profile that starts on the parent road (`connects`), and cut/fill
    # of the terrain under them. Roads without a class are draped (old specs).
    rcfg = load_json("configs/roads.json")
    from scipy import ndimage as _ndi
    grid.z_smooth = (dmeta["core_z_smooth"] if dmeta and "core_z_smooth" in dmeta else _ndi.gaussian_filter(z, 4.0))
    roads = road_records(ent.get("roads", []), grid, rcfg)
    if any(r_["profiled"] for r_ in roads):
        pads = []   # turning circles: a round level pad at the street end
        for r_ in roads:
            if r_.get("end_cap"):
                e = r_["centerline"][-1]
                pads.append({"centerline": [e, [e[0] + 0.01, e[1], e[2]]], "width": 2 * r_["end_cap"], "shoulder": 0.5, "blend_m": 3.0})
        rplan.apply_corridors(grid, [r_ for r_ in roads if r_["profiled"]] + pads)
        z = grid.z
    paths = []
    for pa in ent.get("paths", []):
        pc_ = rcfg["paths"][pa.get("surface", "dirt")]
        p = _densify(np.asarray(pa["centerline"], float), 1.0)
        paths.append({"id": pa["id"], "kind": pc_["kind"], "surface": pa.get("surface", "dirt"),
                      "width": float(pa.get("width", sum(pc_["width"]) / 2)),
                      "centerline": [[float(x), float(y), float(zz)] for (x, y), zz in zip(p, grid.sample(p[:, 0], p[:, 1]))]})

    # -------------------------------------------------------------- buildings
    buildings = [plan_record(b, seed, grid, _rooftop) for b in ent.get("buildings", [])]
    fillets = junction_fillets(roads, ent, grid, rcfg)
    pcat = load_json("configs/props.json")["types"]
    props = [dict(p_, kind=pcat[p_["type"]]["kind"], z=float(grid.sample(p_["x"], p_["y"])[0]),
                  yaw=math.radians(p_.get("yaw_deg", 0.0)), height=float(p_.get("height", 8.0)),
                  arm=float(pcat[p_["type"]].get("arm_m", 1.5))) for p_ in ent.get("props", [])]

    # ------------------------------------------------------------ yard walls
    # One record per wall segment, on the ground after road cut/fill; all
    # segments of a wall entity share its instance id (`label_id`).
    walls = []
    for w in ent.get("walls", []):
        for k, sg in enumerate(w["segments"]):
            p = np.asarray(sg, float)
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(seg)])
            t = np.unique(np.concatenate([s, np.arange(0, s[-1], 1.0)]))   # keep the corners
            xy = np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])
            walls.append({"id": f"{w['id']}_{k}", "label_id": w["id"], "kind": f"wall.{'fence' if w['type'] == 'fence' else 'yard'}",
                          "type": w["type"], "role": WALL_ROLE[w["type"]], "height": float(w.get("height", 1.6)),
                          "thickness": float(w.get("thickness", 0.25)), "tint": list(w.get("tint", [1.0, 1.0, 1.0])),
                          "points": [[float(x), float(y), float(z)] for (x, y), z in zip(xy, grid.sample(xy[:, 0], xy[:, 1]))]})

    # ------------------------------------------------------------------ trees
    # Single trees and groves (configs/vegetation.json species). A grove is one
    # entity: its trees share the grove's id (`label_id`) and kind veg.grove.<species>.
    vcfg = load_json("configs/vegetation.json")["species"]
    reg = load_json("configs/assets.json")["assets"]
    veg = []

    def models_for(role, usd):
        """Model files of the assets playing `role`: USD trees (not their branch
        libraries), or meshes (highest detail level when LODs exist)."""
        ms = [m for a in reg.values() if role in a.get("roles", []) and a.get("status") == "ok" for m in a["models"]]
        if usd:
            return sorted(m for m in ms if m.lower().endswith(".usd") and "Branches" not in m)
        ms = [m for m in ms if not m.lower().endswith(".usd")]
        return sorted([m for m in ms if "LOD0" in m] or ms)

    def tree_item(item_id, label_id, species, kind, x, y, r, t):
        sp = vcfg[species]
        z = float(grid.sample(x, y)[0])
        yaw = float(math.radians(t.get("yaw_deg", r.uniform(0, 360))))
        base = {"id": item_id, "label_id": label_id, "kind": kind, "species": species, "x": float(x), "y": float(y)}
        if sp["source"] == "usd":
            ms = models_for(sp["role"], True)
            return dict(base, type="usd_tree", role=sp["role"], variant=t.get("variant") or ms[int(r.integers(len(ms)))],
                        twig_keep=0.3, z=z - 0.1, rot=[0.0, 0.0, yaw], scale=float(t.get("scale", r.uniform(*sp["scale"]))))
        if sp["source"] == "model":
            ms = models_for(sp["role"], False)
            return dict(base, type="model", role=sp["role"], variant=ms[int(r.integers(len(ms)))], z=z - 0.05,
                        rot=[0.0, 0.0, yaw], scale=float(t.get("scale", r.uniform(*sp["scale"]))))
        if sp["source"] == "date_palm":
            return dict(base, type="date_palm", role=sp["role"], trunk_role=sp["trunk_role"], z=z - 0.15, yaw=yaw,
                        height=float(t.get("height", r.uniform(*sp["height_m"]))), lean_deg=float(abs(r.normal(0, 3))),
                        lean_dir=float(r.uniform(0, 2 * math.pi)), fronds=int(r.integers(18, 30)), seed=int(r.integers(1 << 30)))
        return dict(base, type="proc_tree", z=z - 0.05, yaw=yaw, variant=int(r.integers(5)),
                    height=float(t.get("height", r.uniform(*sp["height_m"]))), crown=float(t.get("crown", r.uniform(*sp["crown_m"]))))

    for t in ent.get("trees", []):
        r = rng(seed, "spec_tree", t["id"])
        kind = vcfg[t["species"]]["kind"]
        veg.append(tree_item(t["id"], t["id"], t["species"], kind, t["x"], t["y"], r, t))
    from synth.spec.vegetation import grove_trees
    for g in ent.get("groves", []):
        kind = f"veg.grove.{g['species']}"
        for k, (x, y) in enumerate(grove_trees(g, seed)):
            r = rng(seed, "grove_tree", g["id"], k)
            veg.append(tree_item(f"{g['id']}_{k:03d}", g["id"], g["species"], kind, x, y, r, {}))

    # ------------------------------------------- terrain attributes + splat maps
    free = np.ones_like(z)
    X, Y = grid.xy()
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import unary_union
    blockers = [LineString(np.asarray(r_["centerline"])[:, :2]).buffer(r_["width"] / 2 + r_["shoulder"] + 0.5) for r_ in roads]
    blockers += [LineString(np.asarray(p_["centerline"])[:, :2]).buffer(p_["width"] / 2 + 0.3) for p_ in paths]
    blockers += [Polygon(fp).buffer(5.0) for b_ in buildings for fp in (b_["footprint"], b_["annex"]) if fp]   # yards: no wild shrubs, no stones
    blockers += [Polygon(f["polygon"]) for f in ent.get("fields", [])]   # farm fields are cleared: no stones, no shrubs
    yards = [Polygon(w["yard"]) for w in ent.get("walls", []) if len(w.get("yard", [])) >= 3]
    blockers += [y.buffer(0.5) for y in yards]   # walled yards and the wall foot: cleared
    blockers += [LineString(sg).buffer(0.6) for w in ent.get("walls", []) for sg in w["segments"]]
    blk = unary_union(blockers)
    from shapely import contains_xy
    free[contains_xy(blk, X, Y)] = 0.0
    wet = terrain_wetness(dmeta.pop("core_z_smooth")) if region["type"] == "dem" else np.zeros_like(z)
    near_roads = unary_union([LineString(np.asarray(r_["centerline"])[:, :2]).buffer(r_["width"] / 2 + 2.5) for r_ in roads])
    near_bld = unary_union([Polygon(fp).buffer(6.5) for b_ in buildings for fp in (b_["footprint"], b_["annex"]) if fp] + yards)
    shoulder = contains_xy(near_roads, X, Y).astype(np.float32)
    trampled = contains_xy(near_bld, X, Y).astype(np.float32)
    from scipy import ndimage
    shoulder = ndimage.gaussian_filter(shoulder, 1.2)
    trampled = ndimage.gaussian_filter(trampled, 2.0)
    # B = drainage wetness (wadi beds), core only
    splat = (np.stack([shoulder, trampled, wet.astype(np.float32), np.ones_like(z)], -1)[::-1] * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(splat).save(os.path.join(out_dir, "splat0.png"))
    fl = [{"polygon": f["polygon"], "state": f["crop"], "row_angle": math.radians(f.get("row_deg", 0.0)) % math.pi}
          for f in ent.get("fields", [])]
    from synth.planner.agriculture import fields_raster
    Image.fromarray(fields_raster(grid, fl, 1.0) if fl else np.zeros((n, n, 4), np.uint8)).save(os.path.join(out_dir, "fields.png"))
    np.savez_compressed(os.path.join(out_dir, "terrain.npz"), core_z=z.astype(np.float32), mid_z=mid, far_z=far,
                        free=free.astype(np.float32), wet=wet.astype(np.float32))

    # ---------------------------------------------------------------- scatter
    scatter = []
    for name, sc in spec.get("scatter", {}).items():
        params = {k: v for k, v in sc.items() if k not in ("role", "kind")}
        f = spec.get("scatter_scale", {}).get(sc["kind"].split(".")[0], 1.0)   # per-seed: veg / rock
        if "Density" in params:
            params["Density"] = params["Density"] * f
        params.setdefault("Seed", int(rng(seed, "scatter", name).integers(1 << 30)))
        scatter.append({"name": name, "role": sc["role"], "kind": sc["kind"], "params": params})

    # ------------------------------------------------------------ sun, cameras
    env = spec["environment"]
    lat, lon = dmeta["origin_latlon"] if dmeta else spec["region"]["latlon"]
    local = datetime.fromisoformat(f"{env['date']}T{env['local_time']}")
    utc = (local - timedelta(hours=env["utc_offset_h"])).replace(tzinfo=timezone.utc)
    el, az = solar_position(lat, lon, utc)
    sun = {"datetime_local": local.isoformat(), "datetime_utc": utc.isoformat(), "utc_offset_h": env["utc_offset_h"],
           "elevation_deg": el, "azimuth_deg": az}
    cams = []
    for c in spec["cameras"]:
        res = c.get("resolution", [1024, 1024])
        R = cm.slant_range_for_gsd(c["gsd_m"], c.get("focal_mm", 50), c.get("sensor_width_mm", 36), res[0])
        tz = float(grid.sample(*c["target"])[0])
        pos, eul = cm.camera_pose([c["target"][0], c["target"][1], tz], R, c["tilt_deg"], c.get("heading_deg", 0.0))
        across, along = cm.gsd_at_centre(R * math.cos(math.radians(c["tilt_deg"])), c.get("focal_mm", 50), c.get("sensor_width_mm", 36), res[0], c["tilt_deg"])
        cams.append({"name": c["id"], "tilt_deg": c["tilt_deg"], "heading_deg": c.get("heading_deg", 0.0), "slant_range_m": R,
                     "location": [float(v) for v in pos], "rotation_euler": [float(e) for e in eul],
                     "focal_mm": c.get("focal_mm", 50), "sensor_width_mm": c.get("sensor_width_mm", 36), "resolution": res,
                     "gsd_centre_m": {"across": across, "along": along}, "height_above_target_m": float(pos[2] - tz),
                     "K": cm.intrinsics(c.get("focal_mm", 50), c.get("sensor_width_mm", 36), res[0], res[1]).tolist()})

    # --------------------------------------------------------------- assemble
    if dmeta:
        dtm = dict(dmeta)
        dtm["source"] = region.get("source", "dem")
        water = region.get("water_level_abs")
        water = None if water is None else water - dmeta["origin_utm"][2]
    else:
        import pyproj
        epsg = 32636
        ex, ey = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        dtm = {"source": "flat_test", "epsg": epsg, "origin_utm": [ex, ey, region.get("elevation_m", 0.0)],
               "origin_latlon": [lat, lon], "core_res_m": 1.0, "core_n": n, "core_half_m": half, "mid_res_m": 5.0,
               "mid_n": mid.shape[0], "mid_half_m": 1600.0, "far_res_m": 30.0, "far_n": far.shape[0], "far_half_m": 9000.0,
               "site_rank": 0}
        water = None
    cfg = {"render": spec["render"], "post": spec["post"], "terrain_material": spec["terrain_material"],
           "missing_asset_policy": "error", "camera": {}}
    build = {
        "version": 1, "seed": seed, "scene_config": f"spec:{spec['scene_id']}", "config": cfg,
        "dtm": dtm,
        "terrain": {"file": "terrain.npz", "splat": ["splat0.png"], "fields": "fields.png", "splat_res_m": 1.0, "water_level": water},
        "roads": roads, "junction_fillets": fillets, "paths": paths, "props": props, "buildings": buildings, "parcels": [], "walls": walls, "fields": [],
        "vegetation": veg, "rocks": [], "scatter_systems": scatter, "sun": sun, "cameras": cams,
        "counts": {"roads": len(roads), "paths": len(paths), "buildings": len(buildings), "trees": len(veg),
                   "walls": len(ent.get("walls", []))},
        "spec": {"scene_id": spec["scene_id"], "spec_version": spec["spec_version"]},
    }
    save_json(build, os.path.join(out_dir, "plan.json"))
    save_json(spec, os.path.join(out_dir, "spec.json"))
    return os.path.join(out_dir, "plan.json")


def _densify(p, step):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1], step)
    t = np.append(t, s[-1])
    xy = np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])
    # gentle smoothing of hand-drawn polylines (keeps the ends)
    if len(xy) > 7:
        from scipy import ndimage
        sm = ndimage.gaussian_filter1d(xy, 3.0, axis=0, mode="nearest")
        w = np.minimum(np.arange(len(xy)), np.arange(len(xy))[::-1]) / 6.0
        w = np.clip(w, 0, 1)[:, None]
        xy = xy * (1 - w) + sm * w
    return xy
