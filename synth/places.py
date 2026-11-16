"""Place + style + seed -> a complete scene spec (pure Python).

A place file (configs/places/<place>.json) holds the whole logic for one kind of
place, per style: terrain (area, landform, ground presets), roads, buildings,
footpaths, vegetation, fields. `make_spec` runs the generators in order:

  1 terrain    the seed picks one of the best `site_choices` real sites for the
               style's landform in its area (synth/prep/dtm.rank_landform)
  2 roads      main road, contour streets, cross streets, tracks (synth/spec/network.py)
  3 buildings  along the streets, landmarks first (synth/spec/placement.py)
  4 access     every building opens onto a road or gets an access path to one (synth/spec/access.py)
  4a yards     walls / fences around the house plots, gates on the way out (synth/spec/yards.py)
  4b fields    farm fields on gentle open ground (synth/spec/vegetation.plant_fields)
  5 trees      groves, forest, street and yard trees (synth/spec/vegetation.plant)
  6 footpaths  desire lines to roads, tracks or outlying buildings (synth/spec/network.plan_footpaths)
  7 props      streetlights (synth/spec/props.py)

Everything is written as explicit entities into the spec; the same place,
style and seed always give the same spec.
"""
import numpy as np

from synth.common.config import load_json
from synth.common.rng import rng
from synth.spec.access import plan_access
from synth.spec.network import plan_footpaths, plan_network
from synth.spec.placement import Site, fill_blocks, grow_along_streets
from synth.spec.scene_spec import validate
from synth.spec.to_build import dem_terrain, terrain_wetness
from synth.spec.vegetation import plant, plant_fields
from synth.spec.props import plan_streetlights
from synth.spec.yards import open_gaps, plan_yards


def places():
    import glob
    import os
    from synth.common.config import project_path
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(project_path("configs/places/*.json")))


def load_place(place):
    return load_json(f"configs/places/{place}.json")


def styles(place):
    return sorted(load_place(place)["styles"])


def _site(region):
    z, half, _, _, meta = dem_terrain(region)
    site = Site(z, half, terrain_wetness(meta["core_z_smooth"]))
    site.grid.z_smooth = meta["core_z_smooth"]
    return site, meta


def _cameras(centre, extent, tz=None, heading=30.0):
    gsd = max(0.2, round(2 * extent / 1024, 2))
    c = [round(float(centre[0]), 1), round(float(centre[1]), 1)]
    return [
        {"id": "cam_nadir", "target": c, "tilt_deg": 0, "heading_deg": 0, "gsd_m": gsd, "resolution": [1024, 1024], "focal_mm": 50, "sensor_width_mm": 36},
        {"id": "cam_oblique", "target": c, "tilt_deg": 45, "heading_deg": round(heading % 360, 1), "gsd_m": round(gsd * 0.75, 2), "resolution": [1024, 1024], "focal_mm": 50, "sensor_width_mm": 36},
        {"id": "cam_low", "target": c, "tilt_deg": 62, "heading_deg": round((heading + 170) % 360, 1), "gsd_m": round(gsd * 0.35, 2), "resolution": [1024, 1024], "focal_mm": 50, "sensor_width_mm": 36},
    ]


def set_sections(spec, roads_style, r, log=print):
    """Road cross-sections (configs/roads.json `sections`): the main road is
    `main_section` (two_lane, or divided with a median, wider); `marked_streets`
    = share of the streets (longest first) painted as two-lane streets."""
    sec = load_json("configs/roads.json")["sections"]
    roads = spec["entities"]["roads"]
    kind = roads_style.get("main_section", "two_lane")
    for rd in roads:
        if rd["class"] != "main":
            continue
        rd["section"] = kind
        if kind == "divided":
            d = sec["divided"]
            rd["lanes"], rd["median_m"] = d["lanes"], d["median_m"]
            rd["width"] = round(2 * d["lanes"] * d["lane_m"] + d["median_m"], 2)
    streets = sorted((rd for rd in roads if rd["class"] == "street"),
                     key=lambda rd: -float(np.sum(np.linalg.norm(np.diff(np.asarray(rd["centerline"])[:, :2], axis=0), axis=1))))
    for rd in streets[:int(round(roads_style.get("marked_streets", 0.0) * len(streets)))]:
        rd["section"] = "two_lane"
    log(f"main road {kind}; {sum(rd.get('section') == 'two_lane' for rd in streets)} two-lane streets")


def _draw(v, r):
    """One value from a `vary` entry: {"range": [lo, hi]} (integers if both are),
    {"choice": [a, b, ...]} or {"scale": [lo, hi]} (returned as a factor)."""
    if "choice" in v:
        return v["choice"][int(r.integers(len(v["choice"])))]
    lo, hi = v.get("range", v.get("scale"))
    if isinstance(lo, int) and isinstance(hi, int) and "range" in v:
        return int(r.integers(lo, hi + 1))
    return round(float(r.uniform(lo, hi)), 3)


def vary(st, r):
    """The seed's own character: a copy of the style with every `vary` entry
    drawn ("a.b.c": value path in the style; "...scale" entries multiply the
    number there). `types_jitter` = sigma of a log-normal factor on every
    building type weight. Returns (style, drawn values)."""
    import copy
    st = copy.deepcopy(st)
    drawn = {}
    for path, spec_v in st.get("vary", {}).items():
        if path.startswith("_"):
            continue
        val = _draw(spec_v, r)
        keys = path.split(".")
        d = st
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        if "scale" in spec_v and isinstance(d.get(keys[-1]), (int, float)):
            val = type(d[keys[-1]])(round(d[keys[-1]] * val, 3))
        d[keys[-1]] = val
        drawn[path] = val
    sig = st.get("vary", {}).get("_types_jitter", 0.0)
    if sig:
        t = st["buildings"]["types"]
        for k in t:
            t[k] = round(t[k] * float(np.exp(r.normal(0, sig))), 3)
        drawn["buildings.types"] = dict(t)
    return st, drawn


def make_spec(place, style, seed, log=print):
    pf = load_place(place)
    r = rng(seed, "place", place, style)
    st, character = vary(pf["styles"][style], r)
    base = load_json("configs/specs/step3_ground.json")   # environment, render and post settings
    tr = st["terrain"]
    aois = tr.get("aois", [tr["aoi"]] if "aoi" in tr else [])
    aoi = aois[int(r.integers(len(aois)))]
    region = {"type": "dem", "source": "copernicus_glo30", "aoi": aoi, "core_size_m": tr.get("core_size_m", 1000),
              "site_rank": int(r.integers(tr.get("site_choices", 10))),
              "selection": {"mode": tr["mode"], "target_slope_deg": tr["target_slope_deg"]}}
    env = dict(base["environment"])
    ev = st.get("environment", {})   # light: dry-season date and time of day drawn per seed
    if "months" in ev:
        env["date"] = f"2025-{int(r.integers(ev['months'][0], ev['months'][1] + 1)):02d}-{int(r.integers(1, 29)):02d}"
    if "hours" in ev:
        m = int(r.uniform(ev["hours"][0], ev["hours"][1]) * 60)
        env["local_time"] = f"{m // 60:02d}:{m % 60:02d}"
    if "haze_km" in ev:
        env["haze_visibility_km"] = int(r.integers(ev["haze_km"][0], ev["haze_km"][1] + 1))
    spec = {"spec_version": 1, "scene_id": f"{place}_{style}_{seed:04d}", "seed": int(seed),
            "place": {"place": place, "style": style, "seed": int(seed), "character": character},
            "region": region, "environment": env, "terrain_material": tr["ground_material"],
            "scatter": tr["ground_cover"], "render": base["render"], "post": base["post"], "cameras": [],
            "entities": {k: [] for k in ("roads", "paths", "buildings", "trees", "groves", "fields", "walls", "props")}, "history": []}
    if "rules" in st:
        spec["rules"] = st["rules"]
    if "ground_cover_scale" in st:   # sparse to lush wild bushes, few to many stones
        spec["scatter_scale"] = {k: _draw(v, r) for k, v in st["ground_cover_scale"].items()}
    site, meta = _site(region)
    log(f"{place}/{style} seed {seed}: site rank {region['site_rank']} at {[round(v, 4) for v in meta['origin_latlon']]}")
    centre = np.array([0.0, 0.0])   # the site search centres the core on the landform
    net = dict(st["roads"], name=style)
    plan_network(spec, site, net, seed, centre, log=log)
    set_sections(spec, st["roads"], r, log=log)
    bc = st["buildings"]
    n = int(r.integers(bc["count"][0], bc["count"][1] + 1))
    added, rejected = grow_along_streets(spec, site, n, dict(bc, name=style), seed, centre)
    log(f"{len(added)} of {n} buildings along the streets; rejected candidates: {rejected}")
    if len(added) < n and bc.get("infill", True):
        more, rej2 = fill_blocks(spec, site, n - len(added), dict(bc, name=style), seed, centre)
        added += more
        log(f"{len(more)} more by block infill (now {len(added)} of {n}); rejected: {rej2}")
    plan_access(spec, site, seed, st.get("access"), log=log)
    if st.get("yards", {}) is not False:
        plan_yards(spec, site, seed, st.get("yards"), log=log)
    fl = plant_fields(spec, site, st.get("fields"), seed)
    log(f"{len(fl)} field strips")
    plant(spec, site, dict(st["vegetation"], name=style), seed, centre, log=log)
    paths = plan_footpaths(spec, site, st["footpaths"]["count"], seed)
    open_gaps(spec, paths)
    if st.get("props", {}).get("streetlights", True) is not False:
        plan_streetlights(spec, site, seed, st.get("props", {}).get("streetlights"), log=log)
    log(f"{len(paths)} footpaths")
    xy = np.array([b["center"] for b in added] or [centre])
    c = np.median(xy, axis=0)
    ext = float(np.percentile(np.abs(xy - c).max(axis=1), 90)) + 40
    spec["cameras"] = _cameras(c, min(ext, 480.0), heading=float(r.uniform(0, 360)))
    spec["_comment"] = f"Generated by synth/places.py from configs/places/{place}.json, style '{style}', seed {seed}. Edit freely."
    validate(spec)
    return spec
