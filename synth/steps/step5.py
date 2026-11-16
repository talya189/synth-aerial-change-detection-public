"""Step 5: roads and paths in the scene spec (no vegetation; that is Step 6).

    ~/synth_aerial_work/venv/bin/python -m synth.steps.step5 [--regenerate] [--only moshav|arab_village|town] [--plan-only]

For each settlement style: a road network on the Korazim site (regional road,
access road, streets or lanes, dirt tracks; synth/spec/network.py), buildings
along the streets facing them (placement.grow_along_streets), footpaths as
desire lines. Everything is written as explicit entities into
configs/specs/step5_<style>.json (generated once, then reused; --regenerate
draws again). The spec sets "vegetation": false.

Output: ~/synth_aerial_work/outputs/steps/step5/<style>/
  network_map.png   shaded relief + roads by class + paths + buildings, with ids
  roads.json        per road: class, length, max grade (vs limit), junction
  scene.blend, views/, contact_sheet.jpg, id_map.png, buildings.json
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

from generate import blender_build
from synth.common.config import load_json, project_path
from synth.spec.buildings import footprint
from synth.spec.network import crossings, min_radius, plan_footpaths, plan_network
from synth.spec.placement import Site, grow_along_streets, village_centre
from synth.spec.scene_spec import clone, load, save
from synth.spec.to_build import build_data, dem_terrain, terrain_wetness
from synth.steps.step2 import hillshade
from synth.steps.step4 import id_map, report

STYLES = {"moshav": (40, 51), "arab_village": (45, 52), "town": (28, 53)}   # style: (buildings, seed)
CLASS_COL = {"main": (40, 40, 40), "street": (90, 90, 90), "lane": (120, 120, 120), "track": (170, 110, 40)}


def site_for(spec):
    z, half, _, _, meta = dem_terrain(spec["region"])
    site = Site(z, half, terrain_wetness(meta["core_z_smooth"]))
    site.grid.z_smooth = meta["core_z_smooth"]
    return site


def make_spec(style, n, seed, out_path, log=print):
    spec = clone(load(project_path("configs/specs/step3_ground.json")))
    spec.update(scene_id=f"step5_{style}", seed=seed, vegetation=False, scatter="galilee_summer")
    spec.pop("_comment", None)
    spec["region"].pop("resolved", None)
    for k in ("roads", "paths", "buildings", "trees"):
        spec["entities"][k] = []
    site = site_for(spec)
    centre = village_centre(site, margin_frac=0.55)   # well inside the core: room for streets all round
    plan_network(spec, site, style, seed, centre, log=log)
    added, rejected = grow_along_streets(spec, site, n, style, seed, centre)
    log(f"{len(added)} buildings along the streets; rejected candidates: {rejected}")
    paths = plan_footpaths(spec, site, load_json("configs/roads.json")["networks"][style]["footpaths"], seed)
    log(f"{len(paths)} footpaths")
    log(f"road crossings without a junction: {crossings(spec)}")
    xy = np.array([b["center"] for b in added] or [centre])
    c = [round(float(v), 1) for v in xy.mean(0)]
    ext = float(np.abs(xy - c).max()) + 40
    gsd = max(0.2, round(2 * ext / 1024, 2))
    spec["cameras"] = [
        {"id": "cam_nadir", "target": c, "tilt_deg": 0, "heading_deg": 0, "gsd_m": gsd, "resolution": [1024, 1024], "focal_mm": 50, "sensor_width_mm": 36},
        {"id": "cam_oblique", "target": c, "tilt_deg": 45, "heading_deg": 30, "gsd_m": round(gsd * 0.8, 2), "resolution": [1024, 1024], "focal_mm": 50, "sensor_width_mm": 36},
        {"id": "cam_street", "target": c, "tilt_deg": 62, "heading_deg": 200, "gsd_m": round(gsd * 0.4, 2), "resolution": [1024, 1024], "focal_mm": 50, "sensor_width_mm": 36},
    ]
    spec["_comment"] = (f"Step 5, style '{style}': road network (synth/spec/network.py), {len(added)} buildings along the "
                        f"streets, {len(paths)} footpaths; no vegetation. Generated once; edit freely.")
    save(spec, project_path(out_path))
    return spec


def network_map(spec, site, out_png, px=1000):
    hs = hillshade(site.grid.z, 1.0)
    im = Image.fromarray((np.stack([hs * 0.85 + 0.1] * 3, -1)[::-1] * 255).astype(np.uint8)).convert("RGB")
    im = im.resize((px, px))
    dr = ImageDraw.Draw(im)
    s = px / (2 * site.half)
    P = lambda x, y: ((x + site.half) * s, (site.half - y) * s)
    rc = load_json("configs/roads.json")["classes"]
    for rd in spec["entities"]["roads"]:
        pts = [P(x, y) for x, y in rd["centerline"]]
        dr.line(pts, fill=CLASS_COL[rd["class"]], width=max(2, int(rc[rd["class"]]["width"] * s)))
        dr.text(pts[len(pts) // 2], rd["id"], fill=(0, 0, 160))
    for pa in spec["entities"]["paths"]:
        dr.line([P(x, y) for x, y in pa["centerline"]], fill=(230, 200, 40), width=1)
    for b in spec["entities"]["buildings"]:
        fp = footprint(b)
        for g in getattr(fp, "geoms", [fp]):
            dr.polygon([P(x, y) for x, y in g.exterior.coords], fill=(220, 40, 60))
    dr.text((8, 8), "roads: main dark grey, street grey, lane light grey, track brown; paths yellow; buildings red", fill=(0, 0, 0))
    im.save(out_png)


def road_report(scene_dir):
    """Per road from the built plan: length, steepest grade (vs its limit), junction."""
    plan = load_json(f"{scene_dir}/plan.json")
    labels = load_json(f"{scene_dir}/labels.json")
    meta = load_json(f"{scene_dir}/metadata.json")
    iid = {v["plan_id"]: k for k, v in labels.items()}
    rows = []
    for rd in plan["roads"] + plan["paths"]:
        p = np.asarray(rd["centerline"])
        seg = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
        L = float(seg.sum())
        row = {"id": rd["id"], "class": rd.get("class") or rd["kind"], "length_m": round(L, 1),
               "pixels": {v["name"]: v["masks"]["instance_pixels"].get(iid.get(rd["id"], "-"), 0) for v in meta["views"]}}
        if rd.get("profiled"):
            # grade over 10 m windows (the profile itself is limited per metre)
            s = np.concatenate([[0], np.cumsum(seg)])
            k = np.searchsorted(s, s + 10.0).clip(0, len(s) - 1)
            g = np.abs(p[k, 2] - p[:, 2]) / np.maximum(s[k] - s, 1e-6)
            g = g[s[k] - s > 5]
            row.update(max_grade=round(float(g.max()), 3) if len(g) else 0.0, limit=rd["max_grade"],
                       junction_with=rd.get("junction_with"))
            row["grade_ok"] = row["max_grade"] <= rd["max_grade"] + 0.005
        rows.append(row)
    with open(f"{scene_dir}/roads.json", "w") as f:
        json.dump(rows, f, indent=1)
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--only", choices=list(STYLES))
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    for style, (n, seed) in STYLES.items():
        if a.only and a.only != style:
            continue
        sp = f"configs/specs/step5_{style}.json"
        print(f"{style}:")
        if a.regenerate or not os.path.exists(project_path(sp)):
            make_spec(style, n, seed, sp)
        spec = load(project_path(sp))
        out = project_path(f"outputs/steps/step5/{style}")
        os.makedirs(out, exist_ok=True)
        site = site_for(spec)
        network_map(spec, site, f"{out}/network_map.png")
        if a.plan_only:
            continue
        blender_build(build_data(spec, out), log=print)
        id_map(out, spec, f"{out}/id_map.png")
        brows, meta = report(out, spec, site)
        rows, meta = road_report(out)
        for r_ in rows:
            g = f"max grade {r_['max_grade']:.3f} / {r_['limit']:.2f} {'ok' if r_['grade_ok'] else 'TOO STEEP'}" if "max_grade" in r_ else ""
            print(f"  {r_['id']:9s} {r_['class']:13s} {r_['length_m']:6.0f} m  {g}  nadir px {r_['pixels'].get('cam_nadir', 0)}")
        bad = [b["id"] for b in brows if b.get("problems")]
        print(f"  buildings {len(brows)}, rule problems {bad}, grounding {meta['grounding_check']}")
        print("  class fractions (nadir):", next(v for v in meta["views"] if v["name"] == "cam_nadir")["masks"]["class_fraction"])
    print(f"step 5 done in {time.time() - t0:.0f}s -> {project_path('outputs/steps/step5')}")


if __name__ == "__main__":
    main()
