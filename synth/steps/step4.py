"""Step 4: buildings in the scene spec.

    ~/synth_aerial_work/venv/bin/python -m synth.steps.step4 [--regenerate] [--only catalogue|villages|<style>]

Scenes:
  catalogue        configs/specs/step4_catalogue.json: every building type and option on a flat plot
  village_<style>  configs/specs/step4_village_<style>.json: the Korazim site plus a settlement grown
                   by the placement rules (synth/spec/placement.py) in one of the styles of
                   configs/buildings.json (moshav, arab_village, kibbutz, old_town, town), with the
                   Galilee ground cover. Generated once and then reused; --regenerate draws again.

Output: ~/synth_aerial_work/outputs/steps/step4/{catalogue,village_<style>}/
  scene.blend, views/, contact_sheet.jpg, id_map.png (nadir view with building ids),
  buildings.json (per building: placement checks, visible pixels per view)
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
from synth.spec.placement import Site, check_building, grow_village
from synth.spec.scene_spec import clone, load, save
from synth.spec.to_build import build_data, dem_terrain, terrain_wetness

VILLAGES = {"moshav": (22, 41), "arab_village": (34, 42), "kibbutz": (28, 43), "old_town": (22, 44), "town": (24, 45)}   # style: (buildings, seed)


def site_for(spec):
    z, half, _, _, meta = dem_terrain(spec["region"])
    return Site(z, half, terrain_wetness(meta["core_z_smooth"]))


def make_village_spec(base_path, out_path, style, n, seed, log=print):
    spec = clone(load(project_path(base_path)))
    spec["scene_id"] = f"step4_village_{style}"
    spec["seed"] = seed
    spec.pop("_comment", None)
    spec["region"].pop("resolved", None)
    spec["scatter"] = "galilee_summer"
    site = site_for(spec)
    added, rejected = grow_village(spec, site, n, style, seed)
    log(f"village: {len(added)} buildings placed; rejected candidates: {rejected}")
    xy = np.array([b["center"] for b in added])
    c = [round(float(v), 1) for v in xy.mean(0)]
    ext = float(np.abs(xy - c).max()) + 25
    tz = float(site.grid.sample(c[0], c[1])[0])
    gsd = max(0.15, round(2 * ext / 1024, 2))
    spec["cameras"] = [
        {"id": "cam_nadir", "target": c, "tilt_deg": 0, "heading_deg": 0, "gsd_m": gsd, "resolution": [1024, 1024], "focal_mm": 50, "sensor_width_mm": 36},
        {"id": "cam_oblique", "target": c, "tilt_deg": 45, "heading_deg": 30, "gsd_m": round(gsd * 0.8, 2), "resolution": [1024, 1024], "focal_mm": 50, "sensor_width_mm": 36},
        {"id": "cam_low", "target": c, "tilt_deg": 68, "heading_deg": 200, "gsd_m": round(gsd * 0.5, 2), "resolution": [1024, 1024], "focal_mm": 50, "sensor_width_mm": 36},
    ]
    spec["_comment"] = (f"Step 4 village, style '{style}': the Step 3 site plus {len(added)} buildings placed by "
                        f"synth/spec/placement.py. Generated once; edit freely. Village centre {c}, ground {tz:.1f} m.")
    save(spec, project_path(out_path))
    return spec


def id_map(scene_dir, spec, out_png, view="cam_nadir"):
    """The view's RGB with each building's id at its projected centre."""
    meta = load_json(f"{scene_dir}/metadata.json")
    plan = load_json(f"{scene_dir}/plan.json")
    v = next(x for x in meta["views"] if x["name"] == view)
    W, K = np.array(v["world_to_cam_opencv"]), np.array(v["K"])
    sx = v["render_resolution"][0] / v["resolution"][0]
    im = Image.open(f"{scene_dir}/views/{view}/{view}_rgb.png").convert("RGB")
    dr = ImageDraw.Draw(im)
    for b in plan["buildings"]:
        c = np.mean(b["footprint"], axis=0)
        p = W @ np.array([c[0], c[1], b["floor_z"] + b["floors"] * b["floor_height"], 1.0])
        if p[2] <= 0:
            continue
        u, vv = (K @ (p[:3] / p[2]))[:2] * sx
        dr.rectangle([u - 26, vv - 8, u + 26, vv + 8], fill=(0, 0, 0))
        dr.text((u - 22, vv - 6), b["id"], fill=(255, 255, 0))
    im.save(out_png)


def report(scene_dir, spec, site=None):
    labels = load_json(f"{scene_dir}/labels.json")
    meta = load_json(f"{scene_dir}/metadata.json")
    iid = {v["plan_id"]: k for k, v in labels.items() if v["kind"].startswith("building.")}
    rows = []
    for b in spec["entities"]["buildings"]:
        row = {"id": b["id"], "type": b.get("type", "house"), "floors": b["floors"], "roof": b.get("roof"),
               "wall": b.get("wall"), "footprint_m2": round(footprint(b).area, 1)}
        if site is not None:
            problems, vals = check_building(b, spec, site)
            row.update(vals, problems=problems)
        row["pixels"] = {v["name"]: v["masks"]["instance_pixels"].get(iid.get(b["id"], "-"), 0) for v in meta["views"]}
        rows.append(row)
    with open(f"{scene_dir}/buildings.json", "w") as f:
        json.dump(rows, f, indent=1)
    return rows, meta


def run_scene(spec_path, out, site=None, log=print):
    spec = load(project_path(spec_path))
    out_dir = project_path(out)
    blender_build(build_data(spec, out_dir), log=log)
    id_map(out_dir, spec, f"{out_dir}/id_map.png")
    rows, meta = report(out_dir, spec, site)
    for r_ in rows:
        log(f"  {r_['id']} {r_['type']:10s} {r_['floors']}F {r_['roof']:11s} {r_['wall']:13s} "
            f"{r_['footprint_m2']:6.0f} m2  " + (f"slope {r_['slope_deg']:4.1f}  relief {r_['relief_m']:4.1f} m  "
                                                if 'slope_deg' in r_ else "")
            + f"pixels {r_['pixels']}" + (f"  PROBLEMS {r_['problems']}" if r_.get("problems") else ""))
    log(f"  grounding: {meta['grounding_check']}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--only", choices=["catalogue", "villages", *VILLAGES])
    a = ap.parse_args()
    t0 = time.time()
    if a.only != "village":
        print("catalogue:")
        run_scene("configs/specs/step4_catalogue.json", "outputs/steps/step4/catalogue")
    for style, (n, seed) in VILLAGES.items():
        if a.only not in (None, "villages", style):
            continue
        vpath = f"configs/specs/step4_village_{style}.json"
        if a.regenerate or not os.path.exists(project_path(vpath)):
            make_village_spec("configs/specs/step3_ground.json", vpath, style, n, seed)
        spec = load(project_path(vpath))
        print(f"village {style}:")
        run_scene(vpath, f"outputs/steps/step4/village_{style}", site_for(spec))
    print(f"step 4 done in {time.time() - t0:.0f}s -> {project_path('outputs/steps/step4')}")


if __name__ == "__main__":
    main()
