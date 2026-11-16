"""Step 6: vegetation (trees, groves, wild ground cover) in the scene spec.

    ~/synth_aerial_work/venv/bin/python -m synth.steps.step6 [--regenerate] [--only catalogue|moshav|arab_village|town] [--plan-only]

Scenes:
  catalogue   configs/specs/step6_catalogue.json: every tree species (two sizes), an
              olive grove and a pine-forest patch on a flat plot
  <style>     configs/specs/step6_<style>.json: the Step 5 settlement (same roads and
              buildings) planted by synth/spec/vegetation.py (yard and street trees,
              groves, forest) with the Galilee ground cover (wild shrubs, rocks)

Output: ~/synth_aerial_work/outputs/steps/step6/<scene>/
  scene.blend, views/, contact_sheet.jpg, id_map.png, vegetation.json (counts, ids, pixels)
"""
import argparse
import collections
import json
import os
import time

from generate import blender_build
from synth.common.config import load_json, project_path
from synth.spec.scene_spec import clone, load, save
from synth.spec.to_build import build_data
from synth.spec.vegetation import grove_trees, plant
from synth.steps.step4 import id_map
from synth.steps.step5 import site_for

STYLES = ("moshav", "arab_village", "town")


def make_spec(style, out_path, log=print):
    spec = clone(load(project_path(f"configs/specs/step5_{style}.json")))
    spec["scene_id"] = f"step6_{style}"
    spec.pop("vegetation", None)
    spec["scatter"] = "galilee_summer"
    spec["entities"]["trees"], spec["entities"]["groves"] = [], []
    site = site_for(spec)
    plant(spec, site, style, spec["seed"], spec["cameras"][0]["target"], log=log)
    spec["_comment"] = (f"Step 6, style '{style}': the Step 5 settlement planted by synth/spec/vegetation.py "
                        "(yard and street trees, groves, forest) with the wild ground cover. Generated once; edit freely.")
    save(spec, project_path(out_path))
    return spec


def make_catalogue(out_path):
    base = load(project_path("configs/specs/step4_catalogue.json"))
    sp = load_json("configs/vegetation.json")["species"]
    spec = clone(base)
    spec["scene_id"] = "step6_catalogue"
    spec["region"]["size_m"] = 200
    # rocks only: wild shrubs would hide the species being compared
    spec["scatter"] = {k: v for k, v in load_json("configs/presets/ground_cover/galilee_summer.json").items()
                       if isinstance(v, dict) and not v.get("kind", "").startswith("veg.")}
    trees = []
    x = -70.0
    for name, s in sp.items():
        for big in (False, True):
            t = {"id": f"tree_{len(trees) + 1:03d}", "species": name, "x": round(x, 1), "y": 30.0, "yaw_deg": 0.0}
            if "height_m" in s:
                t["height"] = s["height_m"][1] if big else s["height_m"][0]
            if "crown_m" in s:
                t["crown"] = s["crown_m"][1] if big else s["crown_m"][0]
            if "scale" in s:
                t["scale"] = s["scale"][1] if big else s["scale"][0]
            trees.append(t)
            x += 10.0 if not big else 13.0
    spec["entities"] = {"roads": [], "paths": [], "buildings": [], "trees": trees,
                        "groves": [{"id": "grove_001", "species": "olive", "polygon": [[-60, -20], [-5, -20], [-5, -60], [-60, -60]],
                                    "spacing_m": 7.0, "row_deg": 0.0, "forest": False},
                                   {"id": "grove_002", "species": "pine", "polygon": [[10, -15], [65, -15], [65, -65], [10, -65]],
                                    "spacing_m": 6.0, "row_deg": 0.0, "forest": True}]}
    spec["cameras"] = [
        {"id": "cam_nadir", "target": [0, -5], "tilt_deg": 0, "heading_deg": 0, "gsd_m": 0.12, "resolution": [1536, 1152], "focal_mm": 50, "sensor_width_mm": 36},
        {"id": "cam_species", "target": [0, 30], "tilt_deg": 62, "heading_deg": 0, "gsd_m": 0.05, "resolution": [1536, 768], "focal_mm": 50, "sensor_width_mm": 36},
        {"id": "cam_groves", "target": [0, -40], "tilt_deg": 50, "heading_deg": 10, "gsd_m": 0.09, "resolution": [1536, 1024], "focal_mm": 50, "sensor_width_mm": 36}]
    spec["_comment"] = ("Step 6 species catalogue: each species small and large (row, west to east, in configs/vegetation.json order), "
                        "an olive grove (rows) and a pine-forest patch (irregular).")
    save(spec, project_path(out_path))
    return spec


def report(scene_dir, spec):
    labels = load_json(f"{scene_dir}/labels.json")
    meta = load_json(f"{scene_dir}/metadata.json")
    iid = {v["plan_id"]: k for k, v in labels.items()}
    nadir = next(v for v in meta["views"] if v["name"] == "cam_nadir")
    rows = {"trees": dict(collections.Counter(t["species"] for t in spec["entities"]["trees"])),
            "groves": [{"id": g["id"], "species": g["species"], "trees": len(grove_trees(g, spec["seed"])),
                        "nadir_px": nadir["masks"]["instance_pixels"].get(iid.get(g["id"], "-"), 0)}
                       for g in spec["entities"].get("groves", [])],
            "trees_visible_in_nadir": sum(1 for t in spec["entities"]["trees"]
                                          if nadir["masks"]["instance_pixels"].get(iid.get(t["id"], "-"), 0) > 0),
            "class_fraction_nadir": nadir["masks"]["class_fraction"], "grounding": meta["grounding_check"]}
    with open(f"{scene_dir}/vegetation.json", "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--only", choices=["catalogue", *STYLES])
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    for name in ("catalogue", *STYLES):
        if a.only and a.only != name:
            continue
        sp = f"configs/specs/step6_{name}.json"
        print(f"{name}:")
        if a.regenerate or not os.path.exists(project_path(sp)):
            make_catalogue(sp) if name == "catalogue" else make_spec(name, sp)
        spec = load(project_path(sp))
        if a.plan_only:
            continue
        out = project_path(f"outputs/steps/step6/{name}")
        blender_build(build_data(spec, out), log=print)
        id_map(out, spec, f"{out}/id_map.png")
        rows = report(out, spec)
        print(f"  trees {rows['trees']}, visible in nadir {rows['trees_visible_in_nadir']}/{len(spec['entities']['trees'])}")
        print(f"  groves {[(g['id'], g['species'], g['trees'], g['nadir_px']) for g in rows['groves']]}")
        print(f"  class fractions (nadir) {rows['class_fraction_nadir']}  grounding {rows['grounding']}")
    print(f"step 6 done in {time.time() - t0:.0f}s -> {project_path('outputs/steps/step6')}")


if __name__ == "__main__":
    main()
