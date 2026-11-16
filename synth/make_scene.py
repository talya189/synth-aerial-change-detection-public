"""Make a scene from a place, a style and a seed.

    ~/synth_aerial_work/venv/bin/python -m synth.make_scene --list
    ~/synth_aerial_work/venv/bin/python -m synth.make_scene --place arab_village --style hillside --seed 7
    ~/synth_aerial_work/venv/bin/python -m synth.make_scene --place arab_city --style basin_dense --seed 3 --plan-only

Writes the spec to configs/specs/places/<place>_<style>_<seed>.json (kept; re-running
the same command reuses it unless --regenerate) and builds it to
~/synth_aerial_work/outputs/scenes/<place>/<style>/seed_<seed>/:
  plan_map.png      shaded relief with roads, fields, groves, trees, buildings, paths
  scene.blend, views/, contact_sheet.jpg, id_map.png
  report.json       counts, rule checks (crossings, connectivity, grades, buildings, access, walls)
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw


def plan_map(spec, site, out_png, px=1000):
    from synth.spec.buildings import footprint
    from synth.steps.step2 import hillshade
    from synth.common.config import load_json
    hs = hillshade(site.grid.z, 1.0)
    im = Image.fromarray((np.stack([hs * 0.8 + 0.15] * 3, -1)[::-1] * 255).astype(np.uint8)).convert("RGB").resize((px, px))
    dr = ImageDraw.Draw(im)
    s = px / (2 * site.half)
    P = lambda x, y: ((x + site.half) * s, (site.half - y) * s)
    ent = spec["entities"]
    for f in ent.get("fields", []):
        dr.polygon([P(x, y) for x, y in f["polygon"]], fill={"ploughed": (150, 110, 80), "stubble": (225, 205, 150), "fallow": (200, 190, 160)}[f["crop"]])
    for g in ent.get("groves", []):
        dr.polygon([P(x, y) for x, y in g["polygon"]], outline=(40, 110, 40), fill=(130, 165, 110) if not g.get("forest") else (70, 120, 70))
    rc = load_json("configs/roads.json")["classes"]
    col = {"main": (30, 30, 30), "street": (80, 80, 80), "lane": (120, 120, 120), "track": (170, 110, 40)}
    for rd in ent["roads"]:
        dr.line([P(x, y) for x, y in rd["centerline"]], fill=col[rd["class"]], width=max(2, int(rc[rd["class"]]["width"] * s)))
    for pa in ent["paths"]:
        dr.line([P(x, y) for x, y in pa["centerline"]], fill=(230, 200, 40), width=1)
    wcol = {"stone": (120, 70, 30), "block": (90, 90, 110), "plaster": (150, 60, 150), "fence": (20, 140, 170)}
    for w in ent.get("walls", []):
        for sg in w["segments"]:
            dr.line([P(x, y) for x, y in sg], fill=wcol[w["type"]], width=2)
    for b in ent["buildings"]:
        fp = footprint(b)
        for g in getattr(fp, "geoms", [fp]):
            dr.polygon([P(x, y) for x, y in g.exterior.coords], fill=(40, 120, 200) if b.get("type") == "mosque" else (220, 40, 60))
    for pr in ent.get("props", []):
        x, y = P(pr["x"], pr["y"])
        dr.rectangle([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 140, 0))
    for t in ent["trees"]:
        x, y = P(t["x"], t["y"])
        dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(20, 120, 20))
    dr.text((8, 8), f"{spec['scene_id']}: roads main/street/lane/track, paths yellow, buildings red (mosque blue), "
                    "fields brown/straw, groves green, yard walls dark lines", fill=(0, 0, 0))
    im.save(out_png)


def checks(spec, site):
    from synth.common.config import load_json
    from synth.spec.access import access_report, paths_through_buildings
    from synth.spec.network import crossings, dangling_paths, radius_violations, unreachable
    from synth.spec.placement import check_building
    from synth.spec.to_build import grade_violations, road_records
    from synth.spec.props import prop_report
    from synth.spec.yards import wall_report
    import collections
    ent = spec["entities"]
    return {
        "counts": {k: len(v) for k, v in ent.items()},
        "building_types": dict(collections.Counter(b.get("type", "house") for b in ent["buildings"])),
        "road_classes": dict(collections.Counter(r["class"] for r in ent["roads"])),
        "tree_species": dict(collections.Counter(t["species"] for t in ent["trees"])),
        "crossings_without_junction": crossings(spec),
        "unreachable_roads": unreachable(spec),
        "roads_too_steep": grade_violations(road_records(ent["roads"], site.grid, load_json("configs/roads.json"))),
        "buildings_breaking_rules": [b["id"] for b in ent["buildings"] if check_building(b, spec, site)[0]],
        "buildings_without_access": access_report(spec),
        "access_paths_through_buildings": paths_through_buildings(spec),
        "paths_ending_nowhere": dangling_paths(spec),
        "mosques_closer_than_200m": [(a["id"], b["id"]) for i, a in enumerate(ent["buildings"]) for b in ent["buildings"][i + 1:]
                                     if a.get("type") == b.get("type") == "mosque"
                                     and np.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1]) < 200],
        "roads_tighter_than_class_radius": radius_violations(spec, load_json("configs/roads.json")),
        "wall_types": dict(collections.Counter(w["type"] for w in ent.get("walls", []))),
        **wall_report(spec),
        "props": dict(collections.Counter(p["type"] for p in ent.get("props", []))),
        "props_on_buildings_roads_paths": prop_report(spec),
    }


def main():
    from synth.common.config import project_path
    from synth.places import _site, make_spec, places, styles
    from synth.spec.scene_spec import load, save
    ap = argparse.ArgumentParser(description="Make a scene: place + style + seed")
    ap.add_argument("--list", action="store_true", help="list places and their styles")
    ap.add_argument("--place")
    ap.add_argument("--style")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--regenerate", action="store_true", help="draw the spec again even if it exists")
    ap.add_argument("--plan-only", action="store_true", help="spec, plan map and checks only (no Blender)")
    ap.add_argument("--samples", type=int)
    a = ap.parse_args()
    if a.list or not (a.place and a.style):
        for p in places():
            print(p, ":", ", ".join(styles(p)))
        return
    t0 = time.time()
    sp = f"configs/specs/places/{a.place}_{a.style}_{a.seed:04d}.json"
    os.makedirs(project_path("configs/specs/places"), exist_ok=True)
    if a.regenerate or not os.path.exists(project_path(sp)):
        save(make_spec(a.place, a.style, a.seed), project_path(sp))
    spec = load(project_path(sp))
    out = project_path(f"outputs/scenes/{a.place}/{a.style}/seed_{a.seed:04d}")
    os.makedirs(out, exist_ok=True)
    site, _ = _site(spec["region"])
    plan_map(spec, site, f"{out}/plan_map.png")
    rep = checks(spec, site)
    with open(f"{out}/report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items()}, indent=None)[:1500])
    if not a.plan_only:
        from generate import blender_build
        from synth.spec.to_build import build_data
        from synth.steps.step4 import id_map
        blender_build(build_data(spec, out), samples=a.samples, log=print)
        id_map(out, spec, f"{out}/id_map.png")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
