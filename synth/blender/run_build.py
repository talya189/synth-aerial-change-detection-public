"""Build, render and annotate one planned scene (runs inside Blender 5.2).

    Blender -b --factory-startup -P synth/blender/run_build.py -- \
        --plan outputs/scene_0042/plan.json [--res 1024 1024] [--samples 32] \
        [--views nadir oblique45] [--no-render]
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from synth.blender import building_builder, lighting, render, road_builder, scatter_builder, terrain_builder, tree_builder  # noqa: E402
from synth.blender.asset_manager import AssetManager  # noqa: E402
from synth.blender.labels import IdRegistry, label_table  # noqa: E402
from synth.blender.meshutil import collection  # noqa: E402
from synth.common.config import load_json, project_path, save_json  # noqa: E402
from synth.common.rng import rng  # noqa: E402
from synth.post import annotations, sensor_model  # noqa: E402


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--res", type=int, nargs=2)
    ap.add_argument("--samples", type=int)
    ap.add_argument("--views", nargs="*")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


def log(*a):
    print("[build]", *a, flush=True)


def validate_grounding(plan, T):
    """Automated check that nothing floats: every scattered item must sit at or
    below the terrain surface, every building's foundation below its lowest corner."""
    items = [i for i in plan["vegetation"] + plan["rocks"] + plan.get("props", []) if "z" in i]
    xs = np.array([i["x"] for i in items])
    ys = np.array([i["y"] for i in items])
    zs = np.array([i["z"] for i in items])
    gap = zs - T.height(xs, ys)
    bad = int((gap > 0.05).sum())
    bfloat = 0
    for b in plan["buildings"]:
        fp = np.asarray(b["footprint"])
        if b["foundation_z"] > T.height(fp[:, 0], fp[:, 1]).min() - 0.05:
            bfloat += 1
    return {"items_checked": len(items), "items_floating_gt_5cm": bad, "max_item_gap_m": float(gap.max()) if len(gap) else 0.0,
            "buildings_checked": len(plan["buildings"]), "buildings_floating": bfloat}


def main():
    a = parse()
    t0 = time.time()
    plan_path = project_path(a.plan)
    plan_dir = os.path.dirname(plan_path)
    plan = load_json(plan_path)
    cfg = plan["config"]
    rcfg = dict(cfg["render"])
    if a.samples:
        rcfg["samples"] = a.samples

    template = project_path(cfg.get("template", "templates/template.blend"))
    if os.path.exists(template):  # scene logic + prepared assets live in the template
        bpy.ops.wm.open_mainfile(filepath=template, load_ui=False)
        log("template", template)
        # the template's demo playground must never appear in a generated scene
        demo = bpy.data.collections.get("TEMPLATE_DEMO")
        if demo is not None:
            for ob in list(demo.objects):
                bpy.data.objects.remove(ob, do_unlink=True)
            bpy.data.collections.remove(demo)
    else:
        log("no template found, building everything from scratch:", template)
        bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    am = AssetManager(registry_path=cfg.get("assets_config", "configs/assets.json"), tier=rcfg.get("texture_tier", 1024), missing_policy=cfg.get("missing_asset_policy", "error"), log=log)
    classes = load_json("configs/classes.json")
    ids = IdRegistry(classes)

    cols = {k: collection(k) for k in ("01_TERRAIN", "02_ROADS_PATHS", "03_BUILDINGS", "04_VEGETATION", "05_ROCKS", "06_LIGHT_CAMERA")}
    T = terrain_builder.build_terrain(plan, plan_dir, am, cols["01_TERRAIN"], ids)
    log("terrain", round(time.time() - t0, 1))
    road_builder.build_roads(plan, am, cols["02_ROADS_PATHS"], ids, T)
    from synth.blender import props_builder
    n_props = len(props_builder.build_props(plan, cols["02_ROADS_PATHS"], ids))
    log(f"roads, {n_props} props", round(time.time() - t0, 1))
    building_builder.build_buildings(plan, am, cols["03_BUILDINGS"], ids)
    building_builder.build_walls(plan, am, cols["03_BUILDINGS"], ids)
    log("buildings", round(time.time() - t0, 1))
    n_models, n_palms = scatter_builder.build_scatter(plan, am, cols["04_VEGETATION"], cols["05_ROCKS"], ids)
    log(f"scatter {n_models} models, {n_palms} palms", round(time.time() - t0, 1))
    scat = scatter_builder.build_scatter_systems(plan, am, cols["04_VEGETATION"], ids, T.core_object)
    log(f"scatter systems {[o.name for o in scat]}", round(time.time() - t0, 1))
    n_trees = tree_builder.build_trees(plan, cols["04_VEGETATION"], ids)
    from synth.blender import proc_veg
    n_trees += proc_veg.build_proc_trees(plan, cols["04_VEGETATION"], ids, am.library)
    log(f"trees {n_trees}", round(time.time() - t0, 1))
    grounding = validate_grounding(plan, T)
    log("grounding check", grounding)

    atmo = lighting.setup_lighting(scene, plan, rng(plan["seed"], "lighting"))
    for o in scene.collection.objects:
        if o.type == "LIGHT":
            scene.collection.objects.unlink(o)
            cols["06_LIGHT_CAMERA"].objects.link(o)
    render.configure_cycles(scene, rcfg, plan["seed"], use_gpu=not a.cpu)
    cams = render.make_cameras(scene, plan)
    for _, ob in cams:
        scene.collection.objects.unlink(ob)
        cols["06_LIGHT_CAMERA"].objects.link(ob)
    scene.camera = cams[0][1]

    labels = label_table(ids, classes)
    save_json({str(k): v for k, v in labels.items()}, os.path.join(plan_dir, "labels.json"))
    # keep the plan inside the .blend for inspection
    txt = bpy.data.texts.new("plan.json")
    txt.write(json.dumps({k: v for k, v in plan.items() if k not in ("vegetation", "rocks")}, indent=1))
    blend = os.path.join(plan_dir, "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend, relative_remap=True, compress=True)
    log("saved", blend, round(time.time() - t0, 1))

    views_meta = []
    fallbacks = []
    if not a.no_render:
        for v, ob in cams:
            if a.views and v["name"] not in a.views:
                continue
            res = tuple(a.res) if a.res else tuple(v["resolution"])
            vdir = os.path.join(plan_dir, "views", v["name"])
            t1 = time.time()
            try:
                exr, png = render.render_view(scene, ob, res, vdir, v["name"])
            except RuntimeError as e:
                if "memory" not in str(e).lower() or scene.cycles.device == "CPU":
                    raise
                log(f"view {v['name']}: GPU out of memory, retrying on CPU")
                scene.cycles.device = "CPU"
                fallbacks.append(v["name"])
                exr, png = render.render_view(scene, ob, res, vdir, v["name"])
            rt = time.time() - t1
            masks = annotations.make_masks(exr, labels, classes, os.path.join(vdir, v["name"]))
            # sensor model on the display-referred render
            import OpenImageIO as oiio
            img = oiio.ImageBuf(png).get_pixels(oiio.FLOAT)[..., :3]
            layers = annotations.read_exr_layers(exr)
            sky = np.rint(layers["Object Index"][..., 0]) == 0
            out, sp = sensor_model.apply(img, layers["Depth"][..., 0], sky, rng(plan["seed"], "sensor", v["name"]), cfg["post"])
            annotations.write_png(os.path.join(vdir, f"{v['name']}_rgb.png"), out, np.uint8)
            scale = res[0] / v["resolution"][0]
            vm = dict(v)
            vm.update({"render_resolution": list(res), "render_seconds": rt, "masks": masks, "sensor_model": sp,
                       "gsd_centre_m_at_render_res": {k: g / scale for k, g in v["gsd_centre_m"].items()},
                       **render.camera_matrices(scene, ob, res)})
            views_meta.append(vm)
            log(f"view {v['name']} rendered in {rt:.1f}s", masks["class_fraction"])

    meta = {
        "seed": plan["seed"], "scene_config": plan["scene_config"], "plan_version": plan["version"],
        "blender_version": bpy.app.version_string, "blender_build_hash": bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else str(bpy.app.build_hash),
        "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
        "dtm": plan["dtm"], "sun": plan["sun"], "atmosphere": atmo, "render": rcfg,
        "device": scene.cycles.device, "cpu_fallback_views": fallbacks, "counts": plan["counts"], "instances": len(labels),
        "assets_used": am.used, "missing_roles": am.missing, "grounding_check": grounding,
        "classes": classes, "views": views_meta, "build_seconds": time.time() - t0,
        "valid_for_dataset": not am.missing and grounding["items_floating_gt_5cm"] == 0 and grounding["buildings_floating"] == 0,
    }
    save_json(meta, os.path.join(plan_dir, "metadata.json"))
    log("done", round(time.time() - t0, 1))


main()
