"""Dataset builder (standard Python venv).

    ~/synth_aerial_work/venv/bin/python -m synth.dataset.build configs/dataset/galilee_v1.json
    ... --count 3                 # override number of scenes
    ... --export-only             # rebuild exports from scenes already rendered

Renders scenes seed by seed with RESUME (a scene with a valid metadata.json
and all its view images is skipped) and FAILURE ISOLATION (a failing scene is
logged to failures.jsonl and the batch continues), then exports:

  <out>/scenes/scene_NNNNN/...           full per-scene outputs (+ scene.blend)
  <out>/images|semantic|instance/<id>.png  hard links to the per-view files
  <out>/cameras/<id>.json                K, extrinsics (OpenCV + Blender), GSD, pose
  <out>/manifest.jsonl                   one record per image
  <out>/splits/{train,val,test}.txt      deterministic, split by scene (no leakage)
  <out>/coco/instances_<split>.json      COCO instance annotations (polygons)
  <out>/dataset_info.json                classes, counts, class frequencies, provenance
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image

from synth.common.config import load_json, project_path, save_json
from synth.common.rng import rng


def scene_complete(scene_dir, expected_cfg=None):
    """Rendered, all view images present, and (if given) planned with the
    current configuration, so a config change triggers regeneration."""
    mp = os.path.join(scene_dir, "metadata.json")
    if not os.path.exists(mp):
        return False
    if expected_cfg is not None:
        pp = os.path.join(scene_dir, "plan.json")
        if not os.path.exists(pp) or json.loads(json.dumps(load_json(pp).get("config"))) != json.loads(json.dumps(expected_cfg)):
            return False
    m = load_json(mp)
    if not m.get("views"):
        return False
    return all(os.path.exists(os.path.join(scene_dir, "views", v["name"], f"{v['name']}_rgb.png")) for v in m["views"])


def split_of(seed, splits):
    u = rng(seed, "split").random()
    acc = 0.0
    for name, frac in splits.items():
        acc += frac
        if u < acc:
            return name
    return list(splits)[-1]


def build(dcfg_path, count=None, export_only=False, log=print):
    from generate import generate_scene
    d = load_json(dcfg_path)
    out = project_path(d["output"])
    os.makedirs(out, exist_ok=True)
    save_json(d, os.path.join(out, "dataset_config.json"))
    start, n = d["seeds"]["start"], count or d["seeds"]["count"]
    seeds = list(range(start, start + n))
    fail_path = os.path.join(out, "failures.jsonl")
    from synth.common.config import load_scene_config
    expected = load_scene_config(d["scene_config"], d.get("overrides"))
    if not export_only:
        if os.path.exists(fail_path):
            os.replace(fail_path, fail_path + ".prev")
        for i, seed in enumerate(seeds):
            sdir = os.path.join(out, "scenes", f"scene_{seed:05d}")
            if scene_complete(sdir, expected):
                log(f"[{i + 1}/{n}] seed {seed}: complete, skipped")
                continue
            t0 = time.time()
            try:
                generate_scene(seed, d["scene_config"], output_path=sdir, overrides=d.get("overrides"), log=lambda *a: None)
                log(f"[{i + 1}/{n}] seed {seed}: done in {time.time() - t0:.0f}s")
            except Exception as e:  # isolate: record and continue with the next scene
                with open(fail_path, "a") as f:
                    f.write(json.dumps({"seed": seed, "error": str(e), "trace": traceback.format_exc()[-2000:],
                                        "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                log(f"[{i + 1}/{n}] seed {seed}: FAILED ({e}); continuing")
    export(d, out, seeds, log)


def _link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export(d, out, seeds, log=print):
    from synth.common.config import load_scene_config
    expected_cfg = load_scene_config(d["scene_config"], d.get("overrides"))
    classes = load_json("configs/classes.json")
    cname = {c["id"]: c["name"] for c in classes["classes"]}
    coco_names = d.get("coco_classes", [])
    cats = [{"id": i + 1, "name": nm, "supercategory": "scene"} for i, nm in enumerate(coco_names)]
    cat_of = {nm: c["id"] for nm, c in zip(coco_names, cats)}
    coco = {s: {"images": [], "annotations": [], "categories": cats,
                "info": {"description": d["name"], "version": "1"}} for s in d["splits"]}
    manifest, splits = [], {s: [] for s in d["splits"]}
    class_px = np.zeros(256, np.int64)
    ann_id = 1
    n_scenes = 0
    for seed in seeds:
        sdir = os.path.join(out, "scenes", f"scene_{seed:05d}")
        if not scene_complete(sdir, expected_cfg):
            continue
        n_scenes += 1
        m = load_json(os.path.join(sdir, "metadata.json"))
        labels = {int(k): v for k, v in load_json(os.path.join(sdir, "labels.json")).items()}
        split = split_of(seed, d["splits"])
        for v in m["views"]:
            iid = f"s{seed:05d}_{v['name']}"
            vd = os.path.join(sdir, "views", v["name"])
            files = {"image": f"{v['name']}_rgb.png", "image_raw": f"{v['name']}_render.png",
                     "semantic": f"{v['name']}_semantic.png", "instance": f"{v['name']}_instance.png"}
            for k in ("image", "semantic", "instance"):
                _link(os.path.join(vd, files[k]), os.path.join(out, {"image": "images"}.get(k, k), f"{iid}.png"))
            cam = {"image_id": iid, "width": v["render_resolution"][0], "height": v["render_resolution"][1],
                   "K": _scale_K(v["K"], v["resolution"], v["render_resolution"]),
                   "world_to_cam_opencv": v["world_to_cam_opencv"], "cam_to_world_blender": v["cam_to_world_blender"],
                   "focal_mm": v["focal_mm"], "sensor_width_mm": v["sensor_width_mm"], "tilt_deg": v["tilt_deg"],
                   "heading_deg": v["heading_deg"], "gsd_centre_m": v["gsd_centre_m_at_render_res"],
                   "location_local_m": v["location"], "origin_utm": m["dtm"]["origin_utm"], "epsg": m["dtm"]["epsg"]}
            save_json(cam, os.path.join(out, "cameras", f"{iid}.json"))
            sem = np.asarray(Image.open(os.path.join(vd, files["semantic"])))
            class_px += np.bincount(sem.ravel(), minlength=256)[:256]
            rec = {"id": iid, "seed": seed, "split": split, "view": v["name"],
                   "image": f"images/{iid}.png", "semantic": f"semantic/{iid}.png", "instance": f"instance/{iid}.png",
                   "camera": f"cameras/{iid}.json", "depth_exr": os.path.relpath(os.path.join(vd, f"{v['name']}_passes.exr"), out),
                   "image_raw_16bit": os.path.relpath(os.path.join(vd, files["image_raw"]), out),
                   "tilt_deg": v["tilt_deg"], "gsd_m": v["gsd_centre_m_at_render_res"]["across"],
                   "sun_elevation_deg": m["sun"]["elevation_deg"], "sun_azimuth_deg": m["sun"]["azimuth_deg"],
                   "site_latlon": m["dtm"]["origin_latlon"], "site_rank": m["dtm"].get("site_rank"),
                   "valid_for_dataset": m["valid_for_dataset"],
                   "class_fraction": v["masks"]["class_fraction"]}
            manifest.append(rec)
            splits[split].append(iid)
            # COCO
            img_id = len(coco[split]["images"]) + 1
            coco[split]["images"].append({"id": img_id, "file_name": f"images/{iid}.png", "width": cam["width"],
                                          "height": cam["height"], "seed": seed})
            inst = np.asarray(Image.open(os.path.join(vd, files["instance"]))).astype(np.int64)
            for inst_id, npx in v["masks"]["instance_pixels"].items():
                lab = labels.get(int(inst_id))
                if not lab or lab.get("shared") or npx < d.get("min_instance_pixels", 12):
                    continue
                name = cname.get(lab["class_id"])
                if name not in cat_of:
                    continue
                mask = inst == int(inst_id)
                if not mask.any():
                    continue
                ann = _coco_ann(mask, ann_id, img_id, cat_of[name], lab)
                if ann:
                    coco[split]["annotations"].append(ann)
                    ann_id += 1
    with open(os.path.join(out, "manifest.jsonl"), "w") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")
    for s, ids in splits.items():
        os.makedirs(os.path.join(out, "splits"), exist_ok=True)
        with open(os.path.join(out, "splits", f"{s}.txt"), "w") as f:
            f.write("\n".join(ids) + ("\n" if ids else ""))
        save_json(coco[s], os.path.join(out, "coco", f"instances_{s}.json"))
    tot = class_px.sum() or 1
    failures = []
    fp = os.path.join(out, "failures.jsonl")
    if os.path.exists(fp):
        failures = [json.loads(l)["seed"] for l in open(fp) if l.strip()]
    info = {"name": d["name"], "description": d.get("description"), "scenes": n_scenes, "images": len(manifest),
            "split_images": {s: len(v) for s, v in splits.items()},
            "coco_annotations": {s: len(c["annotations"]) for s, c in coco.items()},
            "classes": classes["classes"], "sky_class": classes["sky_class"],
            "class_pixel_fraction": {cname[k]: float(class_px[k] / tot) for k in cname if class_px[k]},
            "failed_seeds": sorted(set(failures) - {r["seed"] for r in manifest}),
            "config_sha256": hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(),
            "scene_config_sha256": hashlib.sha256(open(project_path(d["scene_config"]), "rb").read()).hexdigest(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_json(info, os.path.join(out, "dataset_info.json"))
    log(f"exported {len(manifest)} images from {n_scenes} scenes -> {out}")
    return info


def _scale_K(K, planned_res, render_res):
    s = render_res[0] / planned_res[0]
    K = np.array(K, float)
    K[:2] *= s
    return K.tolist()


def _coco_ann(mask, ann_id, img_id, cat_id, lab):
    from skimage.measure import find_contours
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    polys = []
    for c in find_contours(np.pad(mask, 1).astype(float), 0.5):
        c = c - 1
        if len(c) < 4:
            continue
        step = max(1, len(c) // 200)
        poly = np.column_stack([c[::step, 1], c[::step, 0]]).ravel().round(1).tolist()
        if len(poly) >= 6:
            polys.append(poly)
    if not polys:
        return None
    return {"id": ann_id, "image_id": img_id, "category_id": cat_id, "segmentation": polys,
            "area": int(mask.sum()), "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)], "iscrowd": 0,
            "kind": lab["kind"], "plan_id": lab["plan_id"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_config")
    ap.add_argument("--count", type=int)
    ap.add_argument("--export-only", action="store_true")
    a = ap.parse_args()
    build(a.dataset_config, a.count, a.export_only)


if __name__ == "__main__":
    sys.exit(main())
