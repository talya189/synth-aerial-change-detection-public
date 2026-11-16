"""Step 3: ground materials (same Korazim site as Step 2).

    ~/synth_aerial_work/venv/bin/python -m synth.steps.step3 [--calibrate] [--spec configs/specs/step3_ground.json]

--calibrate renders every terrain layer alone (synth/blender/swatches.py), compares
its mean colour with the layer's `target_srgb` and corrects the layer's `adjust`
(hue / saturation / value), a few rounds; the result is written back into the spec.

Output: ~/synth_aerial_work/outputs/steps/step3/
  swatches.png          each layer alone: render | measured colour | target colour
  wetness_overlay.png   where the wadi-bed layer goes (drainage wetness)
  closeups.png          before (Step 2 materials) | after, at aerial GSDs
  spec.json, plan.json, scene.blend, views/, contact_sheet.jpg
"""
import argparse
import colorsys
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from generate import BLENDER, PROJECT_ROOT, blender_build
from synth.common.config import load_json, project_path
from synth.spec.scene_spec import load, preset_path, resolved, save
from synth.spec.to_build import build_data

# close-up views: (name, look-at, range m, tilt deg, heading deg); resolution 1024, f 50, sensor 36
#   nadir GSD = range * 36 / (50 * 1024)
CLOSEUPS = [("plateau_nadir_0.18m", "73,-19,7.1", 260, 0, 30),
            ("plateau_oblique", "73,-19,7.1", 260, 50, 30)]


def _lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def update_adjust(adj, measured, target):
    """One calibration step of a layer's Hue/Saturation/Value correction.
    Works in decoded (linear-ish) colour; the view transform is monotone, so a
    few rounds converge."""
    hm, sm, vm = colorsys.rgb_to_hsv(*_lin(measured))
    ht, st, vt = colorsys.rgb_to_hsv(*_lin(target))
    dh = (ht - hm + 0.5) % 1.0 - 0.5
    adj = dict(adj or {"hue": 0.5, "saturation": 1.0, "value": 1.0})
    adj["hue"] = round(float(np.clip(adj.get("hue", 0.5) + dh, 0.0, 1.0)), 4)
    adj["saturation"] = round(float(np.clip(adj.get("saturation", 1.0) * st / max(sm, 1e-3), 0.05, 3.0)), 4)
    adj["value"] = round(float(np.clip(adj.get("value", 1.0) * vt / max(vm, 1e-3), 0.1, 8.0)), 4)
    return adj


def layer_list(tm):
    return [dict(name=tm["base"].get("name", "base"), role=tm["base"]["role"], adjust=tm["base"].get("adjust"),
                 target_srgb=tm["base"].get("target_srgb"))] + \
           [dict(name=L["name"], role=L["role"], adjust=L.get("adjust"), target_srgb=L.get("target_srgb"))
            for L in tm["layers"]]


def render_swatches(blend, layers, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lj = os.path.join(out_dir, "layers.json")
    with open(lj, "w") as f:
        json.dump(layers, f)
    cmd = [BLENDER, "-b", blend, "--python-exit-code", "1", "-P",
           os.path.join(PROJECT_ROOT, "synth", "blender", "swatches.py"), "--", "--layers", lj, "--out-dir", out_dir]
    p = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout[-3000:] + p.stderr[-3000:])
    return load_json(os.path.join(out_dir, "swatches.json"))


def calibrate(spec, spec_path, blend, out_dir, rounds=4, log=print):
    tm = resolved(spec)["terrain_material"]
    for r in range(rounds):
        layers = layer_list(tm)
        meas = render_swatches(blend, layers, os.path.join(out_dir, "swatches"))
        err = {}
        for L, cfg in zip(layers, [tm["base"]] + tm["layers"]):
            if not L["target_srgb"]:
                continue
            m = meas[L["name"]]["mean_srgb"]
            err[L["name"]] = round(float(np.abs(np.subtract(m, L["target_srgb"])).max()), 1)
            cfg["adjust"] = update_adjust(cfg.get("adjust"), m, L["target_srgb"])
        log(f"calibration round {r + 1}: max channel error {err}")
        if max(err.values()) < 4:
            break
    # the last update was not measured: measure the final values for the report
    if isinstance(spec["terrain_material"], str):   # a shared preset: write the preset file
        p = project_path(preset_path("terrain_material", spec["terrain_material"]))
        with open(p, "w", encoding="utf-8") as f:
            json.dump(tm, f, indent=2, ensure_ascii=False)
    else:
        orig = load(spec_path)   # write back only the material (not values resolved during the build)
        orig["terrain_material"] = tm
        save(orig, spec_path)
    return render_swatches(blend, layer_list(tm), os.path.join(out_dir, "swatches"))


def swatch_board(layers, meas, sw_dir, out_png, tile=180):
    W = tile * 2 + 20
    g = Image.new("RGB", (W * 4, (tile + 50) * ((len(layers) + 3) // 4)), (255, 255, 255))
    dr = ImageDraw.Draw(g)
    for k, L in enumerate(layers):
        x0, y0 = (k % 4) * W, (k // 4) * (tile + 50)
        g.paste(Image.open(os.path.join(sw_dir, f"{L['name']}.png")).convert("RGB").resize((tile, tile)), (x0, y0 + 30))
        m = tuple(int(v) for v in meas[L["name"]]["mean_srgb"])
        dr.rectangle([x0 + tile, y0 + 30, x0 + tile + tile // 2, y0 + 30 + tile], fill=m)
        if L["target_srgb"]:
            dr.rectangle([x0 + tile + tile // 2, y0 + 30, x0 + 2 * tile, y0 + 30 + tile], fill=tuple(L["target_srgb"]))
        dr.text((x0 + 4, y0 + 4), f"{L['name']} ({L['role']})", fill=(0, 0, 0))
        dr.text((x0 + 4, y0 + 16), f"measured {m}  target {tuple(L['target_srgb'] or ())}", fill=(0, 0, 0))
    g.save(out_png)


def wetness_overlay(out_dir, wet_range, out_png):
    from synth.steps.step2 import hillshade
    d = np.load(f"{out_dir}/terrain.npz")
    hs = hillshade(d["core_z"].astype(float), 1.0)
    w = np.clip((d["wet"] - wet_range[0]) / (wet_range[1] - wet_range[0]), 0, 1)
    rgb = np.stack([hs * (1 - w) + 0.1 * w, hs * (1 - w) + 0.45 * w, hs * (1 - w) + 0.95 * w], -1)
    Image.fromarray((rgb[::-1] * 255).astype(np.uint8)).resize((700, 700)).save(out_png)
    return float((w > 0.5).mean())


def closeups(blend, out_dir, tag):
    paths = []
    for name, tgt, rng_m, tilt, hd in CLOSEUPS:
        out = os.path.join(out_dir, "closeups", f"{name}_{tag}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cmd = [BLENDER, "-b", blend, "--python-exit-code", "1", "-P", os.path.join(PROJECT_ROOT, "synth", "blender", "set_params.py"),
               "--", "--look-at", tgt, "--range", str(rng_m), "--tilt", str(tilt), "--heading", str(hd),
               "--res", "1024", "--samples", "32", "--out", out]
        p = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
        paths.append(out)
    return paths


def side_by_side(before, after, out_png, w=640):
    rows = []
    for b, a in zip(before, after):
        ib, ia = (Image.open(p).convert("RGB").resize((w, w)) for p in (b, a))
        r = Image.new("RGB", (2 * w + 10, w + 24), (255, 255, 255))
        r.paste(ib, (0, 24)), r.paste(ia, (w + 10, 24))
        dr = ImageDraw.Draw(r)
        dr.text((6, 6), "before: " + os.path.basename(b), fill=(0, 0, 0))
        dr.text((w + 16, 6), "after: " + os.path.basename(a), fill=(0, 0, 0))
        rows.append(r)
    g = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for r in rows:
        g.paste(r, (0, y))
        y += r.height
    g.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/specs/step3_ground.json")
    ap.add_argument("--out", default="outputs/steps/step3")
    ap.add_argument("--before-blend", default="outputs/steps/step2/scene.blend")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    spec_path = project_path(a.spec)
    spec = load(spec_path)
    out_dir = project_path(a.out)
    blend = os.path.join(out_dir, "scene.blend")
    if a.calibrate:
        blender_build(build_data(spec, out_dir), render=False, log=print)
        calibrate(spec, spec_path, blend, out_dir)
    plan = build_data(spec, out_dir)
    blender_build(plan, log=print)
    spec = load(spec_path)
    tm = resolved(spec)["terrain_material"]
    layers = layer_list(tm)
    meas = render_swatches(blend, layers, os.path.join(out_dir, "swatches"))
    swatch_board(layers, meas, os.path.join(out_dir, "swatches"), os.path.join(out_dir, "swatches.png"))
    wadi = next((L for L in tm["layers"] if "wet" in L), None)
    if wadi:
        print(f"wadi-bed layer covers {100 * wetness_overlay(out_dir, wadi['wet'], os.path.join(out_dir, 'wetness_overlay.png')):.1f}% of the core")
    after = closeups(blend, out_dir, "after")
    before_blend = project_path(a.before_blend)
    if os.path.exists(before_blend):
        before = closeups(before_blend, out_dir, "before")
        side_by_side(before, after, os.path.join(out_dir, "closeups.png"))
    print(f"step 3 done in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
