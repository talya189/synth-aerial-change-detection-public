"""Render each terrain layer alone and measure its mean colour (runs inside Blender).

    Blender -b <scene.blend> -P synth/blender/swatches.py -- \
        --layers layers.json --out-dir DIR [--gsd 0.125] [--size 24] [--samples 16]

layers.json: [{"name": "grass", "role": "terrain.grass", "adjust": {...}}, ...]
The scene's sun, sky and colour management are kept; everything else is hidden.
Each layer is a flat 24 m square far above the terrain, seen straight down by an
orthographic camera at aerial GSD. Writes DIR/<name>.png and DIR/swatches.json
(mean sRGB 0-255 of the rendered pixels, i.e. after the view transform).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from synth.blender.asset_manager import AssetManager  # noqa: E402
from synth.blender.terrain_builder import layer_material  # noqa: E402

Z = 3000.0   # above any terrain: no terrain shadows, same sun and sky


def plane(name, size):
    h = size / 2
    me = bpy.data.meshes.new(name)
    me.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name="UVMap")
    for loop in me.loops:   # UV in metres, like the terrain
        co = me.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (co.x, co.y)
    ob = bpy.data.objects.new(name, me)
    ob.location = (0, 0, Z)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--layers", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--gsd", type=float, default=0.125)
    a.add_argument("--size", type=float, default=24.0)
    a.add_argument("--samples", type=int, default=16)
    a.add_argument("--tier", type=int, default=1024)
    a = a.parse_args(sys.argv[sys.argv.index("--") + 1:])
    layers = json.load(open(a.layers))
    os.makedirs(a.out_dir, exist_ok=True)
    sc = bpy.context.scene
    from synth.blender.render import select_device
    sc.cycles.device = select_device()   # Metal / OptiX / CUDA / HIP, else CPU
    for ob in sc.objects:
        if ob.type != "LIGHT":
            ob.hide_render = True
    cd = bpy.data.cameras.new("swatch_cam")
    cd.type, cd.ortho_scale, cd.clip_end = "ORTHO", a.size, 100.0
    cam = bpy.data.objects.new("swatch_cam", cd)
    cam.location = (0, 0, Z + 50)
    sc.collection.objects.link(cam)
    sc.camera = cam
    sc.render.resolution_x = sc.render.resolution_y = int(round(a.size / a.gsd))
    sc.render.resolution_percentage = 100
    sc.cycles.samples = a.samples
    s = sc.render.image_settings
    s.media_type, s.file_format, s.color_depth, s.color_mode = "IMAGE", "PNG", "8", "RGB"
    am = AssetManager(tier=a.tier, missing_policy="error", log=print)
    pl = plane("swatch", a.size)
    res = {}
    for k, L in enumerate(layers):
        pl.data.materials.clear()
        pl.data.materials.append(layer_material(f"swatch_{L['name']}", am, L, seed=0.37 * (k + 1)))
        bpy.ops.render.render()
        path = os.path.join(a.out_dir, f"{L['name']}.png")
        bpy.data.images["Render Result"].save_render(path, scene=sc)
        img = bpy.data.images.load(path)
        px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, img.channels)[:, :3]
        bpy.data.images.remove(img)
        res[L["name"]] = {"mean_srgb": [round(float(v) * 255, 1) for v in px.mean(0)],
                          "std_srgb": [round(float(v) * 255, 1) for v in px.std(0)]}
        print(f"[swatch] {L['name']}: mean sRGB {res[L['name']]['mean_srgb']}")
    with open(os.path.join(a.out_dir, "swatches.json"), "w") as f:
        json.dump(res, f, indent=1)


main()
