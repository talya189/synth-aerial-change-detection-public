"""Change scene logic parameters in a generated .blend and render (runs inside Blender).

    Blender -b <scene.blend> -P synth/blender/set_params.py -- \
        --set '{"scatter_dead_shrub": {"Density": 0.1}, "scatter_basalt": {"Density": 0}}' \
        --camera cam_low_oblique --res 768 --samples 32 --out out.png [--save edited.blend]

Every scatter object's modifier inputs are addressed by object name and input
name, exactly as shown in Blender's modifier panel.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import bpy  # noqa: E402

from synth.blender.scatter_nodes import set_inputs  # noqa: E402


def apply(changes):
    for obj_name, values in changes.items():
        ob = bpy.data.objects[obj_name]
        md = next(m for m in ob.modifiers if m.type == "NODES")
        set_inputs(md, {k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items()})


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="{}")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save")
    ap.add_argument("--look-at", help="x,y,z target for a temporary camera")
    ap.add_argument("--range", type=float, default=70.0)
    ap.add_argument("--tilt", type=float, default=50.0)
    ap.add_argument("--heading", type=float, default=30.0)
    a = ap.parse_args(argv)
    apply(json.loads(a.set))
    sc = bpy.context.scene
    from synth.blender.render import select_device
    sc.cycles.device = select_device()   # Metal / OptiX / CUDA / HIP, else CPU
    if a.camera:
        sc.camera = bpy.data.objects[a.camera]
    if a.look_at:
        from synth.common.camera_math import camera_pose
        pos, eul = camera_pose([float(v) for v in a.look_at.split(",")], a.range, a.tilt, a.heading)
        cd = bpy.data.cameras.new("look")
        cd.lens = 50
        cam = bpy.data.objects.new("look", cd)
        sc.collection.objects.link(cam)
        cam.location, cam.rotation_euler = tuple(pos), eul
        sc.camera = cam
    sc.render.resolution_x = sc.render.resolution_y = a.res
    sc.cycles.samples = a.samples
    t = time.time()
    bpy.ops.render.render()
    s = sc.render.image_settings
    s.media_type, s.file_format, s.color_depth = "IMAGE", "PNG", "8"
    bpy.data.images["Render Result"].save_render(a.out, scene=sc)
    print(f"[set_params] rendered {a.out} in {time.time() - t:.1f}s")
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=a.save, relative_remap=True)


main()
