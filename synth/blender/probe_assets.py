"""Measure every model variant's real dimensions (runs inside Blender).

    Blender -b --factory-startup -P synth/blender/probe_assets.py

Writes cache/asset_dims.json: {model_path: [dx, dy, dz]} in metres, used by
the planner (outside Blender) for burial depth and spacing.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import bpy  # noqa: E402

from synth.blender.asset_manager import AssetManager  # noqa: E402
from synth.common.config import project_path  # noqa: E402

SKIP_KINDS = ("surface", "atlas", "decal")


def main():
    out_path = project_path("cache/asset_dims.json")
    dims = json.load(open(out_path)) if os.path.exists(out_path) else {}
    am = AssetManager(missing_policy="skip")
    for stem, a in am.reg.items():
        if a.get("status") != "ok" or a.get("kind") in SKIP_KINDS or not a.get("models"):
            continue
        if any(r.startswith("kit.") for r in a.get("roles", [])) or any(m.lower().endswith(".usd") for m in a["models"]):
            continue  # kits are not scattered as single models; USD trees are measured below
        for m in am.variants(stem, lod=0):
            if m in dims:
                continue
            try:
                p = am.prototype(stem, m)
                dims[m] = [round(v, 4) for v in p["dims_m"]]
                print("DIMS", m, dims[m])
            except Exception as e:  # report, never silently substitute
                print("PROBE-ERROR", m, e)
    # USD trees: measured from their evaluated (instanced) geometry, one clean scene each
    for stem, a in am.reg.items():
        if a.get("status") != "ok":
            continue
        for m in a.get("models", []):
            if not m.lower().endswith(".usd") or "Branches" in m or m in dims:
                continue
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.wm.usd_import(filepath=project_path(m))
            dims[m] = _instanced_dims()
            print("DIMS", m, dims[m])
    with open(out_path, "w") as f:
        json.dump(dims, f, indent=1)


def _instanced_dims():
    from mathutils import Vector
    dg = bpy.context.evaluated_depsgraph_get()
    lo, hi = [1e9] * 3, [-1e9] * 3
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH":
            continue
        for c in ob.bound_box:
            p = inst.matrix_world @ Vector(c)
            lo = [min(lo[i], p[i]) for i in range(3)]
            hi = [max(hi[i], p[i]) for i in range(3)]
    return [round(hi[i] - lo[i], 3) for i in range(3)] + [round(lo[2], 3)]


main()
