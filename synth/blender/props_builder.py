"""Street furniture (runs inside Blender): streetlights.

One prototype mesh per (type, height bucket), placed as linked duplicates.
Local frame: pole at the origin, arm along +x (over the road), z up from the
ground. Each prop carries its own instance id (kind prop.*).
"""
import math

import bpy
import numpy as np

from synth.blender import material_factory as mf
from synth.blender.building_builder import MeshAcc, _cyl
from synth.blender.meshutil import new_object

POLE, LAMP, GLASS = range(3)


def _arm(acc, x0, x1, z, rise, r=0.045, seg=10):
    """Gently rising arm from (x0, z) to (x1, z + rise), as a chain of short cylinders."""
    for k in range(seg):
        a, b = x0 + (x1 - x0) * k / seg, x0 + (x1 - x0) * (k + 1) / seg
        za = z + rise * math.sin(0.5 * math.pi * k / seg)
        zb = z + rise * math.sin(0.5 * math.pi * (k + 1) / seg)
        acc.box(min(a, b), -r, min(za, zb) - r, max(a, b), r, max(za, zb) + r, POLE)


def _head(acc, x, z):
    """LED luminaire: flat box, glass underneath."""
    acc.box(x - 0.35, -0.16, z - 0.07, x + 0.35, 0.16, z + 0.05, LAMP)
    acc.poly([(x - 0.3, -0.12, z - 0.071), (x - 0.3, 0.12, z - 0.071), (x + 0.3, 0.12, z - 0.071), (x + 0.3, -0.12, z - 0.071)],
             [(0, 0), (0, 1), (1, 1), (1, 0)], GLASS)


def streetlight_mesh(name, height, arm, double):
    acc = MeshAcc()
    acc.box(-0.18, -0.18, -0.5, 0.18, 0.18, 0.35, POLE)                  # concrete footing / base plate
    _cyl(acc, 0, 0, 0.35, 0.085, height * 0.55, POLE, seg=12)            # tapered pole: two sections
    _cyl(acc, 0, 0, 0.35 + height * 0.55, 0.06, height * 0.45 - 0.35, POLE, seg=12)
    sides = (1, -1) if double else (1,)
    for sgn in sides:
        _arm(acc, 0, sgn * arm, height - 0.25, 0.25)
        _head(acc, sgn * arm, height + 0.02)
    return acc.mesh(name)


def materials():
    pole = mf.dirty_flat_material("M_galvanised", (0.55, 0.56, 0.55), 0.45, metallic=0.85, dirt=0.25, noise_scale=3)
    lamp = mf.dirty_flat_material("M_lamp_housing", (0.32, 0.33, 0.34), 0.4, metallic=0.6, dirt=0.15)
    glass = mf.flat_material("M_lamp_glass", (0.75, 0.75, 0.72), 0.15, specular=0.5)
    return pole, lamp, glass


def build_props(plan, col, ids):
    props = plan.get("props", [])
    if not props:
        return []
    mats = materials()
    protos = {}
    out = []
    for p in props:
        if not p["type"].startswith("streetlight"):
            continue
        h = round(p["height"] * 2) / 2   # prototypes per half metre of height
        key = (p["type"], h)
        if key not in protos:
            me = streetlight_mesh(f"proto_{p['type']}_{h}", h, p["arm"], p["type"] == "streetlight_double")
            for m in mats:
                me.materials.append(m)
            protos[key] = me
        ob = new_object(p["id"], protos[key], col, ids.new(p["kind"], p["id"]), {"kind": p["kind"], "type": p["type"]})
        ob.location = (p["x"], p["y"], p["z"] + (0.2 if p["type"] == "streetlight_double" else 0.0))
        ob.rotation_euler = (0, 0, p["yaw"])
        out.append(ob)
    return out
