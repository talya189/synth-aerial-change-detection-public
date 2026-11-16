"""Procedural trees and shrubs for species / roles without a scanned asset (runs inside Blender).

Built from leaf clusters (synth/blender/foliage.py), like real crowns and the
scanned pine: branches inside, clumps of small leaves on the crown shell, gaps
and self-shadowing. Smooth blob crowns read as plastic from the air.

  proc.shrub_cushion   low cushion shrubs (thorny burnet), ~1 m
  proc.shrub_large     bushes (Ziziphus, Atlantic pistachio), ~1.8 m before scaling
  tree species         olive, oak, carob, cypress, Ziziphus, citrus (configs/vegetation.json)

Leaf materials honour the scatter tint (INSTANCER attribute `tint_off`), so
Live Fraction / Live Tint still work on shrubs.
"""
import math
import zlib

import bpy
import numpy as np

from synth.blender import material_factory as mf

VARIANTS = 5


SPECS = {   # role: (shape, height m, width m, leaf colour, shadowed leaf colour, leaf length, leaf width)
    "proc.shrub_cushion": ("cushion", 0.6, 1.0, (0.10, 0.12, 0.06), (0.035, 0.045, 0.022), 0.03, 0.012),
    "proc.shrub_large": ("bush", 1.6, 1.8, (0.06, 0.11, 0.04), (0.018, 0.035, 0.014), 0.05, 0.03),
}


def leaf_material(name, light, dark):
    """Leaves: per-leaf random mix (face attribute leaf_rand) of the light and the
    shadowed leaf colour, x object colour x scatter tint, with a little translucency."""
    mat = mf.new_material(name)
    nb = mf.NodeBuilder(mat)
    lr = nb.node("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="leaf_rand").outputs["Fac"]
    col = nb.mix(lr, (*dark, 1.0), (*light, 1.0))
    col = nb.mix(1.0, col, nb.node("ShaderNodeObjectInfo").outputs["Color"], "RGBA", "MULTIPLY")
    toff = nb.node("ShaderNodeAttribute", attribute_type="INSTANCER", attribute_name="tint_off")
    tint = nb.node("ShaderNodeVectorMath", operation="ADD")
    nb.link(toff.outputs["Vector"], tint.inputs[0])
    tint.inputs[1].default_value = (1.0, 1.0, 1.0)
    col = nb.mix(1.0, col, tint.outputs[0], "RGBA", "MULTIPLY")
    bsdf = mf.principled(nb, col, 0.55, None, specular=0.35)
    tr = nb.node("ShaderNodeBsdfTranslucent")
    nb.link(col, tr.inputs["Color"])
    mix = nb.node("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.12
    nb.link(bsdf.outputs[0], mix.inputs[1])
    nb.link(tr.outputs[0], mix.inputs[2])
    mf.output(nb, mix.outputs[0])
    return mat


def plant_mesh(name, geo):
    """Mesh from foliage.build_plant output (verts, tris, material index, leaf_rand per face)."""
    from synth.blender.meshutil import mesh_from_arrays
    verts, tris, mat, lr = geo
    me = mesh_from_arrays(name, verts, tris, mat_idx=mat, smooth=False)
    at = me.attributes.new("leaf_rand", "FLOAT", "FACE")
    at.data.foreach_set("value", lr)
    return me


def prototypes(role, library):
    """Create (or reuse) the shrub prototypes of a procedural role (leaf-cluster shrubs)."""
    from synth.blender.foliage import build_plant
    shape, h, w, light, dark, ll, lw = SPECS[role]
    mat = bpy.data.materials.get(f"M_{role}") or leaf_material(f"M_{role}", light, dark)
    bark = bpy.data.materials.get("M_bark_shrub") or mf.dirty_flat_material("M_bark_shrub", (0.12, 0.1, 0.08), 0.85, dirt=0.3)
    out = []
    for k in range(VARIANTS):
        name = f"PROTO_{role}_{k}"
        ob = bpy.data.objects.get(name)
        if ob is None:
            geo = build_plant(shape, h * (0.85 + 0.08 * k), w * (0.9 + 0.06 * k), 0.0, ll, lw, seed=500 + k,
                              density=1.0, leaves_per_cluster=10, shrub=True)
            ob = bpy.data.objects.new(name, plant_mesh(name, geo))
            ob.data.materials.append(mat)
            ob.data.materials.append(bark)
            library.objects.link(ob)
        ob.color = (1.0 + 0.12 * (k - 2) / 2,) * 3 + (1.0,)
        out.append(ob)
    return out


# ------------------------------------------------------------- procedural trees
# Prototypes are built at unit size (height 1, crown width 1) and scaled per tree
# to (crown, crown, height). Crown shapes: round (olive, carob, citrus), broad
# (oak), column (cypress), irregular (Ziziphus). Slot 0 = leaves, 1 = bark.

def tree_prototypes(species, sp, library):
    """Five variants of a procedural species at its typical size (mean height and
    crown width), built from leaf clusters; trees scale them modestly."""
    from synth.blender.foliage import build_plant
    leaf = bpy.data.materials.get(f"M_tree_{species}") or leaf_material(f"M_tree_{species}", tuple(sp["leaf"]), tuple(sp["leaf_dark"]))
    bark = bpy.data.materials.get(f"M_bark_{species}") or mf.dirty_flat_material(f"M_bark_{species}", tuple(sp["bark"]), 0.85, dirt=0.3)
    h, w = sum(sp["height_m"]) / 2, sum(sp["crown_m"]) / 2
    lf = sp.get("leaf_size_m", [0.07, 0.03])
    out = []
    for k in range(VARIANTS):
        name = f"PROTO_tree_{species}_{k}"
        ob = bpy.data.objects.get(name)
        if ob is None:
            geo = build_plant(sp["shape"], h, w, sp["trunk_frac"], lf[0], lf[1], seed=1000 + 37 * k + len(species),
                              density=sp.get("foliage_density", 0.6), leaves_per_cluster=12, gnarled=species == "olive")
            ob = bpy.data.objects.new(name, plant_mesh(name, geo))
            ob.data.materials.append(leaf)
            ob.data.materials.append(bark)
            ob["unit_h"], ob["unit_w"] = h, w
            library.objects.link(ob)
        out.append(ob)
    return out


def build_proc_trees(plan, col, ids, library):
    """One linked-duplicate object per procedural tree (grove trees share their grove's instance id)."""
    from synth.common.config import load_json
    species = load_json("configs/vegetation.json")["species"]
    protos = {}
    n = 0
    for t in plan["vegetation"]:
        if t["type"] != "proc_tree":
            continue
        sp = species[t["species"]]
        if t["species"] not in protos:
            protos[t["species"]] = tree_prototypes(t["species"], sp, library)
        proto = protos[t["species"]][t["variant"] % VARIANTS]
        ob = bpy.data.objects.new(t["id"], proto.data)
        ob.location = (t["x"], t["y"], t["z"])
        ob.rotation_euler = (0.0, 0.0, t["yaw"])
        ob.scale = (t["crown"] / proto["unit_w"], t["crown"] / proto["unit_w"], t["height"] / proto["unit_h"])
        v = 0.88 + 0.24 * ((zlib.crc32(t["id"].encode()) % 1000) / 1000.0)   # per-tree brightness variation (stable)
        ob.color = (v, v * 1.02, v * 0.97, 1.0)
        ob.pass_index = ids.new(t["kind"], t.get("label_id", t["id"]))
        ob["kind"] = t["kind"]
        col.objects.link(ob)
        n += 1
    return n
