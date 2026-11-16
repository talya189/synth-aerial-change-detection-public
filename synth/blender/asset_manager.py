"""Asset registry access and model import/caching (runs inside Blender).

Generators and the planner refer to *roles* ("veg.dead_shrub"); this module
resolves roles to registered assets, imports each model file once into a
hidden `_ASSET_LIBRARY` collection, and hands out linked duplicates
(new Object, shared mesh datablock) for placement.
"""
import os
import re

import bpy
import mathutils

from synth.blender import material_factory as mf
from synth.common.config import load_json, project_path


class MissingAssetError(RuntimeError):
    pass


class AssetManager:
    def __init__(self, registry_path="configs/assets.json", tier=1024, missing_policy="error", log=print):
        self.reg = load_json(registry_path)["assets"]
        self.tier = tier
        self.missing_policy = missing_policy
        self.log = log
        self.used = {}           # stem -> version, for metadata
        self.missing = []        # roles that could not be resolved
        self._protos = {}        # (stem, model_path) -> prototype object
        self._mats = {}
        self.library = bpy.data.collections.get("_ASSET_LIBRARY") or bpy.data.collections.new("_ASSET_LIBRARY")
        if self.library.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(self.library)
        self.library.hide_render = True
        self.library.hide_viewport = True

    # ---------------------------------------------------------------- roles
    def assets_for_role(self, role):
        return [k for k, a in self.reg.items() if a.get("status") == "ok" and role in a.get("roles", [])]

    def resolve(self, role, required=True):
        stems = self.assets_for_role(role)
        if not stems:
            self.missing.append(role)
            msg = f"no usable asset for role '{role}'"
            if required and self.missing_policy == "error":
                raise MissingAssetError(msg)
            self.log("MISSING " + msg)
            return None
        stem = sorted(stems)[0]
        self.used[stem] = self.reg[stem].get("version")
        return stem

    def asset(self, stem):
        return self.reg[stem]

    # ------------------------------------------------------------ materials
    def surface_material(self, role, **kw):
        stem = self.resolve(role)
        key = ("surface", stem, tuple(sorted(kw.items())))
        if key not in self._mats:
            a = dict(self.reg[stem])
            a["height_m"] = _height_m(a)
            self._mats[key] = mf.surface_material(f"S_{role}", a, self.tier, **kw)
        return self._mats[key]

    def model_material(self, stem, **kw):
        key = ("model", stem, tuple(sorted(kw.items())))
        if key not in self._mats:
            self._mats[key] = mf.model_material(f"M_{stem}", self.reg[stem], max(self.tier, 2048), **kw)
        return self._mats[key]

    # --------------------------------------------------------------- models
    def variants(self, stem, lod=0):
        """Model files of an asset; for Megascans plants, one per Var at the given LOD."""
        models = self.reg[stem]["models"]
        lods = [m for m in models if re.search(rf"_LOD{lod}\.", m)]
        return lods or [m for m in models if "LOD" not in os.path.basename(m)] or models

    def prototype(self, stem, model_path, foliage=False, alpha=False, value_jitter=0.0):
        key = (stem, model_path)
        if key in self._protos:
            return self._protos[key]
        name = f"PROTO_{stem}_{os.path.splitext(os.path.basename(model_path))[0][-12:]}"
        existing = bpy.data.objects.get(name)
        if existing is not None:  # already in the template
            self._protos[key] = existing
            self.used[stem] = self.reg[stem].get("version")
            return existing
        path = project_path(model_path)
        before = set(bpy.data.objects)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path)
        elif ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=path)
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=path)
        elif ext.startswith(".usd"):
            bpy.ops.wm.usd_import(filepath=path)
        else:
            raise ValueError(f"unsupported model format {ext}")
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH"]
        if not meshes:
            raise RuntimeError(f"no mesh in {path}")
        proto = _join_and_normalise(meshes, [o for o in new if o.type != "MESH"])
        proto.name = name
        mat = self.model_material(stem, foliage=foliage, alpha=alpha, value_jitter=value_jitter)
        proto.data.materials.clear()
        proto.data.materials.append(mat)
        for c in list(proto.users_collection):
            c.objects.unlink(proto)
        self.library.objects.link(proto)
        self._protos[key] = proto
        self.used[stem] = self.reg[stem].get("version")
        return proto

    def instance(self, proto, name, collection, location, rotation=(0, 0, 0), scale=1.0, pass_index=0, props=None):
        ob = bpy.data.objects.new(name, proto.data)  # linked duplicate: shared mesh
        ob.location = location
        ob.rotation_euler = rotation
        ob.scale = (scale, scale, scale) if isinstance(scale, (int, float)) else scale
        ob.pass_index = pass_index
        for k, v in (props or {}).items():
            if k == "tint":
                ob.color = (*v, 1.0)
            else:
                ob[k] = v
        collection.objects.link(ob)
        return ob


def _height_m(asset):
    """Physical displacement range of a scanned surface, used as bump distance."""
    return float(asset.get("height_m") or 0.03)


def _join_and_normalise(meshes, others):
    """Bake import transforms (FBX centimetre scale) into one mesh with its
    origin at the bottom centre, so placement uses true metres."""
    for o in meshes:
        o.data = o.data.copy() if o.data.users > 1 else o.data
        o.data.transform(o.matrix_world)
        o.matrix_world = mathutils.Matrix.Identity(4)
    base = meshes[0]
    if len(meshes) > 1:
        with bpy.context.temp_override(active_object=base, selected_editable_objects=meshes, object=base):
            bpy.ops.object.join()
    for o in others:
        bpy.data.objects.remove(o, do_unlink=True)
    vs = base.data.vertices
    xs = [v.co.x for v in vs]
    ys = [v.co.y for v in vs]
    zs = [v.co.z for v in vs]
    off = mathutils.Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, min(zs)))
    base.data.transform(mathutils.Matrix.Translation(-off))
    base["dims_m"] = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]
    return base
