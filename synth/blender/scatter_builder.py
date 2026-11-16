"""Vegetation and rocks from the plan: Megascans models as linked duplicates,
and date palms assembled from the trunk model + frond atlas (runs inside Blender)."""
import math
import os

import bpy
import numpy as np

from synth.blender import material_factory as mf
from synth.blender.meshutil import mesh_from_arrays, new_object
from synth.common.rng import rng


def _stem_of(path):
    parts = path.replace("\\", "/").split("/")
    return parts[parts.index("assets") + 1]


def build_scatter(plan, am, col_veg, col_rock, ids):
    n = 0
    for item in plan["vegetation"] + plan["rocks"]:
        if item["type"] != "model":
            continue
        stem = _stem_of(item["variant"])
        is_veg = item["role"].startswith("veg.")
        proto = am.prototype(stem, item["variant"], foliage=is_veg, alpha=is_veg, value_jitter=0.12 if is_veg else 0.08)
        col = col_veg if is_veg else col_rock
        am.instance(proto, item["id"], col, (item["x"], item["y"], item["z"]), tuple(item["rot"]), item["scale"],
                    ids.new(item.get("kind", item["role"]), item.get("label_id", item["id"])),
                    {"kind": item.get("kind", item["role"]), **({"tint": item["tint"]} if "tint" in item else {})})
        n += 1
    palms = [v for v in plan["vegetation"] if v["type"] == "date_palm"]
    for p in palms:
        build_date_palm(p, am, col_veg, ids)
    return n, len(palms)


# ------------------------------------------------------------------ date palm
_FROND_MATS = {}


def _frond_materials(am, role):
    if not _FROND_MATS:
        stem = am.resolve(role)
        _FROND_MATS["green"] = am.model_material(stem, foliage=True, alpha=True, uv_name="UVMap", value_jitter=0.1)
        dead = mf.model_material("M_frond_dead", am.asset(stem), 2048, alpha=True, foliage=True, uv_name="UVMap",
                                 tint=(0.95, 0.7, 0.45))
        _FROND_MATS["dead"] = dead
    return _FROND_MATS


# frond card inside the 1x1 atlas (base at the bottom, tip at the top)
ATLAS_U = (0.20, 0.86)
ATLAS_V = (0.03, 0.97)


def _frond(verts, faces, uvs, mats, base, azimuth, elev, droop, length, width, fold, mat, seg=9):
    f_h = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
    lat = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    p = np.array(base, float)
    ds = length / seg
    rows = []
    for k in range(seg + 1):
        t = k / seg
        ang = elev - droop * t * t
        d = f_h * math.cos(ang) + np.array([0, 0, 1.0]) * math.sin(ang)
        up = np.cross(lat, d)
        up /= np.linalg.norm(up) + 1e-9
        hw = width / 2
        rows.append([p - lat * hw + up * fold * hw, p.copy(), p + lat * hw + up * fold * hw])
        p = p + d * ds
    b = len(verts)
    for k, row in enumerate(rows):
        verts.extend(row)
    us = (ATLAS_U[0], (ATLAS_U[0] + ATLAS_U[1]) / 2, ATLAS_U[1])
    for k in range(seg):
        for j in range(2):
            a = b + k * 3 + j
            faces.append([a, a + 1, a + 4, a + 3])
            v0 = ATLAS_V[0] + (ATLAS_V[1] - ATLAS_V[0]) * k / seg
            v1 = ATLAS_V[0] + (ATLAS_V[1] - ATLAS_V[0]) * (k + 1) / seg
            uvs.extend([(us[j], v0), (us[j + 1], v0), (us[j + 1], v1), (us[j], v1)])
            mats.append(mat)


def build_date_palm(p, am, col, ids):
    r = np.random.default_rng(p["seed"])
    iid = ids.new(p.get("kind", "veg.palm.date"), p.get("label_id", p["id"]))
    H = p["height"]
    # trunk: stacked scanned trunk segments (shared mesh), slightly narrowed, random yaw
    stem = am.resolve(p["trunk_role"])
    model = am.variants(stem)[0]
    trunk = am.prototype(stem, model, value_jitter=0.05)
    seg_h0 = trunk["dims_m"][2]
    nseg = max(2, int(round(H / 2.1)))
    sz = H / (nseg * seg_h0)
    sxy = float(r.uniform(0.45, 0.55))
    lean = math.radians(p["lean_deg"])
    ld = np.array([math.cos(p["lean_dir"]), math.sin(p["lean_dir"])])
    base = np.array([p["x"], p["y"], p["z"]])
    for k in range(nseg):
        z = k * seg_h0 * sz
        off = ld * math.tan(lean) * z
        am.instance(trunk, f"{p['id']}_trunk{k}", col, (base[0] + off[0], base[1] + off[1], base[2] + z),
                    (lean * ld[1] * -1, lean * ld[0], float(r.uniform(0, 2 * math.pi))), (sxy, sxy, sz), iid,
                    {"kind": "veg.palm.date"})
    top = base + np.array([*(ld * math.tan(lean) * H), H])
    # crown
    mats = _frond_materials(am, p["role"])
    verts, faces, uvs, mi = [], [], [], []
    nf = p["fronds"]
    L0 = float(np.clip(H * 0.42, 2.8, 4.6))
    golden = math.radians(137.5)
    az0 = p["yaw"]
    for i in range(nf):
        t = i / max(nf - 1, 1)  # 0 = youngest (top), 1 = oldest
        az = az0 + i * golden + r.normal(0, 0.15)
        if t < 0.2:
            elev, droop = math.radians(r.uniform(55, 75)), r.uniform(0.2, 0.5)
        elif t < 0.75:
            elev, droop = math.radians(r.uniform(10, 45)), r.uniform(0.6, 1.1)
        else:
            elev, droop = math.radians(r.uniform(-35, -5)), r.uniform(0.3, 0.7)
        L = L0 * r.uniform(0.8, 1.05) * (0.8 if t < 0.2 else 1.0)
        b0 = top + np.array([0.18 * math.cos(az), 0.18 * math.sin(az), -0.6 * t])
        _frond(verts, faces, uvs, mi, b0, az, elev, droop, L, L * 0.62, float(r.uniform(0.25, 0.45)), 0)
    for i in range(int(r.integers(2, 6))):  # dead fronds hanging under the crown
        az = az0 + r.uniform(0, 2 * math.pi)
        b0 = top + np.array([0.15 * math.cos(az), 0.15 * math.sin(az), -0.8 - r.uniform(0, 0.4)])
        _frond(verts, faces, uvs, mi, b0, az, math.radians(r.uniform(-80, -60)), 0.1, L0 * 0.85, L0 * 0.5, 0.2, 1)
    me = mesh_from_arrays(f"{p['id']}_crown", np.asarray(verts), faces, uvs={"UVMap": np.asarray(uvs)}, mat_idx=mi)
    me.materials.append(mats["green"])
    me.materials.append(mats["dead"])
    new_object(f"{p['id']}_crown", me, col, iid, {"kind": "veg.palm.date"})


def build_scatter_systems(plan, am, col, ids, terrain_obj):
    """One object per scatter system, driven by the SYN_Scatter node group.
    Prototypes of a role live in an ASSETS_<role> collection; their pass_index
    is the shared id of the kind (scatter instances report the prototype's id)."""
    from synth.blender.scatter_nodes import scatter_object
    lib = am.library
    out = []
    role_cols = {}
    for sys_ in plan.get("scatter_systems", []):
        role, kind = sys_["role"], sys_["kind"]
        if role not in role_cols:
            rc = role_collection(am, role)
            sid = ids.new(kind, f"<scatter {kind}>")
            for p in rc.objects:
                p.pass_index = sid
            role_cols[role] = rc
        params = dict(sys_["params"])
        if "Live Tint" in params:
            params["Live Tint"] = tuple(params["Live Tint"])
        ob = scatter_object(f"scatter_{sys_['name']}", col, terrain_obj, role_cols[role], params)
        ob["kind"] = kind
        out.append(ob)
    return out


def role_collection(am, role):
    """ASSETS_<role>: the prototypes a scatter system picks from (made once, kept in the template)."""
    rc = bpy.data.collections.get(f"ASSETS_{role}")
    if rc is None:
        rc = bpy.data.collections.new(f"ASSETS_{role}")
        am.library.children.link(rc)
    if role.startswith("proc."):   # procedural prototypes (no scanned asset for this role)
        from synth.blender import proc_veg
        for p in proc_veg.prototypes(role, am.library):
            if p.name not in rc.objects:
                rc.objects.link(p)
        return rc
    is_veg = role.startswith("veg.")
    for stem in am.assets_for_role(role):
        for m in am.variants(stem, lod=0):
            p = am.prototype(stem, m, foliage=is_veg, alpha=is_veg, value_jitter=0.12 if is_veg else 0.08)
            if p.name not in rc.objects:
                rc.objects.link(p)
    return rc
