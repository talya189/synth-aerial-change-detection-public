"""Road and path meshes (runs inside Blender).

Each road is a ribbon swept along its planned centreline: carriageway with 2 %
camber and a short skirt that tucks under the cut terrain. UVs:
  UVMap   metres (u across, v along)  -> physically sized textures
  RoadUV  (u / half_width, v metres)  -> edge masks, lane markings
Dirt roads and footpaths are wider than their nominal width and fade out with
a noisy alpha edge so they melt into the soil (labels use the alpha threshold).
"""
import numpy as np

from synth.blender import material_factory as mf
from synth.blender.meshutil import mesh_from_arrays, new_object


def _frames(pts):
    p = np.asarray(pts, float)
    t = np.gradient(p[:, :2], axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.column_stack([-t[:, 1], t[:, 0]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))])
    return p, n, s


def ribbon(name, pts, offsets, dz_fn, terrain=None, drape=False, lift=0.0, width_scale=None, uv_centre=0.0, uv_hw=None):
    """offsets: lateral positions (m). dz_fn(off) -> height offset from centreline z.
    drape=True puts every vertex on the terrain (+lift) instead of the road profile.
    RoadUV u = (offset - uv_centre) / uv_hw (default: the ribbon's own half width),
    so one carriageway of a divided road gets its own edges and markings."""
    p, n, s = _frames(pts)
    offs = np.asarray(offsets, float)
    m = len(offs)
    ws = np.ones(len(p)) if width_scale is None else np.asarray(width_scale, float)
    X = p[:, None, 0] + n[:, None, 0] * offs[None] * ws[:, None]
    Y = p[:, None, 1] + n[:, None, 1] * offs[None] * ws[:, None]
    if drape:
        Z = terrain.height(X.ravel(), Y.ravel()).reshape(X.shape) + lift
    else:
        Z = p[:, None, 2] + dz_fn(offs)[None]
        if terrain is not None:  # carriageway never below the ground (skirt columns excluded)
            hw = np.abs(offs).max() - 0.25
            cols = np.abs(offs) <= hw + 1e-6
            T = terrain.height(X[:, cols].ravel(), Y[:, cols].ravel()).reshape(X[:, cols].shape)
            Z[:, cols] = np.maximum(Z[:, cols], T + 0.03)
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    i, j = np.mgrid[0:len(p) - 1, 0:m - 1]
    a = i * m + j
    faces = np.stack([a, a + 1, a + m + 1, a + m], -1).reshape(-1, 4)
    hw = uv_hw or np.abs(offs).max()
    uv_m = np.column_stack([np.tile(offs, len(p)), np.repeat(s, m)])
    uv_r = np.column_stack([np.tile((offs - uv_centre) / hw, len(p)), np.repeat(s, m)])
    fl = faces.ravel()
    return mesh_from_arrays(name, verts, faces, uvs={"UVMap": uv_m[fl], "RoadUV": uv_r[fl]})


CAMBER = 0.02  # must match roads.apply_corridors


def junction_trim(rd, by_id, terrain=None, flare_m=6.0, flare=0.0):
    """A branch road starts on top of its parent. Trim it where it reaches the
    parent's edge (tucked 0.25 m under it) and bring its start height to the
    parent's cambered edge. The rounded kerb corners are separate fillet meshes
    (plan["junction_fillets"]); `flare` optionally widens the mouth instead.
    Returns (points, width_scale)."""
    pts = np.asarray(rd["centerline"], float)
    parent = by_id.get(rd.get("junction_with"))
    if parent is None:
        return pts, None
    pc = np.asarray(parent["centerline"], float)
    phw = parent["width"] / 2
    k0 = int(np.argmin(np.linalg.norm(pc[:, :2] - pts[0, :2], axis=1)))
    if k0 < 3 or k0 > len(pc) - 4:
        flare = 0.0  # continuation from the parent's end, not a side junction
    d = np.min(np.linalg.norm(pts[:, None, :2] - pc[None, :, :2], axis=2), axis=1)
    out = np.nonzero(d > phw - 0.25)[0]
    if not len(out) or out[0] == 0 or out[0] > len(pts) // 2:
        return pts, None
    i0 = out[0] - 1
    pts = pts[i0:].copy()
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))])
    k = int(np.argmin(np.linalg.norm(pc[:, :2] - pts[0, :2], axis=1)))
    edge_z = pc[k, 2] - CAMBER * phw - 0.01
    blend = np.clip(1 - s / 8.0, 0, 1) ** 2
    pts[:, 2] = pts[:, 2] * (1 - blend) + edge_z * blend
    if terrain is not None:  # never let the cut terrain hide the mouth of the branch
        near = s < 12.0
        pts[near, 2] = np.maximum(pts[near, 2], terrain.height(pts[near, 0], pts[near, 1]) + 0.03)
    ws = 1 + flare * np.clip(1 - s / flare_m, 0, 1) ** 2
    return pts, ws


def _trim_both(rd, by_id, terrain):
    """Junction trim at the start (`junction_with`) and at the end (`junction_end`)."""
    pts, ws = junction_trim(rd, by_id, terrain)
    if rd.get("junction_end") in by_id:
        rev = dict(rd, centerline=np.asarray(pts)[::-1], junction_with=rd["junction_end"])
        p2, w2 = junction_trim(rev, by_id, terrain)
        if w2 is not None or len(p2) != len(pts):
            n_cut = len(pts) - len(p2)
            pts = np.asarray(p2)[::-1]
            w_end = np.ones(len(pts)) if w2 is None else np.asarray(w2)[::-1]
            w_start = np.ones(len(pts)) if ws is None else np.asarray(ws)[:len(pts)]
            ws = w_start * w_end
    return pts, ws


def _turning_circle(name, centre, z, radius, seg=28):
    """Round asphalt pad at a cul-de-sac end, falling slightly to its edge like the camber."""
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.column_stack([centre[0] + radius * np.cos(ang), centre[1] + radius * np.sin(ang),
                            np.full(seg, z - CAMBER * radius)])
    skirt = np.column_stack([centre[0] + (radius + 0.25) * np.cos(ang), centre[1] + (radius + 0.25) * np.sin(ang),
                             np.full(seg, z - CAMBER * radius - 0.15)])
    verts = np.vstack([[centre[0], centre[1], z], ring, skirt])
    faces = [(0, 1 + k, 1 + (k + 1) % seg) for k in range(seg)]
    faces += [(1 + k, 1 + seg + k, 1 + seg + (k + 1) % seg, 1 + (k + 1) % seg) for k in range(seg)]
    import bpy
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces)
    for layer in ("UVMap", "RoadUV"):
        uvl = me.uv_layers.new(name=layer)
        for i, lp in enumerate(me.loops):
            x, y = verts[lp.vertex_index][:2]
            uvl.data[i].uv = (x, y) if layer == "UVMap" else (0.0, 0.0)
    return me


def build_roads(plan, am, col, ids, terrain):
    mats = {}
    out = []
    by_id = {r["id"]: r for r in plan["roads"]}
    for rd in plan["roads"]:
        hw = rd["width"] / 2
        pts, ws = _trim_both(rd, by_id, terrain)
        if rd["surface"] == "asphalt" and rd.get("section") == "divided":
            me, extra = _divided(rd, pts, plan["roads"], am, mats, terrain)
            iid = ids.new(rd["kind"], rd["id"])
            for k, m2 in enumerate(extra):   # median pieces: same road, same instance
                out.append(new_object(f"{rd['id']}_median_{k}", m2, col, iid, {"kind": rd["kind"]}))
        elif rd["surface"] == "asphalt":
            offs = np.concatenate([[-hw - 0.25], np.linspace(-hw, hw, 9), [hw + 0.25]])
            dz = lambda o, hw=hw: np.where(np.abs(o) > hw, -CAMBER * hw - 0.15, -CAMBER * np.abs(o))
            me = ribbon(rd["id"], pts, offs, dz, terrain=terrain, width_scale=ws)
            me.materials.append(_asphalt(am, mats, rd.get("section") == "two_lane", hw))
        else:
            ext = 1.0  # fade zone beyond nominal width
            offs = np.linspace(-hw - ext, hw + ext, 11)
            me = ribbon(rd["id"], pts, offs, None, terrain=terrain, drape=True, lift=0.025, width_scale=ws)
            if "dirt" not in mats:
                mats["dirt"] = dirt_road_material(am, fade_from=hw / (hw + ext))
            me.materials.append(mats["dirt"])
        iid = ids.new(rd["kind"], rd["id"])
        ob = new_object(rd["id"], me, col, iid, {"kind": rd["kind"], "width": rd["width"]})
        out.append(ob)
        if rd.get("end_cap"):   # turning circle: same road, same instance id
            e = np.asarray(rd["centerline"][-1], float)
            cm = _turning_circle(f"{rd['id']}_circle", e[:2], float(e[2]), float(rd["end_cap"]))
            cm.materials.append(me.materials[0])
            out.append(new_object(f"{rd['id']}_circle", cm, col, iid, {"kind": rd["kind"], "width": 2 * rd["end_cap"]}))
    for f in plan.get("junction_fillets", []):   # rounded corners: same material and instance as the joining road
        tris = np.asarray(f["triangles"], float).reshape(-1, 3)
        faces = np.arange(len(tris)).reshape(-1, 3)
        uv_m = tris[:, :2]
        uv_r = np.column_stack([np.full(len(tris), 0.6), tris[:, 1]])
        me = mesh_from_arrays(f["id"], tris, faces, uvs={"UVMap": uv_m, "RoadUV": uv_r})
        me.materials.append(_asphalt(am, mats, False, 3.0))
        out.append(new_object(f["id"], me, col, ids.new(f["kind"], f["label_id"]), {"kind": f["kind"]}))
    for p in plan["paths"]:
        hw = p["width"] / 2
        if p.get("surface") == "paved":   # concrete walkway: crisp edges, slightly raised
            me = ribbon(p["id"], p["centerline"], np.linspace(-hw, hw, 5), None, terrain=terrain, drape=True, lift=0.06)
            if "paved" not in mats:
                mats["paved"] = am.surface_material("path.sidewalk")
            me.materials.append(mats["paved"])
        else:
            offs = np.linspace(-hw - 0.5, hw + 0.5, 7)
            me = ribbon(p["id"], p["centerline"], offs, None, terrain=terrain, drape=True, lift=0.02)
            if "path" not in mats:
                mats["path"] = dirt_road_material(am, fade_from=hw / (hw + 0.5), role="path.dirt", ruts=False, name="M_footpath")
            me.materials.append(mats["path"])
        out.append(new_object(p["id"], me, col, ids.new(p["kind"], p["id"]), {"kind": p["kind"], "width": p["width"]}))
    return out


def _asphalt(am, mats, markings, hw):
    """Asphalt material, shared per (markings, half width): line widths are in RoadUV units."""
    key = f"asphalt_marked_{hw:.2f}" if markings else "asphalt_street"
    if key not in mats:
        mats[key] = asphalt_material(am, markings, hw, name=f"M_{key}")
    return mats[key]


def _divided(rd, pts, roads, am, mats, terrain):
    """Divided road: two carriageways (one mesh, each with its own edges and
    lane markings) either side of a raised median with kerbs, opened where
    another road joins. Returns (carriageway mesh, [median meshes])."""
    W, med = rd["width"], rd["median_m"]
    cw = (W - med) / 2              # one carriageway
    c = med / 2 + cw / 2            # its centre offset
    parts = []
    for sgn in (-1, 1):
        offs = sgn * c + np.concatenate([[-cw / 2 - 0.25], np.linspace(-cw / 2, cw / 2, 7), [cw / 2 + 0.25]])
        # each carriageway falls from the median to its outer edge; skirts tuck under
        dz = lambda o, sgn=sgn: np.where(np.abs(o - sgn * c) > cw / 2, -CAMBER * cw - 0.15, -CAMBER * (np.abs(o) - med / 2))
        parts.append(ribbon(f"{rd['id']}_{'L' if sgn < 0 else 'R'}", pts, offs, dz, terrain=terrain, uv_centre=sgn * c, uv_hw=cw / 2))
    me = _join(rd["id"], parts)
    me.materials.append(_asphalt(am, mats, True, cw / 2))
    # median: kerbed island, broken where roads join (gap = their width + 2 corner radii)
    p, n, s = _frames(pts)
    joins = []
    for o in roads:
        for key, end in (("junction_with", 0), ("junction_end", -1)):
            if o.get(key) == rd["id"]:
                q = np.asarray(o["centerline"][end][:2])
                k = int(np.argmin(np.linalg.norm(p[:, :2] - q, axis=1)))
                joins.append((s[k], o["width"] / 2 + 7.0))
    keep = np.ones(len(p), bool)
    for sj, half in joins:
        keep &= np.abs(s - sj) > half
    keep[:2] = keep[-2:] = False     # the median starts and ends with the built-up stretch
    pieces = []
    k = 0
    while k < len(p):
        if not keep[k]:
            k += 1
            continue
        k1 = k
        while k1 < len(p) and keep[k1]:
            k1 += 1
        if k1 - k >= 4:
            hm = med / 2
            offs = np.array([-hm, -hm + 0.01, -hm + 0.2, hm - 0.2, hm - 0.01, hm])
            kh = 0.18
            dz = lambda o: np.where(np.abs(o) > hm - 0.005, -0.02, kh)
            m2 = ribbon(f"{rd['id']}_median_{len(pieces)}", pts[k:k1], offs, dz, terrain=None)
            m2.materials.append(mats.setdefault("kerb", mf.dirty_flat_material("M_kerb", (0.62, 0.6, 0.55), 0.8, dirt=0.35, noise_scale=5)))
            m2.materials.append(mats.setdefault("median_soil", am.surface_material("terrain.soil")))
            fi = np.arange(len(m2.polygons)) % (len(offs) - 1)
            m2.polygons.foreach_set("material_index", np.where(fi == 2, 1, 0).astype(np.int32))
            pieces.append(m2)
        k = k1
    return me, pieces


def _join(name, meshes):
    """One mesh from several (same UV layers)."""
    import bmesh
    import bpy
    bm = bmesh.new()
    for m in meshes:
        bm.from_mesh(m)
        bpy.data.meshes.remove(m)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def asphalt_material(am, markings, half_width, name=None):
    """Dusty Mediterranean asphalt: two scanned asphalts blended, lighter worn
    wheel tracks, soil dust creeping in from the edges, faded paint."""
    stem = am.resolve("road.asphalt")
    edge_stem = am.resolve("terrain.dirt")
    mat = mf.new_material(name or ("M_asphalt_main" if markings else "M_asphalt_street"))
    nb = mf.NodeBuilder(mat)
    uv = nb.uv("UVMap")
    a = mf.SurfaceSample(nb, _h(am, stem), am.tier, uv, seed=0.1)
    color, rough, height = a.color, a.rough, a.height
    # sparse repair patches: newer asphalt is darker and smoother (sharp-edged, blocky)
    cell = nb.node("ShaderNodeTexVoronoi")
    nb.link(nb.scale_vec(uv, 1 / 4.0), cell.inputs["Vector"])
    cell.inputs["Randomness"].default_value = 0.6
    cell.distance = "CHEBYCHEV"
    sep_c = nb.node("ShaderNodeSeparateColor")
    nb.link(cell.outputs["Color"], sep_c.inputs[0])
    pick = nb.map_range(sep_c.outputs[0], 0.93, 0.94)  # ~6 % of 4 m cells
    patch = nb.math("MULTIPLY", pick, nb.map_range(nb.noise(nb.scale_vec(uv, 1 / 40.0), 1.0, detail=2).outputs["Fac"], 0.5, 0.6))
    color = nb.mix(nb.math("MULTIPLY", patch, 0.85), color, nb.mix(1.0, color, (0.62, 0.62, 0.64, 1), "RGBA", "MULTIPLY"))
    rough = nb.mix(patch, rough, nb.math("MULTIPLY", rough, 0.8), "FLOAT")
    # road coordinates
    ruv = nb.uv("RoadUV")
    sep = nb.node("ShaderNodeSeparateXYZ")
    nb.link(ruv, sep.inputs[0])
    u = sep.outputs["X"]
    au = nb.math("ABSOLUTE", u)
    # wheel tracks: smoother, slightly lighter bands
    tracks = nb.math("MULTIPLY", nb.map_range(nb.math("ABSOLUTE", nb.math("SUBTRACT", au, 0.55)), 0.18, 0.05), 0.05)
    color = nb.mix(tracks, color, (0.42, 0.41, 0.39, 1), "RGBA", "SCREEN")
    # dust from the shoulders
    e = mf.SurfaceSample(nb, _h(am, edge_stem), am.tier, uv, seed=0.3, anti_tile=False)
    dn = nb.noise(nb.scale_vec(uv, 1 / 1.3), 1.0, detail=5).outputs["Fac"]
    dust = nb.map_range(nb.math("ADD", au, nb.math("MULTIPLY", nb.math("SUBTRACT", dn, 0.5), 0.35)), 0.8, 1.02)
    color = nb.mix(dust, color, e.color)
    rough = nb.mix(dust, rough, e.rough, "FLOAT")
    if markings:
        # Israeli rural road: dashed white centre line, continuous edge lines; worn
        v = sep.outputs["Y"]
        dash = nb.map_range(nb.math("MODULO", v, 12.0), 3.0, 3.05, 1.0, 0.0)
        centre = nb.math("MULTIPLY", nb.map_range(au, 0.06 / half_width, 0.04 / half_width, 0, 1), dash)
        edge = nb.map_range(nb.math("ABSOLUTE", nb.math("SUBTRACT", au, 1 - 0.45 / half_width)), 0.08 / half_width, 0.06 / half_width, 0, 1)
        paint = nb.math("MAXIMUM", centre, edge)
        wear = nb.map_range(nb.noise(nb.scale_vec(uv, 1 / 0.4), 1.0, detail=6).outputs["Fac"], 0.35, 0.6)
        paint = nb.math("MULTIPLY", paint, nb.math("MULTIPLY", wear, 0.55))
        color = nb.mix(paint, color, (0.62, 0.61, 0.57, 1))
        rough = nb.mix(paint, rough, 0.55, "FLOAT")
    n = mf.bump(nb, height, 0.6, a.height_m)
    mf.output(nb, mf.principled(nb, color, rough, n, specular=0.4).outputs[0])
    return mat


def dirt_road_material(am, fade_from, role="road.dirt", ruts=True, name="M_dirt_road"):
    """Compacted dirt with optional wheel ruts; alpha fades to the terrain at the edges."""
    stem = am.resolve(role)
    mat = mf.new_material(name)
    nb = mf.NodeBuilder(mat)
    uv = nb.uv("UVMap")
    s = mf.SurfaceSample(nb, _h(am, stem), am.tier, uv, seed=0.5)
    color, rough = s.color, s.rough
    ruv = nb.uv("RoadUV")
    sep = nb.node("ShaderNodeSeparateXYZ")
    nb.link(ruv, sep.inputs[0])
    au = nb.math("ABSOLUTE", sep.outputs["X"])
    if ruts:
        rut = nb.map_range(nb.math("ABSOLUTE", nb.math("SUBTRACT", au, 0.38 * fade_from)), 0.12 * fade_from, 0.03, 0, 0.25)
        color = nb.mix(rut, color, (0.55, 0.47, 0.38, 1), "RGBA", "OVERLAY")
    # compaction: dirt roads are lighter and smoother than the surrounding soil
    hsv = nb.node("ShaderNodeHueSaturation")
    nb.link(color, hsv.inputs["Color"])
    hsv.inputs["Value"].default_value = 1.35   # compacted limestone/basalt dust: pale beige-grey
    hsv.inputs["Saturation"].default_value = 0.55
    dn = nb.noise(nb.scale_vec(uv, 1 / 0.9), 1.0, detail=5).outputs["Fac"]
    edge = nb.math("ADD", au, nb.math("MULTIPLY", nb.math("SUBTRACT", dn, 0.5), 0.5 * (1 - fade_from) + 0.15))
    alpha = nb.map_range(edge, 1.0, fade_from, 0, 1)
    bsdf = mf.principled(nb, hsv.outputs["Color"], rough, mf.bump(nb, s.height, 0.7, s.height_m), alpha=alpha, specular=0.35)
    mf.output(nb, bsdf.outputs[0])
    return mat


def _h(am, stem):
    a = dict(am.asset(stem))
    a["height_m"] = a.get("height_m") or 0.03
    am.used[stem] = a.get("version")
    return a
