"""Terrain meshes (core / mid ring / far field), terrain materials, water."""
import bpy
import numpy as np

from synth.blender import material_factory as mf
from synth.blender.meshutil import grid_faces, mesh_from_arrays, new_object, per_loop
from synth.common.config import project_path


def bilinear(arr, half, res, x, y):
    """Sample a south-row-0 grid at local (x, y); clamps at the border."""
    n = arr.shape[0]
    fi = np.clip((np.asarray(y) + half) / res, 0, n - 1.000001)
    fj = np.clip((np.asarray(x) + half) / res, 0, n - 1.000001)
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    ti, tj = fi - i0, fj - j0
    return (arr[i0, j0] * (1 - ti) * (1 - tj) + arr[i0, j0 + 1] * (1 - ti) * tj
            + arr[i0 + 1, j0] * ti * (1 - tj) + arr[i0 + 1, j0 + 1] * ti * tj)


class Terrain:
    def __init__(self, plan, plan_dir):
        d = np.load(f"{plan_dir}/{plan['terrain']['file']}")
        m = plan["dtm"]
        self.core, self.mid, self.far = d["core_z"], d["mid_z"], d["far_z"]
        self.free = d["free"] if "free" in d.files else np.ones_like(self.core)
        self.wet = d["wet"] if "wet" in d.files else np.zeros_like(self.core)
        self.core_half, self.core_res = m["core_half_m"], m["core_res_m"]
        self.mid_half, self.mid_res = m["mid_half_m"], m["mid_res_m"]
        self.far_half, self.far_res = m["far_half_m"], m["far_res_m"]

    def height(self, x, y):
        """Best available terrain height at local (x, y)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = bilinear(self.far, self.far_half, self.far_res, x, y)
        in_mid = (np.abs(x) < self.mid_half) & (np.abs(y) < self.mid_half)
        z = np.where(in_mid, bilinear(self.mid, self.mid_half, self.mid_res, x, y), z)
        in_core = (np.abs(x) <= self.core_half) & (np.abs(y) <= self.core_half)
        return np.where(in_core, bilinear(self.core, self.core_half, self.core_res, x, y), z)


def _slope_deg(z, res, smooth_m=6.0):
    """Slope from heights smoothed to a common scale, so every ring (1 m core
    with micro-relief, 5 m mid, 30 m far) drives slope-based cover the same way."""
    from math import pi
    k = max(1, int(round(smooth_m / res)))
    zz = z
    if k > 1:  # box blur (numpy only)
        pad = np.pad(z, k, mode="edge")
        c = np.cumsum(np.cumsum(pad, 0), 1)
        c = np.pad(c, ((1, 0), (1, 0)))
        n = 2 * k + 1
        zz = (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)
    gy, gx = np.gradient(zz, res)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def _ring_mesh(name, z, half, res, inner_half, inner_height_fn, blend_m, core_half):
    """Grid with the inner square removed; vertices near the inner edge blend
    toward the inner terrain so the rings meet.
    If the inner edge lies on this grid's lines, the hole is exact (no face
    under the inner terrain), so border normals come only from visible faces:
    no shading crease. Otherwise the ring overlaps slightly, its hidden
    vertices pushed under the inner terrain (only used far from the camera)."""
    n = z.shape[0]
    c = np.arange(n) * res - half
    X, Y = np.meshgrid(c, c)
    Z = z.astype(np.float64).copy()
    cheb = np.maximum(np.abs(X), np.abs(Y))
    inner = inner_height_fn(np.clip(X, -inner_half, inner_half), np.clip(Y, -inner_half, inner_half))
    w = np.clip(1 - (cheb - inner_half) / blend_m, 0, 1)
    w = w * w * (3 - 2 * w)
    Z = Z * (1 - w) + inner * w
    corner_min = np.minimum.reduce([cheb[:-1, :-1], cheb[1:, 1:], cheb[:-1, 1:], cheb[1:, :-1]])
    corner_max = np.maximum.reduce([cheb[:-1, :-1], cheb[1:, 1:], cheb[:-1, 1:], cheb[1:, :-1]])
    aligned = abs((half - inner_half) / res - round((half - inner_half) / res)) < 1e-6
    if aligned:
        keep = corner_min >= inner_half - 1e-6
    else:
        under = cheb < inner_half
        Z[under] = inner_height_fn(X[under], Y[under]) - 2.0
        keep = corner_max >= inner_half - res
    faces = grid_faces(n, n, keep)
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    uv_m = per_loop(verts[:, :2], faces)
    uv_s = per_loop((verts[:, :2] + core_half) / (2 * core_half), faces)
    return mesh_from_arrays(name, verts, faces, uvs={"UVMap": uv_m, "SplatUV": uv_s},
                            point_attrs={"slope_deg": _slope_deg(z, res)})


def _skirt_mesh(name, z, half, res, depth=1.0):
    """Vertical curtain under the core's border (separate object, so it never
    bends the core's smooth normals); hides T-junction cracks at the ring join."""
    n = z.shape[0]
    c = np.arange(n) * res - half
    edge = [(c, np.full(n, -half), z[0, :]), (np.full(n, half), c, z[:, -1]),
            (c[::-1], np.full(n, half), z[-1, ::-1]), (np.full(n, -half), c[::-1], z[::-1, 0])]
    xs = np.concatenate([e[0] for e in edge])
    ys = np.concatenate([e[1] for e in edge])
    zs = np.concatenate([e[2] for e in edge])
    m = len(xs)
    verts = np.vstack([np.column_stack([xs, ys, zs - 0.02]), np.column_stack([xs, ys, zs - depth])])
    i = np.arange(m - 1)
    faces = np.stack([i, i + m, i + m + 1, i + 1], -1)
    uv = per_loop(verts[:, [0, 2]], faces)
    return mesh_from_arrays(name, verts, faces, uvs={"UVMap": uv, "SplatUV": uv * 0 - 1}, smooth=False,
                            point_attrs={"slope_deg": np.full(len(verts), 90.0)})


def build_terrain(plan, plan_dir, am, col, ids):
    T = Terrain(plan, plan_dir)
    # ---- core
    n = T.core.shape[0]
    c = np.arange(n) * T.core_res - T.core_half
    X, Y = np.meshgrid(c, c)
    verts = np.column_stack([X.ravel(), Y.ravel(), T.core.ravel()])
    faces = grid_faces(n, n)
    uv_m = per_loop(verts[:, :2], faces)
    uv_s = per_loop((verts[:, :2] + T.core_half) / (2 * T.core_half), faces)
    # slope for land cover comes from the 5 m mid grid on the core too, so the
    # rock/grass pattern is continuous across the core/mid join
    mid_slope = _slope_deg(T.mid, T.mid_res)
    core_slope = bilinear(mid_slope, T.mid_half, T.mid_res, verts[:, 0], verts[:, 1])
    me = mesh_from_arrays("terrain_core", verts, faces, uvs={"UVMap": uv_m, "SplatUV": uv_s},
                          point_attrs={"slope_deg": core_slope, "free": T.free.ravel(), "wet": T.wet.ravel()})
    mat = terrain_material("T_terrain", am, plan, plan_dir)
    me.materials.append(mat)
    core = new_object("terrain_core", me, col, ids.new("terrain.core", "terrain.core"), {"kind": "terrain.core"})
    sk = _skirt_mesh("terrain_core_skirt", T.core, T.core_half, T.core_res)
    sk.materials.append(mat)
    new_object("terrain_core_skirt", sk, col, core.pass_index, {"kind": "terrain.core"})

    # ---- mid ring and far field
    core_h = lambda x, y: bilinear(T.core, T.core_half, T.core_res, x, y)
    me = _ring_mesh("terrain_mid", T.mid, T.mid_half, T.mid_res, T.core_half, core_h, 60.0, T.core_half)
    me.materials.append(mat)
    new_object("terrain_mid", me, col, ids.new("terrain.mid", "terrain.mid"), {"kind": "terrain.mid"})
    mid_h = lambda x, y: T.height(x, y)
    me = _ring_mesh("terrain_far", T.far, T.far_half, T.far_res, T.mid_half, mid_h, 300.0, T.core_half)
    me.materials.append(mat)
    new_object("terrain_far", me, col, ids.new("terrain.far", "terrain.far"), {"kind": "terrain.far"})

    # ---- water (Sea of Galilee etc.): one plane at the lake level; land above hides it
    wl = plan["terrain"].get("water_level")
    if wl is not None and (T.far < wl).any():
        h = T.far_half
        wv = np.array([[-h, -h, wl], [h, -h, wl], [h, h, wl], [-h, h, wl]])
        wme = mesh_from_arrays("water", wv, np.array([[0, 1, 2, 3]]), uvs={"UVMap": wv[[0, 1, 2, 3], :2]}, smooth=False)
        wme.materials.append(water_material())
        new_object("water", wme, col, ids.new("water", "water"), {"kind": "water"})
    T.core_object = core
    return T


def water_material():
    mat = mf.new_material("Water")
    nb = mf.NodeBuilder(mat)
    tc = nb.node("ShaderNodeTexCoord")
    nz = nb.noise(nb.scale_vec(tc.outputs["Object"], 0.15), 1.0, detail=6, roughness=0.6)
    b = mf.bump(nb, nz.outputs["Fac"], 0.08, 1.0)
    bsdf = mf.principled(nb, (0.035, 0.07, 0.075, 1.0), 0.06, b, specular=0.5)
    mf.output(nb, bsdf.outputs[0])
    return mat


def terrain_material(name, am, plan, plan_dir):
    """One layered terrain material for core, mid ring and far field (no seams).

    Natural land cover comes from world-space noise and slope, identical on
    every ring. The splat map (core only; zero outside via CLIP) adds the
    human-made layers: road-shoulder gravel and trampled ground near houses
    and paths. Layers blend by weight AND their scanned height maps, so gravel
    fills soil crevices instead of cross-fading. Each layer is anti-tiled."""
    tcfg = plan["config"]["terrain_material"]
    tier = am.tier
    mat = mf.new_material(name)
    nb = mf.NodeBuilder(mat)
    uv = nb.uv("UVMap")
    base = mf.SurfaceSample(nb, _with_h(am, am.resolve(tcfg["base"]["role"])), tier, uv, seed=0.0)
    color = _tone(nb, uv, _adjust(nb, base.color, tcfg["base"].get("adjust")), tcfg["base"].get("tone"), 0)
    rough, height = base.rough, base.height
    splat = _splat_channels(nb, plan, plan_dir)
    slope_deg = nb.node("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="slope_deg").outputs["Fac"]
    edge = nb.math("SUBTRACT", nb.noise(nb.scale_vec(uv, 1 / 2.2), 1.0, detail=5, roughness=0.6).outputs["Fac"], 0.5)
    for k, L in enumerate(tcfg["layers"]):
        s = mf.SurfaceSample(nb, _with_h(am, am.resolve(L["role"])), tier, uv, seed=0.37 * (k + 1))
        w = nb.value(0.0)
        if L.get("coverage", 0) > 0:
            n1 = nb.noise_m(uv, L["noise_m"], seed=1.0 + k, detail=5, roughness=0.55)
            n2 = nb.noise_m(uv, L["noise_m"] / 5, seed=11.0 + k, detail=3, roughness=0.5)
            n = nb.math("ADD", nb.math("MULTIPLY", n1, 0.55), nb.math("MULTIPLY", n2, 0.45))
            thr = 0.5 + (0.5 - L["coverage"]) * 0.32
            w = nb.map_range(n, thr - 0.05, thr + 0.05)
        if "slope_deg" in L:   # slope-driven (rock on steep ground); noise-jittered edge
            a0, a1 = L["slope_deg"]
            sl = slope_deg
            if L.get("slope_jitter_deg"):
                nj = nb.noise_m(uv, 6.0, seed=21.0 + k, detail=4, roughness=0.6)
                sl = nb.math("ADD", sl, nb.math("MULTIPLY", nb.math("SUBTRACT", nj, 0.5), 2 * L["slope_jitter_deg"]))
            w = nb.math("MAXIMUM", w, nb.map_range(sl, a0, a1))
        if "slope_max_deg" in L:   # e.g. grass thins out on steep slopes
            w = nb.math("MULTIPLY", w, nb.map_range(slope_deg, L["slope_max_deg"], L["slope_max_deg"] - 12))
        if "splat" in L:
            w = nb.math("MAXIMUM", w, splat[L["splat"]])
        if "wet" in L:   # drainage lines (splat B): wadi beds
            a0, a1 = L["wet"]
            w = nb.math("MAXIMUM", w, nb.map_range(splat[2], a0, a1))
        if "density" in L:   # smooth amount of this layer (lo..hi), varying at wavelength m; no islands
            dn = L["density"]
            dens = nb.map_range(_snoise(nb, uv, dn["m"], 91.0 + k, 3), -0.5, 0.5, dn["lo"], dn["hi"])
            w = nb.math("MAXIMUM", w, dens)
        if "lines" in L:   # thin meandering network (cattle and goat trails): the zero set of a noise field
            ln = L["lines"]
            n_l = nb.noise_m(uv, ln["m"], seed=101.0 + k, detail=1.0, roughness=0.3)   # smooth: long, gently curving trails
            d_l = nb.math("ABSOLUTE", nb.math("SUBTRACT", n_l, 0.5))
            line = nb.map_range(d_l, ln["width"] * 0.4, ln["width"])          # 1 on the line, 0 off it
            keep = nb.map_range(_snoise(nb, uv, ln["m"] * 2.5, 111.0 + k, 2), ln.get("keep", 0.0), ln.get("keep", 0.0) + 0.25)   # trails fade out in places
            w = nb.math("MAXIMUM", w, nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, line), keep))
        if "coverage_var" in L:   # large-scale change of cover (more grass here, barer there)
            w = nb.math("ADD", w, nb.math("MULTIPLY", _snoise(nb, uv, L["coverage_var"]["m"], 51.0 + k, 3), L["coverage_var"]["amp"]))
        if "breakup" in L:   # fine patchiness: the layer frays into clumps instead of a solid sheet
            w = nb.math("ADD", w, nb.math("MULTIPLY", _snoise(nb, uv, L["breakup"]["m"], 31.0 + k, 6), L["breakup"]["amp"]))
        if "dither" in L:   # tuft-scale noise: edges dissolve into scattered tufts instead of a crisp line
            w = nb.math("ADD", w, nb.math("MULTIPLY", _snoise(nb, uv, L["dither"]["m"], 61.0 + k, 2), L["dither"]["amp"]))
        if L.get("core_only"):   # splat alpha: 1 inside the detailed core, 0 on the outer rings
            w = nb.math("MULTIPLY", w, splat[3])
        if L.get("far_only"):    # outer rings only: stands in for the 3D ground cover scattered in the core
            w = nb.math("MULTIPLY", w, nb.math("SUBTRACT", 1.0, splat[3]))
        w = nb.math("ADD", w, nb.math("MULTIPLY", edge, 0.3))
        t = nb.math("ADD", w, nb.math("MULTIPLY", nb.math("SUBTRACT", s.height, height), 0.3))
        f = nb.map_range(t, 0.38, 0.68)
        color = nb.mix(f, color, _tone(nb, uv, _adjust(nb, s.color, L.get("adjust")), L.get("tone"), k + 1))
        rough = nb.mix(f, rough, s.rough, "FLOAT")
        height = nb.mix(f, height, s.height, "FLOAT")
    # fine stone speckle: small dark cobbles and their shadows, sub-pixel at aerial GSD
    vor = nb.node("ShaderNodeTexVoronoi")
    nb.link(nb.scale_vec(uv, 1 / 0.35), vor.inputs["Vector"])
    vs = nb.node("ShaderNodeSeparateColor")
    nb.link(vor.outputs["Color"], vs.inputs[0])
    dens = nb.map_range(nb.noise_m(uv, 30.0, seed=41.0, detail=2), 0.35, 0.7, 0.03, 0.14)
    stone = nb.math("LESS_THAN", vs.outputs[0], dens)
    stone = nb.math("MULTIPLY", stone, nb.math("SUBTRACT", 1.0, splat[0]))
    color = nb.mix(nb.math("MULTIPLY", stone, 0.55), color, (0.16, 0.15, 0.14, 1))
    color, rough, height = _field_overlay(nb, plan, plan_dir, uv, color, rough, height)
    # gentle macro variation (moisture / mineral patches): value and saturation only
    m1 = nb.noise(nb.scale_vec(uv, 1 / 140.0), 1.0, detail=3).outputs["Fac"]
    m2 = nb.noise(nb.scale_vec(uv, 1 / 31.0), 1.0, detail=2).outputs["Fac"]
    hsv = nb.node("ShaderNodeHueSaturation")
    nb.link(color, hsv.inputs["Color"])
    nb.link(nb.map_range(m1, 0.3, 0.7, 0.88, 1.08), hsv.inputs["Saturation"])
    nb.link(nb.map_range(m2, 0.3, 0.7, 0.92, 1.06), hsv.inputs["Value"])
    n = mf.bump(nb, height, 0.9, 0.05)
    bsdf = mf.principled(nb, hsv.outputs["Color"], rough, n, specular=0.35)
    mf.output(nb, bsdf.outputs[0])
    return mat


def layer_material(name, am, L, seed=0.0):
    """One terrain layer alone (asset + its `adjust`), shaded like in the full
    terrain material. Used by the swatch renderer to measure layer colours."""
    mat = mf.new_material(name)
    nb = mf.NodeBuilder(mat)
    stem = L.get("stem") or am.resolve(L["role"])   # a role, or one asset directly (asset survey)
    s = mf.SurfaceSample(nb, _with_h(am, stem), am.tier, nb.uv("UVMap"), seed=seed)
    bsdf = mf.principled(nb, _adjust(nb, s.color, L.get("adjust")), s.rough, mf.bump(nb, s.height, 0.9, 0.05), specular=0.35)
    mf.output(nb, bsdf.outputs[0])
    return mat


def _field_overlay(nb, plan, plan_dir, uv, color, rough, height):
    """Field plots from the planner raster (R state, G row angle/pi, B mask):
    ploughed = dark furrowed soil, stubble = light straw with combine tracks,
    fallow = pale weedy ground. Row geometry is computed in world metres."""
    f = plan["terrain"].get("fields")
    if not f:
        return color, rough, height
    img = bpy.data.images.load(project_path(f"{plan_dir}/{f}"), check_existing=True)
    img.colorspace_settings.name = "Non-Color"
    t = nb.tex(img, nb.uv("SplatUV"), interpolation="Closest")
    t.extension = "CLIP"
    sep = nb.node("ShaderNodeSeparateColor")
    nb.link(t.outputs["Color"], sep.inputs[0])
    state, ang_n, mask = sep.outputs[0], sep.outputs[1], sep.outputs[2]
    ang = nb.math("MULTIPLY", ang_n, 3.14159265)
    uvs = nb.node("ShaderNodeSeparateXYZ")
    nb.link(uv, uvs.inputs[0])
    across = nb.math("SUBTRACT", nb.math("MULTIPLY", uvs.outputs["Y"], nb.math("COSINE", ang)), nb.math("MULTIPLY", uvs.outputs["X"], nb.math("SINE", ang)))
    is_st = nb.math("MULTIPLY", mask, nb.map_range(nb.math("ABSOLUTE", nb.math("SUBTRACT", state, 0.333)), 0.1, 0.05))
    is_pl = nb.math("MULTIPLY", mask, nb.map_range(nb.math("ABSOLUTE", nb.math("SUBTRACT", state, 0.667)), 0.1, 0.05))
    is_fa = nb.math("MULTIPLY", mask, nb.map_range(state, 0.9, 0.95))
    wob = nb.math("MULTIPLY", nb.math("SUBTRACT", nb.noise(nb.scale_vec(uv, 1 / 12.0), 1.0, detail=2).outputs["Fac"], 0.5), 0.6)
    # ploughed: furrows every 0.9 m across the row direction
    furrow = nb.math("ADD", nb.math("MULTIPLY", nb.math("SINE", nb.math("MULTIPLY", nb.math("ADD", across, wob), 6.98)), 0.5), 0.5)
    ploughed = nb.mix(1.0, color, (0.62, 0.55, 0.5, 1.0), "RGBA", "MULTIPLY")
    ploughed = nb.mix(1.0, ploughed, nb.math("ADD", 0.82, nb.math("MULTIPLY", furrow, 0.25)), "RGBA", "MULTIPLY")
    # stubble: bright straw, darker combine tracks every ~6 m along the rows
    track = nb.map_range(nb.math("ABSOLUTE", nb.math("SUBTRACT", nb.math("FRACT", nb.math("DIVIDE", nb.math("ADD", across, wob), 6.0)), 0.5)), 0.04, 0.0)
    stubble = nb.mix(1.0, color, (1.28, 1.2, 1.0, 1.0), "RGBA", "MULTIPLY")
    stubble = nb.mix(nb.math("MULTIPLY", track, 0.25), stubble, (0.45, 0.4, 0.3, 1.0))
    fallow = nb.mix(1.0, color, (1.12, 1.1, 0.98, 1.0), "RGBA", "MULTIPLY")
    color = nb.mix(is_pl, color, ploughed)
    color = nb.mix(is_st, color, stubble)
    color = nb.mix(is_fa, color, fallow)
    height = nb.mix(is_pl, height, nb.math("ADD", nb.math("MULTIPLY", height, 0.4), nb.math("MULTIPLY", furrow, 0.6)), "FLOAT")
    return color, rough, height


def _splat_channels(nb, plan, plan_dir):
    """R, G, B, A of the core splat image; CLIP extension -> 0 outside the core."""
    img = bpy.data.images.load(project_path(f"{plan_dir}/{plan['terrain']['splat'][0]}"), check_existing=True)
    img.colorspace_settings.name = "Non-Color"
    t = nb.tex(img, nb.uv("SplatUV"))
    t.extension = "CLIP"
    sep = nb.node("ShaderNodeSeparateColor")
    nb.link(t.outputs["Color"], sep.inputs[0])
    return [sep.outputs[0], sep.outputs[1], sep.outputs[2], t.outputs["Alpha"]]


def _snoise(nb, uv, m, seed, detail=4):
    """Noise of feature size m (metres), stretched to about -0.5..0.5."""
    return nb.map_range(nb.noise_m(uv, m, seed=seed, detail=detail, roughness=0.6), 0.32, 0.68, -0.5, 0.5)


def _tone(nb, uv, color, tone, k):
    """In-layer tonal variation: `fine_m`/`fine_amp` = tuft shading (value),
    `patch_m`/`patch_amp` = greyer vs more saturated patches (saturation and value)."""
    if not tone:
        return color
    hsv = nb.node("ShaderNodeHueSaturation")
    nb.link(color, hsv.inputs["Color"])
    v = nb.math("ADD", 1.0, nb.math("MULTIPLY", _snoise(nb, uv, tone["fine_m"], 71.0 + k, 3), 2 * tone["fine_amp"]))
    p = _snoise(nb, uv, tone["patch_m"], 81.0 + k, 4)
    v = nb.math("MULTIPLY", v, nb.math("ADD", 1.0, nb.math("MULTIPLY", p, tone["patch_amp"])))
    nb.link(nb.math("ADD", 1.0, nb.math("MULTIPLY", p, 2 * tone["patch_amp"])), hsv.inputs["Saturation"])
    nb.link(v, hsv.inputs["Value"])
    return hsv.outputs["Color"]


def _adjust(nb, color, adj):
    """Seasonal colour correction of a scanned layer (config `layer_adjust`)."""
    if not adj:
        return color
    hsv = nb.node("ShaderNodeHueSaturation")
    nb.link(color, hsv.inputs["Color"])
    hsv.inputs["Hue"].default_value = adj.get("hue", 0.5)
    hsv.inputs["Saturation"].default_value = adj.get("saturation", 1.0)
    hsv.inputs["Value"].default_value = adj.get("value", 1.0)
    return hsv.outputs["Color"]


def _with_h(am, stem):
    a = dict(am.asset(stem))
    a["height_m"] = a.get("height_m") or 0.03
    am.used[stem] = a.get("version")
    return a
