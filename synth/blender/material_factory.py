"""PBR material construction (runs inside Blender 5.2).

Conventions
- Base colour / translucency: sRGB. Everything else: Non-Color.
- Gloss maps are inverted into roughness.
- Procedural meshes carry a UV map in METRES ("UVMap"), so a surface of real
  size S metres is mapped with scale 1/S and is physically sized everywhere.
- Tiling surfaces are sampled twice (second sample rotated, scaled, offset) and
  blended with low-frequency noise, which breaks visible repetition.
- Height maps drive Bump (rotation-invariant), so anti-tiling stays correct.
"""
import os

import bpy

from synth.common.config import project_path

_IMG_CACHE = {}


def image(path, non_color):
    path = project_path(path)
    key = (path, non_color)
    if key not in _IMG_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"texture missing: {path}")
        img = bpy.data.images.load(path, check_existing=True)
        img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
        _IMG_CACHE[key] = img
    return _IMG_CACHE[key]


def map_path(asset, key, tier):
    m = asset["maps"].get(key)
    if not m:
        return None
    return m.get(str(tier)) or m["full"]


class NodeBuilder:
    def __init__(self, mat):
        self.mat = mat
        self.nt = mat.node_tree
        self.n = self.nt.nodes
        self.l = self.nt.links
        self._x = -1400

    def node(self, kind, **props):
        nd = self.n.new(kind)
        for k, v in props.items():
            setattr(nd, k, v)
        nd.location = (self._x, len(self.n) * -40 % 1200)
        self._x += 6
        return nd

    def link(self, a, b):
        self.l.new(a, b)

    def value(self, v):
        nd = self.node("ShaderNodeValue")
        nd.outputs[0].default_value = v
        return nd.outputs[0]

    def math(self, op, a, b=None, clamp=False):
        nd = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self._in(nd.inputs[0], a)
        if b is not None:
            self._in(nd.inputs[1], b)
        return nd.outputs[0]

    def mix(self, fac, a, b, data_type="RGBA", blend="MIX"):
        nd = self.node("ShaderNodeMix", data_type=data_type, blend_type=blend)
        idx = {"FLOAT": (2, 3), "RGBA": (6, 7), "VECTOR": (4, 5)}[data_type]
        self._in(nd.inputs[0], fac)
        self._in(nd.inputs[idx[0]], a)
        self._in(nd.inputs[idx[1]], b)
        out = {"FLOAT": 0, "VECTOR": 1, "RGBA": 2}[data_type]
        return nd.outputs[out]

    def _in(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.link(v, sock)
        elif v is not None:
            sock.default_value = v

    def tex(self, img, vector, interpolation="Linear"):
        nd = self.node("ShaderNodeTexImage", image=img, interpolation=interpolation)
        self.link(vector, nd.inputs["Vector"])
        return nd

    def uv(self, name="UVMap"):
        nd = self.node("ShaderNodeUVMap")
        nd.uv_map = name
        return nd.outputs[0]

    def scale_vec(self, vec, s):
        nd = self.node("ShaderNodeVectorMath", operation="SCALE")
        self.link(vec, nd.inputs[0])
        self._in(nd.inputs["Scale"], s)
        return nd.outputs[0]

    def mapping(self, vec, loc=(0, 0, 0), rot_z=0.0, scale=1.0):
        nd = self.node("ShaderNodeMapping")
        self.link(vec, nd.inputs["Vector"])
        nd.inputs["Location"].default_value = loc
        nd.inputs["Rotation"].default_value = (0, 0, rot_z)
        nd.inputs["Scale"].default_value = (scale, scale, scale)
        return nd.outputs[0]

    def noise_m(self, uv_m, wavelength_m, seed=0.0, detail=3.0, roughness=0.5):
        """Noise with a feature size in metres; `seed` shifts the pattern (it is
        NOT a scale) so layers using the same wavelength stay uncorrelated."""
        v = self.mapping(uv_m, loc=(seed * 17.31, seed * 31.73, 0), scale=1.0 / wavelength_m)
        return self.noise(v, 1.0, detail, roughness).outputs["Fac"]

    def noise(self, vec, scale, detail=3.0, roughness=0.5):
        nd = self.node("ShaderNodeTexNoise")
        if vec is not None:
            self.link(vec, nd.inputs["Vector"])
        nd.inputs["Scale"].default_value = scale
        nd.inputs["Detail"].default_value = detail
        nd.inputs["Roughness"].default_value = roughness
        return nd

    def map_range(self, v, a, b, c=0.0, d=1.0, clamp=True, interp="LINEAR"):
        nd = self.node("ShaderNodeMapRange", clamp=clamp, interpolation_type=interp)
        self._in(nd.inputs[0], v)
        nd.inputs[1].default_value = a
        nd.inputs[2].default_value = b
        nd.inputs[3].default_value = c
        nd.inputs[4].default_value = d
        return nd.outputs[0]


def new_material(name):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat


class SurfaceSample:
    """Anti-tiled sample of all maps of one surface asset.
    Exposes sockets: color, rough, height (0..1), and asset height in metres."""

    def __init__(self, nb, asset, tier, uv_m, anti_tile=True, seed=0.0, cell_m=None):
        self.asset = asset
        size = (asset.get("real_size_m") or [2.0, 2.0])[0]
        base = nb.mapping(uv_m, loc=(seed * 3.17, seed * 1.91, 0), scale=1.0 / size)
        coords = [base]
        if anti_tile:
            coords.append(nb.mapping(uv_m, loc=(seed * 7.3 + 0.37, seed * 5.1 + 0.71, 0), rot_z=0.66 + seed, scale=0.83 / size))
            cell = cell_m or size * 3.0
            mask_n = nb.noise(nb.scale_vec(uv_m, 1.0 / cell), 1.0, detail=2.0)
            self.mask = nb.map_range(mask_n.outputs["Fac"], 0.42, 0.58)
        self.color = self._sample(nb, map_path(asset, "base_color", tier), False, coords)
        rough_p = map_path(asset, "roughness", tier)
        if rough_p:
            self.rough = self._sample(nb, rough_p, True, coords, scalar=True)
        else:
            g = self._sample(nb, map_path(asset, "gloss", tier), True, coords, scalar=True)
            self.rough = nb.math("SUBTRACT", 1.0, g)
        h = map_path(asset, "height", tier)
        self.height = self._sample(nb, h, True, coords, scalar=True) if h else nb.value(0.5)
        self.ao = None
        ao = map_path(asset, "ao", tier)
        if ao:
            self.ao = self._sample(nb, ao, True, coords, scalar=True)
        self.height_m = float(asset.get("height_m") or 0.03)

    def _sample(self, nb, path, non_color, coords, scalar=False):
        img = image(path, non_color)
        outs = [nb.tex(img, c).outputs["Color"] for c in coords]
        if len(outs) == 1:
            out = outs[0]
        else:
            out = nb.mix(self.mask, outs[0], outs[1], "RGBA")
        if scalar:
            sep = nb.node("ShaderNodeSeparateColor")
            nb.link(out, sep.inputs[0])
            return sep.outputs[0]
        return out


def principled(nb, color, rough, normal=None, alpha=None, specular=0.5):
    bsdf = nb.node("ShaderNodeBsdfPrincipled")
    nb._in(bsdf.inputs["Base Color"], color)
    nb._in(bsdf.inputs["Roughness"], rough)
    bsdf.inputs["Specular IOR Level"].default_value = specular
    if normal is not None:
        nb.link(normal, bsdf.inputs["Normal"])
    if alpha is not None:
        nb._in(bsdf.inputs["Alpha"], alpha)
    return bsdf


def output(nb, shader):
    out = nb.node("ShaderNodeOutputMaterial")
    nb.link(shader, out.inputs["Surface"])
    return out


def bump(nb, height, strength, distance, normal=None):
    b = nb.node("ShaderNodeBump")
    nb._in(b.inputs["Height"], height)
    b.inputs["Strength"].default_value = strength
    b.inputs["Distance"].default_value = distance
    if normal is not None:
        nb.link(normal, b.inputs["Normal"])
    return b.outputs["Normal"]


def surface_material(name, asset, tier=1024, tint=None, bump_strength=0.8, anti_tile=True,
                     random_variation=0.0, uv_name="UVMap", ao_strength=0.4):
    """Generic tiling surface (roads, walls, roofs, paving). UV must be in metres."""
    mat = new_material(name)
    nb = NodeBuilder(mat)
    s = SurfaceSample(nb, asset, tier, nb.uv(uv_name), anti_tile=anti_tile)
    color = s.color
    if s.ao is not None and ao_strength > 0:
        aof = nb.map_range(s.ao, 0, 1, 1 - ao_strength, 1)
        color = nb.mix(1.0, color, aof, "RGBA", "MULTIPLY")
    if tint is not None:
        color = nb.mix(1.0, color, (*tint, 1.0), "RGBA", "MULTIPLY")
    if random_variation > 0:
        oi = nb.node("ShaderNodeObjectInfo")
        hsv = nb.node("ShaderNodeHueSaturation")
        nb.link(color, hsv.inputs["Color"])
        nb.link(nb.map_range(oi.outputs["Random"], 0, 1, 1 - random_variation, 1 + random_variation * 0.5), hsv.inputs["Value"])
        color = hsv.outputs["Color"]
    n = bump(nb, s.height, bump_strength, s.height_m)
    sh = principled(nb, color, s.rough, n)
    output(nb, sh.outputs[0])
    return mat


def model_material(name, asset, tier=2048, alpha=False, foliage=False, uv_name=None, tint=None,
                   value_jitter=0.0):
    """Material for an imported Megascans model/plant, using its own UVs."""
    mat = new_material(name)
    nb = NodeBuilder(mat)
    uv = nb.uv(uv_name) if uv_name else nb.node("ShaderNodeTexCoord").outputs["UV"]
    img = lambda k, nc: image(map_path(asset, k, tier), nc) if map_path(asset, k, tier) else None
    col = nb.tex(img("base_color", False), uv).outputs["Color"]
    # per-instance colour: object colour (linked duplicates, white by default) and the
    # GN scatter attribute tint_off = tint - 1 (missing attribute reads 0 -> no change)
    col = nb.mix(1.0, col, nb.node("ShaderNodeObjectInfo").outputs["Color"], "RGBA", "MULTIPLY")
    toff = nb.node("ShaderNodeAttribute", attribute_type="INSTANCER", attribute_name="tint_off")
    inst_tint = nb.node("ShaderNodeVectorMath", operation="ADD")
    nb.link(toff.outputs["Vector"], inst_tint.inputs[0])
    inst_tint.inputs[1].default_value = (1.0, 1.0, 1.0)
    col = nb.mix(1.0, col, inst_tint.outputs[0], "RGBA", "MULTIPLY")
    if tint is not None:
        col = nb.mix(1.0, col, (*tint, 1.0), "RGBA", "MULTIPLY")
    if value_jitter > 0:
        oi = nb.node("ShaderNodeObjectInfo")
        hsv = nb.node("ShaderNodeHueSaturation")
        nb.link(col, hsv.inputs["Color"])
        nb.link(nb.map_range(oi.outputs["Random"], 0, 1, 1 - value_jitter, 1 + value_jitter), hsv.inputs["Value"])
        nb.link(nb.map_range(oi.outputs["Random"], 0, 1, 1 - value_jitter, 1 + value_jitter * 0.5), hsv.inputs["Saturation"])
        col = hsv.outputs["Color"]
    if img("roughness", True):
        rough = nb.tex(img("roughness", True), uv).outputs["Color"]
    elif img("gloss", True):
        rough = nb.math("SUBTRACT", 1.0, nb.tex(img("gloss", True), uv).outputs["Color"])
    else:
        rough = 0.8
    normal = None
    if img("normal", True):
        nm = nb.node("ShaderNodeNormalMap")
        if uv_name:
            nm.uv_map = uv_name
        nb.link(nb.tex(img("normal", True), uv).outputs["Color"], nm.inputs["Color"])
        normal = nm.outputs["Normal"]
    alpha_s = None
    if alpha and img("opacity", True):
        alpha_s = nb.tex(img("opacity", True), uv).outputs["Color"]
    bsdf = principled(nb, col, rough, normal, alpha=None if foliage else alpha_s)
    if foliage:
        # thin-leaf look: mix in translucent BSDF driven by the translucency map
        tr_img = img("translucency", False)
        tcol = nb.tex(tr_img, uv).outputs["Color"] if tr_img else col
        tl = nb.node("ShaderNodeBsdfTranslucent")
        nb.link(tcol, tl.inputs["Color"])
        mix = nb.node("ShaderNodeMixShader")
        mix.inputs[0].default_value = 0.3
        nb.link(bsdf.outputs[0], mix.inputs[1])
        nb.link(tl.outputs[0], mix.inputs[2])
        shader = mix.outputs[0]
        if alpha_s is not None:
            tr = nb.node("ShaderNodeBsdfTransparent")
            m2 = nb.node("ShaderNodeMixShader")
            nb.link(alpha_s, m2.inputs[0])
            nb.link(tr.outputs[0], m2.inputs[1])
            nb.link(shader, m2.inputs[2])
            shader = m2.outputs[0]
        output(nb, shader)
    else:
        output(nb, bsdf.outputs[0])
    return mat


def flat_material(name, color, rough=0.6, metallic=0.0, emission=None, specular=0.5):
    mat = new_material(name)
    nb = NodeBuilder(mat)
    b = principled(nb, (*color, 1.0), rough, specular=specular)
    b.inputs["Metallic"].default_value = metallic
    output(nb, b.outputs[0])
    return mat


def dirty_flat_material(name, color, rough=0.6, metallic=0.0, dirt=0.25, noise_scale=6.0):
    """Painted metal / plastic with procedural dust and value variation (rooftop equipment)."""
    mat = new_material(name)
    nb = NodeBuilder(mat)
    oi = nb.node("ShaderNodeObjectInfo")
    tc = nb.node("ShaderNodeTexCoord")
    nz = nb.noise(tc.outputs["Object"], noise_scale, detail=6)
    d = nb.map_range(nz.outputs["Fac"], 0.35, 0.75, 0, dirt)
    jitter = nb.map_range(oi.outputs["Random"], 0, 1, 0.85, 1.1)
    col = nb.mix(d, (*color, 1.0), (0.42, 0.37, 0.31, 1.0), "RGBA")
    hsv = nb.node("ShaderNodeHueSaturation")
    nb.link(col, hsv.inputs["Color"])
    nb.link(jitter, hsv.inputs["Value"])
    b = principled(nb, hsv.outputs["Color"], nb.map_range(d, 0, dirt + 1e-3, rough, min(1, rough + 0.3)))
    b.inputs["Metallic"].default_value = metallic
    output(nb, b.outputs[0])
    return mat
