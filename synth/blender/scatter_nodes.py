"""SYN_Scatter: the Geometry Nodes scatter system (runs inside Blender 5.2).

One node group, used by every scatter object (grass, shrubs, wadi shrubs,
stones, boulders, slabs ...). All behaviour is exposed as modifier inputs, so
it can be tuned in Blender's modifier panel or set from Python:

    Terrain (object)      the terrain mesh; reads its point attributes
                          slope_deg, free (0/1 allowed), wet (0..1)
    Assets (collection)   prototype meshes; one is picked at random per point
    Seed, Density (/m2), Min Distance (m)
    Cluster Size (m), Coverage (0..1)        natural patchiness (noise mask)
    Slope Min / Max (deg), Slope Gain (0..1) where it grows, more on slopes
    Wet Boost, Wet Min                       more (or only) in drainage lines
    Scale Min / Max, Power Law               size distribution (many small, few large)
    Align (0..1), Sink (m)                   tilt to the ground, bury depth
    Live Fraction, Live Tint                 part of the instances tinted (live shrubs)

Instances are object references to the prototypes, so in the Object Index
pass they report the prototype's pass_index (verified in 5.2): prototypes
carry the shared "stuff" id of their kind.
"""
import bpy

GROUP = "SYN_Scatter"

INPUTS = [  # (name, socket type, default, min, max)
    ("Terrain", "NodeSocketObject", None, None, None),
    ("Assets", "NodeSocketCollection", None, None, None),
    ("Seed", "NodeSocketInt", 0, 0, 1 << 30),
    ("Density", "NodeSocketFloat", 0.1, 0.0, 20.0),
    ("Min Distance", "NodeSocketFloat", 0.5, 0.0, 50.0),
    ("Cluster Size", "NodeSocketFloat", 20.0, 0.5, 500.0),
    ("Coverage", "NodeSocketFloat", 0.5, 0.0, 1.0),
    ("Slope Min", "NodeSocketFloat", 0.0, 0.0, 90.0),
    ("Slope Max", "NodeSocketFloat", 35.0, 0.0, 90.0),
    ("Slope Gain", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Wet Boost", "NodeSocketFloat", 0.0, 0.0, 20.0),
    ("Wet Min", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Scale Min", "NodeSocketFloat", 1.0, 0.01, 20.0),
    ("Scale Max", "NodeSocketFloat", 1.0, 0.01, 20.0),
    ("Power Law", "NodeSocketFloat", 0.0, 0.0, 4.0),
    ("Align", "NodeSocketFloat", 0.5, 0.0, 1.0),
    ("Sink", "NodeSocketFloat", 0.03, -1.0, 5.0),
    ("Live Fraction", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Live Tint", "NodeSocketColor", (1.0, 1.0, 1.0, 1.0), None, None),
]


class G:
    """Tiny helper for building node trees by code."""

    def __init__(self, ng):
        self.ng, self.N, self.L = ng, ng.nodes, ng.links
        self.x = 0

    def node(self, idn, **props):
        n = self.N.new(idn)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = (self.x, 0)
        self.x += 180
        return n

    def link(self, a, b):
        self.L.new(a, b)

    def math(self, op, a, b=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self._in(n.inputs[0], a)
        if b is not None:
            self._in(n.inputs[1], b)
        return n.outputs[0]

    def mrange(self, v, a, b, c=0.0, d=1.0):
        n = self.node("ShaderNodeMapRange", clamp=True)
        self._in(n.inputs[0], v)
        for i, x in zip((1, 2, 3, 4), (a, b, c, d)):
            self._in(n.inputs[i], x)
        return n.outputs[0]

    def _in(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.link(v, sock)
        elif v is not None:
            sock.default_value = v

    def attr(self, name, dtype="FLOAT"):
        n = self.node("GeometryNodeInputNamedAttribute", data_type=dtype)
        n.inputs["Name"].default_value = name
        return n.outputs["Attribute"]

    def rand(self, dtype, lo=None, hi=None, seed=None, seed_offset=0, prob=None):
        n = self.node("FunctionNodeRandomValue", data_type=dtype)
        ins = [s for s in n.inputs if s.enabled]
        by = {s.name: s for s in ins}
        if lo is not None:
            self._in(by["Min"], lo)
        if hi is not None:
            self._in(by["Max"], hi)
        if prob is not None:
            self._in(by["Probability"], prob)
        self._in(by["Seed"], self.math("ADD", seed, float(seed_offset)) if seed is not None else seed_offset)
        return [s for s in n.outputs if s.enabled][0]


def build_scatter_group():
    if GROUP in bpy.data.node_groups:
        return bpy.data.node_groups[GROUP]
    ng = bpy.data.node_groups.new(GROUP, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for name, st, dv, lo, hi in INPUTS:
        s = ng.interface.new_socket(name, in_out="INPUT", socket_type=st)
        if dv is not None:
            s.default_value = dv
        if lo is not None:
            s.min_value, s.max_value = lo, hi
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    g = G(ng)
    gi = g.node("NodeGroupInput")
    go = g.node("NodeGroupOutput")
    I = {s.name: s for s in gi.outputs}

    terrain = g.node("GeometryNodeObjectInfo", transform_space="ORIGINAL")
    g.link(I["Terrain"], terrain.inputs["Object"])

    # --- where things may grow (a 0..1 density factor on the terrain)
    slope = g.attr("slope_deg")
    free = g.attr("free")
    wet = g.attr("wet")
    s_ok = g.math("MULTIPLY",
                  g.mrange(slope, g.math("SUBTRACT", I["Slope Min"], 3.0), I["Slope Min"]),
                  g.mrange(slope, I["Slope Max"], g.math("ADD", I["Slope Max"], 5.0), 1.0, 0.0))
    steep = g.mrange(slope, 4.0, 30.0)
    s_gain = g.math("ADD", g.math("SUBTRACT", 1.0, I["Slope Gain"]), g.math("MULTIPLY", I["Slope Gain"], steep))
    pos = g.node("GeometryNodeInputPosition").outputs[0]
    vscale = g.node("ShaderNodeVectorMath", operation="DIVIDE")
    g.link(pos, vscale.inputs[0])
    comb = g.node("ShaderNodeCombineXYZ")
    for k in ("X", "Y", "Z"):
        g.link(I["Cluster Size"], comb.inputs[k])
    g.link(comb.outputs[0], vscale.inputs[1])
    # the seed shifts the noise field so each seed gets its own patch pattern
    shift = g.node("ShaderNodeVectorMath", operation="ADD")
    g.link(vscale.outputs[0], shift.inputs[0])
    sv = g.node("ShaderNodeCombineXYZ")
    g.link(g.math("MULTIPLY", g.math("MODULO", I["Seed"], 9973.0), 7.31), sv.inputs["X"])
    g.link(g.math("MULTIPLY", g.math("MODULO", I["Seed"], 7919.0), 3.17), sv.inputs["Y"])
    g.link(sv.outputs[0], shift.inputs[1])
    noise = g.node("ShaderNodeTexNoise")
    g.link(shift.outputs[0], noise.inputs["Vector"])
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 3.0
    thr = g.math("ADD", 0.5, g.math("MULTIPLY", g.math("SUBTRACT", 0.5, I["Coverage"]), 0.32))
    cluster = g.mrange(noise.outputs[0], g.math("SUBTRACT", thr, 0.05), g.math("ADD", thr, 0.05))
    wet_need = g.mrange(wet, I["Wet Min"], g.math("ADD", I["Wet Min"], 0.2))
    wet_need = g.math("MAXIMUM", wet_need, g.math("LESS_THAN", I["Wet Min"], 0.001))
    wet_mul = g.math("ADD", 1.0, g.math("MULTIPLY", wet, I["Wet Boost"]))
    factor = g.math("MULTIPLY", g.math("MULTIPLY", s_ok, s_gain), g.math("MULTIPLY", cluster, free))
    factor = g.math("MULTIPLY", factor, g.math("MULTIPLY", wet_need, wet_mul))
    dens_max = g.math("MULTIPLY", I["Density"], g.math("ADD", 1.0, I["Wet Boost"]))
    factor = g.math("DIVIDE", factor, g.math("ADD", 1.0, I["Wet Boost"]))

    dist = g.node("GeometryNodeDistributePointsOnFaces", distribute_method="POISSON")
    g.link(terrain.outputs["Geometry"], dist.inputs["Mesh"])
    g.link(I["Min Distance"], dist.inputs["Distance Min"])
    g.link(dens_max, dist.inputs["Density Max"])
    g.link(factor, dist.inputs["Density Factor"])
    g.link(I["Seed"], dist.inputs["Seed"])

    # --- orientation: partly aligned to the ground normal, random yaw
    align = g.node("FunctionNodeAlignRotationToVector")
    g.link(dist.outputs["Normal"], align.inputs["Vector"])
    g.link(I["Align"], align.inputs["Factor"])
    axis = g.node("FunctionNodeAxisAngleToRotation")
    axis.inputs["Axis"].default_value = (0, 0, 1)
    g.link(g.rand("FLOAT", 0.0, 6.2831853, I["Seed"], 11), axis.inputs["Angle"])
    rot = g.node("FunctionNodeRotateRotation")
    g.link(align.outputs[0], rot.inputs["Rotation"])
    g.link(axis.outputs[0], rot.inputs["Rotate By"])

    # --- size: min * (max/min) ^ (u ^ (1 + power))
    u = g.rand("FLOAT", 0.0, 1.0, I["Seed"], 23)
    up = g.math("POWER", u, g.math("ADD", 1.0, I["Power Law"]))
    ratio = g.math("DIVIDE", I["Scale Max"], I["Scale Min"])
    size = g.math("MULTIPLY", I["Scale Min"], g.math("POWER", ratio, up))

    assets = g.node("GeometryNodeCollectionInfo", transform_space="ORIGINAL")
    g.link(I["Assets"], assets.inputs["Collection"])
    assets.inputs["Separate Children"].default_value = True
    assets.inputs["Reset Children"].default_value = True
    iop = g.node("GeometryNodeInstanceOnPoints")
    g.link(dist.outputs["Points"], iop.inputs["Points"])
    g.link(assets.outputs[0], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True
    g.link(g.rand("INT", 0, 100000, I["Seed"], 37), iop.inputs["Instance Index"])
    g.link(rot.outputs[0], iop.inputs["Rotation"])
    g.link(size, iop.inputs["Scale"])

    sink = g.node("GeometryNodeTranslateInstances")
    g.link(iop.outputs[0], sink.inputs["Instances"])
    tr = g.node("ShaderNodeCombineXYZ")
    g.link(g.math("MULTIPLY", I["Sink"], -1.0), tr.inputs["Z"])
    g.link(tr.outputs[0], sink.inputs["Translation"])
    sink.inputs["Local Space"].default_value = False

    # --- per-instance tint (live shrubs): offset read by the material as (tint - 1)
    live = g.rand("BOOLEAN", seed=I["Seed"], seed_offset=53, prob=I["Live Fraction"])
    off = g.node("ShaderNodeVectorMath", operation="SUBTRACT")
    g.link(I["Live Tint"], off.inputs[0])
    off.inputs[1].default_value = (1.0, 1.0, 1.0)
    sw = g.node("GeometryNodeSwitch", input_type="VECTOR")
    g.link(live, sw.inputs["Switch"])
    sw.inputs["False"].default_value = (0.0, 0.0, 0.0)
    g.link(off.outputs[0], sw.inputs["True"])
    store = g.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT_VECTOR", domain="INSTANCE")
    store.inputs["Name"].default_value = "tint_off"
    g.link(sink.outputs[0], store.inputs["Geometry"])
    g.link(sw.outputs[0], store.inputs["Value"])
    g.link(store.outputs[0], go.inputs[0])
    return ng


def socket_ids(ng):
    return {it.name: it.identifier for it in ng.interface.items_tree
            if getattr(it, "in_out", None) == "INPUT" and it.item_type == "SOCKET"}


def set_inputs(md, values):
    """Set modifier inputs by name. Blender 5.2: md.properties.inputs.<id>.value
    (the older md['Socket_N'] item access is rejected)."""
    ids = socket_ids(md.node_group)
    for name, v in values.items():
        if name not in ids:
            raise KeyError(f"{md.node_group.name} has no input '{name}'")
        getattr(md.properties.inputs, ids[name]).value = v


def scatter_object(name, collection, terrain, assets, params, pass_index=0):
    ob = bpy.data.objects.new(name, bpy.data.meshes.new(name + "_anchor"))
    collection.objects.link(ob)
    md = ob.modifiers.new("Scatter", "NODES")
    md.node_group = build_scatter_group()
    set_inputs(md, {"Terrain": terrain, "Assets": assets, **params})
    ob.pass_index = pass_index
    return ob
