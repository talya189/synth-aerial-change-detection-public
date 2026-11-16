"""Scanned USD trees placed as Geometry-Nodes instancers (runs inside Blender).

Each tree variant is imported once into a prototype collection that is
excluded from the view layer. Every planned tree is a tiny object whose GN
modifier instances that collection. Verified in Blender 5.2: geometry
instanced by a GN modifier reports the *owner's* pass_index in the Object
Index pass, so each tree gets its own instance id while all geometry stays
shared (one Aleppo pine expands to ~120 M faces, but costs ~1 GB once).
"""
import bpy

from synth.blender import material_factory as mf
from synth.common.config import project_path

_PROTOS = {}
_GROUPS = {}


def _instancer_group(proto):
    """One GN group per tree variant. Each top-level prototype object enters as
    Object Info *geometry* (not an object reference), transformed into place.
    Verified in 5.2: GN geometry reports the owning object's pass_index, while
    object references (Collection Info) keep the prototype's own index."""
    if proto.name in _GROUPS:
        return _GROUPS[proto.name]
    if f"TreeInstance_{proto.name}" in bpy.data.node_groups:
        _GROUPS[proto.name] = bpy.data.node_groups[f"TreeInstance_{proto.name}"]
        return _GROUPS[proto.name]
    ng = bpy.data.node_groups.new(f"TreeInstance_{proto.name}", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    ng.nodes.new("NodeGroupInput")
    go = ng.nodes.new("NodeGroupOutput")
    join = ng.nodes.new("GeometryNodeJoinGeometry")
    for ob in proto.objects:
        if ob.type not in ("MESH", "POINTCLOUD"):
            continue
        oi = ng.nodes.new("GeometryNodeObjectInfo")
        oi.transform_space = "ORIGINAL"
        oi.inputs["Object"].default_value = ob
        # wrap as an instance before transforming: transforming the instance keeps the
        # mesh data implicitly shared across all trees (transforming the geometry
        # itself would copy the whole tree per object and exhaust memory)
        g2i = ng.nodes.new("GeometryNodeGeometryToInstance")
        ng.links.new(oi.outputs["Geometry"], g2i.inputs[0])
        tr = ng.nodes.new("GeometryNodeTransform")
        ng.links.new(g2i.outputs[0], tr.inputs["Geometry"])
        for k in ("Location", "Rotation", "Scale"):
            ng.links.new(oi.outputs[k], tr.inputs["Translation" if k == "Location" else k])
        ng.links.new(tr.outputs[0], join.inputs[0])
    ng.links.new(join.outputs[0], go.inputs[0])
    _GROUPS[proto.name] = ng
    return ng


def _realize_object_refs(ng, socket):
    """Collection Info -> geometry: depth 2 realizes the collection instance and
    its object references (verified), so they inherit the owner's pass_index."""
    ri = ng.nodes.new("GeometryNodeRealizeInstances")
    ri.inputs["Realize All"].default_value = False
    ri.inputs["Depth"].default_value = 2
    ng.links.new(socket, ri.inputs["Geometry"])
    return ri.outputs[0]


def _patch_importer_groups(groups, twig_keep=1.0):
    """The USD importer instances prototype *collections* by `Pick Instance`
    (Collection Info with Separate Children, chosen per point by `proto_index`).
    Object references keep the prototype's own pass_index (0), so each tree
    would lose its id. Rebuild each group equivalently: every prototype child
    (alphabetical, the Separate Children order) is realized once to geometry and
    instanced on the points whose proto_index matches. Placements are identical;
    per-point instancing (the expensive part) stays shared."""
    for ng in groups:
        cis = [n for n in ng.nodes if n.bl_idname == "GeometryNodeCollectionInfo"]
        iops = [n for n in ng.nodes if n.bl_idname == "GeometryNodeInstanceOnPoints"]
        if len(cis) != 1 or len(iops) != 1:
            raise RuntimeError(f"unexpected USD instancer layout in {ng.name}")
        ci, iop = cis[0], iops[0]
        src = lambda name: iop.inputs[name].links[0].from_socket if iop.inputs[name].is_linked else None
        points, sel, idx, rot, scl = src("Points"), src("Selection"), src("Instance Index"), src("Rotation"), src("Scale")
        out_sock = iop.outputs[0].links[0].to_socket
        col = ci.inputs["Collection"].default_value
        children = sorted(list(col.children) + list(col.objects), key=lambda c: c.name)
        join = ng.nodes.new("GeometryNodeJoinGeometry")
        keep_sel, scl_out = None, scl
        if twig_keep < 0.999:
            # thin the twig placements (Cycles syncs every instance on every render)
            # and enlarge the kept ones so the canopy keeps about the same cover
            rv = ng.nodes.new("FunctionNodeRandomValue")
            rv.data_type = "BOOLEAN"
            [x for x in rv.inputs if x.enabled and x.name == "Probability"][0].default_value = twig_keep
            [x for x in rv.inputs if x.enabled and x.name == "Seed"][0].default_value = 7
            keep_sel = [x for x in rv.outputs if x.enabled][0]
            if scl is not None:
                vm = ng.nodes.new("ShaderNodeVectorMath")
                vm.operation = "SCALE"
                ng.links.new(scl, vm.inputs[0])
                vm.inputs["Scale"].default_value = twig_keep ** -0.4
                scl_out = vm.outputs[0]
        for k, child in enumerate(children):
            if isinstance(child, bpy.types.Collection):
                info = ng.nodes.new("GeometryNodeCollectionInfo")
                info.inputs["Collection"].default_value = child
            else:
                info = ng.nodes.new("GeometryNodeObjectInfo")
                info.inputs["Object"].default_value = child
            info.transform_space = "ORIGINAL"
            geo = _realize_object_refs(ng, info.outputs["Instances" if isinstance(child, bpy.types.Collection) else "Geometry"])
            cmp = ng.nodes.new("FunctionNodeCompare")
            cmp.data_type = "INT"
            cmp.operation = "EQUAL"
            a_in, b_in = [i for i in cmp.inputs if i.name in ("A", "B") and i.type == "INT"][:2]
            ng.links.new(idx, a_in)
            b_in.default_value = k
            pick = cmp.outputs[0]
            if keep_sel is not None:
                bk = ng.nodes.new("FunctionNodeBooleanMath")
                bk.operation = "AND"
                ng.links.new(keep_sel, bk.inputs[0])
                ng.links.new(pick, bk.inputs[1])
                pick = bk.outputs[0]
            if sel is not None:
                band = ng.nodes.new("FunctionNodeBooleanMath")
                band.operation = "AND"
                ng.links.new(sel, band.inputs[0])
                ng.links.new(pick, band.inputs[1])
                pick = band.outputs[0]
            new = ng.nodes.new("GeometryNodeInstanceOnPoints")
            ng.links.new(points, new.inputs["Points"])
            ng.links.new(pick, new.inputs["Selection"])
            ng.links.new(geo, new.inputs["Instance"])
            if rot is not None:
                ng.links.new(rot, new.inputs["Rotation"])
            if scl_out is not None:
                ng.links.new(scl_out, new.inputs["Scale"])
            ng.links.new(new.outputs[0], join.inputs[0])
        ng.nodes.remove(iop)
        ng.nodes.remove(ci)
        ng.links.new(join.outputs[0], out_sock)


def _improve_materials(objs, seen):
    """The USD ships flat colours only. Add per-tree and per-cluster value/hue
    variation, needle translucency and a bark noise, keeping its base colours."""
    for ob in objs:
        for slot in getattr(ob, "material_slots", []):
            m = slot.material
            if not m or m.name in seen:
                continue
            seen.add(m.name)
            bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            base = tuple(bsdf.inputs["Base Color"].default_value)
            foliage = base[1] > base[0]  # green vs brown bark
            mat = mf.new_material(m.name)
            nb = mf.NodeBuilder(mat)
            oi = nb.node("ShaderNodeObjectInfo")
            tc = nb.node("ShaderNodeTexCoord")
            fine = nb.noise(tc.outputs["Object"], 3.0 if foliage else 12.0, detail=4).outputs["Fac"]
            hsv = nb.node("ShaderNodeHueSaturation")
            hsv.inputs["Color"].default_value = base
            nb.link(nb.map_range(oi.outputs["Random"], 0, 1, 0.485, 0.515), hsv.inputs["Hue"])
            nb.link(nb.map_range(nb.math("ADD", oi.outputs["Random"], fine), 0.3, 1.7, 0.75, 1.3), hsv.inputs["Value"])
            nb.link(nb.map_range(oi.outputs["Random"], 0, 1, 0.8, 1.1), hsv.inputs["Saturation"])
            if foliage:
                b = mf.principled(nb, hsv.outputs["Color"], 0.55, specular=0.4)
                tl = nb.node("ShaderNodeBsdfTranslucent")
                nb.link(nb.mix(1.0, hsv.outputs["Color"], (1.4, 1.5, 0.9, 1), "RGBA", "MULTIPLY"), tl.inputs["Color"])
                mix = nb.node("ShaderNodeMixShader")
                mix.inputs[0].default_value = 0.25
                nb.link(b.outputs[0], mix.inputs[1])
                nb.link(tl.outputs[0], mix.inputs[2])
                mf.output(nb, mix.outputs[0])
            else:
                b = mf.principled(nb, hsv.outputs["Color"], 0.85, mf.bump(nb, fine, 0.6, 0.02), specular=0.3)
                mf.output(nb, b.outputs[0])
            m.user_remap(mat)


def _prototype(path, root, twig_keep=1.0):
    if path in _PROTOS:
        return _PROTOS[path]
    cname = f"PROTO_{bpy.path.display_name_from_filepath(path)}"
    if cname in bpy.data.collections:  # already prepared in the template
        _PROTOS[path] = bpy.data.collections[cname]
        return _PROTOS[path]
    col = bpy.data.collections.new(cname)
    root.children.link(col)
    before = set(bpy.data.objects)
    groups_before = set(bpy.data.node_groups)
    bpy.ops.wm.usd_import(filepath=project_path(path))
    new_groups = [g for g in bpy.data.node_groups if g not in groups_before]
    new_objs = [o for o in bpy.data.objects if o not in before]
    # the tree-level group is the one on the tree's own instancer ("Tree_..._Instancer");
    # only that level (twig placement on the tree) is thinned
    top = {m.node_group for o in new_objs if o.name.startswith("Tree_") and o.name.endswith("_Instancer")
           for m in o.modifiers if m.type == "NODES" and m.node_group}
    _patch_importer_groups([g for g in new_groups if g in top], twig_keep)
    _patch_importer_groups([g for g in new_groups if g not in top])
    new = [o for o in bpy.data.objects if o not in before]
    scene_col = bpy.context.scene.collection
    for o in new:
        if o.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(o)
            continue
        # move only top-level objects; the importer's own proto_* collections hold
        # the twig prototypes its point-cloud instancers reference, so leave them
        if scene_col in o.users_collection:
            scene_col.objects.unlink(o)
            col.objects.link(o)
    _improve_materials(new, set())
    _PROTOS[path] = col
    return col


def tree_prototype_root():
    root = bpy.data.collections.get("_TREE_PROTOTYPES")
    if root is None:
        root = bpy.data.collections.new("_TREE_PROTOTYPES")
    if root.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(root)
    bpy.context.view_layer.layer_collection.children[root.name].exclude = True
    return root


def build_trees(plan, col, ids):
    trees = [v for v in plan["vegetation"] if v["type"] == "usd_tree"]
    if not trees:
        return 0
    root = tree_prototype_root()
    anchor = bpy.data.meshes.new("tree_anchor")  # empty mesh: the GN output replaces it
    for t in trees:
        proto = _prototype(t["variant"], root, t.get("twig_keep", 1.0))
        ob = bpy.data.objects.new(t["id"], anchor)
        md = ob.modifiers.new("tree", "NODES")
        md.node_group = _instancer_group(proto)
        ob.location = (t["x"], t["y"], t["z"])
        ob.rotation_euler = tuple(t["rot"])
        ob.scale = (t["scale"],) * 3
        ob.pass_index = ids.new(t["kind"], t.get("label_id", t["id"]))
        ob["kind"] = t["kind"]
        col.objects.link(ob)
    return len(trees)
