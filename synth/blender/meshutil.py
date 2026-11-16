"""Bulk mesh construction from numpy arrays (Blender 5.2)."""
import bpy
import numpy as np


def mesh_from_arrays(name, verts, faces, uvs=None, mat_idx=None, smooth=True, point_attrs=None):
    """verts (N,3); faces: (F,k) int array or list of index lists;
    uvs: {uv_name: (L,2) per-loop array in loop order}; mat_idx: (F,)"""
    verts = np.asarray(verts, np.float32)
    if isinstance(faces, np.ndarray):
        sizes = np.full(len(faces), faces.shape[1], np.int32)
        loops = faces.astype(np.int32).ravel()
    else:
        sizes = np.fromiter((len(f) for f in faces), np.int32, len(faces))
        loops = np.fromiter((i for f in faces for i in f), np.int32, int(sizes.sum()))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(len(loops))
    me.loops.foreach_set("vertex_index", loops)
    me.polygons.add(len(sizes))
    me.polygons.foreach_set("loop_start", starts)
    if mat_idx is not None:
        me.polygons.foreach_set("material_index", np.asarray(mat_idx, np.int32))
    for uv_name, arr in (uvs or {}).items():
        layer = me.uv_layers.new(name=uv_name)
        layer.data.foreach_set("uv", np.asarray(arr, np.float32).ravel())
    for attr_name, arr in (point_attrs or {}).items():
        at = me.attributes.new(attr_name, "FLOAT", "POINT")
        at.data.foreach_set("value", np.asarray(arr, np.float32).ravel())
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.full(len(sizes), bool(smooth)))
    me.validate(clean_customdata=False)
    return me


def grid_faces(nr, nc, keep=None):
    """Quad faces of an nr x nc vertex grid (row-major). keep: (nr-1, nc-1) bool."""
    i, j = np.mgrid[0:nr - 1, 0:nc - 1]
    a = i * nc + j
    f = np.stack([a, a + 1, a + nc + 1, a + nc], -1)
    if keep is not None:
        f = f[keep]
    return f.reshape(-1, 4)


def per_loop(vert_attr, faces):
    """Expand a per-vertex attribute (N,k) into per-loop order for quad faces."""
    return vert_attr[faces.ravel()]


def new_object(name, mesh, collection, pass_index=0, props=None):
    ob = bpy.data.objects.new(name, mesh)
    ob.pass_index = pass_index
    for k, v in (props or {}).items():
        ob[k] = v
    collection.objects.link(ob)
    return ob


def collection(name, parent=None):
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if col.name not in parent.children:
        parent.children.link(col)
    return col
