"""Build template.blend: the reusable scene logic + prepared assets (runs inside Blender).

    Blender -b --factory-startup -P synth/blender/build_template.py -- [--scene-config configs/scene/galilee_village.json]

The template contains
  * SYN_Scatter         the Geometry Nodes scatter system (grass, shrubs, stones ...)
  * ASSETS_<role>       prototype meshes per scatter role (materials included)
  * _TREE_PROTOTYPES    the scanned pine variants, prepared for instancing
                        (importer groups rebuilt for per-tree labels, twigs thinned)
  * TEMPLATE_DEMO       one example object per scatter system on a small test
                        terrain, so the logic can be explored in Blender directly

Scenes are built by opening this file and filling it with a planned layout
(run_build.py). Open it in Blender, change a node group or an asset, save it,
and every scene generated afterwards uses the change.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from synth.blender import scatter_builder, scatter_nodes, tree_builder  # noqa: E402
from synth.blender.asset_manager import AssetManager  # noqa: E402
from synth.blender.meshutil import collection, grid_faces, mesh_from_arrays  # noqa: E402
from synth.common.config import load_json, load_scene_config, project_path  # noqa: E402


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-config", default="configs/scene/galilee_village.json")
    ap.add_argument("--out", default="templates/template.blend")
    a = ap.parse_args(argv)
    cfg = load_scene_config(a.scene_config)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    am = AssetManager(tier=cfg["render"].get("texture_tier", 1024), missing_policy="error")

    scatter_nodes.build_scatter_group()
    roles = sorted({s["role"] for s in cfg.get("scatter", {}).values()} | {"rock.small_dark"})
    for role in roles:
        rc = scatter_builder.role_collection(am, role)
        print("[template] scatter role", role, len(rc.objects), "prototypes")

    root = tree_builder.tree_prototype_root()
    for name, t in cfg["vegetation"].get("trees", {}).items():
        reg = am.asset(am.resolve(t["role"]))
        for m in reg["models"]:
            if m.lower().endswith(".usd") and "Branches" not in m:
                col = tree_builder._prototype(m, root, t.get("twig_keep", 1.0))
                tree_builder._instancer_group(col)
                print("[template] tree prototype", col.name)

    # a small demo terrain with every scatter system, to explore the logic in Blender
    demo = collection("TEMPLATE_DEMO")
    n, res = 121, 1.0
    c = np.arange(n) * res - 60
    X, Y = np.meshgrid(c, c)
    Z = 2.0 * np.sin(X / 25.0) * np.cos(Y / 31.0) + 0.02 * X
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = grid_faces(n, n)
    gy, gx = np.gradient(Z, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    me = mesh_from_arrays("demo_terrain", verts, faces, uvs={"UVMap": verts[faces.ravel(), :2]},
                          point_attrs={"slope_deg": slope.ravel(), "free": np.ones(n * n), "wet": np.clip((Z.ravel() + 2) / 4, 0, 1)})
    terrain = bpy.data.objects.new("demo_terrain", me)
    terrain.hide_render = True  # playground only; scene builds delete TEMPLATE_DEMO anyway
    demo.objects.link(terrain)
    for name, sc in cfg.get("scatter", {}).items():
        params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in sc.items() if k not in ("role", "kind")}
        ob = scatter_nodes.scatter_object(f"demo_scatter_{name}", demo, terrain, bpy.data.collections[f"ASSETS_{sc['role']}"], params)
        ob.hide_render = True  # the demo never appears in scene renders

    out = project_path(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out, relative_remap=True, compress=True)
    print("[template] saved", out, round(os.path.getsize(out) / 1e6, 1), "MB")


main()
