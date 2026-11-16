"""Cycles configuration, aerial cameras, per-view rendering (runs inside Blender)."""
import math
import os

import bpy


def select_device(metal_kernel_optimization="OFF"):
    """Best available Cycles device: Metal (Mac), OptiX / CUDA / HIP (Linux and
    Windows GPUs), else the CPU (e.g. a GPU-less cloud container). Returns
    "GPU" or "CPU" for scene.cycles.device."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("METAL", "OPTIX", "CUDA", "HIP"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue   # backend not available in this Blender build / OS
        if backend == "METAL" and hasattr(prefs, "kernel_optimization_level"):
            # scene-specialised Metal kernels are compiled in a background thread;
            # in 5.2.2 that thread crashed (nil NSURL in MetalKernelPipeline::compile)
            # during long renders. Generic kernels avoid it at a small speed cost.
            prefs.kernel_optimization_level = metal_kernel_optimization
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        for d in prefs.devices:
            d.use = d.type == backend
        if gpus:
            return "GPU"
    return "CPU"


def configure_cycles(scene, rcfg, seed, use_gpu=True):
    scene.render.engine = "CYCLES"
    c = scene.cycles
    c.device = select_device(rcfg.get("metal_kernel_optimization", "OFF")) if use_gpu else "CPU"
    c.samples = int(rcfg["samples"])
    c.use_adaptive_sampling = True
    c.adaptive_threshold = float(rcfg["adaptive_threshold"])
    c.use_denoising = bool(rcfg["denoise"])
    c.seed = int(seed) % (2 ** 31)
    c.use_animated_seed = False
    c.max_bounces = int(rcfg["max_bounces"])
    c.diffuse_bounces = 3
    c.glossy_bounces = 2
    c.transmission_bounces = 4
    c.transparent_max_bounces = 16
    c.sample_clamp_indirect = 10.0
    c.pixel_filter_type = "BLACKMAN_HARRIS"
    c.filter_width = float(rcfg["filter_width"])
    scene.view_settings.view_transform = rcfg["view_transform"]
    try:
        scene.view_settings.look = rcfg["look"]
    except TypeError:
        scene.view_settings.look = "None"
    scene.view_settings.exposure = float(rcfg["exposure"])
    scene.render.film_transparent = False
    # keep the synced scene / BVH on the device between views (camera-only changes)
    scene.render.use_persistent_data = bool(rcfg.get("persistent_data", True))
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_normal = True
    vl.use_pass_position = True
    vl.use_pass_object_index = True
    vl.use_pass_material_index = True
    vl.pass_alpha_threshold = 0.5   # alpha-cut leaves count as solid only where opaque


def make_cameras(scene, plan, res_override=None):
    cams = []
    for v in plan["cameras"]:
        cam_name = v["name"] if v["name"].startswith("cam_") else f"cam_{v['name']}"
        cd = bpy.data.cameras.new(cam_name)
        cd.lens = v["focal_mm"]
        cd.sensor_fit = "HORIZONTAL"
        cd.sensor_width = v["sensor_width_mm"]
        cd.clip_start = 1.0
        cd.clip_end = 40000.0
        ob = bpy.data.objects.new(cam_name, cd)
        ob.location = v["location"]
        ob.rotation_euler = v["rotation_euler"]
        scene.collection.objects.link(ob)
        cams.append((v, ob))
    return cams


def render_view(scene, cam_ob, resolution, out_dir, name):
    scene.camera = cam_ob
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    bpy.ops.render.render(write_still=False)
    img = bpy.data.images["Render Result"]
    os.makedirs(out_dir, exist_ok=True)
    s = scene.render.image_settings
    # 1) all passes, linear float
    s.media_type = "MULTI_LAYER_IMAGE"
    s.file_format = "OPEN_EXR_MULTILAYER"
    s.color_depth = "32"
    s.exr_codec = "ZIP"
    exr = os.path.join(out_dir, f"{name}_passes.exr")
    img.save_render(exr, scene=scene)
    # 2) display-referred RGB with the scene's view transform
    s.media_type = "IMAGE"
    s.file_format = "PNG"
    s.color_mode = "RGB"
    s.color_depth = "16"
    png = os.path.join(out_dir, f"{name}_render.png")
    img.save_render(png, scene=scene)
    return exr, png


def camera_matrices(scene, cam_ob, resolution):
    """World->camera (OpenCV and Blender conventions) and projection matrix."""
    M = cam_ob.matrix_world
    dg = bpy.context.evaluated_depsgraph_get()
    P = cam_ob.calc_matrix_camera(dg, x=resolution[0], y=resolution[1])
    w2c_blender = M.inverted()
    flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    import numpy as np
    w2c_cv = np.array(flip) @ np.array(w2c_blender)
    return {"cam_to_world_blender": [list(r) for r in M], "world_to_cam_opencv": w2c_cv.tolist(),
            "projection_blender": [list(r) for r in P]}
