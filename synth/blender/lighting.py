"""Sun lamp + physically based sky with a calibrated diffuse/direct ratio.

Sky Texture `sun_rotation` equals the compass azimuth (0 = north/+Y,
90 deg = east/+X); verified with an equirectangular test render in 5.2.
The sky's own sun disc is disabled and a Sun lamp carries direct light, so
shadows are sharp and controllable. The sky strength is calibrated by
rendering a small white plane lit by the sky alone, so that diffuse sky
irradiance / direct sun irradiance on flat ground equals `sky_ratio`.
"""
import math

import bpy
import numpy as np
from mathutils import Vector

SUN_STRENGTH = 3.2   # relative units; with AgX + exposure 0 gives correct mid-tones for arid ground


def sun_vector(el_deg, az_deg):
    el, az = math.radians(el_deg), math.radians(az_deg)
    return Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))


def _world(scene, sun, atmo):
    w = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = w
    nt = w.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = False
    sky.sun_elevation = math.radians(sun["elevation_deg"])
    sky.sun_rotation = math.radians(sun["azimuth_deg"])
    sky.altitude = float(atmo["altitude_m"])
    sky.air_density = float(atmo["air_density"])
    sky.aerosol_density = float(atmo["aerosol_density"])
    sky.ozone_density = float(atmo["ozone_density"])
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(sky.outputs[0], bg.inputs[0])
    nt.links.new(bg.outputs[0], out.inputs[0])
    return bg


def _measure_sky_irradiance(sun, atmo):
    """Mean radiance of a white Lambertian plane lit by the sky only (strength 1)."""
    sc = bpy.data.scenes.new("_sky_calib")
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 64
    sc.cycles.use_denoising = False
    sc.render.resolution_x = sc.render.resolution_y = 16
    sc.view_settings.view_transform = "Standard"
    bg = _world(sc, sun, atmo)
    bg.inputs[1].default_value = 1.0
    me = bpy.data.meshes.new("_calib_plane")
    me.from_pydata([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("_calib_plane", me)
    sc.collection.objects.link(ob)
    m = bpy.data.materials.new("_calib_white")
    m.use_nodes = True
    m.node_tree.nodes.clear()
    dif = m.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
    dif.inputs["Color"].default_value = (1, 1, 1, 1)
    mo = m.node_tree.nodes.new("ShaderNodeOutputMaterial")
    m.node_tree.links.new(dif.outputs[0], mo.inputs["Surface"])
    me.materials.append(m)
    cd = bpy.data.cameras.new("_calib_cam")
    cd.type = "ORTHO"
    cd.ortho_scale = 10
    cam = bpy.data.objects.new("_calib_cam", cd)
    cam.location = (0, 0, 5)
    sc.collection.objects.link(cam)
    sc.camera = cam
    # render into the scene's result and read pixels
    win_scene = bpy.context.window.scene if bpy.context.window else None
    with bpy.context.temp_override(scene=sc):
        bpy.ops.render.render(scene=sc.name)
    img = bpy.data.images["Render Result"]
    path = bpy.app.tempdir + "sky_calib.exr"
    sc.render.image_settings.file_format = "OPEN_EXR"
    img.save_render(path, scene=sc)
    im = bpy.data.images.load(path, check_existing=False)
    px = np.array(im.pixels[:]).reshape(-1, 4)[:, :3].mean()
    bpy.data.images.remove(im)
    for o in (ob, cam):
        bpy.data.objects.remove(o)
    bpy.data.meshes.remove(me)
    bpy.data.cameras.remove(cd)
    bpy.data.materials.remove(m)
    w = sc.world
    bpy.data.scenes.remove(sc)
    if w and w.users == 0:
        bpy.data.worlds.remove(w)
    return float(px) * math.pi  # irradiance E = pi * L for a white Lambertian plane


def setup_lighting(scene, plan, r):
    sun = plan["sun"]
    atmo = {"altitude_m": max(0.0, plan["dtm"]["origin_utm"][2] + 300.0),
            "air_density": 1.0,
            "aerosol_density": float(r.uniform(1.0, 3.0)),   # Levant summer: dusty
            "ozone_density": 1.0,
            "sky_ratio": float(r.uniform(0.14, 0.24)),
            "sun_angle_deg": float(r.uniform(0.53, 0.9))}
    e_sky = _measure_sky_irradiance(sun, atmo)
    bg = _world(scene, sun, atmo)
    e_sun = SUN_STRENGTH * max(math.sin(math.radians(sun["elevation_deg"])), 0.05)
    k = atmo["sky_ratio"] * e_sun / max(e_sky, 1e-6)
    bg.inputs[1].default_value = k
    ld = bpy.data.lights.new("Sun", "SUN")
    ld.energy = SUN_STRENGTH
    ld.angle = math.radians(atmo["sun_angle_deg"])
    ld.color = (1.0, 0.975, 0.94)
    sob = bpy.data.objects.new("Sun", ld)
    scene.collection.objects.link(sob)
    sob.rotation_euler = (-sun_vector(sun["elevation_deg"], sun["azimuth_deg"])).to_track_quat("-Z", "Y").to_euler()
    atmo.update({"sky_strength": k, "sky_irradiance_unit": e_sky, "sun_strength": SUN_STRENGTH})
    return atmo
