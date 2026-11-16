"""Aerial camera geometry (no bpy). Units: metres, millimetres for sensor/focal.

Conventions
-----------
tilt_deg     angle between optical axis and nadir (0 = straight down, 90 = horizontal)
heading_deg  compass direction the camera looks toward (0 = north/+Y, 90 = east/+X)
GSD is defined at the principal point (image centre) on flat ground at the target.
"""
import math

import numpy as np


def pixel_pitch_mm(sensor_width_mm, res_x):
    return sensor_width_mm / res_x


def slant_range_for_gsd(gsd_m, focal_mm, sensor_width_mm, res_x):
    """Distance camera->ground along the optical axis that gives `gsd_m`
    across-track at the image centre."""
    return gsd_m * focal_mm / pixel_pitch_mm(sensor_width_mm, res_x)


def altitude_for_gsd(gsd_m, focal_mm, sensor_width_mm, res_x, tilt_deg=0.0):
    r = slant_range_for_gsd(gsd_m, focal_mm, sensor_width_mm, res_x)
    return r * math.cos(math.radians(tilt_deg))


def gsd_at_centre(height_m, focal_mm, sensor_width_mm, res_x, tilt_deg=0.0):
    """Returns (across_track, along_track) GSD at the principal point."""
    t = math.radians(tilt_deg)
    r = height_m / max(math.cos(t), 1e-6)
    across = r * pixel_pitch_mm(sensor_width_mm, res_x) / focal_mm
    return across, across / max(math.cos(t), 1e-6)


def intrinsics(focal_mm, sensor_width_mm, res_x, res_y):
    """OpenCV K matrix (square pixels, principal point at image centre)."""
    f_px = focal_mm / pixel_pitch_mm(sensor_width_mm, res_x)
    return np.array([[f_px, 0.0, res_x / 2.0], [0.0, f_px, res_y / 2.0], [0.0, 0.0, 1.0]])


def camera_pose(target_xyz, slant_range_m, tilt_deg, heading_deg):
    """Camera position looking at `target_xyz` with given tilt and heading.

    Returns (position, blender_euler_xyz_radians). A Blender camera looks down
    its local -Z with +Y up; rotation X = tilt, Z = -heading gives the right view.
    """
    t = math.radians(tilt_deg)
    h = math.radians(heading_deg)
    look = np.array([math.sin(t) * math.sin(h), math.sin(t) * math.cos(h), -math.cos(t)])
    pos = np.asarray(target_xyz, float) - look * slant_range_m
    euler = (t, 0.0, -h)
    return pos, euler


def footprint_halfwidth(slant_range_m, focal_mm, sensor_width_mm):
    return slant_range_m * (sensor_width_mm / 2.0) / focal_mm
