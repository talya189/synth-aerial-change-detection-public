"""Procedural buildings (runs inside Blender).

Each building is one mesh in its own local frame (x across the street facade,
y away from the street, z absolute height), placed at the footprint centre and
rotated by the planned yaw. Material slots:
  0 wall  1 glass  2 metal frame  3 flat roof  4 plinth  5 concrete
  6 roller shutter  7 door  8 roof tiles  9 corrugated metal roof  10 dark interior
  11 railing metal  12 solar panel  13 wood (pergola)  14 greenhouse film  15 coop curtain
Rooftop equipment is instanced from shared prototype meshes (linked duplicates)
and carries the building's instance id, so instance masks include it.
"""
import math

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

from synth.blender import material_factory as mf
from synth.blender.meshutil import mesh_from_arrays, new_object
from synth.common.rng import rng

WALL, GLASS, FRAME, ROOF, PLINTH, CONCRETE, SHUTTER, DOOR, TILES, METAL, INTERIOR, RAIL, PV, WOOD, FILM, CURTAIN = range(16)


class MeshAcc:
    """Accumulates quads/polygons with per-loop UVs (metres) and material ids."""

    def __init__(self):
        self.v, self.f, self.uv, self.m = [], [], [], []

    def poly(self, pts, uvs, mat):
        base = len(self.v)
        self.v.extend(pts)
        self.f.append(list(range(base, base + len(pts))))
        self.uv.extend(uvs)
        self.m.append(mat)

    def quad_wall(self, A, t, n, u0, u1, z0, z1, depth, mat, flip=False):
        """Vertical quad on a wall plane: A origin, t unit along wall, n outward
        normal, spans u0..u1 along the wall, z0..z1 in height, recessed by depth."""
        P = lambda u, z: (A[0] + t[0] * u - n[0] * depth, A[1] + t[1] * u - n[1] * depth, z)
        pts = [P(u0, z0), P(u1, z0), P(u1, z1), P(u0, z1)]
        uvs = [(u0, z0), (u1, z0), (u1, z1), (u0, z1)]
        if flip:
            pts, uvs = pts[::-1], uvs[::-1]
        self.poly(pts, uvs, mat)

    def side_face(self, A, t, n, u, z0, z1, d0, d1, mat, facing):
        """Reveal (jamb) face perpendicular to the wall at position u."""
        P = lambda d, z: (A[0] + t[0] * u - n[0] * d, A[1] + t[1] * u - n[1] * d, z)
        pts = [P(d0, z0), P(d1, z0), P(d1, z1), P(d0, z1)]
        uvs = [(d0, z0), (d1, z0), (d1, z1), (d0, z1)]
        if facing < 0:
            pts, uvs = pts[::-1], uvs[::-1]
        self.poly(pts, uvs, mat)

    def horiz(self, A, t, n, u0, u1, z, d0, d1, mat, up=True):
        P = lambda u, d: (A[0] + t[0] * u - n[0] * d, A[1] + t[1] * u - n[1] * d, z)
        pts = [P(u0, d0), P(u1, d0), P(u1, d1), P(u0, d1)]
        uvs = [(p[0], p[1]) for p in pts]
        # orientation: pts go along t then inward (-n); normal = t x (-n)
        cross = t[0] * (-n[1]) - t[1] * (-n[0])
        if (cross > 0) != up:
            pts, uvs = pts[::-1], uvs[::-1]
        self.poly(pts, uvs, mat)

    def box(self, x0, y0, z0, x1, y1, z1, mat, top_mat=None, bottom=False):
        c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for k in range(4):
            a, b = c[k], c[(k + 1) % 4]
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            self.poly([(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)],
                      [(0, z0), (L, z0), (L, z1), (0, z1)], mat)
        self.poly([(x, y, z1) for x, y in c], [(x, y) for x, y in c], top_mat if top_mat is not None else mat)
        if bottom:
            self.poly([(x, y, z0) for x, y in c[::-1]], [(x, y) for x, y in c[::-1]], mat)

    def mesh(self, name):
        faces = self.f
        uv = np.asarray(self.uv, float)
        return mesh_from_arrays(name, np.asarray(self.v, float), faces, uvs={"UVMap": uv}, mat_idx=self.m, smooth=False)


def _local_rect(points, centre, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    p = np.asarray(points, float) - centre
    loc = np.column_stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c])
    return loc[:, 0].min(), loc[:, 1].min(), loc[:, 0].max(), loc[:, 1].max()


def _block(acc, r, b, rect, floors, z_floor, z_found, front_door, parapet, is_annex=False):
    """Walls with openings, plinth, roof slab, parapet for one rectangular block."""
    x0, y0, x1, y1 = rect
    fh = b["floor_height"]
    z_roof = z_floor + floors * fh
    open_bays = b.get("window_style") == "open"
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # CCW, front edge first (y = y0 faces the street)
    for k in range(4):
        A, B = np.array(corners[k]), np.array(corners[(k + 1) % 4])
        L = float(np.linalg.norm(B - A))
        t = (B - A) / L
        n = np.array([t[1], -t[0]])  # outward for CCW
        # plinth (stone), slightly proud of the wall
        acc.quad_wall(A, t, n, 0, L, z_found, z_floor, -0.04, PLINTH)
        if b.get("door") == "curtain":   # poultry house: curtain sides, fan wall at one end
            _coop_wall(acc, A, t, n, L, k, z_floor, fh * floors)
            continue
        # windows: rhythm per facade, fewer on side walls
        spacing = b["window_spacing"] * (1.0 if k in (0, 2) else 1.35)
        nwin = max(0, int((L - 1.2) / spacing))
        ww, wh = b["window_w"], b["window_h"]
        us = [(L / 2 + (i - (nwin - 1) / 2) * spacing) for i in range(nwin)] if nwin else []
        for f in range(floors):
            zf = z_floor + f * fh
            openings = []
            for i, uc in enumerate(us):
                er = rng(b["id"], "win", is_annex, k, f, i)
                if er.random() < 0.12:  # blank wall bay
                    continue
                is_door = front_door and k == 0 and f == 0 and i == len(us) // 2
                roller = b.get("door") == "roller"
                if b.get("door") == "strip":   # poultry house: long curtain openings, no door on the long side
                    openings.append((uc - spacing / 2 + 0.3, uc + spacing / 2 - 0.3, 0.6, fh - 0.5, "win", 0.0))
                    continue
                if is_door:
                    dw, dh = (2.0, min(4.2, fh - 0.6)) if roller else (0.55, 2.15)
                    openings.append((uc - dw, uc + dw, 0.0, dh, "door", 0.0))
                elif roller:   # sheds: narrow high windows
                    openings.append((uc - 0.9, uc + 0.9, fh - 1.6, fh - 0.7, "win", 0.0))
                else:
                    w = ww * (0.6 if er.random() < 0.2 else 1.0)  # small bathroom windows
                    sill = b.get("sill", 0.9) if w > 0.8 else 1.45
                    h = wh if w > 0.8 else 0.6
                    shut = er.random() * 0.9 if er.random() < b["shutters_p"] else 0.0
                    openings.append((uc - w / 2, uc + w / 2, sill, sill + h, "win", shut))
            openings = [o for o in openings if o[0] > 0.35 and o[1] < L - 0.35]
            u = 0.0
            for (u0, u1, v0, v1, kind, shut) in sorted(openings):
                acc.quad_wall(A, t, n, u, u0, zf, zf + fh, 0, WALL)  # pier
                if v0 > 0:
                    acc.quad_wall(A, t, n, u0, u1, zf, zf + v0, 0, WALL)
                acc.quad_wall(A, t, n, u0, u1, zf + v1, zf + fh, 0, WALL)
                dep = 0.18
                acc.side_face(A, t, n, u0, zf + v0, zf + v1, 0, dep, WALL, facing=1)
                acc.side_face(A, t, n, u1, zf + v0, zf + v1, 0, dep, WALL, facing=-1)
                acc.horiz(A, t, n, u0, u1, zf + v1, 0, dep, WALL, up=False)
                if v0 > 0:
                    acc.horiz(A, t, n, u0, u1, zf + v0, 0, dep, CONCRETE, up=True)
                if kind == "door":
                    acc.quad_wall(A, t, n, u0, u1, zf + v0, zf + v1, dep,
                                  SHUTTER if b.get("door") == "roller" else (INTERIOR if open_bays else DOOR))
                elif open_bays or b.get("door") == "strip":   # empty openings / curtains: dark interior behind
                    acc.quad_wall(A, t, n, u0, u1, zf + v0, zf + v1, dep + 0.25, INTERIOR)
                else:
                    zs = zf + v1 - (v1 - v0) * shut
                    acc.quad_wall(A, t, n, u0, u1, zf + v0, zs, dep, GLASS)
                    if shut > 0:
                        acc.quad_wall(A, t, n, u0, u1, zs, zf + v1, dep - 0.02, SHUTTER)
                    # shutter box above the window (typical Israeli "trisim")
                    acc.quad_wall(A, t, n, u0 - 0.05, u1 + 0.05, zf + v1, zf + v1 + 0.22, -0.02, FRAME)
                u = u1
            acc.quad_wall(A, t, n, u, L, zf, zf + fh, 0, WALL)
        # parapet outer face (continuous wall), coping, inner face
        if parapet > 0:
            acc.quad_wall(A, t, n, 0, L, z_roof, z_roof + parapet, 0, WALL)
            acc.horiz(A, t, n, 0, L, z_roof + parapet, 0, 0.2, CONCRETE, up=True)
            # inner face (normal pointing inward)
            P0 = A - n * 0.2 + t * 0.2
            acc.quad_wall(P0, t, -n, 0, L - 0.4, z_roof, z_roof + parapet, 0, WALL, flip=True)
    # cantilevered front balconies on upper floors (slab + solid parapet railing)
    bal = b.get("balcony") if not is_annex else None
    if bal:
        w = x1 - x0
        bw = w * bal["width_frac"]
        cx = (x0 + x1) / 2 + w * bal["offset_frac"]
        bx0, bx1 = max(x0 + 0.2, cx - bw / 2), min(x1 - 0.2, cx + bw / 2)
        dpt = bal["depth"]
        for f in range(1, floors):
            zf = z_floor + f * fh
            acc.box(bx0, y0 - dpt, zf - 0.18, bx1, y0, zf, CONCRETE, bottom=True)
            acc.box(bx0, y0 - dpt, zf, bx1, y0 - dpt + 0.12, zf + 1.0, WALL)
            acc.box(bx0, y0 - dpt + 0.12, zf, bx0 + 0.12, y0, zf + 1.0, WALL)
            acc.box(bx1 - 0.12, y0 - dpt + 0.12, zf, bx1, y0, zf + 1.0, WALL)
    # roof slab
    if parapet > 0:
        ins = 0.2
        acc.poly([(x0 + ins, y0 + ins, z_roof), (x1 - ins, y0 + ins, z_roof), (x1 - ins, y1 - ins, z_roof), (x0 + ins, y1 - ins, z_roof)],
                 [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], ROOF)
    elif b["roof"] == "flat":   # no parapet (unfinished): bare slab with a visible edge
        acc.box(x0 - 0.05, y0 - 0.05, z_roof - 0.2, x1 + 0.05, y1 + 0.05, z_roof, CONCRETE, top_mat=ROOF)
    return z_roof


def _coop_wall(acc, A, t, n, L, k, z0, h):
    """Poultry-house wall. Long sides (k 0, 2): knee wall, one continuous curtain,
    thin steel columns. Ends: k 1 = tunnel-ventilation fans, k 3 = service door."""
    if k in (0, 2):
        acc.quad_wall(A, t, n, 0, L, z0, z0 + 0.6, 0, WALL)
        acc.quad_wall(A, t, n, 0, L, z0 + 0.6, z0 + h - 0.35, 0.06, CURTAIN)
        acc.quad_wall(A, t, n, 0, L, z0 + h - 0.35, z0 + h, 0, WALL)
        for u in np.arange(0.0, L + 0.01, 6.0):
            uu = min(max(u, 0.08), L - 0.08)
            acc.quad_wall(A, t, n, uu - 0.08, uu + 0.08, z0 + 0.6, z0 + h - 0.35, -0.02, RAIL)
        return
    acc.quad_wall(A, t, n, 0, L, z0, z0 + h, 0, WALL)
    if k == 1:
        nf = max(1, int((L - 1.0) / 1.7))
        rows = 2 if h > 2.7 else 1
        for i in range(nf):
            uc = L / 2 + (i - (nf - 1) / 2) * 1.7
            for j in range(rows):
                zc = z0 + 0.4 + 0.7 + j * 1.45
                acc.quad_wall(A, t, n, uc - 0.7, uc + 0.7, zc - 0.7, zc + 0.7, -0.03, FRAME)
                acc.quad_wall(A, t, n, uc - 0.58, uc + 0.58, zc - 0.58, zc + 0.58, -0.035, INTERIOR)
    else:
        acc.quad_wall(A, t, n, L / 2 - 0.6, L / 2 + 0.6, z0, z0 + 2.1, -0.03, DOOR)


def curtain_material():
    """Poultry-house side curtain: off-white / yellowish tarp with vertical folds."""
    mat = mf.new_material("M_curtain")
    nb = mf.NodeBuilder(mat)
    uv = nb.uv("UVMap")
    wave = nb.node("ShaderNodeTexWave")
    nb.link(uv, wave.inputs["Vector"])
    wave.inputs["Scale"].default_value = 2.0
    wave.inputs["Distortion"].default_value = 1.5
    dust = nb.map_range(nb.noise_m(uv, 2.5, seed=4.0, detail=4), 0.4, 0.7, 0.0, 0.35)
    color = nb.mix(dust, (0.78, 0.74, 0.6, 1.0), (0.55, 0.5, 0.4, 1.0))
    color = nb.mix(nb.math("MULTIPLY", wave.outputs["Fac"], 0.25), color, (0.4, 0.38, 0.3, 1.0))
    mf.output(nb, mf.principled(nb, color, 0.85, mf.bump(nb, wave.outputs["Fac"], 0.3, 0.02), specular=0.3).outputs[0])
    return mat


def _hip_roof(acc, rect, z, overhang=0.35, pitch_deg=28):
    x0, y0, x1, y1 = rect
    x0, y0, x1, y1 = x0 - overhang, y0 - overhang, x1 + overhang, y1 + overhang
    w, d = x1 - x0, y1 - y0
    h = min(w, d) / 2 * math.tan(math.radians(pitch_deg))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if w >= d:
        r0, r1 = (x0 + d / 2, cy, z + h), (x1 - d / 2, cy, z + h)
    else:
        r0, r1 = (cx, y0 + w / 2, z + h), (cx, y1 - w / 2, z + h)
    c = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    if w >= d:
        faces = [[c[0], c[1], r1, r0], [c[1], c[2], r1], [c[2], c[3], r0, r1], [c[3], c[0], r0]]
    else:
        faces = [[c[0], c[1], r0], [c[1], c[2], r1, r0], [c[2], c[3], r1], [c[3], c[0], r0, r1]]
    for f in faces:
        p0, p1 = np.array(f[0]), np.array(f[1])
        e = (p1 - p0)[:2]
        e /= np.linalg.norm(e)
        up = np.array([-e[1], e[0]])
        uvs = []
        for p in f:
            q = np.array(p)
            run = float(np.dot(q[:2] - p0[:2], up))
            uvs.append((float(np.dot(q[:2] - p0[:2], e)), (run if run > 0 else 0) / math.cos(math.radians(pitch_deg))))
        acc.poly([tuple(p) for p in f], uvs, TILES)
    # soffit
    acc.poly(c[::-1], [(p[0], p[1]) for p in c[::-1]], CONCRETE)


def _gable_roof(acc, rect, z, overhang=0.3, pitch_deg=12, mat=None, pv_dir=None):
    """Gable roof (metal sheds: low pitch; tiled cottages: ~30 deg); ridge along
    the longer side, triangular gable walls closing the ends. `pv_dir` (local
    2-D south) puts a solar panel field on the slope facing it."""
    mat = METAL if mat is None else mat
    x0, y0, x1, y1 = rect
    along_x = (x1 - x0) >= (y1 - y0)
    if not along_x:   # work in a frame where the ridge runs along x
        x0, y0, x1, y1 = y0, x0, y1, x1
    span = (y1 - y0) / 2
    h = span * math.tan(math.radians(pitch_deg))
    yc = (y0 + y1) / 2
    P = (lambda x, y, zz: (x, y, zz)) if along_x else (lambda x, y, zz: (y, x, zz))
    ox0, ox1 = x0 - overhang, x1 + overhang
    drop = overhang * math.tan(math.radians(pitch_deg))
    slope_len = (span + overhang) / math.cos(math.radians(pitch_deg))
    for ya, flip in ((y0 - overhang, False), (y1 + overhang, True)):
        q = [P(ox0, ya, z - drop), P(ox1, ya, z - drop), P(ox1, yc, z + h), P(ox0, yc, z + h)]
        uv = [(0, 0), (ox1 - ox0, 0), (ox1 - ox0, slope_len), (0, slope_len)]
        if flip ^ (not along_x):
            q, uv = q[::-1], uv[::-1]
        acc.poly(q, uv, mat)
        acc.poly(q[::-1], uv[::-1], mat)   # underside
        facing = -1.0 if ya < yc else 1.0   # this slope faces -y / +y in the roof frame
        if pv_dir is not None and facing * (pv_dir[1] if along_x else pv_dir[0]) > 0.2:
            ny = math.sin(math.radians(pitch_deg)) * facing
            nz = math.cos(math.radians(pitch_deg))
            lift = 0.08
            def on(xx, tt):   # point on the slope, tt = 0 eave .. 1 ridge
                yy = ya + (yc - ya) * tt
                return P(xx, yy + ny * lift, (z - drop) + (z + h - (z - drop)) * tt + nz * lift)
            pq = [on(ox0 + 0.6, 0.06), on(ox1 - 0.6, 0.06), on(ox1 - 0.6, 0.94), on(ox0 + 0.6, 0.94)]
            puv = [(0, 0), (ox1 - ox0 - 1.2, 0), (ox1 - ox0 - 1.2, slope_len * 0.88), (0, slope_len * 0.88)]
            if flip ^ (not along_x):
                pq, puv = pq[::-1], puv[::-1]
            acc.poly(pq, puv, PV)
    for xg, flip in ((x0, True), (x1, False)):
        tri = [P(xg, y0, z), P(xg, y1, z), P(xg, yc, z + h)]
        uvt = [(0, z), (y1 - y0, z), (span, z + h)]
        if flip ^ (not along_x):
            tri, uvt = tri[::-1], uvt[::-1]
        acc.poly(tri, uvt, WALL)


def metal_roof_material():
    """Corrugated galvanised / painted sheet: per-building tint (object attr
    `metal_tint`), dust and rust streaks, corrugation as a bump (sub-pixel from
    the air, but it shapes the specular highlight)."""
    mat = mf.new_material("M_roof_metal")
    nb = mf.NodeBuilder(mat)
    uv = nb.uv("UVMap")
    tint = nb.node("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="metal_tint").outputs["Color"]
    sep = nb.node("ShaderNodeSeparateXYZ")
    nb.link(uv, sep.inputs[0])
    corr = nb.math("SINE", nb.math("MULTIPLY", sep.outputs["X"], 2 * math.pi / 0.076))
    dust = nb.map_range(nb.noise_m(uv, 3.0, seed=2.0, detail=5), 0.4, 0.7, 0.0, 0.35)
    rust = nb.map_range(nb.noise_m(uv, 0.8, seed=5.0, detail=6), 0.6, 0.75, 0.0, 0.5)
    color = nb.mix(dust, tint, (0.55, 0.5, 0.42, 1.0))
    color = nb.mix(rust, color, (0.32, 0.17, 0.08, 1.0))
    rough = nb.math("ADD", 0.32, nb.math("MULTIPLY", nb.math("ADD", dust, rust), 0.8))
    bsdf = mf.principled(nb, color, rough, mf.bump(nb, corr, 0.25, 0.02), specular=0.5)
    bsdf.inputs["Metallic"].default_value = 0.55
    mf.output(nb, bsdf.outputs[0])
    return mat


def pv_material():
    """Solar panel: dark blue-black glass with a light cell/frame grid."""
    mat = mf.new_material("M_solar_pv")
    nb = mf.NodeBuilder(mat)
    uv = nb.uv("UVMap")
    br = nb.node("ShaderNodeTexBrick")
    nb.link(uv, br.inputs["Vector"])
    br.inputs["Scale"].default_value = 1.0
    br.inputs["Brick Width"].default_value = 1.0
    br.inputs["Row Height"].default_value = 1.7
    br.inputs["Mortar Size"].default_value = 0.025
    br.offset = 0.0
    br.inputs["Color1"].default_value = (0.02, 0.03, 0.06, 1)
    br.inputs["Color2"].default_value = (0.025, 0.035, 0.07, 1)
    br.inputs["Mortar"].default_value = (0.55, 0.56, 0.58, 1)
    mf.output(nb, mf.principled(nb, br.outputs["Color"], 0.12, None, specular=0.6).outputs[0])
    return mat


def film_material():
    """Greenhouse polyethylene film: milky white, dusty, slightly glossy."""
    mat = mf.new_material("M_greenhouse_film")
    nb = mf.NodeBuilder(mat)
    uv = nb.uv("UVMap")
    dust = nb.map_range(nb.noise_m(uv, 4.0, seed=3.0, detail=4), 0.35, 0.7, 0.0, 0.3)
    color = nb.mix(dust, (0.66, 0.67, 0.65, 1.0), (0.5, 0.47, 0.41, 1.0))
    mf.output(nb, mf.principled(nb, color, 0.35, None, specular=0.5).outputs[0])
    return mat


def local_south(yaw):
    """World south (-y) expressed in a building's local frame."""
    return (-math.sin(yaw), -math.cos(yaw))


def _pv_rows(acc, rect, z, south, tilt_deg=20.0, depth=1.0, pitch=2.1, inset=0.8):
    """Rows of tilted solar panels on a flat roof, facing (roughly) south."""
    x0, y0, x1, y1 = rect[0] + inset, rect[1] + inset, rect[2] - inset, rect[3] - inset
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    along_x = abs(south[1]) >= abs(south[0])   # rows run along x, panels face +-y
    sgn = math.copysign(1.0, south[1] if along_x else south[0])
    c, h = depth * math.cos(math.radians(tilt_deg)), depth * math.sin(math.radians(tilt_deg))
    a0, a1, b0, b1 = (x0, x1, y0, y1) if along_x else (y0, y1, x0, x1)
    P = (lambda a, b, zz: (a, b, zz)) if along_x else (lambda a, b, zz: (b, a, zz))
    bb = b0
    while bb + c <= b1:
        lo, hi = (bb + c, bb) if sgn < 0 else (bb, bb + c)   # low edge toward the sun
        zl = z + 0.35
        q = [P(a0, lo, zl), P(a1, lo, zl), P(a1, hi, zl + h), P(a0, hi, zl + h)]
        uv = [(0, 0), (a1 - a0, 0), (a1 - a0, depth), (0, depth)]
        if (sgn < 0) ^ (not along_x):
            q, uv = q[::-1], uv[::-1]
        acc.poly(q, uv, PV)
        acc.poly(q[::-1], uv[::-1], RAIL)   # back of the panels
        bb += pitch


def _railing(acc, rect, z, h=1.05, post=1.5):
    """Metal railing around a flat roof: posts and a top rail."""
    x0, y0, x1, y1 = rect[0] + 0.08, rect[1] + 0.08, rect[2] - 0.08, rect[3] - 0.08
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for k in range(4):
        A, B = np.array(pts[k]), np.array(pts[(k + 1) % 4])
        L = float(np.linalg.norm(B - A))
        t = (B - A) / L
        for u in np.linspace(0, L, max(2, int(L / post) + 1))[:-1]:
            px, py = A + t * u
            acc.box(px - 0.03, py - 0.03, z, px + 0.03, py + 0.03, z + h, RAIL)
        (ax, ay), (bx, by) = A, B
        acc.box(min(ax, bx) - 0.03, min(ay, by) - 0.03, z + h - 0.05, max(ax, bx) + 0.03, max(ay, by) + 0.03, z + h, RAIL)
        acc.box(min(ax, bx) - 0.015, min(ay, by) - 0.015, z + 0.5, max(ax, bx) + 0.015, max(ay, by) + 0.015, z + 0.53, RAIL)


def _pergola(acc, rect, z, r):
    """Wooden pergola in front of the main door (front = local -y)."""
    x0, y0, x1, y1 = rect
    w = min(5.0, (x1 - x0) * 0.5)
    cx = (x0 + x1) / 2 + float(r.uniform(-0.15, 0.15)) * (x1 - x0)
    d = float(r.uniform(2.6, 3.6))
    h = 2.6
    for px in (cx - w / 2, cx + w / 2):
        acc.box(px - 0.07, y0 - d, z, px + 0.07, y0 - d + 0.14, z + h, WOOD)
    acc.box(cx - w / 2 - 0.2, y0 - d, z + h, cx + w / 2 + 0.2, y0 - d + 0.14, z + h + 0.18, WOOD)
    for px in np.arange(cx - w / 2, cx + w / 2 + 0.01, 0.45):
        acc.box(px - 0.03, y0 - d - 0.2, z + h + 0.18, px + 0.03, y0, z + h + 0.28, WOOD)


def _greenhouse(acc, rect, z_floor, z_found, h_side, span=8.0, rise=1.9):
    """Multi-span plastic tunnel greenhouse; tunnels run along local y."""
    x0, y0, x1, y1 = rect
    n = max(1, int(round((x1 - x0) / span)))
    sw = (x1 - x0) / n
    ang = np.linspace(0, math.pi, 12)
    for i in range(n):
        cx = x0 + (i + 0.5) * sw
        prof = [(cx - sw / 2 * math.cos(a), z_floor + h_side + rise * math.sin(a)) for a in ang]
        for (xa, za), (xb, zb) in zip(prof[:-1], prof[1:]):
            acc.poly([(xa, y0, za), (xa, y1, za), (xb, y1, zb), (xb, y0, zb)], [(0, xa), (y1 - y0, xa), (y1 - y0, xb), (0, xb)], FILM)
        for yy, rev in ((y0, False), (y1, True)):
            cap = [(prof[0][0], yy, z_found)] + [(x, yy, zz) for x, zz in prof] + [(prof[-1][0], yy, z_found)]
            cap = cap[::-1] if not rev else cap
            acc.poly(cap, [(p[0], p[2]) for p in cap], FILM)
    for xx, rev in ((x0, True), (x1, False)):
        q = [(xx, y0, z_found), (xx, y1, z_found), (xx, y1, z_floor + h_side), (xx, y0, z_floor + h_side)]
        acc.poly(q[::-1] if rev else q, [(0, 0), (y1 - y0, 0), (y1 - y0, h_side), (0, h_side)], FILM)
    return z_floor + h_side + rise


def _water_tower(acc, rect, z_floor, z_found, elevated):
    """Concrete water tank: ground reservoir, or an elevated tank on a shaft."""
    x0, y0, x1, y1 = rect
    r = min(x1 - x0, y1 - y0) / 2
    if not elevated:
        top = z_floor + 5.0
        _cyl(acc, 0, 0, z_found, r, top - z_found, WALL, seg=40, top_mat=ROOF)
        _cyl(acc, 0, 0, top, r * 0.12, 0.6, CONCRETE, seg=12)   # vent
        return top
    shaft_top = z_floor + 15.0
    _cyl(acc, 0, 0, z_found, r * 0.32, shaft_top - z_found, WALL, seg=24)
    _cyl(acc, 0, 0, shaft_top, r, 4.5, WALL, seg=40, top_mat=ROOF)
    acc.box(-r * 0.35, -r * 0.35, shaft_top - 0.3, r * 0.35, r * 0.35, shaft_top, CONCRETE)
    return shaft_top + 4.5


METAL_TINTS = [(0.75, 0.76, 0.76), (0.92, 0.92, 0.9), (0.62, 0.52, 0.44), (0.5, 0.51, 0.52), (0.82, 0.84, 0.85)]


# ---------------------------------------------------------------- prototypes
def _cyl(acc, cx, cy, z0, r, h, mat, seg=16, axis="z", top_mat=None):
    ang = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    if axis == "z":
        ring = lambda z: [(cx + r * math.cos(a), cy + r * math.sin(a), z) for a in ang]
    else:  # horizontal along x, centred at (cx, cy, z0)
        ring = lambda x: [(x, cy + r * math.cos(a), z0 + r * math.sin(a)) for a in ang]
    lo, hi = (ring(z0), ring(z0 + h)) if axis == "z" else (ring(cx - h / 2), ring(cx + h / 2))
    for k in range(seg):
        k2 = (k + 1) % seg
        acc.poly([lo[k], lo[k2], hi[k2], hi[k]], [(k * 0.1, 0), (k2 * 0.1, 0), (k2 * 0.1, h), (k * 0.1, h)], mat)
    acc.poly(hi, [(p[0], p[1]) for p in hi], top_mat if top_mat is not None else mat)
    if axis != "z":
        acc.poly(lo[::-1], [(p[1], p[2]) for p in lo[::-1]], mat)


def _dome(acc, cx, cy, z, r, mat, seg=24, rings=8, drum_mat=None, drum_h=1.2):
    """Drum + hemispherical dome (mosques)."""
    if drum_mat is not None:
        _cyl(acc, cx, cy, z, r * 1.02, drum_h, drum_mat, seg=seg, top_mat=drum_mat)
        z += drum_h
    prev = [(cx + r * math.cos(a), cy + r * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, seg, endpoint=False)]
    for k in range(1, rings + 1):
        phi = (math.pi / 2) * k / rings
        rr, zz = r * math.cos(phi), z + r * math.sin(phi) * 1.05
        cur = [(cx + rr * math.cos(a), cy + rr * math.sin(a), zz) for a in np.linspace(0, 2 * math.pi, seg, endpoint=False)]
        for i in range(seg):
            j = (i + 1) % seg
            q = [prev[i], prev[j], cur[j], cur[i]]
            acc.poly(q, [(i * 0.3, k * 0.3), (j * 0.3, k * 0.3), (j * 0.3, (k + 1) * 0.3), (i * 0.3, (k + 1) * 0.3)], mat)
        prev = cur
    return z + r * 1.05


def _minaret(acc, cx, cy, z0, height, r, wall_mat, cap_mat):
    """Slender minaret: shaft, balcony at ~75 %, narrower upper shaft, cone top."""
    zb = z0 + height * 0.74
    _cyl(acc, cx, cy, z0, r, zb - z0, wall_mat, seg=12)
    _cyl(acc, cx, cy, zb, r * 1.55, 0.35, CONCRETE, seg=16, top_mat=CONCRETE)          # balcony slab
    _cyl(acc, cx, cy, zb + 0.35, r * 1.5, 1.0, RAIL, seg=16)                          # balcony railing band
    _cyl(acc, cx, cy, zb + 0.35, r * 0.8, height * 0.18, wall_mat, seg=12)
    zt = zb + 0.35 + height * 0.18
    ring = [(cx + r * 0.85 * math.cos(a), cy + r * 0.85 * math.sin(a), zt) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
    tip = (cx, cy, zt + height * 0.1)
    for i in range(12):
        acc.poly([ring[i], ring[(i + 1) % 12], tip], [(0, 0), (1, 0), (0.5, 1)], cap_mat)
    return tip[2]


DOME_TINTS = [(0.16, 0.42, 0.26), (0.8, 0.8, 0.8), (0.82, 0.62, 0.25), (0.92, 0.92, 0.9), (0.2, 0.32, 0.5)]


def prototypes(col):
    """Shared rooftop meshes, created once per scene."""
    P = {}
    m_tank_w = mf.dirty_flat_material("M_tank_white", (0.82, 0.81, 0.78), 0.5, dirt=0.35)
    m_tank_b = mf.dirty_flat_material("M_tank_black", (0.03, 0.03, 0.03), 0.45, dirt=0.25)
    m_tank_blue = mf.dirty_flat_material("M_tank_blue", (0.08, 0.16, 0.35), 0.45, dirt=0.3)
    m_panel = mf.flat_material("M_solar_glass", (0.025, 0.035, 0.06), 0.08, specular=0.6)
    m_metal = mf.dirty_flat_material("M_galv_metal", (0.45, 0.46, 0.46), 0.45, metallic=0.6, dirt=0.3)
    m_ac = mf.dirty_flat_material("M_ac_unit", (0.78, 0.77, 0.74), 0.55, dirt=0.3)
    m_dark = mf.flat_material("M_dark_grille", (0.04, 0.04, 0.04), 0.7)
    m_rust = mf.flat_material("M_rebar", (0.18, 0.09, 0.05), 0.85)
    m_stub = mf.dirty_flat_material("M_column_stub", (0.58, 0.57, 0.54), 0.8, dirt=0.3)

    def make(name, build, mats):
        acc = MeshAcc()
        build(acc)
        me = acc.mesh(name)
        for m in mats:
            me.materials.append(m)
        P[name] = me

    # solar water heater: panel tilted ~40 deg facing -y (south when yaw 0), tank on top behind
    def solar(acc):
        tilt = math.radians(40)
        L, W = 2.0, 1.0
        y0, z0 = -0.6, 0.25
        y1, z1 = y0 + L * math.cos(tilt), z0 + L * math.sin(tilt)
        acc.poly([(-W / 2, y0, z0), (W / 2, y0, z0), (W / 2, y1, z1), (-W / 2, y1, z1)], [(0, 0), (1, 0), (1, 2), (0, 2)], 1)
        acc.poly([(-W / 2, y1, z1 - 0.06), (W / 2, y1, z1 - 0.06), (W / 2, y0, z0 - 0.06), (-W / 2, y0, z0 - 0.06)], [(0, 0)] * 4, 2)
        for x in (-W / 2 + 0.05, W / 2 - 0.05):
            acc.box(x - 0.025, y0, 0, x + 0.025, y0 + 0.05, z0, 2)
            acc.box(x - 0.025, y1 - 0.05, 0, x + 0.025, y1, z1, 2)
        _cyl(acc, 0, y1 + 0.32, z1 + 0.05, 0.27, 1.5, 0, seg=14, axis="x")
    make("proto_solar", solar, [m_tank_w, m_panel, m_metal])

    for name, m in (("proto_tank_black", m_tank_b), ("proto_tank_white", m_tank_w), ("proto_tank_blue", m_tank_blue)):
        make(name, lambda acc: (_cyl(acc, 0, 0, 0, 1.0, 1.0, 0, seg=20)), [m])

    def ac(acc):
        acc.box(-0.42, -0.15, 0.08, 0.42, 0.15, 0.68, 0)
        acc.box(-0.42, -0.15, 0, -0.38, 0.15, 0.08, 1)
        acc.box(0.38, -0.15, 0, 0.42, 0.15, 0.08, 1)
        _cyl(acc, -0.08, -0.155, 0.38, 0.22, 0.01, 1, seg=16)
    make("proto_ac", ac, [m_ac, m_dark])

    def dish(acc):
        acc.box(-0.02, -0.02, 0, 0.02, 0.02, 0.7, 1)
        ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        ring = [(0.33 * math.cos(a), 0.12 + 0.33 * math.sin(a) * 0.5, 0.75 + 0.33 * math.sin(a) * 0.85) for a in ang]
        acc.poly(ring, [(p[0], p[2]) for p in ring], 0)
        acc.poly(ring[::-1], [(p[0], p[2]) for p in ring[::-1]], 0)
    make("proto_dish", dish, [m_ac, m_metal])

    def rebar(acc):
        acc.box(-0.15, -0.15, 0, 0.15, 0.15, 0.35, 1)
        for dx in (-0.09, 0.09):
            for dy in (-0.09, 0.09):
                acc.box(dx - 0.008, dy - 0.008, 0.35, dx + 0.008, dy + 0.008, 1.2, 0)
    make("proto_rebar", rebar, [m_rust, m_stub])
    return P


# scanned walls normalised toward plausible local colours before the per-building tint:
# (saturation, value, warm multiplier, bump strength, bump distance x asset height)
WALL_NORMALISE = {
    "wall.stucco": (0.3, 1.9, (1.0, 0.97, 0.9), 0.3, 3.0),      # orange scan -> off-white render
    "wall.stone": (0.55, 1.6, (1.12, 1.04, 0.86), 0.5, 1.0),     # cool grey scan -> cream limestone
    "wall.block": (0.4, 1.15, (1.0, 0.98, 0.94), 0.5, 1.0),      # grey cinder block
    "wall.concrete": (0.4, 1.2, (1.0, 0.98, 0.95), 0.4, 1.0),
    "wall.white": (0.12, 2.3, (1.0, 0.99, 0.97), 0.25, 3.0),     # fresh white plaster (stucco scan)
    "wall.basalt": (0.25, 0.5, (0.95, 0.95, 1.0), 0.6, 1.2),     # dark basalt masonry (stone scan)
}
WALL_SOURCE = {"wall.white": "wall.stucco", "wall.basalt": "wall.stone"}   # roles without their own scan


def wall_materials(am):
    mats = {}
    for role in WALL_NORMALISE:
        if role == "wall.block" and not am.assets_for_role(role):   # block scan missing/corrupt: concrete + joints
            am.log("wall.block: no scan available, using concrete with procedural block joints")
            stem = am.resolve("wall.concrete")
        else:
            stem = am.resolve(WALL_SOURCE.get(role, role))
        mats[role] = _wall_material(f"M_{role}", am, stem, role)
    return mats


def _wall_material(name, am, stem, role):
    """Scanned wall surface x per-building tint, rain streaks under the roof
    line, splash dirt near the ground, AO-like darkening in window reveals."""
    a = dict(am.asset(stem))
    a["height_m"] = a.get("height_m") or 0.01
    am.used[stem] = a.get("version")
    mat = mf.new_material(name)
    nb = mf.NodeBuilder(mat)
    uv = nb.uv("UVMap")
    s = mf.SurfaceSample(nb, a, am.tier, uv, seed=0.2, cell_m=4.0)
    # scanned stucco is orange-beige and the stone is cool grey: normalise both
    # toward neutral off-white / cream limestone, then apply the per-building tint
    base = nb.node("ShaderNodeHueSaturation")
    nb.link(s.color, base.inputs["Color"])
    sat, val, warm, b_str, b_dist = WALL_NORMALISE[role]
    base.inputs["Saturation"].default_value = sat
    base.inputs["Value"].default_value = val
    warm = (*warm, 1.0)
    oi = nb.node("ShaderNodeObjectInfo")
    color = nb.mix(1.0, base.outputs["Color"], warm, "RGBA", "MULTIPLY")
    joints = None
    if role == "wall.block" and "block" not in stem:   # procedural 40 x 20 cm blocks, running bond
        br = nb.node("ShaderNodeTexBrick")
        nb.link(uv, br.inputs["Vector"])
        br.inputs["Scale"].default_value = 1.0
        br.inputs["Brick Width"].default_value = 0.40
        br.inputs["Row Height"].default_value = 0.20
        br.inputs["Mortar Size"].default_value = 0.012
        br.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1)
        br.inputs["Color2"].default_value = (0.9, 0.9, 0.88, 1)
        br.inputs["Mortar"].default_value = (0.62, 0.61, 0.58, 1)
        color = nb.mix(1.0, color, br.outputs["Color"], "RGBA", "MULTIPLY")
        joints = br.outputs["Fac"]
    color = nb.mix(1.0, color, oi.outputs["Color"], "RGBA", "MULTIPLY")
    geo = nb.node("ShaderNodeNewGeometry")
    pos = geo.outputs["Position"]
    fz = nb.node("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="floor_z")
    rz = nb.node("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="roof_z")
    sep = nb.node("ShaderNodeSeparateXYZ")
    nb.link(pos, sep.inputs[0])
    hz = nb.math("SUBTRACT", sep.outputs["Z"], fz.outputs["Fac"])
    below_roof = nb.math("SUBTRACT", rz.outputs["Fac"], sep.outputs["Z"])
    streak_n = nb.node("ShaderNodeTexNoise")
    sv = nb.node("ShaderNodeMapping")
    nb.link(pos, sv.inputs["Vector"])
    sv.inputs["Scale"].default_value = (5.0, 5.0, 0.25)
    nb.link(sv.outputs[0], streak_n.inputs["Vector"])
    streak_n.inputs["Detail"].default_value = 4
    streak = nb.math("MULTIPLY", nb.map_range(streak_n.outputs["Fac"], 0.5, 0.75, 0, 0.22),
                     nb.map_range(below_roof, 0.0, 3.5, 1.0, 0.25))
    ground = nb.map_range(hz, -0.2, 0.9, 0.3, 0.0)
    dirt = nb.math("ADD", streak, ground)
    color = nb.mix(dirt, color, (0.33, 0.29, 0.24, 1.0))
    rough = nb.math("ADD", s.rough, nb.math("MULTIPLY", dirt, 0.2))
    n = mf.bump(nb, s.height, b_str, a["height_m"] * b_dist)
    if joints is not None:
        n = mf.bump(nb, nb.math("SUBTRACT", 1.0, joints), 0.6, 0.008, normal=n)
    mf.output(nb, mf.principled(nb, color, rough, n, specular=0.35).outputs[0])
    return mat


def roof_material(am):
    stem = am.resolve("roof.flat")
    a = dict(am.asset(stem))
    a["height_m"] = a.get("height_m") or 0.01
    am.used[stem] = a.get("version")
    mat = mf.new_material("M_roof_flat")
    nb = mf.NodeBuilder(mat)
    uv = nb.uv("UVMap")
    s = mf.SurfaceSample(nb, a, am.tier, uv, seed=0.9)
    # the scan is a dark brownish floor; flat roofs read light grey from the air
    hsv = nb.node("ShaderNodeHueSaturation")
    nb.link(s.color, hsv.inputs["Color"])
    hsv.inputs["Saturation"].default_value = 0.9
    hsv.inputs["Value"].default_value = 2.55
    tint = nb.node("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="roof_tint")
    color = nb.mix(1.0, hsv.outputs["Color"], tint.outputs["Color"], "RGBA", "MULTIPLY")
    # water stains and dust drifts on flat roofs
    stain = nb.map_range(nb.noise(nb.scale_vec(uv, 1 / 2.5), 1.0, detail=6).outputs["Fac"], 0.55, 0.75, 0, 0.18)
    color = nb.mix(stain, color, (0.36, 0.32, 0.27, 1.0))
    mf.output(nb, mf.principled(nb, color, s.rough, mf.bump(nb, s.height, 0.4, 0.01), specular=0.3).outputs[0])
    return mat


ROOF_TINTS = [(0.95, 0.95, 0.94), (0.82, 0.82, 0.8), (0.7, 0.69, 0.66), (1.0, 0.98, 0.95), (0.6, 0.6, 0.6)]


def build_buildings(plan, am, col, ids):
    wall_mats = wall_materials(am)
    m_roof = roof_material(am)
    m_plinth = am.surface_material("wall.stone", tint=(1.05, 0.98, 0.84))
    m_conc = am.surface_material("wall.concrete")
    m_tiles = am.surface_material("roof.tiles", tint=(5.5, 1.8, 0.9), random_variation=0.1)  # grey slate scan -> terracotta
    m_glass = mf.flat_material("M_glass", (0.015, 0.018, 0.02), 0.04, specular=0.5)
    m_frame = mf.dirty_flat_material("M_frame", (0.72, 0.71, 0.68), 0.5, dirt=0.2)
    m_shut = mf.dirty_flat_material("M_shutter", (0.68, 0.66, 0.62), 0.55, dirt=0.3, noise_scale=12)
    m_door = mf.dirty_flat_material("M_door", (0.16, 0.14, 0.12), 0.5, metallic=0.3, dirt=0.15)
    m_metal = metal_roof_material()
    m_interior = mf.flat_material("M_interior", (0.035, 0.034, 0.032), 0.9)
    m_rail = mf.dirty_flat_material("M_railing", (0.12, 0.12, 0.12), 0.45, metallic=0.6, dirt=0.15)
    m_pv = pv_material()
    m_wood = mf.dirty_flat_material("M_wood", (0.26, 0.17, 0.1), 0.7, dirt=0.25, noise_scale=8)
    m_film = film_material()
    m_curtain = curtain_material()
    protos = prototypes(col)
    objs = []
    for b in plan["buildings"]:
        r = rng(plan["seed"], "building_geo", b["id"])
        fp = np.asarray(b["footprint"])
        centre = fp.mean(axis=0)
        yaw = b["yaw"]
        rect = _local_rect(fp, centre, yaw)
        acc = MeshAcc()
        flat = b["roof"] == "flat"
        parapet = (0.25 if b.get("railing") else b["parapet"]) if flat else 0.0
        top_rect = rect
        if b["roof"] == "tunnel":
            z_roof = z_items = _greenhouse(acc, rect, b["floor_z"], b["foundation_z"], b["floor_height"])
        elif b["roof"] == "tank":
            z_roof = z_items = _water_tower(acc, rect, b["floor_z"], b["foundation_z"], b["floors"] >= 2)
        elif b["roof"] == "dome":   # mosque: prayer hall, dome on a drum, minaret at a front corner
            z_items = _block(acc, r, b, rect, b["floors"], b["floor_z"], b["foundation_z"], True, 0.6)
            x0, y0, x1, y1 = rect
            dr = 0.33 * min(x1 - x0, y1 - y0)
            z_roof = _dome(acc, (x0 + x1) / 2, (y0 + y1) / 2 + 0.1 * (y1 - y0), z_items, dr, METAL, drum_mat=WALL)
            side = 1 if r.random() < 0.5 else -1
            mx = x1 - 1.8 if side > 0 else x0 + 1.8
            _minaret(acc, mx, y0 + 1.8, b["foundation_z"], float(r.uniform(22, 34)) + (z_items - b["floor_z"]), 1.25, WALL, METAL)
        else:
            z0, zf0, nfl, door = b["floor_z"], b["foundation_z"], b["floors"], True
            if b.get("podium"):   # wider commercial base: tall storeys, shopfront glazing, entrances
                prect = _local_rect(np.asarray(b["podium"]), centre, yaw)
                bp = dict(b, floor_height=4.2, window_spacing=4.0, window_w=3.3, window_h=2.7, sill=0.25,
                          shutters_p=0.0, balcony=None)
                z0 = _block(acc, r, bp, prect, b["podium_floors"], b["floor_z"], b["foundation_z"], True, 0.9)
                zf0, nfl, door = z0, b["floors"] - b["podium_floors"], False
            if b.get("pilotis") and nfl >= 2:   # open ground floor on columns (parking), the house above
                x0, y0, x1, y1 = rect
                fh = b["floor_height"]
                nx, ny = max(2, int(round((x1 - x0) / 4.0)) + 1), max(2, int(round((y1 - y0) / 4.0)) + 1)
                for cx in np.linspace(x0 + 0.2, x1 - 0.2, nx):
                    for cy in np.linspace(y0 + 0.2, y1 - 0.2, ny):
                        acc.box(cx - 0.18, cy - 0.18, zf0, cx + 0.18, cy + 0.18, z0 + fh, CONCRETE)
                acc.box(x0, y0, z0 + fh - 0.25, x1, y1, z0 + fh, CONCRETE, bottom=True)   # slab over the open floor
                z0, zf0, nfl, door = z0 + fh, z0 + fh, nfl - 1, False
            if flat and b.get("top_setback") and nfl >= 2:
                # lower floors over the whole footprint, the top floor over the back part only
                z_items = _block(acc, r, b, rect, nfl - 1, z0, zf0, door, parapet)
                x0, y0, x1, y1 = rect
                top_rect = (x0, y0 + (y1 - y0) * b["top_setback"], x1, y1)
                z_roof = _block(acc, r, b, top_rect, 1, z_items, z_items, False, parapet, is_annex=True)
            else:
                z_roof = z_items = _block(acc, r, b, rect, nfl, z0, zf0, door, parapet)
        if b.get("annex"):
            arect = _local_rect(np.asarray(b["annex"]), centre, yaw)
            za = _block(acc, r, b, arect, b["annex_floors"], b["floor_z"], b["foundation_z"], False,
                        parapet if flat else 0.3, is_annex=True)
            # (pitched-roof houses: the annex gets a flat slab behind a low parapet from _block;
            #  a second coplanar slab here would shadow it black in Cycles)
            if flat and b.get("railing"):
                _railing(acc, arect, za)
        south = local_south(yaw)
        if b["roof"] == "tiled_hip":
            _hip_roof(acc, rect, z_roof)
        elif b["roof"] == "tiled_gable":
            _gable_roof(acc, rect, z_roof, overhang=0.45, pitch_deg=28, mat=TILES)
        elif b["roof"] == "gable_metal":
            _gable_roof(acc, rect, z_roof, pv_dir=south if b.get("pv") else None)
        if flat:
            if b.get("railing"):
                _railing(acc, top_rect, z_roof)
                if top_rect is not rect:
                    _railing(acc, rect, z_items)
            if b.get("pv"):
                _pv_rows(acc, top_rect, z_roof, south)
        if b.get("pergola"):
            _pergola(acc, rect, b["floor_z"] - 0.2, r)
        stair = None
        for it in b.get("rooftop", []):
            if it["type"] == "stairwell":
                x, y, w, d, h = it["x"], it["y"], it["w"], it["d"], it["h"]
                acc.box(x - w / 2, y - d / 2, z_items, x + w / 2, y + d / 2, z_items + h, WALL, top_mat=ROOF)
                acc.box(x - w / 2 - 0.05, y - d / 2 - 0.05, z_items + h, x + w / 2 + 0.05, y + d / 2 + 0.05, z_items + h + 0.12, CONCRETE)
                acc.poly([(x - 0.45, y - d / 2 - 0.01, z_items), (x + 0.45, y - d / 2 - 0.01, z_items),
                          (x + 0.45, y - d / 2 - 0.01, z_items + 2.05), (x - 0.45, y - d / 2 - 0.01, z_items + 2.05)],
                         [(0, 0), (1, 0), (1, 2), (0, 2)], DOOR)
                stair = it
            elif it["type"] == "clutter":
                x, y, w, d, h = it["x"], it["y"], it["w"], it["d"], it["h"]
                acc.box(x - w / 2, y - d / 2, z_items, x + w / 2, y + d / 2, z_items + h,
                        FRAME if it["shade"] > 0.7 else (CONCRETE if it["shade"] > 0.5 else DOOR))
        if b.get("roof_hat") and flat:   # a room on the roof under a small tiled pyramid
            x0, y0, x1, y1 = top_rect
            w_, d_ = min(5.5, (x1 - x0) * 0.45), min(5.0, (y1 - y0) * 0.45)
            hx = x1 - w_ / 2 - 0.6 if r.random() < 0.5 else x0 + w_ / 2 + 0.6
            hy = y1 - d_ / 2 - 0.6
            acc.box(hx - w_ / 2, hy - d_ / 2, z_roof, hx + w_ / 2, hy + d_ / 2, z_roof + 2.6, WALL, top_mat=ROOF)
            _hip_roof(acc, (hx - w_ / 2, hy - d_ / 2, hx + w_ / 2, hy + d_ / 2), z_roof + 2.6, overhang=0.3, pitch_deg=30)
        me = acc.mesh(f"bld_{b['id']}")
        for m in (wall_mats[b["wall_role"]], m_glass, m_frame, m_roof, m_plinth, m_conc, m_shut, m_door, m_tiles, m_metal,
                  m_interior, m_rail, m_pv, m_wood, m_film, m_curtain):
            me.materials.append(m)
        iid = ids.new(b["kind"], b["id"])
        ob = new_object(b["id"] if b["id"].startswith("bld_") else f"building_{b['id']}", me, col, iid,
                        {"kind": b["kind"], "floors": b["floors"], "roof": b["roof"], "plan_id": b["id"],
                         "floor_z": b["floor_z"], "roof_z": z_roof, "type": b.get("type", "house")})
        ob.location = (centre[0], centre[1], 0)
        ob.rotation_euler = (0, 0, yaw)
        if "roof_tint" in b:   # spec buildings: facade and roof colours chosen per building from the palettes
            tint = b["tint"]
            ob["roof_tint"] = list(b["roof_tint"])
        else:                  # older planner buildings
            tint = (b["tint"] if b["wall_role"] in ("wall.stucco", "wall.white") else
                    (1.0, 0.95, 0.86) if b["wall_role"] == "wall.stone" else (1.0, 1.0, 1.0))
            ob["roof_tint"] = list(ROOF_TINTS[int(r.integers(len(ROOF_TINTS)))])
        ob.color = (*tint, 1.0)
        ob["metal_tint"] = list((DOME_TINTS if b["roof"] == "dome" else METAL_TINTS)[int(r.integers(len(METAL_TINTS)))])
        objs.append(ob)
        # rooftop equipment
        M = Matrix.Translation((centre[0], centre[1], 0)) @ Matrix.Rotation(yaw, 4, "Z")
        for it in b.get("rooftop", []):
            t = it["type"]
            if t == "stairwell":
                continue
            if t == "clutter":
                continue  # built into the building mesh below
            if t == "rebar":
                x0, y0, x1, y1 = top_rect
                sp = it["spacing"]
                for x in np.linspace(x0 + 0.15, x1 - 0.15, max(2, int((x1 - x0) / sp) + 1)):
                    for y in np.linspace(y0 + 0.15, y1 - 0.15, max(2, int((y1 - y0) / sp) + 1)):
                        if x0 + 0.2 < x < x1 - 0.2 and y0 + 0.2 < y < y1 - 0.2:
                            continue  # perimeter columns only
                        _inst(protos["proto_rebar"], f"rebar_{b['id']}", col, M, (x, y, z_roof), 0.0, 1.0, iid)
                continue
            if t == "solar_heater":
                p = protos["proto_solar"]
                s = 1.0
            elif t == "water_tank":
                p = protos[f"proto_tank_{it['color']}"]
                s = (it["radius"], it["radius"], it["h"])
            elif t == "ac_unit":
                p, s = protos["proto_ac"], 1.0
            else:
                p, s = protos["proto_dish"], 1.0
            _inst(p, f"{t}_{b['id']}", col, M, (it["x"], it["y"], z_items), it.get("yaw", 0.0), s, iid)
    return objs


def _inst(mesh, name, col, M, local, yaw, scale, iid):
    ob = bpy.data.objects.new(name, mesh)
    loc = M @ Vector(local)
    ob.location = loc
    ob.rotation_euler = (0, 0, M.to_euler().z + yaw)
    ob.scale = (scale, scale, scale) if isinstance(scale, (int, float)) else scale
    ob.pass_index = iid
    ob["kind"] = "building.rooftop"
    col.objects.link(ob)
    return ob


def _strip(acc, P, zb, zt, th, mat, uv_z=True):
    """Extrude polyline P (n x 2) into a wall of thickness th between zb and zt
    (per point): both faces, top, and end caps. UVs in metres (along, height)."""
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.column_stack([t[:, 1], -t[:, 0]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    L, R = P + n * th / 2, P - n * th / 2
    for k in range(len(P) - 1):
        for side, Q in ((1, L), (-1, R)):
            q = [(Q[k, 0], Q[k, 1], zb[k]), (Q[k + 1, 0], Q[k + 1, 1], zb[k + 1]), (Q[k + 1, 0], Q[k + 1, 1], zt[k + 1]), (Q[k, 0], Q[k, 1], zt[k])]
            uv = [(s[k], zb[k]), (s[k + 1], zb[k + 1]), (s[k + 1], zt[k + 1]), (s[k], zt[k])]
            if side < 0:
                q, uv = q[::-1], uv[::-1]
            acc.poly(q, uv, mat)
        top = [(L[k, 0], L[k, 1], zt[k]), (L[k + 1, 0], L[k + 1, 1], zt[k + 1]), (R[k + 1, 0], R[k + 1, 1], zt[k + 1]), (R[k, 0], R[k, 1], zt[k])]
        acc.poly(top, [(p[0], p[1]) for p in top], mat)
    for k, sgn in ((0, -1), (len(P) - 1, 1)):
        q = [(L[k, 0], L[k, 1], zb[k]), (L[k, 0], L[k, 1], zt[k]), (R[k, 0], R[k, 1], zt[k]), (R[k, 0], R[k, 1], zb[k])]
        acc.poly(q if sgn < 0 else q[::-1], [(0, zb[k]), (0, zt[k]), (th, zt[k]), (th, zb[k])], mat)
    return s


def fence_material():
    """Metal bar fence panel: vertical bars every 12 cm (2.5 cm wide), the rest
    transparent, on UVs in metres along the fence."""
    mat = bpy.data.materials.get("M_fence_bars")
    if mat:
        return mat
    mat = mf.new_material("M_fence_bars")
    nb = mf.NodeBuilder(mat)
    sep = nb.node("ShaderNodeSeparateXYZ")
    nb.link(nb.uv("UVMap"), sep.inputs[0])
    bars = nb.math("LESS_THAN", nb.math("FRACT", nb.math("DIVIDE", sep.outputs["X"], 0.12)), 0.21)
    metal = mf.principled(nb, (0.09, 0.09, 0.085, 1.0), 0.5, None, specular=0.4)
    metal.inputs["Metallic"].default_value = 0.6
    clear = nb.node("ShaderNodeBsdfTransparent")
    mix = nb.node("ShaderNodeMixShader")
    nb.link(bars, mix.inputs["Fac"])
    nb.link(clear.outputs[0], mix.inputs[1])
    nb.link(metal.outputs[0], mix.inputs[2])
    mf.output(nb, mix.outputs[0])
    return mat


def build_walls(plan, am, col, ids):
    """Yard walls and fences: polylines extruded on the terrain.
      stone / block / plaster   wall of the house materials (tint per wall),
                                block and plaster with a concrete coping
      fence                     concrete plinth, posts every 2.5 m, bar panel, top rail
    Older plans (role wall.dry_stone etc., no `type`) keep their simple walls."""
    have = {m.name for m in bpy.data.materials}
    wmats = ({r: bpy.data.materials[f"M_{r}"] for r in WALL_NORMALISE} if all(f"M_{r}" in have for r in WALL_NORMALISE)
             else wall_materials(am))
    old = {"wall.stucco": am.surface_material("wall.stucco", tint=(0.93, 0.9, 0.84)),
           "wall.stone": am.surface_material("wall.stone", tint=(0.95, 0.9, 0.8)),
           # dry-stone terrace/field walls from local basalt: dark, rough, varied
           "wall.dry_stone": am.surface_material("wall.stone", tint=(0.9, 0.82, 0.68), random_variation=0.15, bump_strength=1.2)}
    m_cope = mf.dirty_flat_material("M_coping", (0.62, 0.6, 0.56), 0.8, dirt=0.3, noise_scale=6)
    m_metal = mf.dirty_flat_material("M_fence_metal", (0.1, 0.1, 0.095), 0.45, metallic=0.6, dirt=0.15)
    m_bars = fence_material()
    out = []
    for w in plan["walls"]:
        pts = np.asarray(w["points"])
        if len(pts) < 2:
            continue
        acc = MeshAcc()
        P, zg = pts[:, :2], pts[:, 2]
        th, h = w["thickness"], w["height"]
        zb = zg - 0.25
        if "type" not in w:   # older planner walls
            _strip(acc, P, zb, zg + h, th, 0)
            me = acc.mesh(w["id"])
            me.materials.append(old[w["role"]])
            out.append(new_object(w["id"], me, col, ids.new(w["kind"], w["id"]), {"kind": w["kind"]}))
            continue
        if w["type"] == "fence":
            hp = 0.5
            s = _strip(acc, P, zb, zg + hp, th, 0)
            _strip(acc, P, zg + h - 0.05, zg + h, 0.05, 2)                       # top rail
            for d in np.arange(0.0, s[-1] + 1e-6, 2.5):                          # posts
                x, y, z = (np.interp(d, s, P[:, 0]), np.interp(d, s, P[:, 1]), np.interp(d, s, zg))
                acc.box(x - 0.035, y - 0.035, z + hp, x + 0.035, y + 0.035, z + h, 2)
            # bar panel: one double-sided strip on the centreline
            for k in range(len(P) - 1):
                acc.poly([(P[k, 0], P[k, 1], zg[k] + hp), (P[k + 1, 0], P[k + 1, 1], zg[k + 1] + hp),
                          (P[k + 1, 0], P[k + 1, 1], zg[k + 1] + h), (P[k, 0], P[k, 1], zg[k] + h)],
                         [(s[k], 0), (s[k + 1], 0), (s[k + 1], 1), (s[k], 1)], 3)
            top = zg + h
        else:
            cope = w["type"] != "stone"
            top = zg + h
            _strip(acc, P, zb, top - (0.08 if cope else 0.0), th, 0)
            if cope:
                _strip(acc, P, top - 0.08, top, th + 0.06, 1)
        me = acc.mesh(w["id"])
        for m in (wmats[w["role"]], m_cope, m_metal, m_bars):
            me.materials.append(m)
        ob = new_object(w["id"], me, col, ids.new(w["kind"], w.get("label_id", w["id"])),
                        {"kind": w["kind"], "plan_id": w.get("label_id", w["id"]), "type": w["type"],
                         "floor_z": float(zg.mean()), "roof_z": float(top.max())})
        ob.color = (*w["tint"], 1.0)
        out.append(ob)
    return out
