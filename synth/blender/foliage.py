"""Leaf-cluster trees and shrubs (geometry in numpy, mesh in Blender).

Why: smooth crown blobs read as plastic from the air. Real crowns (and the
scanned pine, the one tree that looked right) are thousands of small leaf
clusters gathered into clumps on the outer shell of the crown, with gaps you
can see through, branches inside, and strong self-shadowing. This module builds
exactly that:

  skeleton   trunk (tapered, leaning, gnarled for olives) and main branches
             reaching into the crown
  clumps     30-90 foliage clumps on the crown shell (ellipsoid, broad, column
             or irregular), each a cloud of leaf clusters; gaps between clumps
  clusters   a short twig carrying 12-30 real leaf polygons (species leaf size
             and shape), oriented outward, randomly rotated
  colour     per-leaf random value (face attribute `leaf_rand`) mixes the
             species' light and dark leaf colours; inner leaves are darker

All in metres at the species' typical size; trees are scaled modestly per
instance. Slot 0 = leaves, slot 1 = bark.
"""
import math

import numpy as np


def _rot_to(v):
    """3x3 rotation taking +z to unit vector v."""
    v = v / (np.linalg.norm(v) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, v))
    if c > 0.9999:
        return np.eye(3)
    if c < -0.9999:
        return np.diag([1.0, -1.0, -1.0])
    a = np.cross(z, v)
    s = np.linalg.norm(a)
    a /= s
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def leaf_cluster(r, leaf_len, leaf_w, n_leaves, stem_len):
    """One twig with leaves along it, stem along +z. Returns (verts, tris)."""
    V, T = [], []
    # leaf outline in its own plane: x along the leaf, y across (6 vertices, 4 triangles)
    shape = np.array([[0, 0], [0.25, 0.42], [0.62, 0.5], [1.0, 0], [0.62, -0.5], [0.25, -0.42]], float)
    tri = np.array([[0, 1, 5], [1, 2, 4], [1, 4, 5], [2, 3, 4]])
    for k in range(n_leaves):
        t = stem_len * (0.15 + 0.85 * k / max(n_leaves - 1, 1))
        az = r.uniform(0, 2 * math.pi)
        el = math.radians(r.uniform(15, 65))
        d = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
        p = np.array([-math.sin(az), math.cos(az), 0.0])
        L = leaf_len * r.uniform(0.75, 1.2)
        W = leaf_w * r.uniform(0.8, 1.15)
        base = np.array([0.0, 0.0, t])
        pts = base + shape[:, :1] * L * d + shape[:, 1:] * W * p
        pts[:, 2] -= (shape[:, 0] ** 2) * L * 0.18          # slight droop toward the tip
        T.append(tri + len(V) * 6)
        V.append(pts)
    return np.concatenate(V), np.concatenate(T)


def _tube(p0, p1, r0, r1, sides=7):
    """Tapered open tube from p0 to p1. Returns (verts, tris)."""
    d = p1 - p0
    R = _rot_to(d)
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    ring = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(sides)])
    a = p0 + (ring * r0) @ R.T
    b = p1 + (ring * r1) @ R.T
    V = np.vstack([a, b])
    T = []
    for i in range(sides):
        j = (i + 1) % sides
        T += [[i, j, sides + j], [i, sides + j, sides + i]]
    return V, np.array(T)


SHAPES = {   # crown envelope: centre height (fraction of crown), radii (fraction of crown width / depth)
    "round": dict(rx=0.5, rz=0.45, cz=0.5, clumps=46, shell=(0.72, 1.0)),
    "broad": dict(rx=0.5, rz=0.33, cz=0.55, clumps=60, shell=(0.7, 1.0)),
    "column": dict(rx=0.5, rz=0.5, cz=0.5, clumps=150, shell=(0.35, 1.0)),
    "irregular": dict(rx=0.5, rz=0.42, cz=0.5, clumps=34, shell=(0.6, 1.0)),
    "cushion": dict(rx=0.5, rz=0.5, cz=0.32, clumps=16, shell=(0.55, 1.0)),
    "bush": dict(rx=0.5, rz=0.45, cz=0.45, clumps=22, shell=(0.55, 1.0)),
}


def _envelope_point(shape, r, u_shell):
    """A point on/inside the crown envelope, in unit crown coords (x, y in [-.5, .5], z in [0, 1])."""
    e = SHAPES[shape]
    while True:
        v = r.normal(0, 1, 3)
        v /= np.linalg.norm(v)
        if shape in ("cushion", "bush") and v[2] < -0.2:
            continue   # shrubs: no foliage under the ground
        break
    s = r.uniform(*e["shell"]) if u_shell is None else u_shell
    if shape == "column":   # tall narrow cone-column
        z = r.uniform(0.02, 0.98)
        rad = 0.5 * (1 - z) ** 0.6 * s
        a = math.atan2(v[1], v[0])
        return np.array([rad * math.cos(a), rad * math.sin(a), z]), np.array([math.cos(a), math.sin(a), 0.25])
    p = np.array([v[0] * e["rx"], v[1] * e["rx"], v[2] * e["rz"]]) * s + np.array([0, 0, e["cz"]])
    if shape == "irregular":
        p[:2] *= 1 + 0.35 * math.sin(3 * math.atan2(v[1], v[0]) + r.uniform(0, 6.28))
    return p, v


def build_plant(shape, height, width, trunk_frac, leaf_len, leaf_w, seed, density=1.0,
                leaves_per_cluster=18, gnarled=False, shrub=False):
    """Geometry of one plant at real size. Returns verts, tris, mat (0 leaf / 1 bark), leaf_rand per face."""
    r = np.random.default_rng(seed)
    V, T, M, LR = [], [], [], []

    def add(v, t, mat, lr):
        T.append(t + sum(len(x) for x in V))
        V.append(v)
        M.append(np.full(len(t), mat, np.int32))
        LR.append(np.broadcast_to(np.asarray(lr, np.float32), (len(t),)).copy() if np.ndim(lr) == 0 else lr)

    crown_h = height * (1 - trunk_frac)
    z0 = height * trunk_frac
    e = SHAPES[shape]
    # --- skeleton
    if not shrub:
        rb = max(0.06, width * (0.035 if not gnarled else 0.06))
        lean = r.normal(0, 0.06, 2)
        pts = [np.array([0.0, 0.0, -0.1])]
        n_seg = 4
        for k in range(1, n_seg + 1):
            f = k / n_seg
            wob = r.normal(0, 0.08 if gnarled else 0.025, 2) * width
            pts.append(np.array([lean[0] * f * height + wob[0], lean[1] * f * height + wob[1], z0 * f + 0.15 * crown_h * f]))
        for k in range(n_seg):
            rr0 = rb * (1 - 0.5 * k / n_seg) * (1.4 if gnarled and k == 0 else 1)
            rr1 = rb * (1 - 0.5 * (k + 1) / n_seg)
            add(*_tube(pts[k], pts[k + 1], rr0, rr1), 1, 0.0)
        top = pts[-1]
        nb = int(r.integers(4, 8)) if shape != "column" else 3
        for _ in range(nb):
            tgt, _ = _envelope_point(shape, r, 0.55)
            tgt = np.array([tgt[0] * width, tgt[1] * width, z0 + tgt[2] * crown_h])
            mid = (top + tgt) / 2 + r.normal(0, 0.05, 3) * width
            add(*_tube(top, mid, rb * 0.55, rb * 0.35, 6), 1, 0.0)
            add(*_tube(mid, tgt, rb * 0.35, rb * 0.12, 5), 1, 0.0)
    # --- foliage: clumps on the crown shell, each a cloud of leaf clusters
    cl_v, cl_t = leaf_cluster(r, leaf_len, leaf_w, leaves_per_cluster, stem_len=leaf_len * 2.2)
    n_clumps = max(6, int(e["clumps"] * density * (width / 5.0) ** 0.6 if not shrub else e["clumps"] * density))
    clump_r = width * (0.16 if not shrub else 0.22) * (1.9 if shape == "column" else 1.0)
    area_per_cluster = (leaf_len * 2.2) ** 2 * 1.2
    for _ in range(n_clumps):
        c, nrm = _envelope_point(shape, r, None)
        centre = np.array([c[0] * width, c[1] * width, (z0 if not shrub else 0.0) + c[2] * (crown_h if not shrub else height)])
        cr = clump_r * r.uniform(0.6, 1.25)
        n_cl = int(np.clip(math.pi * cr * cr / area_per_cluster * 1.4 * density, 6, 220))
        inner = float(np.clip(np.linalg.norm(c[:2]) / 0.5, 0, 1))   # 0 = crown centre
        for _ in range(n_cl):
            off = r.normal(0, cr * 0.45, 3)
            off[2] *= 0.7
            p = centre + off
            if shrub and p[2] < 0.02:
                p[2] = 0.02 + abs(off[2]) * 0.2
            out = (p - np.array([0, 0, z0 + e["cz"] * crown_h])) if not shrub else (p - np.array([0, 0, 0.0]))
            dirv = out / (np.linalg.norm(out) + 1e-9) + r.normal(0, 0.5, 3) + np.array([0, 0, 0.4])
            R = _rot_to(dirv) @ _rz(r.uniform(0, 2 * math.pi))
            s = r.uniform(0.8, 1.25)
            v = (cl_v * s) @ R.T + p
            lr = np.clip(r.uniform(0, 1, len(cl_t)) * 0.7 + 0.3 * inner + r.normal(0, 0.05), 0, 1).astype(np.float32)
            add(v, cl_t, 0, lr)
    verts = np.concatenate(V).astype(np.float32)
    tris = np.concatenate(T).astype(np.int32)
    return verts, tris, np.concatenate(M), np.concatenate(LR)
