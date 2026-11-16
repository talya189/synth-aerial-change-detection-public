"""Road network planning: least-cost routing, smoothing, vertical profile,
and terrain cut/fill along the corridor."""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from synth.planner.fields import Grid

# 16-connectivity ("knight moves") avoids the 45-degree staircase of 8-connectivity
OFFSETS = [(0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)]


class Router:
    """Least-cost router on a coarse grid. Edge cost penalises grade above the
    road's limit (steeply), cross-slope (cut/fill), and optional extra cost."""

    def __init__(self, grid: Grid, res):
        self.res = res
        f = grid.res / res
        self.z = ndimage.zoom(ndimage.gaussian_filter(grid.z, res / grid.res / 2), f, order=1)
        self.n = self.z.shape[0]
        self.half = grid.half
        gy, gx = np.gradient(self.z, res)
        self.slope = np.hypot(gx, gy)

    def node(self, x, y):
        i = int(np.clip(round((y + self.half) / self.res), 0, self.n - 1))
        j = int(np.clip(round((x + self.half) / self.res), 0, self.n - 1))
        return i * self.n + j

    def xy(self, k):
        i, j = divmod(k, self.n)
        return j * self.res - self.half, i * self.res - self.half

    def route(self, start, goal, max_grade, extra_cost=None, reuse=None):
        n = self.n
        rows, cols, w = [], [], []
        ii, jj = np.mgrid[0:n, 0:n]
        cell_pen = 1.0 + 6.0 * np.clip(self.slope / 0.35, 0, 3) ** 2
        if extra_cost is not None:
            cell_pen = cell_pen + extra_cost
        for di, dj in OFFSETS:
            for s in (1, -1):
                a, b = di * s, dj * s
                ok = (ii + a >= 0) & (ii + a < n) & (jj + b >= 0) & (jj + b < n)
                i0, j0 = ii[ok], jj[ok]
                i1, j1 = i0 + a, j0 + b
                L = np.hypot(a, b) * self.res
                g = np.abs(self.z[i1, j1] - self.z[i0, j0]) / L
                cost = L * (1 + 2 * (g / max_grade) ** 2 + 60 * np.maximum(g / max_grade - 1, 0) ** 2)
                cost *= 0.5 * (cell_pen[i0, j0] + cell_pen[i1, j1])
                if reuse is not None:  # cheaper to follow existing roads (merge naturally)
                    cost *= np.where(reuse[i0, j0] & reuse[i1, j1], 0.25, 1.0)
                rows.append((i0 * n + j0).ravel())
                cols.append((i1 * n + j1).ravel())
                w.append(cost.ravel())
        G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n)).tocsr()
        s, t = self.node(*start), self.node(*goal)
        dist, pred = dijkstra(G, indices=s, return_predecessors=True)
        if not np.isfinite(dist[t]):
            return None
        path = [t]
        while path[-1] != s:
            path.append(pred[path[-1]])
        return np.array([self.xy(k) for k in path[::-1]], float)


def chaikin(pts, iters=3):
    for _ in range(iters):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def resample_polyline(pts, step):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < step:
        return pts, s
    t = np.arange(0, s[-1], step)
    t = np.append(t, s[-1])
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])]), t


def smooth_polyline(pts, sigma_m, step=1.0, keep_start=True, keep_end=True):
    p, _ = resample_polyline(chaikin(pts, 3), step)
    if len(p) > 5:
        sig = sigma_m / step
        q = ndimage.gaussian_filter1d(p, sig, axis=0, mode="nearest")
        # pin the ends (junctions must stay on the parent road)
        w = np.ones(len(p))
        ramp = np.clip(np.arange(len(p)) / (3 * sig + 1e-9), 0, 1)
        if keep_start:
            w = np.minimum(w, ramp)
        if keep_end:
            w = np.minimum(w, ramp[::-1])
        p = p * (1 - w[:, None]) + q * w[:, None]
    return resample_polyline(p, step)[0]


def straight_start(pts, origin, direction, straight_m, blend_m=22.0):
    """Blend the first metres of a smoothed branch back onto its straight
    perpendicular stub, so it leaves the parent road cleanly and without kinks."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    line = np.asarray(origin) + np.asarray(direction)[None] * s[:, None]
    w = np.clip(1 - (s - straight_m) / blend_m, 0, 1)
    w = w * w * (3 - 2 * w)
    out = pts * (1 - w[:, None]) + line * w[:, None]
    return resample_polyline(out, 1.0)[0]


def trim_hairpins(pts, window_m=25.0, max_turn_deg=110.0, min_keep_m=40.0):
    """Cut a road at the first place it turns back on itself (a hook toward a
    target behind a slope). Dead-end streets in reality simply stop."""
    if len(pts) < 10:
        return pts
    d = np.diff(pts, axis=0)
    hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    step = np.linalg.norm(d, axis=1).mean()
    w = max(2, int(window_m / step))
    turn = np.abs(hd[w:] - hd[:-w])
    bad = np.nonzero(np.degrees(turn) > max_turn_deg)[0]
    bad = bad[bad * step > min_keep_m]
    return pts[: bad[0] + 1] if len(bad) else pts


def vertical_profile(grid: Grid, pts, max_grade, smooth_m, z_start=None, z_end=None):
    z = grid.sample(pts[:, 0], pts[:, 1], getattr(grid, "z_smooth", None))
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1).mean()
    z = ndimage.gaussian_filter1d(z, smooth_m / step, mode="nearest")
    if z_start is not None:
        z[0] = z_start
    if z_end is not None:
        z[-1] = z_end
    # clamp grade both ways, then average (keeps profile close to terrain)
    dmax = max_grade * step
    f = z.copy()
    for k in range(1, len(f)):
        f[k] = np.clip(f[k], f[k - 1] - dmax, f[k - 1] + dmax)
    b = z.copy()
    for k in range(len(b) - 2, -1, -1):
        b[k] = np.clip(b[k], b[k + 1] - dmax, b[k + 1] + dmax)
    out = 0.5 * (f + b)
    if z_start is not None:
        # the average of two clamped passes can still exceed the limit next to a
        # pinned start: a final forward clamp from the junction guarantees it
        out[0] = z_start
        for k in range(1, len(out)):
            out[k] = np.clip(out[k], out[k - 1] - dmax, out[k - 1] + dmax)
    if z_end is not None:   # a road joined at both ends must also land on the second road
        out[-1] = z_end
        for k in range(len(out) - 2, 0, -1):
            out[k] = np.clip(out[k], out[k + 1] - dmax, out[k + 1] + dmax)
    return out


def road_polygon(pts, half_width):
    from shapely.geometry import LineString
    return LineString(pts).buffer(half_width, cap_style="flat", join_style="round")


def apply_corridors(grid: Grid, roads):
    """Cut/fill the terrain under every road. Inside the carriageway the terrain
    is set just below the road surface; outside it blends back over `blend_m`
    (wider where the embankment is tall, roughly a 1:2 batter)."""
    samples, zs, hw, sh, bl = [], [], [], [], []
    for r in roads:
        p = np.asarray(r["centerline"])
        samples.append(p[:, :2])
        zs.append(p[:, 2])
        hw.append(np.full(len(p), r["width"] / 2))
        sh.append(np.full(len(p), r.get("shoulder", 0.0)))
        bl.append(np.full(len(p), r.get("blend_m", 4.0)))
    P = np.vstack(samples)
    Z, HW, SH, BL = map(np.concatenate, (zs, hw, sh, bl))
    tree = cKDTree(P)
    X, Y = grid.xy()
    reach = float(np.max(HW + SH + BL)) + 25
    d, k = tree.query(np.column_stack([X.ravel(), Y.ravel()]), distance_upper_bound=reach)
    ok = np.isfinite(d)
    k = np.where(ok, k, 0)
    d = d.reshape(X.shape)
    k = k.reshape(X.shape)
    ok = ok.reshape(X.shape)
    zr = Z[k]
    dz = np.abs(grid.z - zr)
    inner = HW[k] + SH[k] + 0.3
    blend = BL[k] + 2.0 * dz
    t = np.clip((d - inner) / blend, 0, 1)
    w = np.where(ok, 1 - t * t * (3 - 2 * t), 0.0)
    camber = 0.02  # carriageway cross-fall, must match the road mesh
    # terrain stays a few cm under the cambered road surface; just outside the
    # edge it meets the road edge height so no soil pokes over the asphalt
    target = np.where(d < HW[k], zr - camber * d - 0.05, zr - camber * HW[k] - 0.03)
    grid.z = grid.z * (1 - w) + target * w
    return w
