"""Terrain fields on the core grid: sampling, slope, flow/wetness, noise."""
import numpy as np
from scipy import ndimage


class Grid:
    """Regular grid in local metres. Row 0 = south (y = -half)."""

    def __init__(self, z, res, half):
        self.z = z
        self.res = res
        self.half = half
        self.n = z.shape[0]

    def ij(self, x, y):
        return (np.asarray(y) + self.half) / self.res, (np.asarray(x) + self.half) / self.res

    def sample(self, x, y, arr=None, order=1):
        arr = self.z if arr is None else arr
        i, j = self.ij(x, y)
        return ndimage.map_coordinates(arr, [np.atleast_1d(i), np.atleast_1d(j)], order=order, mode="nearest")

    def xy(self):
        c = np.arange(self.n) * self.res - self.half
        return np.meshgrid(c, c)

    def inside(self, x, y, margin=0.0):
        lim = self.half - margin
        return (np.abs(x) <= lim) & (np.abs(y) <= lim)


def slope_deg(z, res):
    gy, gx = np.gradient(z, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def aspect_northness(z, res):
    gy, gx = np.gradient(z, res)
    # downslope direction; +1 when the slope faces north
    n = np.hypot(gx, gy) + 1e-9
    return -gy / n


def flow_accumulation(z):
    """D8 flow accumulation (cells draining through each cell)."""
    n, m = z.shape
    order = np.argsort(z, axis=None)[::-1]
    acc = np.ones(n * m)
    zf = z.ravel()
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = np.array([np.hypot(a, b) for a, b in offs])
    zp = np.pad(z, 1, mode="edge")
    drops = np.stack([(z - zp[1 + a:1 + a + n, 1 + b:1 + b + m]) / d for (a, b), d in zip(offs, dist)])
    best = np.argmax(drops, axis=0).ravel()
    has = (np.max(drops, axis=0) > 0).ravel()
    di = np.array([o[0] for o in offs])[best]
    dj = np.array([o[1] for o in offs])[best]
    rows, cols = np.divmod(np.arange(n * m), m)
    tgt = np.clip(rows + di, 0, n - 1) * m + np.clip(cols + dj, 0, m - 1)
    for k in order:
        if has[k]:
            acc[tgt[k]] += acc[k]
    return acc.reshape(n, m)


def mfd_accumulation(z, p=1.1):
    """Multiple-flow-direction accumulation (Freeman 1991): each cell passes its
    flow to ALL lower neighbours, weighted by slope^p. Unlike single-direction
    D8 it does not carve artificial parallel lines on smooth planar slopes."""
    n, m = z.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = np.array([np.hypot(a, b) for a, b in offs])
    zp = np.pad(z, 1, mode="edge")
    drops = np.stack([(z - zp[1 + a:1 + a + n, 1 + b:1 + b + m]) / d for (a, b), d in zip(offs, dist)])
    w = np.where(drops > 0, drops, 0.0) ** p
    tot = w.sum(axis=0)
    w = np.where(tot > 0, w / np.where(tot > 0, tot, 1), 0.0)
    acc = np.ones((n, m))
    order = np.argsort(z, axis=None)[::-1]
    rows, cols = np.divmod(order, m)
    for r, c in zip(rows, cols):
        a_ = acc[r, c]
        for k, (dy, dx) in enumerate(offs):
            wk = w[k, r, c]
            if wk:
                rr, cc = r + dy, c + dx
                if 0 <= rr < n and 0 <= cc < m:
                    acc[rr, cc] += a_ * wk
    return acc


def wetness_index_mfd(z, res):
    acc = mfd_accumulation(z) * res * res
    s = np.radians(np.maximum(slope_deg(z, res), 0.5))
    return np.log(acc / res / np.tan(s))


def wetness_index(z, res):
    acc = flow_accumulation(z) * res * res
    s = np.radians(np.maximum(slope_deg(z, res), 0.5))
    return np.log(acc / res / np.tan(s))


def value_noise(shape, res, wavelength_m, rng, octaves=3):
    out = np.zeros(shape, np.float32)
    amp, wl, tot = 1.0, wavelength_m, 0.0
    for _ in range(octaves):
        n = rng.standard_normal(shape).astype(np.float32)
        n = ndimage.gaussian_filter(n, max(wl / res / 3, 0.5))
        n /= n.std() + 1e-9
        out += amp * n
        tot += amp
        amp *= 0.5
        wl /= 2.2
    out /= tot
    return 0.5 + 0.5 * np.tanh(out)  # ~[0,1]


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def resample(arr, src_res, dst_res, order=1):
    return ndimage.zoom(arr, src_res / dst_res, order=order)
