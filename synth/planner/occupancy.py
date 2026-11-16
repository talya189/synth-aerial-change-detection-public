"""Raster occupancy shared by all generators (collision / clearance checks)."""
import numpy as np
from PIL import Image, ImageDraw

LAYERS = {"road": 1, "building": 2, "yard": 4, "wall": 8, "path": 16, "tree": 32, "rock": 64, "shrub": 128, "field": 256}


class Occupancy:
    def __init__(self, half, res):
        self.half = half
        self.res = res
        self.n = int(round(2 * half / res)) + 1
        self.mask = np.zeros((self.n, self.n), np.uint16)

    def _px(self, xy):
        xy = np.asarray(xy, float)
        return [((x + self.half) / self.res, (y + self.half) / self.res) for x, y in xy]

    def add_polygon(self, poly, layer):
        im = Image.new("L", (self.n, self.n), 0)
        dr = ImageDraw.Draw(im)
        geoms = getattr(poly, "geoms", [poly])
        for g in geoms:
            if g.is_empty:
                continue
            dr.polygon(self._px(g.exterior.coords), fill=1)
            for hole in g.interiors:
                dr.polygon(self._px(hole.coords), fill=0)
        self.mask |= (np.asarray(im, np.uint16) * LAYERS[layer]).astype(np.uint16)

    def add_disc(self, x, y, r, layer):
        i, j = (y + self.half) / self.res, (x + self.half) / self.res
        rr = int(np.ceil(r / self.res))
        i0, i1 = max(0, int(i) - rr), min(self.n, int(i) + rr + 2)
        j0, j1 = max(0, int(j) - rr), min(self.n, int(j) + rr + 2)
        if i0 >= i1 or j0 >= j1:
            return
        ii, jj = np.mgrid[i0:i1, j0:j1]
        m = (ii - i) ** 2 + (jj - j) ** 2 <= (r / self.res) ** 2
        self.mask[i0:i1, j0:j1][m] |= LAYERS[layer]

    def blocked(self, x, y, layers, r=0.0):
        """True where any of `layers` is set within radius r of (x, y). Vectorised over points."""
        bits = sum(LAYERS[l] for l in layers)
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        out = np.zeros(len(x), bool)
        rr = int(np.ceil(r / self.res))
        offs = [(a, b) for a in range(-rr, rr + 1) for b in range(-rr, rr + 1) if a * a + b * b <= rr * rr] or [(0, 0)]
        ci = np.round((y + self.half) / self.res).astype(int)
        cj = np.round((x + self.half) / self.res).astype(int)
        for a, b in offs:
            i = np.clip(ci + a, 0, self.n - 1)
            j = np.clip(cj + b, 0, self.n - 1)
            out |= (self.mask[i, j] & bits) > 0
        return out

    def distance_to(self, layers):
        from scipy import ndimage
        bits = sum(LAYERS[l] for l in layers)
        return ndimage.distance_transform_edt((self.mask & bits) == 0) * self.res
