"""Aerial imaging effects applied to the rendered image (numpy only).

Order: atmospheric haze (from the depth pass, Koschmieder extinction) ->
optics (Gaussian PSF, vignetting) -> sensor (Poisson-Gaussian noise) ->
8-bit quantisation (+ optional JPEG). Labels are never touched.
Haze is applied after the view transform on linearised sRGB values: an
approximation of in-scene aerial perspective that keeps colour management
identical to the raw render.
"""
import numpy as np


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def gaussian_blur(img, sigma):
    if sigma <= 0.05:
        return img
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.zeros_like(img)
    pad = np.pad(img, ((r, r), (0, 0), (0, 0)), mode="edge")
    for i, w in enumerate(k):
        out += w * pad[i:i + img.shape[0]]
    pad = np.pad(out, ((0, 0), (r, r), (0, 0)), mode="edge")
    out2 = np.zeros_like(img)
    for i, w in enumerate(k):
        out2 += w * pad[:, i:i + img.shape[1]]
    return out2


def apply(rgb_display, depth, sky_mask, r, cfg):
    """rgb_display: (H,W,3) float [0,1] after view transform. Returns uint8 image and params."""
    lin = srgb_to_lin(rgb_display)
    vis_km = float(r.uniform(*cfg["visibility_km"]))
    beta = 3.912 / (vis_km * 1000.0)
    d = np.where(sky_mask, 0.0, depth)
    t = np.exp(-beta * d)[..., None]
    if sky_mask.any():
        rows = np.nonzero(sky_mask.any(axis=1))[0]
        band = sky_mask & (np.arange(sky_mask.shape[0])[:, None] >= rows.max() - 20)
        airlight = np.median(lin[band], axis=0) if band.any() else np.median(lin[sky_mask], axis=0)
    else:
        airlight = np.array([0.80, 0.84, 0.90]) * float(np.percentile(lin.mean(-1), 90))
    lin = lin * t + airlight[None, None] * (1 - t)
    sigma = float(r.uniform(*cfg["psf_sigma_px"]))
    lin = gaussian_blur(lin, sigma)
    h, w = lin.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rr = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
    vig = float(r.uniform(*cfg["vignette"]))
    lin = lin * (1 - vig * rr)[..., None]
    nlev = float(r.uniform(*cfg["noise"]))
    noise = r.normal(0, 1, lin.shape) * np.sqrt(nlev * np.clip(lin, 0, 1) * 0.5 + (nlev * 0.4) ** 2)
    lin = lin + noise
    out = np.clip(lin_to_srgb(lin) * 255 + r.uniform(0, 1, lin.shape), 0, 255).astype(np.uint8)
    return out, {"visibility_km": vis_km, "airlight_linear": [float(a) for a in airlight], "psf_sigma_px": sigma,
                 "vignette": vig, "noise_level": nlev}
