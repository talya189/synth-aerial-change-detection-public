"""DTM preprocessing (standard Python venv: rasterio, pyproj, scipy, shapely).

Produces a cached .npz with two regular UTM grids in *local* coordinates
(origin = core centre, so Blender never sees large UTM numbers):

  core_z  (n, n)  at core_res_m   covering [-core/2, core/2]^2
  far_z   (m, m)  at far_res_m    covering [-far_radius, far_radius]^2

Row 0 is the SOUTH edge (row index grows with +Y), column 0 the WEST edge.
"""
import hashlib
import json
import math
import os
import sqlite3
import struct

import numpy as np
import pyproj
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from shapely import wkb

from synth.common.config import load_json, project_path, save_json


def read_gpkg_polygon(path):
    """Reads the first geometry of a GeoPackage without GDAL/fiona."""
    con = sqlite3.connect(project_path(path))
    table, srs = con.execute("select table_name, srs_id from gpkg_contents where data_type='features'").fetchone()
    col = con.execute("select column_name from gpkg_geometry_columns where table_name=?", (table,)).fetchone()[0]
    blob = con.execute(f'select "{col}" from "{table}" limit 1').fetchone()[0]
    con.close()
    flags = blob[3]
    env_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 0b111]
    return wkb.loads(bytes(blob[8 + env_len:])), srs


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _mosaic(files):
    srcs = [rasterio.open(project_path(f)) for f in files]
    arr, transform = merge(srcs, nodata=-32767)
    crs = srcs[0].crs
    for s in srcs:
        s.close()
    return arr[0].astype(np.float32), transform, crs


def _warp(src, src_tf, src_crs, dst_crs, x0, y0, res, n, resampling):
    """Warp into an n x n north-up grid with top-left (x0, y0 + n*res)."""
    dst = np.full((n, n), np.nan, np.float32)
    reproject(src, dst, src_transform=src_tf, src_crs=src_crs, src_nodata=-32767,
              dst_transform=from_origin(x0, y0 + n * res, res, res), dst_crs=dst_crs,
              dst_nodata=np.nan, resampling=resampling)
    return dst[::-1].copy()  # row 0 = south


def _fill_voids(z):
    mask = ~np.isfinite(z)
    if not mask.any():
        return z, 0
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return z[tuple(idx)], int(mask.sum())


def dsm_to_dtm(z, res, cfg):
    """Suppress DSM object bumps (houses, tree canopy) with a morphological opening."""
    w = max(3, int(round(cfg["opening_window_m"] / res)) | 1)
    opened = ndimage.grey_opening(z, size=(w, w))
    opened = ndimage.gaussian_filter(opened, w / 4)
    excess = np.maximum(z - opened, 0)
    thr = cfg["object_threshold_m"]
    # soft, blurred weight: a hard mask leaves step seams at its boundary
    w = np.clip((excess - thr) / thr, 0, 1)
    w = ndimage.gaussian_filter(w, max(1.0, 8.0 / res))
    out = z - w * excess * (1 - cfg["keep_fraction"])
    return out.astype(np.float32), float((w > 0.5).mean())


def slope_deg(z, res):
    gy, gx = np.gradient(z, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def micro_relief(z, res, cfg, seed):
    """Band-limited fractal detail that leaves the 30 m-scale shape untouched.
    Seeded from the site (not the scene seed) so one place always has the same ground."""
    rs = np.random.default_rng(seed)
    acc = np.zeros_like(z)
    for wl in cfg["wavelengths_m"]:
        n = rs.standard_normal(z.shape).astype(np.float32)
        band = ndimage.gaussian_filter(n, wl / res / 2.5)
        band /= band.std() + 1e-9
        acc += band * (wl / max(cfg["wavelengths_m"])) ** 0.8
    acc /= acc.std() + 1e-9
    acc -= ndimage.gaussian_filter(acc, 30 / res)  # high-pass: keep the DTM's own shape
    gain = 1 + cfg["slope_gain"] * np.clip(slope_deg(z, res) / 30, 0, 1)
    # fade out toward the core edge so the core joins the (smooth) mid ring without a shading seam
    n = z.shape[0]
    c = np.abs(np.arange(n) - (n - 1) / 2) * res
    cheb = np.maximum(c[:, None], c[None, :])
    half = (n - 1) / 2 * res
    taper = np.clip((half - cheb) / cfg.get("edge_taper_m", 80.0), 0, 1)
    return z + cfg["amplitude_m"] * acc * gain * taper


def rank_sites(poly_utm, z, x0, y0, res, core_m, sel):
    """Candidate core centres inside the AOI, best first, separated by at least
    `min_separation_frac` x core size (non-maximum suppression). Score: median
    slope in the target range, preferring its middle, plus some relief variety."""
    sl = slope_deg(z, res)
    lo, hi = sel["target_median_slope_deg"]
    half = core_m / 2
    minx, miny, maxx, maxy = poly_utm.bounds
    step = sel["search_step_m"]
    from shapely.geometry import box
    cands = []
    for cx in np.arange(minx + half, maxx - half, step):
        for cy in np.arange(miny + half, maxy - half, step):
            if not poly_utm.contains(box(cx - half, cy - half, cx + half, cy + half)):
                continue
            j0, j1 = int((cx - half - x0) / res), int((cx + half - x0) / res)
            i0, i1 = int((cy - half - y0) / res), int((cy + half - y0) / res)
            win = sl[i0:i1, j0:j1]
            if win.size == 0:
                continue
            med = float(np.median(win))
            score = -abs(med - (lo + hi) / 2) + (0 if lo <= med <= hi else -10) + 0.02 * float(np.percentile(win, 90))
            cands.append((score, float(cx), float(cy)))
    cands.sort(reverse=True)
    sep = sel.get("min_separation_frac", 0.6) * core_m
    kept = []
    for c in cands:
        if all(np.hypot(c[1] - k[1], c[2] - k[2]) >= sep for k in kept):
            kept.append(c)
    return [(cx, cy) for _, cx, cy in kept]


def rank_landform(poly_utm, z, x0, y0, res, core_m, sel):
    """Candidate core centres for a settlement landform, best first (separated by
    min_separation_frac x core). Modes (sel["mode"]):
      hillside  mean slope in target_slope_deg, steady (one main aspect)
      hilltop   centre higher than its surroundings (convex), moderate slope
      basin     centre lower than its surroundings (concave), gentle floor
      ridge     strong relief and steeper slopes (settlement over ridges)
    Works on a coarse grid (30 m) with moving-window statistics, so large AOIs
    are searched in seconds."""
    from scipy import ndimage
    from shapely.geometry import box
    sl = slope_deg(z, res)
    w = max(3, int(core_m / res))
    mean_sl = ndimage.uniform_filter(sl, w)
    zc = ndimage.uniform_filter(z, max(3, int(0.35 * core_m / res)))            # the settlement's own ground
    ring = ndimage.uniform_filter(z, max(5, int(2.2 * core_m / res)))           # its surroundings
    contrast = zc - ring                                                        # + hilltop, - basin
    relief = ndimage.maximum_filter(z, w) - ndimage.minimum_filter(z, w)
    gy, gx = np.gradient(ndimage.gaussian_filter(z, w / 4), res)
    steady = np.hypot(ndimage.uniform_filter(gx, w), ndimage.uniform_filter(gy, w)) / (
        ndimage.uniform_filter(np.hypot(gx, gy), w) + 1e-6)                      # 1 = one aspect, 0 = mixed
    lo, hi = sel.get("target_slope_deg", [4, 14])
    mode = sel.get("mode", "hillside")
    in_band = -np.abs(mean_sl - (lo + hi) / 2) - 10 * ((mean_sl < lo) | (mean_sl > hi))
    if mode == "hillside":
        score = in_band + 6 * steady
    elif mode == "hilltop":
        score = in_band + np.clip(contrast, -50, 60) / 6
    elif mode == "basin":
        score = in_band - np.clip(contrast, -60, 50) / 6
    else:   # ridge
        score = in_band + relief / 30
    half = core_m / 2
    minx, miny, maxx, maxy = poly_utm.bounds
    step = sel.get("search_step_m", 150)
    cands = []
    for cx in np.arange(minx + half, maxx - half, step):
        for cy in np.arange(miny + half, maxy - half, step):
            if not poly_utm.contains(box(cx - half, cy - half, cx + half, cy + half)):
                continue
            i, j = int((cy - y0) / res), int((cx - x0) / res)
            if 0 <= i < z.shape[0] and 0 <= j < z.shape[1]:
                cands.append((float(score[i, j]), float(cx), float(cy)))
    cands.sort(reverse=True)
    sep = sel.get("min_separation_frac", 1.2) * core_m
    kept = []
    for c in cands:
        if all(np.hypot(c[1] - k[1], c[2] - k[2]) >= sep for k in kept):
            kept.append(c)
        if len(kept) >= sel.get("max_sites", 40):
            break
    return [(cx, cy) for _, cx, cy in kept]


def aoi_polygon(aoi_cfg):
    """AOI polygon in lon/lat: from a GeoPackage, or a bbox_latlon [lat0, lon0, lat1, lon1]."""
    if "bbox_latlon" in aoi_cfg:
        from shapely.geometry import box
        la0, lo0, la1, lo1 = aoi_cfg["bbox_latlon"]
        return box(lo0, la0, lo1, la1)
    return read_gpkg_polygon(aoi_cfg["polygon_file"])[0]


def aoi_sites(arr, tf, crs, poly_utm, epsg, core_size_m, sel):
    """Ranked candidate sites over the AOI on a 10 m grid. The single place the
    ranking is computed (terrain preparation and the overview map both use it)."""
    minx, miny, maxx, maxy = poly_utm.bounds
    if "mode" in sel:   # landform search on a 30 m grid (large AOIs)
        n30 = int(max(maxx - minx, maxy - miny) / 30) + 1
        z30 = _warp(arr, tf, crs, f"EPSG:{epsg}", minx, miny, 30.0, n30, Resampling.bilinear)
        z30, _ = _fill_voids(z30)
        return {"sites": rank_landform(poly_utm, z30, minx, miny, 30.0, core_size_m, sel), "z10": z30, "x0": minx, "y0": miny}
    n10 = int(max(maxx - minx, maxy - miny) / 10) + 1
    z10 = _warp(arr, tf, crs, f"EPSG:{epsg}", minx, miny, 10.0, n10, Resampling.cubic)
    z10, _ = _fill_voids(z10)
    return {"sites": rank_sites(poly_utm, z10, minx, miny, 10.0, core_size_m, sel), "z10": z10, "x0": minx, "y0": miny}


def prepare(source="copernicus_glo30", aoi="korazim", core_size_m=700.0, centre_latlon=None, force=False, files=None, site_rank=0,
            selection=None):
    cfg = load_json("configs/terrain.json")
    src_cfg, aoi_cfg, proc = cfg["sources"][source], cfg["aois"][aoi], cfg["processing"]
    if files:  # user-supplied GeoTIFF(s) override the named source
        src_cfg = dict(src_cfg, files=list(files), licence="user supplied")
    sel = dict(aoi_cfg.get("site_selection", {}), **(selection or {}))
    key = json.dumps([source, aoi, core_size_m, centre_latlon, proc, src_cfg["files"], site_rank] + ([sel] if selection else []),
                     sort_keys=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    out_npz = project_path("cache", "dtm", f"{aoi}_{h}.npz")
    if os.path.exists(out_npz) and not force:
        return out_npz

    arr, tf, crs = _mosaic(src_cfg["files"])
    poly = aoi_polygon(aoi_cfg)
    c = poly.centroid
    epsg = utm_epsg(c.x, c.y)
    to_utm = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    from shapely.ops import transform as shp_transform
    poly_utm = shp_transform(lambda x, y, z=None: to_utm.transform(x, y), poly)

    # coarse site search at 10 m over the AOI
    if centre_latlon:
        cx, cy = to_utm.transform(centre_latlon[1], centre_latlon[0])
    else:
        sites = aoi_sites(arr, tf, crs, poly_utm, epsg, core_size_m, sel)["sites"]
        if not sites:
            raise RuntimeError(f"no {core_size_m} m site fits inside AOI {aoi}")
        cx, cy = sites[site_rank % len(sites)]
    cx, cy = round(cx), round(cy)

    # core grid
    res = proc["core_res_m"]
    n = int(round(core_size_m / res)) + 1
    half = (n - 1) * res / 2
    core = _warp(arr, tf, crs, f"EPSG:{epsg}", cx - half - res / 2, cy - half - res / 2, res, n, Resampling.cubic)
    core, voids_core = _fill_voids(core)
    if src_cfg["kind"] == "DSM" and proc["dsm_to_dtm"]["enabled"]:
        core, obj_frac = dsm_to_dtm(core, res, proc["dsm_to_dtm"])
    else:
        obj_frac = 0.0
    raw_core = core.copy()
    if proc["micro_relief"]["enabled"]:
        core = micro_relief(core, res, proc["micro_relief"], seed=int(cx) * 1000003 + int(cy))

    # mid ring (5 m): foreground-to-horizon transition for oblique views
    mres = proc["mid_res_m"]
    mn = int(2 * proc["mid_half_m"] / mres) + 1
    mhalf = (mn - 1) * mres / 2
    mid = _warp(arr, tf, crs, f"EPSG:{epsg}", cx - mhalf - mres / 2, cy - mhalf - mres / 2, mres, mn, Resampling.cubic)
    mid, voids_mid = _fill_voids(mid)
    if src_cfg["kind"] == "DSM" and proc["dsm_to_dtm"]["enabled"]:
        mid, _ = dsm_to_dtm(mid, mres, proc["dsm_to_dtm"])

    # far field
    fres = proc["far_res_m"]
    m = int(2 * proc["far_radius_m"] / fres) + 1
    fhalf = (m - 1) * fres / 2
    far = _warp(arr, tf, crs, f"EPSG:{epsg}", cx - fhalf - fres / 2, cy - fhalf - fres / 2, fres, m, Resampling.bilinear)
    far, voids_far = _fill_voids(far)
    if src_cfg["kind"] == "DSM" and proc["dsm_to_dtm"]["enabled"]:
        far, _ = dsm_to_dtm(far, fres, dict(proc["dsm_to_dtm"], opening_window_m=max(90, 3 * fres)))

    z0 = float(np.round(np.median(core)))
    to_ll = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = to_ll.transform(cx, cy)
    meta = {
        "source": source, "source_files": src_cfg["files"], "source_kind": src_cfg["kind"],
        "source_sha256": [hashlib.sha256(open(project_path(f), "rb").read()).hexdigest() for f in src_cfg["files"]],
        "licence": src_cfg["licence"], "aoi": aoi, "epsg": epsg,
        "origin_utm": [cx, cy, z0], "origin_latlon": [lat, lon], "site_rank": site_rank,
        "core_res_m": res, "core_n": n, "core_half_m": half,
        "far_res_m": fres, "far_n": m, "far_half_m": fhalf,
        "mid_res_m": mres, "mid_n": mn, "mid_half_m": mhalf,
        "voids_filled": voids_core + voids_mid + voids_far, "dsm_object_fraction": obj_frac,
        "core_z_range": [float(core.min()), float(core.max())],
        "row0": "south", "processing": proc,
        "core_bounds_latlon": [list(to_ll.transform(cx - half, cy - half))[::-1], list(to_ll.transform(cx + half, cy + half))[::-1]],
    }
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, core_z=(core - z0).astype(np.float32), core_z_smooth=(raw_core - z0).astype(np.float32),
                        mid_z=(mid - z0).astype(np.float32), far_z=(far - z0).astype(np.float32))
    save_json(meta, out_npz.replace(".npz", ".json"))
    return out_npz


if __name__ == "__main__":
    import sys
    p = prepare(force="--force" in sys.argv)
    print(p)
    print(open(p.replace(".npz", ".json")).read()[:1500])
