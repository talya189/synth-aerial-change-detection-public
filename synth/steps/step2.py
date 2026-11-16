"""Step 2: real terrain for the Korazim area.

    ~/synth_aerial_work/venv/bin/python -m synth.steps.step2 [configs/specs/step2_terrain.json]

Output: ~/synth_aerial_work/outputs/steps/step2/
  site_overview.png   the whole AOI (shaded relief) with the ranked candidate sites
  terrain_maps.png    the chosen 1 km core: shaded relief, slope, wetness
  spec.json, plan.json, scene.blend, views/, contact_sheet.jpg
"""
import json
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from generate import blender_build
from synth.common.config import load_json, project_path
from synth.prep import dtm as dtm_prep
from synth.spec.scene_spec import load
from synth.spec.to_build import build_data


def hillshade(z, res, az=315, alt=45):
    gy, gx = np.gradient(z, res)
    sl = np.arctan(np.hypot(gx, gy))
    asp = np.arctan2(-gx, gy)
    h = np.sin(np.radians(alt)) * np.cos(sl) + np.cos(np.radians(alt)) * np.sin(sl) * np.cos(np.radians(az) - asp)
    return np.clip(h, 0, 1)


def site_overview(region, out_png, top=6):
    """AOI shaded relief + polygon + the top candidate core squares (number = site_rank)."""
    import pyproj
    from shapely.ops import transform as shp_tf
    tcfg = load_json("configs/terrain.json")
    src, aoi = tcfg["sources"][region["source"]], tcfg["aois"][region["aoi"]]
    arr, tf, crs = dtm_prep._mosaic(src["files"])
    poly, _ = dtm_prep.read_gpkg_polygon(aoi["polygon_file"])
    epsg = dtm_prep.utm_epsg(poly.centroid.x, poly.centroid.y)
    to_utm = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    pu = shp_tf(lambda x, y, z=None: to_utm.transform(x, y), poly)
    info = dtm_prep.aoi_sites(arr, tf, crs, pu, epsg, region.get("core_size_m", 1000), aoi["site_selection"])
    sites, z, res = info["sites"], info["z10"], 10.0
    minx, miny, pad = info["x0"], info["y0"], 0.0
    hs = hillshade(z, res)
    zn = (z - z.min()) / (z.max() - z.min())
    rgb = np.stack([0.55 + 0.45 * zn, 0.5 + 0.35 * zn, 0.4 + 0.2 * zn], -1) * hs[..., None]
    im = Image.fromarray((rgb[::-1] * 255).astype(np.uint8)).convert("RGB")
    W, H = im.size
    dr = ImageDraw.Draw(im)
    P = lambda x, y: ((x - (minx - pad)) / res, H - (y - (miny - pad)) / res)
    dr.line([P(x, y) for x, y in pu.exterior.coords], fill=(30, 30, 200), width=2)
    half = region.get("core_size_m", 1000) / 2
    chosen = int(region.get("site_rank", 0))
    for k, (cx, cy) in enumerate(sites[:top]):
        col = (230, 30, 30) if k == chosen else (255, 255, 255)
        dr.rectangle([P(cx - half, cy + half), P(cx + half, cy - half)], outline=col, width=3 if k == chosen else 1)
        dr.text(P(cx - half + 12, cy + half - 12), str(k), fill=col)
    im = im.resize((W * 2, H * 2))
    im.save(out_png)
    return [(float(x), float(y)) for x, y in sites[:top]]


def terrain_maps(scene_dir, out_png):
    d = np.load(f"{scene_dir}/terrain.npz")
    z, wet = d["core_z"].astype(float), d["wet"]
    gy, gx = np.gradient(z, 1.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    a = (hillshade(z, 1.0) * 255).astype(np.uint8)
    b = (np.clip(slope / 35, 0, 1) * 255).astype(np.uint8)
    c = (wet * 255).astype(np.uint8)
    tiles = [Image.fromarray(x[::-1]).convert("RGB").resize((500, 500)) for x in (a, b, c)]
    g = Image.new("RGB", (1500, 530), (255, 255, 255))
    dr = ImageDraw.Draw(g)
    for k, (t, lab) in enumerate(zip(tiles, ("shaded relief (1 m)", "slope 0-35 deg", "wetness (drainage lines)"))):
        g.paste(t, (k * 500, 30))
        dr.text((k * 500 + 8, 8), lab, fill=(0, 0, 0))
    g.save(out_png)


def main(spec_path="configs/specs/step2_terrain.json", out="outputs/steps/step2"):
    t0 = time.time()
    spec = load(project_path(spec_path))
    out_dir = project_path(out)
    plan = build_data(spec, out_dir)
    sp = load_json(f"{out_dir}/spec.json")
    print("site", sp["region"]["resolved"]["origin_latlon"], "rank", sp["region"]["resolved"]["site_rank"])
    sites = site_overview(spec["region"], f"{out_dir}/site_overview.png")
    print("top candidate sites (UTM):", [tuple(round(v) for v in s) for s in sites])
    terrain_maps(out_dir, f"{out_dir}/terrain_maps.png")
    blender_build(plan, log=print)
    print(f"step 2 done in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:])
