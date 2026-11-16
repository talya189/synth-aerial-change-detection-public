"""Gallery of rendered scenes: one sheet per scene (plan map | nadir | oblique | low) + index.html.

    ~/synth_aerial_work/venv/bin/python -m synth.gallery [seed_0011 seed_0012 ...]

Writes ~/synth_aerial_work/outputs/gallery/ (only scenes with all three views rendered)."""
import glob
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

from synth.common.config import project_path

root = project_path("outputs/scenes")
out = project_path("outputs/gallery")
os.makedirs(out, exist_ok=True)
seeds = set(sys.argv[1:])
T = 700
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
except OSError:
    font = ImageFont.load_default()
rows = []
for d in sorted(glob.glob(f"{root}/*/*/seed_*")):
    place, style, seed = d.split("/")[-3:]
    if seeds and seed not in seeds:
        continue
    views = [f"{d}/views/cam_{c}/cam_{c}_rgb.png" for c in ("nadir", "oblique", "low")]
    if not all(os.path.exists(v) for v in views):
        continue
    rep = json.load(open(f"{d}/report.json"))
    tiles = [Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS) for p in [f"{d}/plan_map.png"] + views]
    sheet = Image.new("RGB", (4 * T, T + 50), (255, 255, 255))
    for k, t in enumerate(tiles):
        sheet.paste(t, (k * T, 50))
    c = rep["counts"]
    fails = [k for k in ("crossings_without_junction", "unreachable_roads", "roads_too_steep", "buildings_breaking_rules",
                         "buildings_without_access", "access_paths_through_buildings", "roads_tighter_than_class_radius",
                         "walls_through_buildings", "walls_across_roads_paths", "walled_in_buildings") if rep.get(k)]
    txt = (f"{place} / {style} / {seed}   buildings {c['buildings']}, roads {c['roads']}, walls {c.get('walls', 0)}, "
           f"trees {c['trees']}, groves {c['groves']}   rules: {'all pass' if not fails else 'FAIL ' + ', '.join(fails)}")
    ImageDraw.Draw(sheet).text((12, 10), txt, fill=(0, 0, 0), font=font)
    name = f"{place}_{style}_{seed}.jpg"
    sheet.save(f"{out}/{name}", quality=88)
    rows.append((name, txt))
html = ["<!doctype html><meta charset=utf-8><title>Scene gallery</title>",
        "<style>body{font:15px sans-serif;margin:16px;background:#f4f4f4}img{width:100%;margin-bottom:6px}"
        "h3{margin:22px 0 6px}</style><h1>Scene gallery</h1><p>Each row: plan map | nadir | oblique | low view. Click to open full size.</p>"]
html += [f"<h3>{t}</h3><a href='{n}'><img src='{n}'></a>" for n, t in rows]
open(f"{out}/index.html", "w").write("\n".join(html))
print(len(rows), "scenes ->", out)
