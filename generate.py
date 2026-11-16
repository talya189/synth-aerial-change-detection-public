"""Synthetic aerial scene generator - entry point (standard Python, the venv).

    ~/synth_aerial_work/venv/bin/python generate.py --seed 42
    ~/synth_aerial_work/venv/bin/python generate.py --seeds 1 2 3 --res 1024 1024 --samples 32

or from Python:

    from generate import generate_scene
    generate_scene(seed=42, scene_config="configs/scene/galilee_village.json",
                   output_path="outputs/scene_0042")

Stages: DTM prep (cached) -> layout plan (plan.json) -> Blender build, render
and annotation (headless subprocess) -> review contact sheet.
Heavy outputs go to $SYNTH_WORK_ROOT (default ~/synth_aerial_work), outside iCloud.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from synth.common.config import PROJECT_ROOT, load_json, project_path
from synth.planner.plan import plan_scene

BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")


def generate_scene(seed, scene_config="configs/scene/galilee_village.json", output_path=None,
                   dtm_path=None, assets_config="configs/assets.json", resolution=None, samples=None,
                   views=None, render=True, overrides=None, log=print):
    out = output_path or f"outputs/scene_{seed:04d}"
    ov = dict(overrides or {})
    if dtm_path:
        ov.setdefault("terrain", {})["files"] = [dtm_path]
    if assets_config != "configs/assets.json":
        ov["assets_config"] = assets_config
    t0 = time.time()
    plan_path = plan_scene(seed, scene_config, overrides=ov, out_dir=out, log=log)
    log(f"planned in {time.time() - t0:.1f}s")
    scene_dir = blender_build(plan_path, resolution, samples, views, render, log)
    log(f"scene {seed} done in {time.time() - t0:.1f}s -> {scene_dir}")
    return scene_dir


def blender_build(plan_path, resolution=None, samples=None, views=None, render=True, log=print):
    """Build (and optionally render + annotate) a plan/build JSON in headless Blender."""
    cmd = [BLENDER, "-b", "--factory-startup", "--python-exit-code", "1",
           "-P", os.path.join(PROJECT_ROOT, "synth", "blender", "run_build.py"), "--", "--plan", plan_path]
    if resolution:
        cmd += ["--res", str(resolution[0]), str(resolution[1])]
    if samples:
        cmd += ["--samples", str(samples)]
    if views:
        cmd += ["--views", *views]
    if not render:
        cmd += ["--no-render"]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if proc.returncode < 0:  # killed by a signal (intermittent Metal abort): retry once
        log(f"Blender exited with signal {-proc.returncode}; retrying once")
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    blog = os.path.join(os.path.dirname(plan_path), "blender_log.txt")
    with open(blog, "w") as f:
        f.write(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    for line in proc.stdout.splitlines():
        if line.startswith("[build]"):
            log(line)
    if proc.returncode != 0:
        raise RuntimeError(f"Blender failed (exit {proc.returncode}); see {blog}")
    meta = load_json(os.path.join(os.path.dirname(plan_path), "metadata.json"))
    if render:
        contact_sheet(os.path.dirname(plan_path), meta)
    return os.path.dirname(plan_path)


def contact_sheet(scene_dir, meta):
    """RGB | semantic overlay per view, for quick visual review."""
    from PIL import Image
    tiles = []
    for v in meta["views"]:
        d = os.path.join(scene_dir, "views", v["name"])
        rgb = Image.open(os.path.join(d, f"{v['name']}_rgb.png")).convert("RGB")
        sem = Image.open(os.path.join(d, f"{v['name']}_semantic_color.png")).convert("RGB")
        over = Image.blend(rgb, sem, 0.45)
        w = 640
        h = int(rgb.height * w / rgb.width)
        row = Image.new("RGB", (2 * w, h))
        row.paste(rgb.resize((w, h)), (0, 0))
        row.paste(over.resize((w, h)), (w, 0))
        tiles.append(row)
    if tiles:
        sheet = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)))
        y = 0
        for t in tiles:
            sheet.paste(t, (0, y))
            y += t.height
        sheet.save(os.path.join(scene_dir, "contact_sheet.jpg"), quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", help="batch: several seeds")
    ap.add_argument("--scene-config", default="configs/scene/galilee_village.json")
    ap.add_argument("--dtm", help="override DTM GeoTIFF path")
    ap.add_argument("--out", help="output dir (single seed)")
    ap.add_argument("--res", type=int, nargs=2)
    ap.add_argument("--samples", type=int)
    ap.add_argument("--views", nargs="*")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    seeds = a.seeds or [a.seed]
    for s in seeds:
        generate_scene(s, a.scene_config, a.out if len(seeds) == 1 else None, a.dtm, resolution=a.res,
                       samples=a.samples, views=a.views, render=not a.no_render)


if __name__ == "__main__":
    main()
