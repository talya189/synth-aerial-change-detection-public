"""Reproducibility check: regenerate a scene from its seed and compare.

    ~/synth_aerial_work/venv/bin/python -m synth.dataset.verify_repro \
        configs/dataset/galilee_smoke.json 1000

Compares against the dataset's existing scene: plan.json (must be identical),
labels.json and every semantic / instance mask (must be byte-identical), and
the RGB images (reports mean / max absolute difference; GPU path tracing may
differ at noise level between runs).
"""
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

from synth.common.config import load_json, project_path


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _strip(plan):
    p = dict(plan)
    for k in ("dtm_npz",):
        p.pop(k, None)
    return p


def verify(dcfg_path, seed):
    from generate import generate_scene
    d = load_json(dcfg_path)
    ref = project_path(os.path.join(d["output"], "scenes", f"scene_{seed:05d}"))
    if not os.path.exists(os.path.join(ref, "metadata.json")):
        raise SystemExit(f"reference scene not found: {ref}")
    rerun = project_path(os.path.join(d["output"], "repro_check", f"scene_{seed:05d}"))
    generate_scene(seed, d["scene_config"], output_path=rerun, overrides=d.get("overrides"), log=lambda *a: None)
    report = {"seed": seed,
              "plan_identical": _strip(load_json(os.path.join(ref, "plan.json"))) == _strip(load_json(os.path.join(rerun, "plan.json"))),
              "labels_identical": load_json(os.path.join(ref, "labels.json")) == load_json(os.path.join(rerun, "labels.json")),
              "views": []}
    for v in load_json(os.path.join(ref, "metadata.json"))["views"]:
        a, b = os.path.join(ref, "views", v["name"]), os.path.join(rerun, "views", v["name"])
        row = {"view": v["name"]}
        for k in ("semantic", "instance"):
            f = f"{v['name']}_{k}.png"
            row[f"{k}_identical"] = _sha(os.path.join(a, f)) == _sha(os.path.join(b, f))
        ra = np.asarray(Image.open(os.path.join(a, f"{v['name']}_rgb.png")), np.float32)
        rb = np.asarray(Image.open(os.path.join(b, f"{v['name']}_rgb.png")), np.float32)
        diff = np.abs(ra - rb)
        row["rgb_mean_abs_diff"] = float(diff.mean())
        row["rgb_max_abs_diff"] = float(diff.max())
        report["views"].append(row)
    report["masks_identical"] = all(r["semantic_identical"] and r["instance_identical"] for r in report["views"])
    out = project_path(os.path.join(d["output"], "repro_check", f"report_{seed:05d}.json"))
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    verify(sys.argv[1], int(sys.argv[2]))
