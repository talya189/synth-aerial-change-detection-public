"""Step 1: minimal working setup: scene spec (JSON) → .blend → render.

    ~/synth_aerial_work/venv/bin/python -m synth.steps.step1 [configs/specs/step1_minimal.json]

Output: ~/synth_aerial_work/outputs/steps/step1/
  spec.json        the scene spec that was built (single source of truth)
  plan.json        derived build data (heights, foundations, cameras, sun)
  scene.blend      the Blender scene (open it to inspect every object)
  views/<cam>/     render, semantic / instance masks, passes EXR
  contact_sheet.jpg
"""
import sys
import time

from generate import blender_build
from synth.common.config import project_path
from synth.spec.scene_spec import load
from synth.spec.to_build import build_data


def main(spec_path="configs/specs/step1_minimal.json", out="outputs/steps/step1"):
    t0 = time.time()
    spec = load(project_path(spec_path))
    plan = build_data(spec, project_path(out))
    print(f"spec {spec['scene_id']}: {sum(len(v) for v in spec['entities'].values())} entities -> {plan}")
    scene_dir = blender_build(plan, log=print)
    print(f"step 1 done in {time.time() - t0:.0f}s -> {scene_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:])
