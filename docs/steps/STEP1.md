# Step 1: architecture and minimal working setup

## What was built and why

The three-component architecture from [ARCHITECTURE.md](../ARCHITECTURE.md), reduced to its smallest working loop:

```
configs/specs/step1_minimal.json  ──►  synth/spec/to_build.py  ──►  run_build.py (Blender, from template.blend)
   (scene spec: ids, exact geometry,      (derives heights,           ──►  scene.blend + renders + masks
    environment, cameras, seeds)           foundations, sun, cameras)
```

- `synth/spec/scene_spec.py`: the spec format (documented in the module), validation (required fields, unique ids, id pattern `bld_001` …), `next_id` (ids are never reused), `find`.
- `configs/specs/step1_minimal.json`: a 240 m flat test plot at Yeruham (northern Negev) with:
  - 2 roads (asphalt and dirt), 1 footpath;
  - 3 buildings, 3 trees;
  - 2 scatter systems (sparse desert shrubs, stones);
  - 2 cameras;
  - date 15 June, 10:30 local.
- `synth/spec/to_build.py`: spec → build data. Per-entity random streams are keyed by the entity id.
- `synth/steps/step1.py`: the runner.

## How to run

```bash
cd <project folder>
~/synth_aerial_work/venv/bin/python -m synth.steps.step1
open -a Blender ~/synth_aerial_work/outputs/steps/step1/scene.blend
```

Runtime: about 1 minute (2 views at 1024², 64 samples).

## What to inspect

`~/synth_aerial_work/outputs/steps/step1/`:
- `contact_sheet.jpg`: render plus label overlay for both cameras;
- `views/<camera>/*_rgb.png` and the masks;
- `scene.blend`.

In the `.blend`, object names are the spec ids (`road_001`, `bld_002`, `tree_003`, `scatter_stones`, `cam_nadir`), grouped in collections 01–06.

## Validation done

- All 9 entities appear where the spec places them.
- Labels are valid in both views: no stray sky. The masks overlay the image.
- Grounding check: 0 floating objects or buildings.
- Spec validation passes; 7/7 tests pass.

Bug found and fixed: the template's demo playground terrain was rendering inside generated scenes (a white unlabelled patch). Scene builds now delete the playground. The Galilee test scene built from the template before this fix had the same artefact near its origin.

## Not yet realistic (deliberately, later steps)

- The flat plot stands in for the real DEM (step 2).
- The desert ground layers are not tuned yet: blotchy gravel and dark "trampled" halos around houses (step 3).
- Pines are not Negev vegetation (step 6, which needs acacia/tamarisk assets).
