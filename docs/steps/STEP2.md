# Step 2: real terrain (Korazim area)

## What was built and why

The scene spec now supports `"region": {"type": "dem", ...}`. Terrain comes from real elevation data instead of a flat test plot.

- **Data:** Copernicus DEM GLO-30 (30 m), tiles N32/N33 E035, over the Korazim / Rosh Pina area (`data/aoi_korazim.gpkg`). The DSM's building and tree bumps are softly removed. The files live in `~/synth_aerial_work/data`.
- **Site choice:**
  - candidate 1 km squares inside the area are ranked: moderate median slope, some relief, separated from each other;
  - `site_rank` in the spec picks one;
  - the resolved coordinates are written back into the spec, so the scene is reproducible;
  - one shared function does the ranking, so the overview map and the terrain always agree (a mismatch was found and fixed).
- **Terrain rings:**
  - 1 m core (1 km) with band-limited micro-relief, faded out at the edge;
  - 5 m middle ring (3.2 km);
  - 30 m far field (18 km) to the horizon;
  - the Sea of Galilee as a water plane.
- **Hydrology:** wetness now uses **multiple-direction flow routing** on the smoothed surface. The first version (single-direction routing) produced parallel diagonal streaks that would have become stripes of shrubs.

Chosen site, rank 0:
- 32.9546 °N, 35.6621 °E, about 116 m above sea level, east of the Jordan gorge;
- 117 m of relief inside the 1 km core;
- a plateau cut by a wadi gorge.

## How to run

```bash
~/synth_aerial_work/venv/bin/python -m synth.steps.step2
open -a Blender ~/synth_aerial_work/outputs/steps/step2/scene.blend
```

Runtime: about 2 minutes. To try another site, change `site_rank` in `configs/specs/step2_terrain.json` (0–5 are drawn on the overview).

## What to inspect (`~/synth_aerial_work/outputs/steps/step2/`)

- `site_overview.png`: the whole area, shaded relief, blue boundary, candidates 0–5, the chosen one in red.
- `terrain_maps.png`: the 1 km core: shaded relief, slope, wetness.
- `contact_sheet.jpg`: three views (straight down at 1 m/pixel, 45°, low angle to the horizon), each with its label overlay.
- `scene.blend`: collection `01_TERRAIN` (core, skirt, middle ring, far field, water).

## Validation

- The relief matches the DEM: plateau, gorge, steeper walls. The horizon view shows the far field and the lake.
- Labels: terrain everywhere; the lake is "background"; the sky is detected in the horizon view.
- 7/7 tests pass.

## Known limitations (handled in later steps)

- **Ground materials** read as uniform tan "sand", and the rocky gorge walls are flat blue-grey patches (Step 3).
- **No rocks, grass, shrubs or trees yet**, which is why it looks barren (Step 6).
- **30 m source data** smooths the wadi walls; the fine detail is synthetic.
