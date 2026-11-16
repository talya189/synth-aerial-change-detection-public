# Handoff: where the work stands

Read this first, then `README.md`, `docs/ARCHITECTURE.md`, `docs/PLACES.md` and `docs/LESSONS.md`.

## The project

A photorealistic, reproducible generator of synthetic aerial images with pixel-exact labels, for **change detection** (T1/T2 image pairs).

- **Python** makes every decision.
- A **JSON scene spec** lists every object, each with a permanent ID.
- **Blender 5.2 (Cycles)** builds and renders the scene.

The first region is the Galilee: Arab villages and cities.

**The command:** a scene is a place, a style and a seed.

```bash
python -m synth.make_scene --place arab_village --style hillside --seed 7 [--plan-only] [--regenerate]
python -m synth.gallery seed_0011 seed_0012     # sheets + index.html of rendered scenes
```

Outputs go to `~/synth_aerial_work/outputs/scenes/<place>/<style>/seed_NNNN/`:
- `plan_map.png`;
- `report.json`, with the rule checks;
- `scene.blend`;
- `views/cam_*/` with `*_rgb.png`, semantic and instance masks.

**How the user works:**
- Build step by step.
- Show images (plan map, renders) and the rule report at each step.
- Check realism against real aerial imagery.
- Move to the next step only once everything looks realistic and correct.
- Commit and push only when asked.
- Never commit the licensed Fab assets.
- No large dataset until the pipeline is validated.

## Done in the last session (this PR)

1. **Yard walls and fences** (`synth/spec/yards.py`, entity `walls`, IDs `wall_NNN`).
   - Each plot is the house grown by 3–7 m and shared fairly with the neighbours (Voronoi cells).
   - A gate sits where the house's way to its road crosses the wall.
   - Types: stone, block, plaster (the house's own colour) and fence (bars on a plinth).
   - Built in Blender by `building_builder.build_walls`.
   - New label class `wall_fence` (id 9).
   - Rule checks: `walls_through_buildings`, `walls_across_roads_paths`, `walled_in_buildings`.
2. **Rounded junction corners.**
   - Every asphalt junction gets kerb fillets (`to_build.junction_fillets`), with `corner_radius_m` per road class in `configs/roads.json`.
   - The old linear flare at a junction mouth is turned off.
3. **Footpaths connect.**
   - Outbound trails now end at another road or track, or at the door of an outlying building.
   - Rule check: `paths_ending_nowhere`.
4. **Road cross-sections** (`configs/roads.json` `sections`, spec road `section`).
   - Village main roads are `two_lane`, with a centre line and edge lines.
   - City main roads are `divided`: 2+2 lanes and a kerbed median that opens at junctions.
   - Some city streets are `two_lane`.
5. **Streetlights** (`synth/spec/props.py`, `synth/blender/props_builder.py`, `configs/props.json`).
   - New entity type `props`, IDs `prop_NNN`, label class `pole_street_furniture` (id 10).
   - Types: single, double on medians, small.
   - Rule check: `props_on_buildings_roads_paths`.
   - **The other street-furniture items should be added as more `props` types.**
6. **Mosques never side by side.** Landmarks keep `landmark_spacing_m` (default 220 m) apart. Rule check: `mosques_closer_than_200m`.
7. **Seed variety** (the user's complaint: different seeds looked almost the same).
   - **Areas:** each style lists several real Galilee areas (`terrain.aois`; 5 new areas in `configs/terrain.json`) and up to 30 sites in each.
   - **Character:** each style has a `vary` block; every seed draws its own street count, spacing and length, building count and density, building-type mix, wall share, groves, forest, fields, footpaths and main-road mode (`places.vary`, recorded in `spec.place.character`).
   - **Ground cover:** `ground_cover_scale` makes the wild bushes and stones sparse or dense.
   - **Light and camera:** the date, time of day (08:30–15:30) and haze, plus the camera headings, are drawn per seed.
8. **Tests:** 29 pass (`pytest tests -q`).

## Open issues found

- **Steep roads.** With the wider variety, some scenes have roads slightly over their grade limit: 18–21 % against 17–18 %, and 13 % against 12 %. Planning keeps them because the network depends on them (≤ 1.25× the limit), but `roads_too_steep` reports them.
  - Fix in `network.enforce_grades`, or choose gentler sites (`target_slope_deg`).
  - Two ridge cities also have a main road curve tighter than 50 m: `roads_tighter_than_class_radius`.
- **Render times.** The last local batch (seeds 11–13, all 4 styles) took 3.5–20 min per scene. One scene, hillside 11, was planned just before the footpath door-step fix, so re-plan it with `--regenerate`.
- **Basin city density.** It can fall below its requested building count; a drainage-line rule rejects many candidates.

## Left to do (the user's list, in order)

1. **Check the 12-scene variety render** (seeds 11, 12, 13 × 4 styles): plan and render them, run `python -m synth.gallery seed_0011 seed_0012 seed_0013`, and show the user.
2. **Buildings.**
   - Many more houses, much closer together, some directly adjoining (shared walls, row houses in the old core).
   - More variety: old houses, partly ruined buildings, corrugated-metal homes.
   - Mosque spacing is already done.
3. **Trees and gardens.**
   - Larger trees and more species, including olive.
   - Gardens (planted beds, vegetable plots, vines, pergolas inside yards) and fences throughout.
4. **Agriculture.** More realistic fields, and a mix of cultivated crop fields, fruit orchards and olive groves.
5. **Infrastructure:** utility poles, overhead power lines (a pole chain entity), transformers, road signs, traffic lights, small wooden structures such as stalls. Add these as `props` types.
6. **Ground details:** piles of ash.
7. **Optional, the user's idea:** use the built-up footprint of a real Arab village as the baseline, picking one per seed, for where and how many buildings go.

## Roadmap after the user's list

- **Step 7:** cameras and render settings for the dataset.
- **Step 8:** edit operations and T1/T2 pairs.
- **Step 9:** change masks.
- **Step 10:** automation and an LLM command schema.
- **Step 11:** the Negev.

## Cloud setup

- Linux setup uses local asset ZIPs in `../assets`. See the README for setup instructions.
- Without the assets, `--plan-only` still works: specs, plan maps and rule checks.
- Rendering on CPU is slow: plan first and render few scenes.
