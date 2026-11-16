# Step 3: ground materials (same Korazim site)

## Revision 2: ground matched to real imagery

The first version below used invented colour targets (golden straw) and a dark-green meadow scan as "grass", and its noise produced islands of bare soil ("cow print"). It looked wrong. Revision 2 starts from satellite imagery of the site (Esri World Imagery, ~0.5 m/pixel) and from a survey of every ground asset.

- **What the real site looks like:**
  - open ground grey-brown, about (112, 108, 88), not golden;
  - dense dark-green shrubs and small trees (Ziziphus, Tabor oak), about 20 % cover, crowns 1–6 m;
  - a network of thin light cattle trails;
  - green vegetation lining the wadis and covering the gorge walls;
  - no hard-edged soil patches.
- **Asset survey:** every ground scan rendered raw from the air, at 12.5 and 3 cm/pixel.

  | Use | Asset | Note |
  |---|---|---|
  | soil | dry_soil_ground | |
  | dry grass | trampled_forest_path | recoloured; its clumps read as tufts |
  | trails, shoulders, yards, light stony ground | dirt_road | |
  | rocky slopes | rocky_ground | |
  | wadi beds | dirt_ground | |
  | (dropped) | wild_grass | a dark green meadow tile only 41 cm wide, which needed ×7.9 brightening |

  The other gravel scans are red Mediterranean soils and sand is Negev, so they are not used here.
- **No islands:**
  - each layer has a smooth `density` (how much of it shows) that changes over tens of metres;
  - layers mix at tuft scale (`dither`), so sparse grass shows soil between the tufts instead of forming blobs;
  - `lines` adds the thin trail network.
- **Wadis from drained area:** a cell is a channel when more than ~1,500 m² drains through it. The old wetness index also marked flat ground, which caused the dark blobs on the plateau. Buildings keep out of channels.
- **Ground cover** (`configs/presets/ground_cover/galilee_summer.json`):
  - basalt stones and boulders, darkened to basalt;
  - procedural dense shrubs (`synth/blender/proc_veg.py`), because no green-shrub scan is available: cushion shrubs, large shrubs and small trees (many small, few large), slope maquis on steep ground, wadi vegetation.
- **Shared presets:** the material and ground cover are preset files that specs name (`"terrain_material": "galilee_summer"`), so a season or region change is one line. `--calibrate` writes into the preset.

**Result** (`~/synth_aerial_work/outputs/steps/step3_compare/real_vs_synthetic.png`, same area and scale):

| colour (sRGB) | real | synthetic |
|---|---|---|
| whole image | (86, 88, 70) | (86, 85, 70) |
| open ground (brightest 40 %) | (112, 108, 88) | (111, 107, 88) |
| vegetation cover | ~20 % | 22 % |

Known gaps:
- the real trees are slightly lighter and more varied in green;
- there are no reddish bare-soil areas on the gorge rims;
- the shrubs are procedural until scanned green shrubs and trees are added.

---

# Revision 1 (superseded where it differs)

## What was built and why

Step 2's ground read as uniform tan "sand", with flat blue-grey patches on the gorge walls. Measuring each layer showed why: every layer rendered as nearly the same dull grey-brown, about (105, 92, 78). There was no contrast between grass, soil and rock.

- **Swatch renderer** (`synth/blender/swatches.py`):
  - renders each ground layer alone, on a flat 24 m square, straight down, at 0.125 m/pixel;
  - uses the scene's real sun, sky and colour management;
  - reports the mean colour.
- **Colour targets and calibration:**
  - each layer in the spec has a `target_srgb` (its intended aerial colour, raw render, before the sensor model);
  - `python -m synth.steps.step3 --calibrate` corrects the layer's hue, saturation and value until the render matches;
  - the result is written back into the spec. All 8 layers are within 1 level of their target.
- **Wadi-bed layer:**
  - the drainage wetness map (multiple-direction flow, from Step 2) now feeds the splat map's blue channel;
  - a layer option `"wet": [lo, hi]` places darker, moist soil along drainage lines;
  - it covers 4.7 % of the core and follows the gorge's channel network.
- **Natural layer edges:**
  - `breakup` (`{m, amp}`) frays a layer into clumps of size m instead of a smooth sheet;
  - `slope_jitter_deg` breaks up the rock line on slopes;
  - basalt now appears as broken rock with straw between, getting denser as the slope steepens (20–42°).
- **Bug fixed** (also affected the earlier presets): several noise nodes received a "seed" as their *scale*.
  - Layer noise was shrunk to about 0.15 m, below one pixel, where it averages out and has no effect.
  - New helper `NodeBuilder.noise_m(uv, wavelength_m, seed)` takes the size in metres and uses the seed only to offset the pattern.

Layer targets (sRGB, summer):

| layer | asset role | target | look |
|---|---|---|---|
| soil (base) | terrain.soil | (118, 96, 78) | reddish-brown basalt soil |
| grass | terrain.grass | (172, 152, 108) | dry straw (June) |
| soil_sandy | terrain.soil_sandy | (150, 130, 104) | dusty light soil |
| gravel | terrain.gravel | (108, 98, 86) | basalt gravel |
| wadi | terrain.dirt | (104, 92, 76) | moist channel bed |
| dirt | terrain.dirt | (160, 145, 122) | road-shoulder dust (splat R) |
| trampled | terrain.trampled | (150, 133, 110) | yards and paths (splat G) |
| rocky | terrain.rocky | (78, 72, 66) | dark basalt |

These targets are first estimates for a dry-season basalt plateau. They are easy to change in `configs/specs/step3_ground.json`; re-run with `--calibrate`.

## How to run

```bash
~/synth_aerial_work/venv/bin/python -m synth.steps.step3              # build + renders + report images (~4.5 min)
~/synth_aerial_work/venv/bin/python -m synth.steps.step3 --calibrate  # also re-fit layer colours to their targets
open -a Blender ~/synth_aerial_work/outputs/steps/step3/scene.blend
```

## What to inspect (`~/synth_aerial_work/outputs/steps/step3/`)

- `swatches.png`: each layer alone, with its measured colour next to its target.
- `closeups.png`: before (Step 2) vs after, straight down at 0.18 m/pixel and at 50°.
- `wetness_overlay.png`: where the wadi-bed layer goes.
- `contact_sheet.jpg`: the three standard views (1 m nadir, 45°, near-horizon) with label overlays.
- `scene.blend`: the terrain material `terrain` has the full layer stack.

## Validation

- Swatches: all 8 layers within 1 level (0–255) of their target colour.
- Wadi layer: 4.7 % of the core, along drainage lines, as intended (3–6 %).
- Labels unchanged: terrain everywhere, sky in the horizon view.
- 8/8 tests pass, including a new calibration-step test.

## Known limitations

- **Grass needs a ×7.9 brightness boost:** the "wild grass" scan is a dark green meadow. It works at aerial resolution. A dry-grass ground scan would be better (it is on the asset wish list).
- **Ground only:** the 1 m nadir view still looks bare and mottled. 3D rocks, dry grass tufts, shrubs and trees come in Step 6, and will break up the remaining "leopard" pattern.
- **Wetness on the flat plateau** marks small depressions as well as channels. These read as darker soil patches, which is plausible, but not as true wadis.
- The targets apply to the raw render. The sensor model (haze, blur, noise) brightens the final image slightly.
