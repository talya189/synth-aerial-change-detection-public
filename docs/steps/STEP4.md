# Step 4: buildings in the scene spec

## What was built and why

Buildings are fully described in the scene spec, each with a persistent id. Explicit, checkable rules place them, so later steps can add, remove or change one building by its id (T1/T2 edits).

### Building types (`configs/buildings.json`)

| type | label kind | floors | roof | walls | details |
|---|---|---|---|---|---|
| house | building.house | 1–2 | flat / tiled hip | stucco, stone | annex, balcony, rooftop solar heaters, tanks, AC, stairwell, rebar for expansion |
| moshav_house | building.house | 1–2 | red-tile gable / hip | white, stucco | rural cottage, annex, pergola |
| villa | building.house | 1–2 | flat / tiled | white, stone | L-shaped (large annex), big windows, railing, PV array, pergola |
| arab_house | building.house | 2–4 | flat | stone cladding, stucco, white | Galilee Arab-village style: smaller top floor, rebar for the next floor, many tanks, balconies |
| old_basalt | building.house | 1–2 | tiled hip / flat | dark basalt | Rosh Pina–style: tall storeys, small windows |
| apartment | building.residential | 3–4 | flat | stucco, stone | balconies, rooftop solar heaters |
| midrise | building.residential | 5–8 | flat | stucco, white, stone | optional podium |
| tower | building.residential | 9–28 (9, 10, 12, 14, 16, 18, 20, 24, 28) | flat | white, stone, concrete | optional 1–2 floor commercial podium with shop windows, setback crown |
| public | building.public | 1–2 | flat | white, stucco, stone | school / community / dining hall: long, many windows, AC units, PV |
| unfinished | building.unfinished | 1–2 | bare slab | concrete block | open window bays, rebar stubs |
| shed | building.shed | 1 (4.5–6 m) | corrugated metal gable | block, concrete | roller door, high windows, optional PV slope |
| coop | building.agricultural | 1 | low metal gable | white, concrete | poultry house: continuous light curtain sides, fan wall at one end; always in parallel groups of 2–4 |
| greenhouse | building.greenhouse | 1 | plastic tunnels | — | multi-span tunnel greenhouse |
| water_tower | building.tower | — | concrete tank | concrete | ground reservoir or elevated tank on a shaft |

Every type has the semantic class `building`, and each building gets its own instance id.

**Per-building options** (drawn from the type, seeded by the id, when the spec leaves them out):
- `annex`;
- `balcony`;
- `expandable` (rebar);
- `top_setback` (top floor over the back part only);
- `pv` (solar panel rows on flat roofs facing true south, or a panel field on the south slope of a metal roof);
- `railing` (metal railing instead of a solid parapet);
- `pergola`;
- `podium` (towers).

**Walls:**
- stucco;
- white plaster;
- stone;
- basalt (dark);
- concrete block (procedural joints until the corrupt block-wall scan is re-downloaded);
- concrete.

### Placement rules (`synth/spec/placement.py`)

A building may stand only where:
- it is inside the core;
- slope and height difference under its footprint (including annex and podium) are within the type's limits;
- it keeps clear of other buildings;
- it is off roads and paths;
- it is not in a drainage channel.

On slopes, houses turn their long side to the contour and face downhill.

**Settlement styles** (`grow_village(spec, site, n, "<style>", seed)`):

| style | mix | spacing |
|---|---|---|
| moshav | cottages, houses, villas; poultry houses, greenhouses, sheds and water towers on the outskirts | wide |
| arab_village | mostly multi-storey arab_house, houses, unfinished | dense |
| kibbutz | rows of cottages, public buildings, apartments; agricultural buildings outside | medium |
| old_town | old basalt houses, houses, villas | dense |
| town | towers, mid-rises, apartments, public buildings, houses | medium |

Core types grow next to existing houses. Outskirts types go beyond them. Every drawn type gets up to 40 location attempts, so the final mix follows the style's probabilities even for large footprints. Poultry houses are placed as a whole group or not at all.

## How to run

```bash
~/synth_aerial_work/venv/bin/python -m synth.steps.step4                       # catalogue + 5 villages (~25 min)
~/synth_aerial_work/venv/bin/python -m synth.steps.step4 --only town          # one village
~/synth_aerial_work/venv/bin/python -m synth.steps.step4 --only villages --regenerate   # grow new villages
```

The village specs (`configs/specs/step4_village_<style>.json`) are generated once and then reused. Edit any building in them and re-run; the per-building report shows whether it still passes the rules.

## What to inspect (`~/synth_aerial_work/outputs/steps/step4/`)

- `catalogue/`: 29 buildings, every type and option. Views:
  - `cam_highrise`: 5 to 28 floors;
  - `cam_houses`, `cam_village_types`: house styles;
  - `cam_agri`: shed, greenhouse, water towers, poultry-house group.
- `village_<style>/`: each style on the Korazim site, with the Galilee ground cover. Views: nadir, 45°, 68°.
- `*/id_map.png`: nadir view with each building's id. Edits and LLM commands refer to these ids.
- `*/buildings.json`: per building, its type, size, slope, height difference, rule check, and visible pixels per view.

## Validation

- Catalogue: 29/29 built, no building floats.
- Villages: moshav 22, Arab village 34, kibbutz 28, old town 22, town 24 buildings (town up to 28 floors).
  - Every building passes the placement rules.
  - Every building is visible in its village's nadir view.
  - None float.
- 17/17 tests pass. New ones cover:
  - presets;
  - all types and options;
  - settlement styles;
  - podium footprint and validation;
  - poultry-house groups;
  - the channel map.

## Known limitations

- **No streets yet:** villages are clusters. Step 5 adds roads and paths and orients buildings to them.
- **No yard walls, fences, gardens or cars** (Step 6).
- **Simple facade geometry:** windows, balconies and roof equipment are fine at aerial resolution (≥ 6 cm/pixel) but not for close-ups.
