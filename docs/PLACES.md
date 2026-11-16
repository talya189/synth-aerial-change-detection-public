# Places, styles and seeds

A **place file** holds the whole logic for one kind of place, per **style**:
- terrain;
- buildings;
- asphalt roads and dirt tracks;
- footpaths;
- vegetation;
- farm fields.

A scene is **place + style + seed**. The same three always give the same scene.

```bash
~/synth_aerial_work/venv/bin/python -m synth.make_scene --list
~/synth_aerial_work/venv/bin/python -m synth.make_scene --place arab_village --style hillside --seed 7
~/synth_aerial_work/venv/bin/python -m synth.make_scene --place arab_city --style ridge_hillside --seed 3 --plan-only
```

- The spec goes to `configs/specs/places/<place>_<style>_<seed>.json`. It is kept: re-running reuses it unless you pass `--regenerate`, and you may edit it by hand.
- The scene goes to `~/synth_aerial_work/outputs/scenes/<place>/<style>/seed_<seed>/`:
  - `plan_map.png`;
  - `report.json`, with the rule checks;
  - `scene.blend`, `views/`, `contact_sheet.jpg`, `id_map.png`.
- `--plan-only` makes the spec, plan map and checks in seconds, without Blender.

## Files

| file | what it holds |
|---|---|
| `configs/places/arab_village.json` | Arab village / town: styles `hillside`, `hilltop` |
| `configs/places/arab_city.json` | Arab city: styles `basin_dense`, `ridge_hillside` |
| `configs/buildings.json` | building types (incl. mosque, shop-house, warehouse) and their placement rules |
| `configs/roads.json` | road classes (width, grade limit, curve radius) |
| `configs/vegetation.json` | tree species (scanned and leaf-cluster procedural) |
| `configs/terrain.json` | elevation areas (`lower_galilee`, `wadi_ara`, `korazim`) |
| `configs/presets/` | ground material and ground cover (Galilee summer) |
| `synth/places.py` | place + style + seed → spec (runs the generators in order) |
| `synth/make_scene.py` | the command |

## Styles (from real places seen in aerial imagery)

| place | style | real examples | what defines it |
|---|---|---|---|
| arab_village | hillside | Deir Hanna, Tuba-Zangariyye | main road along the foot of the slope; contour lanes climbing, joined by diagonal cross lanes; mosque near the centre; dense core thinning upslope into olive groves; fields on the valley floor |
| arab_village | hilltop | village by Route 77, Arraba core | ring lanes around the top with radial lanes; mosque at the top; main road passing below; olive groves on the slopes, strip fields in the valleys |
| arab_city | basin_dense | Sakhnin, Arraba | main road through the centre with shops below flats; 9 contour streets with many cross streets; houses 2–3 deep plus block infill; 2–3 mosques; industrial strip on the main road; forest patches and groves around |
| arab_city | ridge_hillside | Umm al-Fahm | main road along the ridge through the centre; serpentine contour streets with switchback connectors; terraced houses on slopes up to 30°; street grades up to 18–20 % |

## What each style controls (keys in the place file)

| key | meaning |
|---|---|
| `terrain` | `aoi` (area), `mode` (hillside / hilltop / basin / ridge), `target_slope_deg`, `site_choices` (the seed picks one of the best N real sites), ground presets |
| `roads` | `main_road` (bypass / downhill / through); `contour` streets (count, spacing, length, class, how strictly they follow the contour); `connectors` (cross streets, dead-end alleys); `access_count`; `tracks`; optional `grade_limits` |
| `buildings` | `count`, `types` (weights), `landmarks` (e.g. mosque), `frontage` (spacing, setback, rows behind the street), `density` (core radius, falloff), `infill_step_m` |
| `rules` | optional, e.g. `max_wetness` (cities build over all but the main drainage channels) |
| `footpaths` | `count` (desire lines: shortcuts across blocks, trails out of the village) |
| `vegetation` | `yard`, `street` trees, `groves` (olive), `forest` patches |
| `fields` | farm fields (count, size, max slope, distance from the village, strips, crops) |
| `yards` | optional overrides for yard walls: `share` per building type, `types` (stone / block / plaster / fence weights), `height_m`, `depth_m`, `gate_m`; `false` for no walls (defaults in `synth/spec/yards.py`) |

## Rules every scene must pass (checked in `report.json`)

- **Smooth roads.** Every road is continuous with differentiable curves, with no kinks.
  - The last step fairs each road (junction points fixed) until its tightest curve is at least its class radius: main 50 m, street 15 m, lane and track 10 m. Centerlines are stored at 2 m spacing, so the curve is kept.
  - A road that must climb a steep slope in hairpins may use 0.6× (at most 0.3×) its class radius. This is recorded on the road as `min_radius_m`.
  - Report key: `roads_tighter_than_class_radius`.
- **Every house is reachable.**
  - A house whose front door is within 9 m of a road edge, with nothing in between, opens onto that road.
  - Every other house gets an access alley (`kind: "access"`, paved, 1–2.6 m, narrowed to fit the gap it passes). It follows one shortest-path tree grown from the road network, so alleys of neighbouring houses merge.
  - Each house records `access` (road or path). Each alley records the houses it `serves` and what it joins (`road_join` / `path_join`).
  - Report keys: `buildings_without_access`, `access_paths_through_buildings`.

- **Yard walls never block anything.**
  - No wall overlaps a building, a road or a path; a footpath planned later opens a gap in the wall it passes.
  - Every walled house can get out: the gate is where its way out (the line from its door to its road, or its access alley) crosses the wall.
  - Report keys: `walls_through_buildings`, `walls_across_roads_paths`, `walled_in_buildings`.

- No road crosses another without a junction. Crossing streets become crossroads; a track ends at the road it meets.
- Every road is reachable from the main road. Isolated streets are joined by a connector: straight, diagonal or serpentine.
- Built road grades are within their limits. A too-steep road is first re-routed; if the network depends on it and it is within 1.25× of its limit, it is kept and listed.
- Every building:
  - inside the core;
  - within its type's slope and height-difference limits;
  - clear of other buildings, roads and paths;
  - out of drainage channels.
- Trees never on buildings, roads or paths; fields and yards free of wild shrubs and stones.

**Measured** (seed sweep, 5 new seeds per style):
- 20 of 20 scenes pass, with no crossings, no disconnected roads and no rule-breaking buildings.
- The only remaining steep roads are in the ridge city: up to 4 streets per scene at 20–22 % against 18 %, kept because the network depends on them.

## Seed variety

Two seeds of one style follow the same rules but give clearly different scenes.

- **Terrain:** `terrain.aois` lists several real Galilee areas. The seed picks one, then one of the best `site_choices` sites there for the landform.
- **Character:** every entry in the style's `vary` block is drawn per seed. Formats:
  - `{"range": [lo, hi]}`;
  - `{"choice": [...]}`;
  - `{"scale": [lo, hi]}`, which multiplies the value there;
  - `_types_jitter`, which spreads the building-type weights.

  It covers streets, building count and density, the building mix, walls, groves, forest, fields and footpaths. The drawn values are saved in `spec.place.character`.
- **Ground cover:** `ground_cover_scale` sets the density of wild bushes (`veg`) and stones (`rock`).
- **Light:** `environment` sets the month range, the time of day and the haze. The camera headings are also drawn per seed.

More rules checked in `report.json`:
- `paths_ending_nowhere`: every footpath ends on a road, a path or a building.
- `mosques_closer_than_200m`: mosques keep `landmark_spacing_m` apart.
- `props_on_buildings_roads_paths`: no streetlight stands on a building, road or path.

## Building variety

Every building gets its own look, drawn from palettes in `configs/buildings.json` (seeded by its id; it can also be written explicitly into the spec).

- **Facade colour (`tint`):** a palette per wall material.
  - stone cladding from cream to honey;
  - plaster in cream, beige, white, light grey and sand, with rare pastels (peach, pink, pale green, pale blue);
  - cement and block greys.
- **Roof colour (`roof_tint`):** warm beige dominates, then cement grey, white coatings and weathered dark roofs, matched to aerial imagery of Sakhnin.

  | | ours | real |
  |---|---|---|
  | beige share | 73 % | 77 % |
  | beige value | (172, 156, 139) | (180, 169, 147) |

  Pure white roofs are dimmed by the camera's tone mapping and haze.
- **`pilotis`:** the ground floor is open on columns (parking under the house). Common in new hill houses: 30 % of `arab_house`.
- **`roof_hat`:** a room on the flat roof under a small terracotta pyramid. Common in Arab villages: 20 % of `arab_house`.
- **Annexes, setback top floors, balconies, railings, rebar for the next floor and rooftop clutter** (solar heaters, black and white tanks, AC units, dishes) vary per building.

## Yard walls and fences

Most houses stand in a walled plot (`walls` entities, ids `wall_NNN`, made by `synth/spec/yards.py`).

- **Plot:** the house footprint grown by 3–7 m, shared fairly with the neighbours (Voronoi cells of the building centres), kept clear of roads, other houses, alleys and every other house's way out. A wall shared by two plots is built once.
- **Gate:** where the house's way out crosses the wall, 1.4–4 m wide (a door or a car gate).
- **Types:**

  | type | look | height |
  |---|---|---|
  | stone | cream limestone (always for `old_basalt` houses), no coping | 1.0–1.8 m |
  | block | grey concrete block with a concrete coping | 1.5–2.3 m |
  | plaster | the house's own facade colour, with coping | 1.3–2.1 m |
  | fence | metal bars on a 0.5 m concrete plinth, posts every 2.5 m | 1.2–1.8 m |

- **Who gets one:** `arab_house` 80 %, `house` 75 %, `villa` 90 %, `old_basalt` 85 %, `public` 90 %, `mosque` 60 %, `unfinished` 35 %, `apartment` 30 %; never shops, sheds, farm or industrial buildings.
- Inside a walled yard there are no wild shrubs or stones, and the ground is trampled. Groves, fields and footpaths stay out of yards.
- **Labels:** walls have their own class, `wall_fence` (id 9), and one instance id per wall.

## Adding a style or a place

1. Copy a style block and change its values. Add the real places it is modelled on under `examples`.
2. For a new area, add an `aoi` (`bbox_latlon`) to `configs/terrain.json`. Copernicus tiles N32/N33 E035 are available.
3. Plan a few seeds with `--plan-only` and look at `plan_map.png` and `report.json` before rendering.
