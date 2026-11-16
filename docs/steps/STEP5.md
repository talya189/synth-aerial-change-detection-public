# Step 5: roads and paths in the scene spec

## What was built and why

Roads and footpaths are spec entities with persistent ids, like buildings. A road in the spec is physical:
- it gets a grade-limited vertical profile;
- it joins the roads it meets at proper junctions (at its start and/or its end);
- the terrain is cut and filled under it.

As requested, these scenes have no vegetation (`"vegetation": false`); Step 6 brings it back.

### Layout logic (revision 2, from real settlements)

The first generator branched streets off each other like a tree. It produced starbursts and parallel streets, and did not look like a real village. Revision 2 is built from aerial imagery of real Galilee settlements next to the site:

| Real example | What it shows | Rule in the generator |
|---|---|---|
| Regional road near Korazim | long sweeping curves, no small wiggles | least-cost route under 8 % grade, then **faired** to a minimum curve radius (50 m switchback); it **bypasses** the village; an access road joins at ~90° |
| Tuba-Zangariyye | streets in rings along the contours, linked by short cross streets into a mesh with loops; houses 2–3 deep | **Arab village**: 5 contour lanes 34–50 m apart, cross lanes every 55–95 m, dead-end alleys off the outer lanes; 2 access roads |
| Korazim moshav | a contour collector street with a **comb of straight cul-de-sacs** up the slope, each ending in a **turning circle**, plots on both sides | **Moshav**: 2 contour collectors ~110 m apart, cul-de-sacs every 40–52 m (mostly uphill), 8 m turning circles, a cross street between the collectors |
| Rosh Pina / towns | contour streets with regular cross streets, forming blocks | **Town**: 4 contour streets 65–80 m apart, cross streets every 85–110 m |
| Field tracks around all three | dirt tracks continue from street ends; field tracks are long straight legs along field edges | tracks continue a street along the contour, or run 2–3 straight legs (70–150 m) with bends of 20–65° |
| Footpaths | desire lines where they save a walk; trails out to the fields | **shortcuts** across a block where walking via the streets is >2.5× longer, and **outbound trails** from the village edge; dropped if they would run alongside a road |

Rules that hold for every network:
- **Streets follow the contours.** They are traced along the terrain, turning at most 6° per 3 m, so grades stay low and curves look natural. Cross streets and cul-de-sacs that climb are rotated toward the contour until their grade is within the class limit.
- **Junctions only.**
  - A street that reaches another road ends there, in a T-junction; both ends of a cross street are joined (`connects`, `connects_end`), with matching heights.
  - Two cul-de-sacs leaving the collector at the same point form a crossroads.
  - `crossings()` checks that no road crosses another without a junction. `resolve_crossings()` cuts a track that runs into a road at that road.
- **One connected network:** `unreachable()` checks that every road can be reached from the main road.
- **Buildings face their street** at a setback. Dense styles add a second row. Agricultural buildings line the tracks. The Step 4 placement rules still apply.

### Road classes (`configs/roads.json`)

| class | surface | width | max grade | min radius | used for |
|---|---|---|---|---|---|
| main | asphalt, centre markings | 7.0 m + 1.5 m shoulders | 8 % | 50 m | regional road |
| street | asphalt | 5.5 m | 11 % | — | moshav, town |
| lane | asphalt | 4.0 m | 14 % | — | Arab village |
| track | dirt | 3.6 m | 15 % | — | tracks |

Footpaths are `dirt` (0.7–1.2 m desire lines) or `paved` (concrete walkway).

Spec format: `{"id": "road_012", "class": "street", "centerline": [[x, y], ...], "connects": "road_002", "connects_end": "road_003", "end_cap": 8.0}`.

## How to run

```bash
~/synth_aerial_work/venv/bin/python -m synth.steps.step5                  # 3 settlements (~10 min)
~/synth_aerial_work/venv/bin/python -m synth.steps.step5 --plan-only      # plan maps only (seconds)
~/synth_aerial_work/venv/bin/python -m synth.steps.step5 --only town --regenerate
```

The specs `configs/specs/step5_<style>.json` are generated once and then reused. Edit any road, path or building, then re-run.

## What to inspect (`~/synth_aerial_work/outputs/steps/step5/<style>/`)

- `network_map.png`: shaded relief with roads by class (with ids), footpaths and buildings.
- `contact_sheet.jpg`, `views/`: nadir, 45°, and a low oblique (62°).
- `id_map.png`: nadir view with building ids.
- `roads.json`: per road, its class, length, steepest grade over 10 m vs its limit, junctions, and pixels per view.
- `scene.blend`: collection `02_ROADS_PATHS`, each road and path named by its id (turning circles as `<id>_circle`, same instance id).

## Validation

- **No crossings, one network:** no road crosses another without a junction, and every road is reachable from the main road (all 3 styles).
- **Main road:** tightest curve 53 m, a single switchback; otherwise long, smooth curves.
- **Grades:** every road's steepest grade is within its class limit.
- **Labels:** every road and footpath has its own instance id in the masks.
- **Buildings:** every building passes the placement rules, including footpath clearance; none floats.
- **Tests:** 22/22 pass. New ones cover:
  - crossings and connectivity on all three generated networks;
  - main-road smoothness;
  - fairing;
  - buildings facing a road;
  - the vegetation switch;
  - road validation.

## Known limitations

- **Junctions:** T-junctions and crossroads only. No roundabouts, and junction corners are flared but not kerbed.
- **No sidewalks** along town streets yet. Paved walkways exist as a path type but are not generated automatically.
- **Main-road placement:** the main road is placed by the generator, not taken from the real road map (the real regional road lies outside this 1 km site).
- **Coming in Step 6:** yard walls, fences, parking and cars, together with vegetation.
