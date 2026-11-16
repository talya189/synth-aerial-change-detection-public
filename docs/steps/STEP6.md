# Step 6: vegetation in the scene spec (first version, for your feedback)

## What was built and why

Trees are now spec entities with persistent ids, like buildings and roads, so change pairs can address them ("cut tree_017", "remove grove_003"). Wild ground cover (shrubs, grass tufts, rocks) stays as seeded "stuff" (scatter systems): it covers the open land around buildings, roads and paths, but never on them.

### Species (`configs/vegetation.json`)

| species | source | look |
|---|---|---|
| pine (Aleppo) | your scanned USD tree | KKL-style plantation pine |
| fan palm | your scanned model | |
| date palm | procedural fronds on your scanned trunk | |
| olive | procedural | silvery grey-green, wide rounded crown on a short trunk |
| oak (Tabor) | procedural | broad dark crown |
| carob | procedural | dense, dark, round |
| cypress | procedural | tall narrow column |
| Ziziphus | procedural | irregular, medium green |
| citrus | procedural | small, dense, bright green |

Procedural trees (`synth/blender/proc_veg.py`):
- built from noise-displaced crown blobs on a trunk, in five variants per species;
- shared meshes scaled per tree (crown width, height), with per-tree colour variation.

They exist because your assets have no olive, oak, carob, cypress or citrus.

### New entity: `groves`

An orchard or planted forest is one entity: a polygon with species, `spacing_m`, `row_deg` (rows) or `forest: true` (irregular, with gaps). It renders as its trees, all carrying the grove's one instance id, so a whole grove can be added or removed in one edit.

### Planting rules (`synth/spec/vegetation.py`, per settlement style)

| | moshav | Arab village | town |
|---|---|---|---|
| yard trees per house | 1–4 (citrus, olive, palms, pine, cypress) | 0–2 (mostly olive) | 0–3 (pine, cypress, palms, carob) |
| street trees | cypress rows on through streets | — | palm avenues |
| groves | olive and citrus orchards | a ring of 7 olive groves | 1 olive grove |
| forest patches | 1 pine patch | — | 1 pine patch |

Rules:
- Trees never stand on buildings, roads, paths or turning circles.
- Groves go on gentle, dry ground (no drainage channels), with rows along the contour.
- Street trees go only on through streets longer than 150 m.

## How to run

```bash
~/synth_aerial_work/venv/bin/python -m synth.steps.step6            # catalogue + 3 villages (~25 min)
~/synth_aerial_work/venv/bin/python -m synth.steps.step6 --only catalogue
~/synth_aerial_work/venv/bin/python -m synth.steps.step6 --only moshav --regenerate
```

## What to inspect (`~/synth_aerial_work/outputs/steps/step6/`)

- `catalogue/views/cam_species`: each species small and large, in a row.
- `catalogue/views/cam_groves`: an olive grove and a pine-forest patch.
- `<style>/`: the Step 5 settlements planted, with wild ground cover.
- `vegetation.json`: counts per species, groves with their tree counts and pixels.

## Validation

- **Labels:** every tree and grove has its own instance id; grove trees share it.
- **Placement:** no tree on a building, road or path; every grove tree inside its grove (tests).
- **Grounding:** nothing floats.

## Waiting for your feedback

This first version is meant for your comments, for example on:
- species looks;
- amounts;
- what grows where;
- whether wild shrubs should be denser or sparser;
- seasons.
