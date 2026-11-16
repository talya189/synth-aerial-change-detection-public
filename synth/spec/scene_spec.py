"""Scene spec: the JSON single source of truth for one scene (pure Python).

Structure (spec_version 1):

{
  "spec_version": 1, "scene_id": "...", "seed": 1,
  "region":      {"type": "flat_test" | "dem", ...},           where (and what ground)
  "environment": {"season", "date", "local_time", "utc_offset_h", "haze_visibility_km"},
  "terrain_material": {...},                                   ground layers (asset roles)
  "cameras":  [{"id", "target", "tilt_deg", "heading_deg", "gsd_m", "resolution", ...}],
  "render":   {...},
  "entities": {                                               countable objects, each with a
    "roads":     [{"id", "class", "centerline": [[x, y], ...], "connects"}],   persistent id
                  (class: configs/roads.json; kind/width/shoulder default from it; connects /
                   connects_end = road joined at the start / end; end_cap = turning-circle radius)
    "paths":     [{"id", "width", "centerline"}],
    "buildings": [{"id", "type", "center": [x, y], "width", "depth", "yaw_deg", "floors",
                   "roof", "wall", "annex", ...}],      see synth/spec/buildings.py
    "trees":     [{"id", "species", "x", "y", "height", "crown", "yaw_deg"}]   species: configs/vegetation.json
    "groves":    [{"id", "species", "polygon", "spacing_m", "row_deg", "forest"}]   orchard / planted forest:
                  one entity (one instance id), trees in rows (or irregular if forest)
    "fields":    [{"id", "polygon", "crop", "row_deg"}]   farm field: crop = ploughed | stubble | fallow
    "walls":     [{"id", "type", "segments", "height", "thickness", "encloses", "yard", "gate", "tint"}]
                  yard wall / fence around a house plot: type = stone | block | plaster | fence,
                  segments = polylines (gaps are gates), see synth/spec/yards.py
    "props":     [{"id", "type", "x", "y", "yaw_deg", "height", "road"}]   street furniture
                  (streetlights ...), types: configs/props.json, see synth/spec/props.py
  },
  "scatter": {"<name>": {"role", "kind", <SYN_Scatter inputs>}},   mass "stuff", seeded
  "terrain_material" and "scatter" may instead name a preset file:
      configs/presets/terrain_material/<name>.json, configs/presets/ground_cover/<name>.json
  "history": [ {edit operation}, ... ]                          what was changed, in order
  "vegetation": false                                           optional: drop all plants (bare scene)
}

Coordinates are local metres (x east, y north) around the scene origin given in
`region`. Ids never change once assigned, so T1/T2 diffs and edits can address
"bld_003" reliably.
"""
import copy
import json
import re

ENTITY_TYPES = ("roads", "paths", "buildings", "trees", "groves", "fields", "walls", "props")
ID_PREFIX = {"roads": "road", "paths": "path", "buildings": "bld", "trees": "tree", "groves": "grove", "fields": "field", "walls": "wall", "props": "prop"}
REQUIRED = {
    "roads": ("id", "centerline"),
    "paths": ("id", "width", "centerline"),
    "buildings": ("id", "center", "width", "depth", "floors"),
    "trees": ("id", "species", "x", "y"),
    "groves": ("id", "species", "polygon", "spacing_m"),
    "fields": ("id", "polygon", "crop"),
    "walls": ("id", "type", "segments"),
    "props": ("id", "type", "x", "y"),
}


class SpecError(ValueError):
    pass


def load(path):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    validate(spec)
    return spec


def save(spec, path):
    validate(spec)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)


def validate(spec):
    """Structural checks (geometric rules are checked by the planner/edit ops)."""
    for key in ("spec_version", "scene_id", "seed", "region", "environment", "cameras", "entities"):
        if key not in spec:
            raise SpecError(f"missing '{key}'")
    seen = set()
    for etype in ENTITY_TYPES:
        for e in spec["entities"].get(etype, []):
            for k in REQUIRED[etype]:
                if k not in e:
                    raise SpecError(f"{etype} entity {e.get('id', '?')} lacks '{k}'")
            if e["id"] in seen:
                raise SpecError(f"duplicate id {e['id']}")
            if not re.fullmatch(rf"{ID_PREFIX[etype]}_\d{{3,}}", e["id"]):
                raise SpecError(f"id {e['id']} does not follow '{ID_PREFIX[etype]}_NNN'")
            seen.add(e["id"])
    for b in spec["entities"].get("buildings", []):
        _check_building(b)
    _check_roads(spec)
    _check_vegetation(spec)
    _check_walls(spec)
    _check_props(spec)
    reg = spec["region"]
    if reg.get("type") == "dem" and "aoi" not in reg:
        raise SpecError("dem region needs 'aoi' (an area from configs/terrain.json)")
    if reg.get("type") == "flat_test" and "latlon" not in reg:
        raise SpecError("flat_test region needs 'latlon'")
    for c in spec["cameras"]:
        for k in ("id", "target", "tilt_deg", "gsd_m"):
            if k not in c:
                raise SpecError(f"camera {c.get('id', '?')} lacks '{k}'")
    return True


def _check_building(b):
    from synth.spec.buildings import ANNEX_SIDES, catalogue
    cat = catalogue()
    bid = b["id"]
    if b.get("type", "house") not in cat["types"]:
        raise SpecError(f"{bid}: unknown type {b.get('type')!r} (known: {sorted(cat['types'])})")
    if "roof" in b and b["roof"] not in cat["roofs"]:
        raise SpecError(f"{bid}: unknown roof {b['roof']!r} (known: {cat['roofs']})")
    if "wall" in b and b["wall"] not in cat["walls"]:
        raise SpecError(f"{bid}: unknown wall {b['wall']!r} (known: {cat['walls']})")
    if not (isinstance(b["floors"], int) and 1 <= b["floors"] <= 40):
        raise SpecError(f"{bid}: floors must be an integer 1-40")
    if b.get("podium") and b["podium"].get("floors", 0) >= b["floors"]:
        raise SpecError(f"{bid}: podium must be lower than the building")
    for k in ("width", "depth"):
        if not 4 <= b[k] <= 120:
            raise SpecError(f"{bid}: {k} must be 4-120 m")
    a = b.get("annex")
    if a:
        if a.get("side") not in ANNEX_SIDES or not all(k in a for k in ("width", "depth", "floors")):
            raise SpecError(f"{bid}: annex needs side {ANNEX_SIDES}, width, depth, floors")
        if a["floors"] > b["floors"]:
            raise SpecError(f"{bid}: annex cannot be taller than the building")


PRESET_DIRS = {"terrain_material": "configs/presets/terrain_material", "scatter": "configs/presets/ground_cover"}


def preset_path(key, name):
    return f"{PRESET_DIRS[key]}/{name}.json"


def resolved(spec):
    """Copy of the spec with preset names replaced by the preset contents."""
    from synth.common.config import load_json
    out = copy.deepcopy(spec)
    for key in PRESET_DIRS:
        if isinstance(out.get(key), str):
            out[key] = {k: v for k, v in load_json(preset_path(key, out[key])).items() if k != "_comment" or key != "scatter"}
    if out.get("vegetation") is False:   # bare scene: no scattered plants, no painted far-field shrubs, no trees
        out["scatter"] = {k: v for k, v in out.get("scatter", {}).items()
                          if isinstance(v, dict) and not v.get("kind", "").startswith("veg.")}
        tm = out.get("terrain_material")
        if isinstance(tm, dict):
            tm["layers"] = [L for L in tm["layers"] if not L.get("vegetation")]
        out["entities"]["trees"] = []
    return out


def _check_roads(spec):
    from synth.common.config import load_json
    classes = load_json("configs/roads.json")["classes"]
    ids = {r["id"] for r in spec["entities"].get("roads", [])}
    for r in spec["entities"].get("roads", []):
        if "class" in r and r["class"] not in classes:
            raise SpecError(f"{r['id']}: unknown road class {r['class']!r} (known: {sorted(classes)})")
        if "class" not in r and not all(k in r for k in ("kind", "width")):
            raise SpecError(f"{r['id']}: needs a class, or kind and width")
        for key in ("connects", "connects_end"):
            if r.get(key) and r[key] not in ids:
                raise SpecError(f"{r['id']}: {key} names unknown road {r[key]}")
        if len(r["centerline"]) < 2:
            raise SpecError(f"{r['id']}: centerline needs at least 2 points")


def _check_vegetation(spec):
    from synth.common.config import load_json
    species = load_json("configs/vegetation.json")["species"]
    for etype in ("trees", "groves"):
        for e in spec["entities"].get(etype, []):
            if e["species"] not in species:
                raise SpecError(f"{e['id']}: unknown species {e['species']!r} (known: {sorted(species)})")
    for f in spec["entities"].get("fields", []):
        if f["crop"] not in ("ploughed", "stubble", "fallow") or len(f["polygon"]) < 3:
            raise SpecError(f"{f['id']}: crop must be ploughed, stubble or fallow, polygon 3+ points")
    for g in spec["entities"].get("groves", []):
        if len(g["polygon"]) < 3 or not 2.0 <= g["spacing_m"] <= 20.0:
            raise SpecError(f"{g['id']}: grove needs a polygon (3+ points) and spacing_m 2-20")


def _check_walls(spec):
    bids = {b["id"] for b in spec["entities"].get("buildings", [])}
    for w in spec["entities"].get("walls", []):
        if w["type"] not in ("stone", "block", "plaster", "fence"):
            raise SpecError(f"{w['id']}: type must be stone, block, plaster or fence")
        if not w["segments"] or any(len(s) < 2 for s in w["segments"]):
            raise SpecError(f"{w['id']}: segments must be polylines of 2+ points")
        if w.get("encloses") and w["encloses"] not in bids:
            raise SpecError(f"{w['id']}: encloses unknown building {w['encloses']}")
        if not 0.3 <= w.get("height", 1.5) <= 4.0:
            raise SpecError(f"{w['id']}: height must be 0.3-4 m")


def _check_props(spec):
    from synth.common.config import load_json
    types = load_json("configs/props.json")["types"]
    for p in spec["entities"].get("props", []):
        if p["type"] not in types:
            raise SpecError(f"{p['id']}: unknown prop type {p['type']!r} (known: {sorted(types)})")


def next_id(spec, etype):
    """New persistent id: one above the highest ever used (ids are never reused)."""
    used = [int(e["id"].split("_")[1]) for e in spec["entities"].get(etype, [])]
    used += [int(h["id"].split("_")[1]) for h in spec.get("history", [])
             if isinstance(h.get("id"), str) and h["id"].startswith(ID_PREFIX[etype] + "_")]
    return f"{ID_PREFIX[etype]}_{(max(used) if used else 0) + 1:03d}"


def find(spec, entity_id):
    for etype in ENTITY_TYPES:
        for e in spec["entities"].get(etype, []):
            if e["id"] == entity_id:
                return etype, e
    raise SpecError(f"no entity {entity_id}")


def clone(spec):
    return copy.deepcopy(spec)
