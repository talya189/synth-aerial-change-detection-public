"""Rule-based building placement on real terrain (pure Python).

`check_building` says why a building may not stand where the spec puts it;
`grow_village` adds buildings one by one, each accepted only if it passes the
same check. Both work on spec entities, so the planned "add_building" edit
operation (Step 8) can reuse them unchanged.

Rules (per building type, configs/buildings.json "placement"):
  - inside the 1 m core, away from its edge;
  - mean slope under the footprint and height difference across it are limited;
  - clearance from every other building (the larger of the two types' gaps);
  - not on roads or paths, not in a wadi bed (drainage wetness).
Houses are turned so their long side follows the contour and the front faces
downhill, as on real hillside villages; on flat ground they follow a neighbour.
"""
import math

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString

from synth.common.rng import rng
from synth.planner.fields import Grid, slope_deg
from synth.spec.buildings import catalogue, footprint, pick
from synth.spec.scene_spec import next_id


class Site:
    """Terrain fields of the core needed by the placement rules."""

    def __init__(self, z, half, wet=None, res=1.0):
        self.grid = Grid(z, res, half)
        self.half = half
        self.slope = slope_deg(z, res)
        gz = ndimage.gaussian_filter(z, 3.0)
        self.gy, self.gx = np.gradient(gz, res)
        self.wet = wet if wet is not None else np.zeros_like(z)

    def sample(self, arr, x, y):
        return self.grid.sample(x, y, arr)

    def downhill(self, x, y):
        gx, gy = float(self.sample(self.gx, x, y)[0]), float(self.sample(self.gy, x, y)[0])
        n = math.hypot(gx, gy)
        return (-gx / n, -gy / n, math.degrees(math.atan(n))) if n > 1e-6 else (0.0, -1.0, 0.0)


_OBS_CACHE = {}
_FP_CACHE = {}


def _fp(b):
    """Footprint, cached by the building's geometry fields (placement checks call it a lot)."""
    key = (b["id"], tuple(b["center"]), b["width"], b["depth"], b.get("yaw_deg", 0.0), b.get("type"),
           repr(b.get("annex")), repr(b.get("podium")))
    if key not in _FP_CACHE:
        if len(_FP_CACHE) > 20000:
            _FP_CACHE.clear()
        _FP_CACHE[key] = footprint(b)
    return _FP_CACHE[key]


def _obstacle_union(spec):
    """All road and path keep-out areas as one prepared geometry, cached until roads/paths change."""
    from shapely import prepare
    from shapely.ops import unary_union
    ent = spec["entities"]
    key = (id(spec), len(ent.get("roads", [])), sum(1 for p in ent.get("paths", []) if p.get("kind") != "access"),
           sum(len(r["centerline"]) for r in ent.get("roads", [])),
           sum(len(p["centerline"]) for p in ent.get("paths", []) if p.get("kind") != "access"))
    if key not in _OBS_CACHE:
        if len(_OBS_CACHE) > 50:
            _OBS_CACHE.clear()
        u = unary_union(_obstacles(spec)) if (ent.get("roads") or ent.get("paths")) else None
        if u is not None:
            prepare(u)
        _OBS_CACHE[key] = u
    return _OBS_CACHE[key]


def _obstacles(spec):
    ent = spec["entities"]
    from synth.common.config import load_json
    rc = load_json("configs/roads.json")["classes"]

    def half(r):   # explicit width, or the road class's width and shoulder
        c = rc.get(r.get("class"), {})
        return r.get("width", c.get("width", 4.0)) / 2 + r.get("shoulder", c.get("shoulder", 0.0))
    # clearance from the road edge: houses stand close to lanes and streets, further from main roads and tracks
    clear = {"lane": 0.8, "street": 0.8}
    obs = [LineString(np.asarray(r["centerline"])[:, :2]).buffer(half(r) + clear.get(r.get("class"), 2.0)) for r in ent.get("roads", [])]
    # footpaths may come right up to a door: only the path itself must stay clear;
    # access alleys may hug walls: only their core (half their width) must stay clear
    # (access alleys are planned around the buildings and checked separately: access.paths_through_buildings)
    obs += [LineString(np.asarray(p["centerline"])[:, :2]).buffer(p.get("width", 1.0) / 2 + 0.3)
            for p in ent.get("paths", []) if p.get("kind") != "access"]
    return obs


def _inside_samples(poly, step=1.0):
    x0, y0, x1, y1 = poly.bounds
    xs, ys = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(y0 + step / 2, y1, step))
    from shapely import contains_xy
    m = contains_xy(poly, xs, ys)
    return xs[m], ys[m]


def check_building(b, spec, site, others=None):
    """List of rule violations (empty = OK), plus measured values."""
    cat = catalogue()
    rules = cat["types"][b.get("type", "house")]["placement"]
    fp = _fp(b)
    problems = []
    lim = site.half - cat["core_margin_m"]
    x0, y0, x1, y1 = fp.bounds
    if min(x0, y0) < -lim or max(x1, y1) > lim:
        problems.append("outside the core")
        return problems, {}
    xs, ys = _inside_samples(fp)
    z = site.sample(site.grid.z, xs, ys)
    slope = float(site.sample(site.slope, xs, ys).mean())
    relief = float(z.max() - z.min())
    wet = float(site.sample(site.wet, xs, ys).max())
    if slope > rules["max_slope_deg"]:
        problems.append(f"slope {slope:.1f} deg > {rules['max_slope_deg']}")
    if relief > rules["max_relief_m"]:
        problems.append(f"height difference {relief:.1f} m > {rules['max_relief_m']}")
    if wet > spec.get("rules", {}).get("max_wetness", cat["max_wetness"]):   # cities build over all but the main channels
        problems.append(f"in a drainage line (wetness {wet:.2f})")
    obs = _obstacle_union(spec)
    if obs is not None and obs.intersects(fp):
        problems.append("on a road or path")
    others = spec["entities"].get("buildings", []) if others is None else others
    c = np.asarray(b["center"], float)
    reach = math.hypot(b["width"], b["depth"]) / 2 + 40.0   # pre-filter: only nearby buildings can be too close
    for ob in others:
        if ob["id"] == b["id"]:
            continue
        oc = ob["center"]
        if abs(oc[0] - c[0]) > reach + ob["width"] or abs(oc[1] - c[1]) > reach + ob["width"]:
            continue
        gap = max(rules["gap_m"], cat["types"][ob.get("type", "house")]["placement"]["gap_m"])
        d = fp.distance(_fp(ob))
        if d < gap:
            problems.append(f"{d:.1f} m from {ob['id']} (< {gap} m)")
    return problems, {"slope_deg": round(slope, 1), "relief_m": round(relief, 2), "wetness": round(wet, 2)}


def village_centre(site, radius_m=60.0, step=10, margin_frac=0.35):
    """Flattest dry spot of the core (mean slope over `radius_m`), at least
    margin_frac x half-size away from the core edge."""
    s = ndimage.uniform_filter(site.slope, size=int(2 * radius_m))
    s = s + 60.0 * (ndimage.uniform_filter(site.wet, size=int(radius_m)) > 0.5)
    n = s.shape[0]
    m = int(site.half * margin_frac / site.grid.res)   # keep the village in the inner part of the core
    sub = s[m:n - m:step, m:n - m:step]
    i, j = np.unravel_index(np.argmin(sub), sub.shape)
    return [float((j * step + m) * site.grid.res - site.half), float((i * step + m) * site.grid.res - site.half)]


def _new_building(r, btype, cx, cy, yaw_deg, bid):
    t = catalogue()["types"][btype]
    floors = int(pick(r, t["floors_p"]))
    w, d = float(r.uniform(*t["width_m"])), float(r.uniform(*t["depth_m"]))
    if btype in ("shed", "coop"):
        w, d = max(w, d), min(w, d)
    if btype == "water_tower":
        w = d = (float(r.uniform(6, 9)) if floors == 2 else float(r.uniform(10, 16)))
    b = {"id": bid, "type": btype, "center": [round(cx, 2), round(cy, 2)], "width": round(w, 1), "depth": round(d, 1),
         "yaw_deg": round(yaw_deg, 1), "floors": floors, "roof": pick(r, t["roof_p"]), "wall": pick(r, t["wall_p"])}
    if r.random() < t["annex_p"]:
        (aw0, aw1), (ad0, ad1) = t.get("annex_size", [[0.35, 0.6], [0.4, 0.65]])
        b["annex"] = {"side": str(r.choice(["left", "right", "back"])), "width": round(w * float(r.uniform(aw0, aw1)), 1),
                      "depth": round(d * float(r.uniform(ad0, ad1)), 1),
                      "floors": int(r.integers(1, floors + 1)) if btype == "villa" else 1}
    b["balcony"] = bool(floors >= 2 and r.random() < t["balcony_p"])
    b["expandable"] = bool(b["roof"] == "flat" and r.random() < t["expandable_p"])
    return b


def _candidate(r, cat, btype, blds, centre, spacing_m, site):
    """One proposed position and orientation for a building of `btype` (or None)."""
    core = [b for b in blds if cat["types"][b.get("type", "house")].get("zone", "core") == "core"]
    if cat["types"][btype].get("zone", "core") == "edge":
        if not core:
            return None   # the outskirts need a settlement first
        rad = max(math.hypot(b["center"][0] - centre[0], b["center"][1] - centre[1]) for b in core)
        ang = r.uniform(0, 2 * math.pi)
        reach = rad + float(r.uniform(25, 90))
        cx, cy = centre[0] + reach * math.cos(ang), centre[1] + reach * math.sin(ang)
        ref_yaw = core[0].get("yaw_deg", 0.0)
    elif core:
        anc = core[int(r.integers(len(core)))]
        ang = r.uniform(0, 2 * math.pi)
        reach = math.hypot(anc["width"], anc["depth"]) / 2 + float(r.uniform(*spacing_m)) + 4.5
        cx, cy = anc["center"][0] + reach * math.cos(ang), anc["center"][1] + reach * math.sin(ang)
        ref_yaw = anc.get("yaw_deg", 0.0)
    else:
        cx, cy, ref_yaw = centre[0], centre[1], float(r.uniform(-45, 45))
    dx, dy, sl = site.downhill(cx, cy)
    if sl > 2.0:   # front faces downhill, long side along the contour
        yaw = math.degrees(math.atan2(dx, -dy)) + float(r.normal(0, 5))
    else:          # flat: follow the neighbour's grid, sometimes turned 90 deg
        yaw = ref_yaw + 90.0 * int(r.integers(-1, 2)) * (r.random() < 0.25) + float(r.normal(0, 4))
    return _new_building(r, btype, cx, cy, yaw, "bld_tmp")


def _group(r, t, b, spec, site, blds, room):
    """`b` plus identical siblings side by side along its depth (poultry houses)."""
    members = [b]
    if "group" not in t:
        return members
    yaw_r = math.radians(b["yaw_deg"])
    ax = (-math.sin(yaw_r), math.cos(yaw_r))   # the building's depth direction
    step = b["depth"] + float(r.uniform(*t["group_gap_m"]))
    m = min(int(r.integers(t["group"][0], t["group"][1] + 1)), room)
    for k in range(1, m):
        sib = dict(b, center=[round(b["center"][0] + ax[0] * step * k, 2), round(b["center"][1] + ax[1] * step * k, 2)])
        if check_building(sib, spec, site, others=blds + members)[0]:
            break
        members.append(sib)
    return members if len(members) >= min(t["group"][0], room) else None


def grow_village(spec, site, n, types_p, seed, centre=None, spacing_m=(6.0, 22.0), tries=3000, attempts=40):
    """Add up to `n` buildings around `centre`, each next to an existing one.
    `types_p` = {type: probability} or a settlement style name from
    configs/buildings.json ("moshav", "arab_village", "kibbutz", "old_town", "town").
    Core types grow next to existing buildings; edge types (sheds, coops,
    greenhouses, water towers) go to the outskirts, beyond the current houses.
    Each drawn type gets up to `attempts` locations, so the final mix follows
    the probabilities even where large footprints rarely fit. Group types
    (coops) are placed as one unit.
    Returns (added entities, rejection counts by reason). Spec is modified."""
    cat = catalogue()
    if isinstance(types_p, str):
        st = cat["settlements"][types_p]
        types_p, spacing_m = st["types"], st["spacing_m"]
    r = rng(seed, "grow_village")
    blds = spec["entities"].setdefault("buildings", [])
    centre = centre or village_centre(site)
    added, rejected = [], {}
    for _ in range(tries):
        if len(added) >= n:
            break
        btype = pick(r, types_p)
        t = cat["types"][btype]
        for _attempt in range(attempts):
            b = _candidate(r, cat, btype, blds, centre, spacing_m, site)
            if b is None:
                break
            problems, _ = check_building(b, spec, site)
            if problems:
                key = ("too close" if problems[0][0].isdigit() else
                       "drainage line" if problems[0].startswith("in a drainage") else problems[0].split(" ")[0])
                rejected[key] = rejected.get(key, 0) + 1
                continue
            members = _group(r, t, b, spec, site, blds, n - len(added))
            if members is None:
                rejected["group does not fit"] = rejected.get("group does not fit", 0) + 1
                continue
            for mb in members:
                mb["id"] = next_id(spec, "buildings")
                blds.append(mb)
                added.append(mb)
            break
    return added, rejected


def _frontage_slots(spec, r, classes, spacing_m, near=None, radius=None):
    """Positions along roads of the given classes, on both sides:
    (point, outward normal, road half width incl. shoulder, road class)."""
    from synth.common.config import load_json
    rc = load_json("configs/roads.json")["classes"]
    slots = []
    for rd in spec["entities"].get("roads", []):
        cls = rd.get("class")
        if cls not in classes:
            continue
        p = np.asarray(rd["centerline"], float)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        hw = rc[cls]["width"] / 2 + rc[cls]["shoulder"]
        for side in (1, -1):
            u = float(r.uniform(10, 18))   # keep clear of the junction
            while u < s[-1] - 8:
                k = int(np.clip(np.searchsorted(s, u), 1, len(p) - 1))
                f = (u - s[k - 1]) / max(seg[k - 1], 1e-9)
                pt = p[k - 1] + (p[k] - p[k - 1]) * f
                t = (p[k] - p[k - 1]) / max(seg[k - 1], 1e-9)
                nrm = np.array([-t[1], t[0]]) * side
                if near is None or np.linalg.norm(pt - near) < radius:
                    slots.append((pt, nrm, hw, cls))
                u += float(r.uniform(*spacing_m))
    return slots


def grow_along_streets(spec, site, n, style, seed, centre, tries=6000):
    """Settlement on a street network. Houses stand along streets and lanes at a
    setback, facing the street. Style keys (place files / settlement styles):
      types        {type: probability}
      frontage     spacing_m, setback_m, second_row_p, rows (max rows behind the street)
      density      core_m, falloff_m: full density inside core_m of the centre,
                   thinning outward over falloff_m (villages fade into the groves)
      landmarks    {type: [min, max]} placed first, as close to the centre as possible (mosques),
                   at least landmark_spacing_m[type] (default 220 m) from each other
    Zones: core types along streets/lanes (and the main road near the centre),
    edge types along dirt tracks, industrial types along the main road outside
    the core. A type with "frontage": [classes] only takes slots on those road
    classes (shop-houses on main roads and streets). Returns (added, rejection counts)."""
    from synth.common.config import load_json
    cat = catalogue()
    if isinstance(style, dict):   # a style definition from a place file
        st, fr, name = style, style["frontage"], style.get("name", "custom")
    else:
        st, fr, name = cat["settlements"][style], load_json("configs/roads.json")["networks"][style]["frontage"], style
    r = rng(seed, "frontage", name)
    centre = np.asarray(centre, float)
    dens = st.get("density", {"core_m": 1e9, "falloff_m": 1.0})
    slots = {
        "core": _frontage_slots(spec, r, ("street", "lane"), fr["spacing_m"])
                + _frontage_slots(spec, r, ("main",), fr["spacing_m"], near=centre, radius=dens.get("main_frontage_m", 180.0)),
        "edge": _frontage_slots(spec, r, ("track",), [v * 1.5 for v in fr["spacing_m"]]),
        "industrial": [sl for sl in _frontage_slots(spec, r, ("main",), [v * 1.6 for v in fr["spacing_m"]])
                       if 180.0 < np.linalg.norm(sl[0] - centre) < 480.0],
    }
    order = {z: list(r.permutation(len(v))) for z, v in slots.items()}
    blds = spec["entities"].setdefault("buildings", [])
    added, rejected = [], {}

    def keep_p(pt):
        d = float(np.linalg.norm(pt - centre))
        return 1.0 if d < dens["core_m"] else max(0.12, 1.0 - (d - dens["core_m"]) / dens["falloff_m"])

    def place(btype, pt, nrm, hw, extra_setback=0.0):
        t = cat["types"][btype]
        b = _new_building(r, btype, 0.0, 0.0, 0.0, "bld_tmp")
        setback = float(r.uniform(*fr["setback_m"])) + extra_setback
        c = pt + nrm * (hw + setback + b["depth"] / 2)
        b["center"] = [round(float(c[0]), 2), round(float(c[1]), 2)]
        b["yaw_deg"] = round(math.degrees(math.atan2(-nrm[0], nrm[1])) + float(r.normal(0, 1.5)), 1)   # front faces the road
        problems, _ = check_building(b, spec, site)
        if problems:
            key = ("too close" if problems[0][0].isdigit() else
                   "drainage line" if problems[0].startswith("in a drainage") else problems[0].split(" ")[0])
            rejected[key] = rejected.get(key, 0) + 1
            return None
        members = _group(r, t, b, spec, site, blds, n - len(added))
        if members is None:
            rejected["group does not fit"] = rejected.get("group does not fit", 0) + 1
            return None
        for mb in members:
            mb["id"] = next_id(spec, "buildings")
            blds.append(mb)
            added.append(mb)
        return b

    def next_slot(zone, classes=None):
        """Pop the next free slot of a zone (optionally only on given road classes)."""
        o = order[zone]
        for i in range(len(o) - 1, -1, -1):
            sl = slots[zone][o[i]]
            if classes is None or sl[3] in classes:
                o.pop(i)
                return sl
        return None

    core_types = {k: v for k, v in st["types"].items() if cat["types"][k].get("zone", "core") == "core"}
    # landmarks (mosques): as central as possible, on a street
    for btype, (lo, hi) in st.get("landmarks", {}).items():
        want = int(r.integers(lo, hi + 1))
        gap = st.get("landmark_spacing_m", {}).get(btype, 220.0)   # each mosque serves its own neighbourhood
        mine = [np.asarray(b["center"]) for b in spec["entities"]["buildings"] if b.get("type") == btype]
        by_d = sorted(range(len(slots["core"])), key=lambda i: np.linalg.norm(slots["core"][i][0] - centre))
        for i in by_d:
            if want <= 0:
                break
            if i not in order["core"]:
                continue
            pt, nrm, hw, _ = slots["core"][i]
            if any(np.linalg.norm(pt - m) < gap for m in mine):
                continue
            b = place(btype, pt, nrm, hw)
            if b is not None:
                order["core"].remove(i)
                mine.append(np.asarray(b["center"]))
                want -= 1
    for _ in range(tries):
        if len(added) >= n or not any(order.values()):
            break
        btype = pick(r, st["types"])
        t = cat["types"][btype]
        zone = t.get("zone", "core")
        sl = next_slot(zone, t.get("frontage"))
        if sl is None:
            continue
        pt, nrm, hw, _ = sl
        if zone == "core" and r.random() > keep_p(pt):
            continue   # thinning toward the village edge
        b = place(btype, pt, nrm, hw)
        if b is None or zone != "core":
            continue
        back = b["depth"]
        for _row in range(fr.get("rows", 2) - 1):   # houses behind the first row (dense villages and towns)
            if len(added) >= n or r.random() > fr["second_row_p"] * keep_p(pt):
                break
            b2 = place(pick(r, core_types), pt, nrm, hw, extra_setback=back + float(r.uniform(2.5, 6)))
            if b2 is None:
                break
            back += b2["depth"] + float(r.uniform(2.5, 6))
    return added, rejected


def fill_blocks(spec, site, n, style, seed, centre, step_m=None, tries_per_cell=2):
    """Infill: houses on the land between the streets (inside the built-up area),
    each turned to face its nearest street, as dense Arab towns grow block by
    block. Candidate points on a jittered grid within the style's density area,
    closest to the centre first. Returns (added, rejection counts)."""
    from shapely.geometry import Point
    from shapely.ops import nearest_points
    cat = catalogue()
    st = style
    step_m = step_m or st.get("infill_step_m", 15.0)
    dens = st.get("density", {"core_m": 300, "falloff_m": 300})
    r = rng(seed, "infill", st.get("name", "custom"))
    centre = np.asarray(centre, float)
    roads = [rd for rd in spec["entities"].get("roads", []) if rd.get("class") in ("street", "lane", "main")]
    if not roads:
        return [], {}
    lines = [LineString(np.asarray(rd["centerline"])[:, :2]) for rd in roads]
    R = dens["core_m"] + dens["falloff_m"]
    xs = np.arange(centre[0] - R, centre[0] + R, step_m)
    ys = np.arange(centre[1] - R, centre[1] + R, step_m)
    pts = [np.array([x, y]) + r.normal(0, step_m * 0.25, 2) for x in xs for y in ys]
    pts.sort(key=lambda p: np.linalg.norm(p - centre))
    core_types = {k: v for k, v in st["types"].items() if cat["types"][k].get("zone", "core") == "core"}
    blds = spec["entities"].setdefault("buildings", [])
    added, rejected = [], {}
    for p in pts:
        if len(added) >= n:
            break
        d = float(np.linalg.norm(p - centre))
        keep = 1.0 if d < dens["core_m"] else max(0.0, 1.0 - (d - dens["core_m"]) / dens["falloff_m"])
        if r.random() > keep:
            continue
        P = Point(p)
        L = min(lines, key=lambda l: l.distance(P))
        if L.distance(P) > 45.0:
            continue   # infill stays within reach of a street
        q = np.asarray(nearest_points(L, P)[0].coords[0])
        to_road = (q - p) / (np.linalg.norm(q - p) + 1e-9)
        for _ in range(tries_per_cell):
            b = _new_building(r, pick(r, core_types), float(p[0]), float(p[1]),
                              math.degrees(math.atan2(to_road[0], -to_road[1])) + float(r.normal(0, 3)), "bld_tmp")
            problems, _ = check_building(b, spec, site)
            if not problems:
                b["id"] = next_id(spec, "buildings")
                blds.append(b)
                added.append(b)
                break
            key = ("too close" if problems[0][0].isdigit() else
                   "drainage line" if problems[0].startswith("in a drainage") else problems[0].split(" ")[0])
            rejected[key] = rejected.get(key, 0) + 1
    return added, rejected
