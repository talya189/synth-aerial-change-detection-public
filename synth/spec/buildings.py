"""Spec buildings: catalogue defaults, footprints, plan records (pure Python).

A spec building (configs/specs/*.json, entities.buildings[]):

  {"id": "bld_004", "type": "house", "center": [x, y], "width": 11, "depth": 10,
   "yaw_deg": 12, "floors": 2, "roof": "flat", "wall": "wall.stucco",
   "annex": {"side": "left", "width": 5, "depth": 6, "floors": 1},   optional
   "balcony": true, "expandable": false, "tint": [r, g, b],          optional
   "top_setback": 0.4, "pv": true, "railing": false, "pergola": true,  optional
   "roof_tint": [r, g, b], "pilotis": false, "roof_hat": true,          optional
   "podium": {"floors": 2, "margin": 3.0}}                           optional (towers)

Types and their ranges: configs/buildings.json. top_setback = the top floor
covers only the back (1 - top_setback) of the depth; pv = solar panels on the
roof; railing = metal railing instead of a solid parapet; pergola at the door.
Water towers: width = diameter; floors 1 = ground reservoir, 2 = elevated tank.
`floors` counts all storeys; with a podium, the podium's floors are part of it.

Local frame: x across the front facade (width), y into the building (depth);
the front faces local -y, i.e. world direction (sin yaw, -cos yaw).
Missing optional fields are drawn from configs/buildings.json, seeded by the
building id, so a building never changes when other buildings are edited.
"""
import math

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from synth.common.config import load_json
from synth.common.rng import rng

ANNEX_SIDES = ("left", "right", "back")
_CAT = None


def catalogue():
    global _CAT
    if _CAT is None:
        _CAT = load_json("configs/buildings.json")
    return _CAT


def pick(r, table):
    keys = list(table)
    p = np.array([table[k] for k in keys], float)
    return keys[int(r.choice(len(keys), p=p / p.sum()))]


def _to_world(pts, center, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.asarray(pts, float)
    return np.stack([center[0] + pts[:, 0] * c - pts[:, 1] * s, center[1] + pts[:, 0] * s + pts[:, 1] * c], -1)


def local_rects(b):
    """Main block and optional annex as local (x0, y0, x1, y1)."""
    w, d = b["width"], b["depth"]
    rects = [(-w / 2, -d / 2, w / 2, d / 2)]
    a = b.get("annex")
    if a:
        aw, ad = a["width"], a["depth"]
        if a["side"] == "left":
            rects.append((-w / 2 - aw, d / 2 - ad, -w / 2, d / 2))
        elif a["side"] == "right":
            rects.append((w / 2, d / 2 - ad, w / 2 + aw, d / 2))
        else:
            rects.append((-aw / 2, d / 2, aw / 2, d / 2 + ad))
    return rects


def podium_rect(b):
    """Local rect of the podium (wider base of a tower), or None."""
    p = b.get("podium")
    if not p:
        return None
    w, d, m = b["width"], b["depth"], p["margin"]
    return (-w / 2 - m, -d / 2 - m, w / 2 + m, d / 2 + m)


def _world(rect, b):
    x0, y0, x1, y1 = rect
    return _to_world([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], b["center"], math.radians(b.get("yaw_deg", 0.0)))


def world_rects(b):
    return [_world(r_, b) for r_ in local_rects(b)]


def footprint(b):
    """Whole footprint (main block + annex + podium) as a shapely polygon, world metres."""
    polys = [Polygon(p) for p in world_rects(b)]
    if podium_rect(b):
        polys.append(Polygon(_world(podium_rect(b), b)))
    u = unary_union(polys)
    if u.geom_type != "Polygon":   # parts that only touch along an edge: weld them into one outline
        u = unary_union([q.buffer(0.005, join_style="mitre") for q in polys]).buffer(-0.005, join_style="mitre")
    return u


def _weighted(r, palette):
    """Pick one [value, weight] entry by weight; returns the value."""
    w = np.array([p[1] for p in palette], float)
    return list(palette[int(r.choice(len(palette), p=w / w.sum()))][0])


def complete(b, seed):
    """Fill every optional field from the catalogue (deterministic per id).
    Returns a new dict; the input is not modified."""
    cat = catalogue()
    t = cat["types"][b.get("type", "house")]
    r = rng(seed, "spec_building", b["id"])
    out = dict(b)
    out.setdefault("type", "house")
    out.setdefault("yaw_deg", 0.0)
    out.setdefault("roof", pick(r, t["roof_p"]))
    out.setdefault("wall", pick(r, t["wall_p"]))
    out.setdefault("floor_height", round(float(r.uniform(*t["floor_height_m"])), 2))
    out.setdefault("parapet", round(float(r.uniform(*t["parapet_m"])), 2) if out["roof"] == "flat" else 0.0)
    out.setdefault("balcony", bool(out["floors"] >= 2 and r.random() < t["balcony_p"]))
    out.setdefault("expandable", bool(out["roof"] == "flat" and r.random() < t["expandable_p"]))
    out.setdefault("tint", _weighted(r, cat["facades"].get(out["wall"], [[[1, 1, 1], 1]])))
    out.setdefault("roof_tint", _weighted(r, cat["roof_tints"]["palette"]))
    out.setdefault("annex", None)
    opt = t.get("options", {})
    flat = out["roof"] == "flat"
    out.setdefault("top_setback", round(float(r.uniform(0.3, 0.5)), 2)
                   if flat and out["floors"] >= 2 and r.random() < opt.get("top_setback_p", 0) else 0.0)
    out.setdefault("pv", bool(out["roof"] in ("flat", "gable_metal") and r.random() < opt.get("pv_p", 0)))
    out.setdefault("railing", bool(flat and r.random() < opt.get("railing_p", 0)))
    out.setdefault("pergola", bool(r.random() < opt.get("pergola_p", 0)))
    out.setdefault("pilotis", bool(flat and out["floors"] >= 2 and r.random() < opt.get("pilotis_p", 0)))
    out.setdefault("roof_hat", bool(flat and r.random() < opt.get("roof_hat_p", 0)))
    if "podium" not in out:
        pf, pm = opt.get("podium_floors", [1, 2]), opt.get("podium_margin_m", [2.5, 5.0])
        out["podium"] = ({"floors": int(r.integers(pf[0], pf[1] + 1)), "margin": round(float(r.uniform(*pm)), 1)}
                         if out["floors"] >= opt.get("podium_min_floors", 5) and r.random() < opt.get("podium_p", 0) else None)
    return out


def plan_record(b, seed, grid, rooftop_fn):
    """Spec building -> the record building_builder consumes (absolute heights)."""
    cat = catalogue()
    b = complete(b, seed)
    t = cat["types"][b["type"]]
    r = rng(seed, "spec_building_detail", b["id"])
    yaw = math.radians(b["yaw_deg"])
    rects = world_rects(b)
    pod = _world(podium_rect(b), b) if b.get("podium") else None
    allpts = np.concatenate(rects + ([pod] if pod is not None else []))
    zc = grid.sample(allpts[:, 0], allpts[:, 1])
    zmain = grid.sample((pod if pod is not None else rects[0])[:, 0], (pod if pod is not None else rects[0])[:, 1])
    rec = {
        "id": b["id"], "type": b["type"], "kind": t["kind"],
        "footprint": rects[0].tolist(), "annex": rects[1].tolist() if len(rects) > 1 else None,
        "annex_floors": int(b["annex"]["floors"]) if b.get("annex") else 0,
        "yaw": yaw, "width": float(b["width"]), "depth": float(b["depth"]),
        "front_normal": [math.sin(yaw), -math.cos(yaw)],
        "floors": int(b["floors"]), "floor_height": float(b["floor_height"]),
        "floor_z": float(zmain.max() + 0.2), "foundation_z": float(zc.min() - 0.4),
        "roof": {"tiled": "tiled_hip"}.get(b["roof"], b["roof"]),   # flat, tiled_hip, tiled_gable, gable_metal, tunnel, tank, dome
        "wall_role": b["wall"], "tint": list(b["tint"]), "roof_tint": list(b["roof_tint"]), "parapet": float(b["parapet"]),
        "pilotis": bool(b["pilotis"]), "roof_hat": bool(b["roof_hat"]),
        "window_spacing": t["window_spacing"], "window_w": t.get("window_w", 1.3), "window_h": t.get("window_h", 1.3),
        "top_setback": float(b["top_setback"]), "pv": bool(b["pv"]), "railing": bool(b["railing"]),
        "pergola": bool(b["pergola"]),
        "podium": pod.tolist() if pod is not None else None,
        "podium_floors": int(b["podium"]["floors"]) if b.get("podium") else 0,
        "shutters_p": t["shutters_p"], "window_style": t["window_style"], "door": t.get("door", "single"),
        "expandable": bool(b["expandable"]), "plinth_role": "wall.stone", "street": 0,
        "seed_path": ["building", b["id"]],
    }
    rec["balcony"] = ({"width_frac": float(r.uniform(0.35, 0.7)), "offset_frac": float(r.uniform(-0.2, 0.2)),
                       "depth": float(r.uniform(1.2, 1.8))} if b["balcony"] and b["floors"] >= 2 else None)
    if rec["roof"] == "flat" and t.get("rooftop"):
        rc = dict(t["rooftop"])
        if rec["pv"]:   # the panel array takes the roof: fewer other items
            rc.update(solar_heaters=[0, 1], clutter=[0, 0], stairwell_p=min(rc.get("stairwell_p", 0), 0.3))
        rec["rooftop"] = b.get("rooftop") or rooftop_fn(r, dict(rec, depth=rec["depth"] * (1 - rec["top_setback"])), rc)
        if rec["top_setback"]:   # items stand on the lower, front part of the roof
            dy = -rec["depth"] * rec["top_setback"] / 2
            for it in rec["rooftop"]:
                if "y" in it:
                    it["y"] = float(it["y"] + dy)
    elif rec["roof"] == "flat" and rec["expandable"]:
        rec["rooftop"] = [{"type": "rebar", "spacing": float(r.uniform(3.5, 5.0))}]
    else:
        rec["rooftop"] = []
    return rec
