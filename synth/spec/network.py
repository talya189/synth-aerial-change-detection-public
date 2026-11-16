"""Road, street, track and footpath networks for a settlement (pure Python).

The layout rules come from real Galilee settlements seen in aerial imagery
(Korazim moshav, Tuba-Zangariyye, Rosh Pina):

  regional road   least-cost route under the grade limit, then *faired* to a
                  minimum curve radius (long sweeping curves, no wiggles); it
                  bypasses the village, an access road joins it at ~90 deg
  streets         follow the contours (low grade, natural curves); several
                  contour streets are linked by short cross streets into a mesh
                  with loops (Arab village, town blocks), or a contour
                  collector carries a comb of straight cul-de-sacs with turning
                  circles running up the slope (moshav)
  junctions       a street that reaches another road ends there in a T-junction
                  (both ends of a cross street are joined: `connects`,
                  `connects_end`); no road crosses another without a junction
  tracks          dirt roads continuing from street ends along the contour, and
                  field tracks of a few long straight legs
  footpaths       desire lines where they save a walk: across blocks between
                  parallel streets far from a cross street, and from the
                  village edge out into the open land; never alongside a road

Everything is written into the spec as explicit entities (class, centerline,
the roads it joins). Styles and classes: configs/roads.json.
"""
import math

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon

from synth.common.config import load_json
from synth.common.rng import rng
from synth.planner import roads as rplan
from synth.planner.plan import _branch_route, _road_entry
from synth.spec.buildings import footprint
from synth.spec.scene_spec import next_id


def _cfg():
    return load_json("configs/roads.json")


# --------------------------------------------------------------------- geometry
def fair(xy, min_radius_m, end_m=60.0):
    """Smooth a road until its tightest curve (ignoring `end_m` at each end,
    which stay pinned to the site boundary) is at least `min_radius_m`."""
    xy = np.asarray(xy, float)
    out = xy
    for sigma in (15, 25, 40, 60, 85, 115, 150):
        out = rplan.smooth_polyline(xy, sigma, keep_start=False, keep_end=False)
        if min_radius(out, trim_m=end_m) >= min_radius_m:
            break
    return out


def min_radius(xy, step=2.0, window=5, trim_m=0.0):
    """Smallest curve radius of a polyline, measured over `window` steps (m),
    ignoring `trim_m` at each end."""
    p = rplan.resample_polyline(np.asarray(xy, float), step)[0]
    k = int(trim_m / step)
    if k and len(p) > 2 * k + 2 * window + 1:
        p = p[k:-k]
    if len(p) < 2 * window + 1:
        return float("inf")
    a, b, c = p[:-2 * window], p[window:-window], p[2 * window:]
    d1, d2 = b - a, c - b
    ang = np.abs(np.arctan2(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], (d1 * d2).sum(1)))
    return float((step * window / np.maximum(ang, 1e-9)).min())


class _Field:
    """Smoothed terrain gradient for tracing streets along contours."""

    def __init__(self, site, sigma_m=12.0):
        z = getattr(site.grid, "z_smooth", site.grid.z)
        self.zs = ndimage.gaussian_filter(z, sigma_m / site.grid.res)
        self.gy, self.gx = np.gradient(self.zs, site.grid.res)
        self.site = site

    def grad(self, p):
        g = self.site.grid
        return np.array([float(g.sample(p[0], p[1], self.gx)[0]), float(g.sample(p[0], p[1], self.gy)[0])])

    def z(self, p):
        return float(self.site.grid.sample(p[0], p[1], self.zs)[0])


def _rot(d, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def _xy(rd):
    return np.asarray(rd["centerline"], float)[:, :2]


def _lines(spec, exclude=()):
    return [(rd["id"], LineString(_xy(rd))) for rd in spec["entities"].get("roads", []) if rd["id"] not in exclude]


def trace(field, spec, start, d0, length, follow=1.0, step=3.0, max_turn_deg=6.0, margin=35.0,
          ignore=(), min_before_hit=10.0, r=None, wander=0.0):
    """Walk from `start` in direction `d0`. follow=1 bends along the contour,
    0 keeps straight. Stops at the core margin, or where it meets another road
    (returns that road's id: the street ends there in a T-junction)."""
    p = np.asarray(start, float)
    d = np.asarray(d0, float) / np.linalg.norm(d0)
    pts, s, hit = [p.copy()], 0.0, None
    lines = _lines(spec, exclude=ignore)
    lim = field.site.half - margin
    max_turn = math.radians(max_turn_deg)
    while s < length:
        g = field.grad(p)
        gn = np.linalg.norm(g)
        tgt = d
        if follow > 0 and gn > 1e-4:
            c = np.array([-g[1], g[0]]) / gn
            if np.dot(c, d) < 0:
                c = -c
            w = follow * min(1.0, gn / 0.02)   # flat ground: no contour to follow, keep going
            tgt = (1 - w) * d + w * c
            tgt /= np.linalg.norm(tgt)
        if wander and r is not None:
            tgt = _rot(tgt, float(r.normal(0, wander)))
        turn = math.atan2(d[0] * tgt[1] - d[1] * tgt[0], float(np.dot(d, tgt)))
        d = _rot(d, float(np.clip(turn, -max_turn, max_turn)))
        q = p + d * step
        if np.any(np.abs(q) > lim):
            break
        if s > min_before_hit:
            seg = LineString([p, q])
            hits = [(rid, seg.intersection(L)) for rid, L in lines if seg.intersects(L)]
            if hits:
                rid, x = hits[0]
                pts.append(np.asarray(x.coords[0]) if x.geom_type == "Point" else np.asarray(x.centroid.coords[0]))
                hit = rid
                break
        p = q
        pts.append(p.copy())
        s += step
    return np.array(pts), hit


def _grade(field, xy, window=20.0):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    if s[-1] < 1:
        return 0.0
    z = np.array([field.z(p) for p in xy])
    k = np.searchsorted(s, s + window).clip(0, len(s) - 1)
    g = np.abs(z[k] - z) / np.maximum(s[k] - s, 1e-6)
    g = g[s[k] - s > window / 2]
    return float(g.max()) if len(g) else abs(z[-1] - z[0]) / s[-1]


def _crowded(xy, spec, min_sep=22.0, skip_m=30.0, max_frac=0.15):
    """Does a new road run alongside existing roads (beyond its first skip_m)?"""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    pts = xy[s > skip_m]
    if not len(pts):
        return False
    others = [L for _, L in _lines(spec)]
    close = [any(o.distance(Point(p)) < min_sep for o in others) for p in pts[::3]]
    return float(np.mean(close)) > max_frac


# ---------------------------------------------------------------------- entities
def _entity(spec, cls, xy, connects=None, connects_end=None, end_cap=None, tol=0.25):
    line = LineString(xy).simplify(tol)
    rd = {"id": next_id(spec, "roads"), "class": cls,
          "centerline": [[round(x, 2), round(y, 2)] for x, y in line.coords]}
    if connects:
        rd["connects"] = connects
    if connects_end:
        rd["connects_end"] = connects_end
    if end_cap:
        rd["end_cap"] = end_cap
    spec["entities"].setdefault("roads", []).append(rd)
    return rd


def _records(spec):
    return [{"id": rd["id"], "centerline": rd["centerline"]} for rd in spec["entities"].get("roads", [])]


def _at(rd, frac=None, s_m=None):
    """Point, unit tangent and total length of a road at a fraction (or distance) of its length."""
    p = _xy(rd)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = s[-1] * frac if s_m is None else min(max(s_m, 0.0), s[-1])
    k = int(np.clip(np.searchsorted(s, u), 1, len(p) - 1))
    f = (u - s[k - 1]) / max(seg[k - 1], 1e-9)
    t = (p[k] - p[k - 1]) / max(seg[k - 1], 1e-9)
    return p[k - 1] + (p[k] - p[k - 1]) * f, t, s[-1]


def _turn_to_grade(field, spec, start, d, length, max_grade, follow=0.0, ignore=(),
                   steps=(0, 12, -12, 24, -24, 36, -36, 50, -50)):
    """Straight street from `start`: the first direction near `d` whose grade is
    within the limit (streets up a slope bend toward the contour if needed)."""
    for deg in steps:
        xy, hit = trace(field, spec, start, _rot(d, math.radians(deg)), length, follow=follow, ignore=ignore)
        if len(xy) > 8 and _grade(field, xy) <= max_grade:
            return xy, hit
    return None, None


# ------------------------------------------------------------------- main road
def _main_road(spec, site, router, centre, cfg, r, log, mode="bypass", field=None):
    """mode: bypass = passes the settlement at ~160 m; downhill = along the foot of
    the slope below it (hillside villages); through = crosses its centre (towns,
    cities: the main street with shops)."""
    mc = cfg["classes"]["main"]
    best = None
    down = None
    if field is not None:
        g = field.grad(centre)
        down = -g / (np.linalg.norm(g) + 1e-9)
    target_d = {"bypass": 160.0, "downhill": 170.0, "through": 0.0}[mode]
    for _ in range(24):
        a, b = _road_entry(r, site.grid, site.slope, mc["max_grade"])
        seg = LineString([a, b])
        score = abs(seg.distance(Point(centre)) - target_d)
        if mode == "downhill" and down is not None:   # the road must lie below the village
            q = np.asarray(seg.interpolate(seg.project(Point(centre))).coords[0])
            score += 0 if np.dot(q - centre, down) > 0 else 200
        if best is None or score < best[0]:
            best = (score, a, b)
    if mode == "through":   # routed via the centre: it becomes the main street
        r1 = router.route(best[1], centre, mc["max_grade"])
        r2 = router.route(centre, best[2], mc["max_grade"])
        raw = None if r1 is None or r2 is None else np.vstack([r1, r2[1:]])
    else:                   # bypass / downhill: routing through the settlement is expensive
        c = np.arange(router.n) * router.res - router.half
        X, Y = np.meshgrid(c, c)
        raw = router.route(best[1], best[2], mc["max_grade"],
                           extra_cost=np.where(np.hypot(X - centre[0], Y - centre[1]) < 110.0, 60.0, 0.0))
    xy = rplan.smooth_polyline(raw, mc["smooth_m"], keep_start=False, keep_end=False)
    xy = fair(xy, mc["min_radius_m"])
    main = _entity(spec, "main", xy, tol=0.08)
    log(f"main road {LineString(xy).length:.0f} m, min radius {min_radius(xy, trim_m=60):.0f} m")
    return main


def _access(spec, router, main, target_rd, target_xy, cls, cfg):
    """Street from the main road to a village street: leaves the main road at
    ~90 deg (straight stub), routed under the grade limit, ends on the street."""
    mxy = _xy(main)
    j = mxy[int(np.argmin(np.linalg.norm(mxy - target_xy, axis=1)))]
    if np.linalg.norm(j - target_xy) < 15:
        return None
    c = cfg["classes"][cls]
    raw, stub = _branch_route(router, _records(spec), mxy, cfg["classes"]["main"]["width"] / 2, j, target_xy, c["max_grade"])
    if raw is None:
        return None
    raw[0] = j
    xy = rplan.smooth_polyline(np.vstack([raw, target_xy]), c["smooth_m"], keep_start=stub is None, keep_end=True)
    if stub is not None:
        xy = rplan.straight_start(xy, *stub)
    return _entity(spec, cls, xy, connects=main["id"], connects_end=target_rd["id"])


# --------------------------------------------------------------------- streets
def _contour_streets(spec, site, field, centre, net, r):
    """Streets along the contours, stacked up and down the slope from the centre."""
    cs = net["contour"]
    g = field.grad(centre)
    v = -g / (np.linalg.norm(g) + 1e-9) if np.linalg.norm(g) > 1e-4 else np.array([0.0, -1.0])   # downhill
    u = np.array([-v[1], v[0]])
    K = cs["count"]
    offs, acc = [], 0.0
    for _ in range(K):
        offs.append(acc)
        acc += float(r.uniform(*cs["spacing_m"]))
    offs = np.array(offs) - (offs[-1] if K > 1 else 0.0) * float(r.uniform(0.35, 0.65))
    streets = []
    for k, off in enumerate(offs):
        frac = abs(k - (K - 1) / 2) / max((K - 1) / 2, 1)
        L = float(r.uniform(*cs["length_m"])) * (1 - 0.45 * frac ** 2)   # outer streets shorter: lens-shaped village
        start = centre + v * off + u * float(r.normal(0, 15))
        lines = [L for _, L in _lines(spec)]
        for shift in (0, 30, -30, 60, -60, 90, -90):   # not on top of an existing road (a main road through the centre)
            cand = start + u * shift
            if all(L.distance(Point(cand)) > 25.0 for L in lines):
                start = cand
                break
        if np.any(np.abs(start) > site.half - 60):
            continue
        sh = float(r.uniform(0.35, 0.65))
        a, hit_a = trace(field, spec, start, u, L * sh, follow=cs["follow"], r=r, wander=0.01)
        b, hit_b = trace(field, spec, start, -u, L * (1 - sh), follow=cs["follow"], r=r, wander=0.01)
        xy = np.vstack([b[::-1], a[1:]])
        if LineString(xy).length < 60:
            continue
        streets.append(_entity(spec, cs["class"], rplan.smooth_polyline(xy, 6.0), connects=hit_b, connects_end=hit_a))
    streets.sort(key=lambda s_: float(np.dot(_xy(s_).mean(0) - centre, v)))   # neighbours adjacent
    return streets


def _switchback(router, spec, p, q, A_id, cls, cfg):
    """Connector routed under the grade limit between two points on two roads: on
    steep ground the least-cost route winds down the slope (serpentine street)."""
    c = cfg["classes"][cls]
    raw = router.route(p, q, c["max_grade"] * 0.7)
    if raw is None or len(raw) < 3:
        return None
    xy = rplan.smooth_polyline(np.vstack([p, raw[1:-1], q]), max(3.0, c["smooth_m"] * 0.6), keep_start=True, keep_end=True)
    if LineString(xy).length > 4.0 * np.linalg.norm(q - p) + 40:
        return None   # absurd detour
    return xy


def _cross_streets(spec, field, streets, net, cfg, r, router=None, max_switchbacks=24):
    """Short streets linking neighbouring contour streets (loops, blocks), and
    dead-end alleys off the outermost streets."""
    cc = net["connectors"]
    cls = net["contour"]["class"]
    maxg = cfg["classes"][cls]["max_grade"]
    out = []
    for A, B in zip(streets[:-1], streets[1:]):
        LB = LineString(_xy(B))
        _, _, LA = _at(A, 0.0)
        u = float(r.uniform(15, cc["every_m"][0]))
        while u < LA - 15:
            p, t, _ = _at(A, s_m=u)
            q = np.asarray(LB.interpolate(LB.project(Point(p))).coords[0])
            dist = float(np.linalg.norm(q - p))
            d = (q - p) / max(dist, 1e-9)
            if 15 < dist < 2.5 * net["contour"]["spacing_m"][1] and abs(np.dot(d, t)) < 0.6:
                # straight across if the slope allows, else diagonally (longer, gentler), like real hill lanes
                done = False
                for deg in (0, 20, -20, 35, -35, 50, -50):
                    dd = _rot(d, math.radians(deg))
                    xy, hit = trace(field, spec, p, dd, dist * 2.2, follow=0.0, ignore=(A["id"],), min_before_hit=4.0)
                    if hit is not None and len(xy) > 3 and _grade(field, xy, 10.0) <= maxg * 0.9:
                        out.append(_entity(spec, cls, xy, connects=A["id"], connects_end=hit))
                        done = True
                        break
                if not done and router is not None and max_switchbacks > 0:   # too steep: a serpentine street
                    xy = _switchback(router, spec, p, q, A["id"], cls, cfg)
                    if xy is not None:
                        out.append(_entity(spec, cls, xy, connects=A["id"], connects_end=B["id"]))
                        max_switchbacks -= 1
            u += float(r.uniform(*cc["every_m"]))
    centre = np.mean([_xy(S).mean(0) for S in streets], axis=0) if streets else np.zeros(2)
    for S in (streets[0], streets[-1]) if len(streets) > 1 else ():
        _, _, L = _at(S, 0.0)
        u = float(r.uniform(10, 30))
        while u < L - 10:
            if r.random() < cc.get("alley_p", 0.0):
                p, t, _ = _at(S, s_m=u)
                n = np.array([-t[1], t[0]])
                if np.dot(n, p - centre) < 0:
                    n = -n   # outward from the village
                xy, _ = _turn_to_grade(field, spec, p, n, float(r.uniform(*cc["alley_m"])), maxg, ignore=(S["id"],))
                if xy is not None:
                    out.append(_entity(spec, "lane", xy, connects=S["id"]))
            u += float(r.uniform(*cc["every_m"]))
    return out


def _culdesacs(spec, field, collector, net, cfg, r):
    """Moshav comb: straight cul-de-sacs off the collector, mostly up the slope,
    ending in a turning circle."""
    cd = net["culdesacs"]
    cls = cd["class"]
    maxg = cfg["classes"][cls]["max_grade"]
    out = []
    _, _, L = _at(collector, 0.0)
    u = float(r.uniform(20, 35))
    while u < L - 20:
        p, t, _ = _at(collector, s_m=u)
        n = np.array([-t[1], t[0]])
        up = -field.grad(p)
        sides = [1, -1] if np.dot(n, up) > 0 else [-1, 1]   # first: the uphill side
        for i, sgn in enumerate(sides):
            if r.random() > (cd["p_up"] if i == 0 else cd["p_down"]):
                continue
            xy, hit = _turn_to_grade(field, spec, p, n * sgn, float(r.uniform(*cd["length_m"])), maxg,
                                     ignore=(collector["id"],), steps=(0, 8, -8, 16, -16, 25, -25))
            if xy is None or LineString(xy).length < 40:
                continue
            out.append(_entity(spec, cls, xy, connects=collector["id"], connects_end=hit,
                               end_cap=None if hit else cd["circle_r"]))
        u += float(r.uniform(*cd["spacing_m"]))
    return out


# ---------------------------------------------------------------------- tracks
def _tracks(spec, site, field, streets, centre, net, cfg, r):
    tc = net["tracks"]
    maxg = cfg["classes"]["track"]["max_grade"]
    out = []
    ends = []
    for S in streets:
        if S.get("end_cap"):
            continue
        xy = _xy(S)
        if len(xy) < 5:
            continue
        if not S.get("connects_end"):
            ends.append((S, xy[-1], (xy[-1] - xy[-4]) / (np.linalg.norm(xy[-1] - xy[-4]) + 1e-9)))
        if not S.get("connects"):
            ends.append((S, xy[0], (xy[0] - xy[3]) / (np.linalg.norm(xy[0] - xy[3]) + 1e-9)))
    ends.sort(key=lambda e: -np.linalg.norm(e[1] - centre))
    for S, end, d in ends:   # the paved street continues as a dirt track along the contour
        if len(out) >= tc["continue"]:
            break
        xy, hit = trace(field, spec, end, d, float(r.uniform(*tc["length_m"])), follow=0.7,
                        ignore=(S["id"],), r=r, wander=0.02)
        if len(xy) > 15 and _grade(field, xy) <= maxg and not _crowded(xy, spec, min_sep=14.0, skip_m=20.0):
            out.append(_entity(spec, "track", xy, connects=S["id"], connects_end=hit))
    n_cont = len(out)
    for _ in range(tc["field"] * 5):   # field tracks: a few long straight legs from the village edge
        if len(out) >= n_cont + tc["field"] or not streets:
            break
        S = streets[int(r.integers(len(streets)))]
        p, t, _ = _at(S, float(r.uniform(0.15, 0.85)))
        n = np.array([-t[1], t[0]])
        if np.dot(n, p - centre) < 0:
            n = -n
        legs, cur, d, hit = [p], p, n, None
        for _ in range(int(r.integers(2, 4))):
            seg, hit = _turn_to_grade(field, spec, cur, d, float(r.uniform(*tc["leg_m"])), maxg, ignore=(S["id"],),
                                      steps=(0, 15, -15, 30, -30, 45, -45))
            if seg is None:
                break
            legs.extend(seg[1:])
            cur, d = seg[-1], _rot(d, math.radians(float(r.choice([-1, 1]) * r.uniform(20, 65))))
            if hit:
                break
        legs = rplan.smooth_polyline(np.array(legs), 4.0) if len(legs) > 20 else None
        if legs is not None and not _crowded(legs, spec, min_sep=14.0, skip_m=20.0):
            out.append(_entity(spec, "track", legs, connects=S["id"], connects_end=hit))
    return out


# ---------------------------------------------------------------------- network
def plan_network(spec, site, style, seed, centre, log=print):
    """Add the main road, village streets and tracks for `style` to the spec."""
    import copy
    cfg = copy.deepcopy(_cfg())
    net = style if isinstance(style, dict) else cfg["networks"][style]   # a style definition, or a style name
    for k, v in net.get("grade_limits", {}).items():   # style-specific limits (e.g. very steep hill towns)
        cfg["classes"][k]["max_grade"] = v
    grid = site.grid
    if not hasattr(grid, "z_smooth"):
        grid.z_smooth = ndimage.gaussian_filter(grid.z, 4.0)
    r = rng(seed, "network", net.get("name", style) if isinstance(style, dict) else style)
    router = rplan.Router(grid, cfg["routing_res_m"])
    field = _Field(site)
    centre = np.asarray(centre, float)

    main = _main_road(spec, site, router, centre, cfg, r, log, mode=net.get("main_road", "bypass"), field=field)
    streets = _contour_streets(spec, site, field, centre, net, r)
    log(f"{len(streets)} contour streets ({net['contour']['class']})")
    extra = []
    if net.get("culdesacs"):
        for S in streets:
            extra += _culdesacs(spec, field, S, net, cfg, r)
        log(f"{len(extra)} cul-de-sacs")
    if net.get("connectors") and len(streets) > 1:
        cross = _cross_streets(spec, field, streets, net, cfg, r, router=router)
        log(f"{len(cross)} cross streets / alleys")
        extra += cross
    acc = []   # access roads from the main road to the nearest streets
    mline = LineString(_xy(main))
    for S in sorted(streets, key=lambda s_: mline.distance(LineString(_xy(s_))))[: net.get("access_count", 1)]:
        LS = LineString(_xy(S))
        q = np.asarray(LS.interpolate(LS.project(mline.interpolate(mline.project(LS.centroid)))).coords[0])
        a = _access(spec, router, main, S, q, net["contour"]["class"], cfg)
        if a is not None:
            acc.append(a)
    log(f"{len(acc)} access roads")
    tracks = _tracks(spec, site, field, streets + extra, centre, net, cfg, r)
    log(f"{len(tracks)} tracks")
    # repair until stable: every crossing a junction, grades within limits, one connected network
    cls = net["contour"]["class"]

    def stamp():   # style limits travel with each road into the build
        for rd in spec["entities"]["roads"]:
            if rd["class"] in net.get("grade_limits", {}):
                rd["max_grade"] = net["grade_limits"][rd["class"]]
    for it in range(4):
        n_cross = resolve_crossings(spec)
        stamp()
        removed = enforce_grades(spec, site, log=log, cfg=cfg)
        joined = connect_islands(spec, site, field, cls, cfg, log=log, router=router)
        stamp()
        if not (n_cross or removed or joined or crossings(spec)):
            break
    # smoothing and grade repair alternate until both hold (smoothing moves curves; re-routing adds bends)
    from synth.spec.to_build import grade_violations, road_records
    for _ in range(3):
        smooth_roads(spec, cfg, site=site)
        resolve_crossings(spec)
        stamp()
        if not grade_violations(road_records(spec["entities"]["roads"], site.grid, cfg)) and not unreachable(spec):
            break
        enforce_grades(spec, site, log=log, cfg=cfg)
        connect_islands(spec, site, field, cls, cfg, log=log, router=router)
        stamp()
    smooth_roads(spec, cfg, site=site)
    final_steep_cleanup(spec, site, cfg, log=log)
    resolve_crossings(spec)
    tidy(spec, log=log)
    drop_unreachable(spec, log=log)
    left = crossings(spec)
    if left:   # last resort: of a pair still touching, remove one the network does not depend on
        def cuts_off(rid):
            trial = dict(spec, entities=dict(spec["entities"], roads=[rd for rd in spec["entities"]["roads"] if rd["id"] != rid]))
            return bool(unreachable(trial))
        gone, kept = [], []
        for a_, b_ in left:
            ids_now = {rd["id"] for rd in spec["entities"]["roads"]}
            if a_ not in ids_now or b_ not in ids_now:
                continue
            for rid in sorted((a_, b_), reverse=True):
                if rid != spec["entities"]["roads"][0]["id"] and not cuts_off(rid):
                    spec["entities"]["roads"] = [rd for rd in spec["entities"]["roads"] if rd["id"] != rid]
                    for rd in spec["entities"]["roads"]:
                        for k in ("connects", "connects_end"):
                            if rd.get(k) == rid:
                                rd.pop(k)
                    gone.append(rid)
                    break
            else:
                kept.append((a_, b_))
        if gone:
            log(f"removed {gone}: touching roads that could not be made junctions")
        if kept:
            spec.setdefault("_warnings", []).append({"touching_roads_kept": kept})
    return {"main": main, "streets": streets, "extra": extra, "access": acc, "tracks": tracks}


# -------------------------------------------------------------------- footpaths
def _keep_clear(xy, polys, clear, iters=3):
    """Push polyline points that come closer than `clear` to a polygon away from it."""
    from shapely.ops import nearest_points
    xy = xy.copy()
    for _ in range(iters):
        moved = False
        for i, (x, y) in enumerate(xy):
            pt = Point(x, y)
            for poly in polys:
                if poly.distance(pt) < clear:
                    q = np.asarray(nearest_points(poly, pt)[0].coords[0])
                    vv = np.array([x, y]) - q
                    nv = np.linalg.norm(vv)
                    vv = vv / nv if nv > 1e-6 else np.array([1.0, 0.0])
                    xy[i] = q + vv * clear
                    x, y = xy[i]
                    pt = Point(x, y)
                    moved = True
        if not moved:
            break
    return xy


def _alongside(xy, roads, near_m=7.0, frac=0.3, skip_m=8.0):
    """Does a path run along a road (beyond its two ends)?"""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    pts = xy[(s > skip_m) & (s < s[-1] - skip_m)]
    if not len(pts):
        return False
    lines = [LineString(_xy(rd)) for rd in roads]
    close = [any(L.distance(Point(p)) < near_m for L in lines) for p in pts[::2]]
    return float(np.mean(close)) > frac


def plan_footpaths(spec, site, n, seed, width=(0.7, 1.2)):
    """Desire lines, planned after the buildings:
      shortcuts  across a block between two parallel streets, where walking via
                 the streets would be more than 2.5x longer
      outbound   from the village edge out into the open land (to fields, views)
    Routed on a 2 m grid around buildings, softly avoiding slopes, with a little
    wobble; dropped if they would run alongside a road."""
    from shapely import contains_xy
    grid = site.grid
    router = rplan.Router(grid, 2.0)
    blds = spec["entities"].get("buildings", [])
    roads = spec["entities"].get("roads", [])
    if not roads:
        return []
    c = np.arange(router.n) * router.res - router.half
    X, Y = np.meshgrid(c, c)
    blocked = np.zeros_like(X, bool)
    for b in blds:
        blocked |= contains_xy(footprint(b).buffer(1.2), X, Y)
    for w in spec["entities"].get("walls", []):   # footpaths go round walled yards
        if len(w.get("yard", [])) >= 3:
            blocked |= contains_xy(Polygon(w["yard"]).buffer(1.5), X, Y)
    extra = np.where(blocked, 400.0, 0.0)
    fps = [footprint(b) for b in blds]
    r = rng(seed, "footpaths")
    streets = [rd for rd in roads if rd["class"] in ("street", "lane")]
    lines = {rd["id"]: LineString(_xy(rd)) for rd in roads}

    def via_streets(A, p, B, q):
        """Walking distance from p on A to q on B along a street joining both."""
        best = 1e9
        for x in roads:
            if {A["id"], B["id"]} <= {x.get("connects"), x.get("connects_end")}:
                LX = lines[x["id"]]
                ja, jb = Point(LX.coords[0]), Point(LX.coords[-1])
                if ja.distance(lines[A["id"]]) > jb.distance(lines[A["id"]]):
                    ja, jb = jb, ja
                best = min(best, Point(p).distance(ja) + LX.length + Point(q).distance(jb))
        return best

    cands = []
    for _ in range(n * 40):   # shortcuts between neighbouring streets
        if len(streets) < 2 or len(cands) > n * 3:
            break
        A = streets[int(r.integers(len(streets)))]
        p, _, _ = _at(A, float(r.uniform(0.1, 0.9)))
        B = min((S for S in streets if S["id"] != A["id"]), key=lambda S: lines[S["id"]].distance(Point(p)))
        q = np.asarray(lines[B["id"]].interpolate(lines[B["id"]].project(Point(p))).coords[0])
        dd = float(np.linalg.norm(q - p))
        if 25 < dd < 120 and via_streets(A, p, B, q) > 2.5 * dd:
            cands.append((p, q, None))
    ctr = np.mean([_xy(S).mean(0) for S in streets], axis=0) if streets else np.zeros(2)
    # outbound trails from the village edge, each to a destination: a point on
    # another road or track, or the door of an outlying building (shed, farm)
    dests = [(rd["id"], q, None) for rd in roads for q in _xy(rd)[::5]]
    from synth.spec.access import door
    dests += [(None, door(b)[0], b["id"]) for b in blds if b.get("type") in ("shed", "coop", "warehouse", "greenhouse", "water_tower")]
    D = np.array([q for _, q, _ in dests])
    for _ in range(n * 10):
        if not streets:
            break
        S = streets[int(r.integers(len(streets)))]
        p, t, _ = _at(S, float(r.uniform(0.1, 0.9)))
        nn = np.array([-t[1], t[0]])
        if np.dot(nn, p - ctr) < 0:
            nn = -nn
        v = D - p
        dist = np.linalg.norm(v, axis=1)
        cosang = (v @ nn) / np.maximum(dist, 1e-6)
        ok = (dist > 60) & (dist < 250) & (cosang > 0.65) & np.array([rid != S["id"] for rid, _, _ in dests])
        ok &= np.all(np.abs(D) < site.half - 30, axis=1)
        if not ok.any():
            continue
        k = int(r.choice(np.nonzero(ok)[0]))
        rid, q, bid = dests[k]
        if rid is not None and lines[S["id"]].distance(Point(q)) < 0.5 * dist[k]:
            continue   # the road it reaches is right beside this one: no shortcut
        cands.append((p, np.asarray(q, float), bid))
    out = []
    for p, q, target in cands:
        if len(out) >= n:
            break
        raw = router.route(p, q, 0.25, extra_cost=extra)
        if raw is None or len(raw) < 10:
            continue
        xy = rplan.smooth_polyline(raw, 2.5)
        tan = np.gradient(xy, axis=0)
        nrm = np.column_stack([-tan[:, 1], tan[:, 0]]) / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9)
        wob = ndimage.gaussian_filter1d(r.normal(0, 0.8, len(xy)), 3)
        wob[:3] = wob[-3:] = 0
        xy = xy + nrm * wob[:, None]
        w = round(float(r.uniform(*width)), 2)
        xy = rplan.resample_polyline(ndimage.gaussian_filter1d(xy, 1.0, axis=0, mode="nearest"), 0.5)[0]
        others = [f for b, f in zip(blds, fps) if b["id"] != target]
        xy = _keep_clear(xy, others, w / 2 + 0.6)
        if target is not None:   # stop at the door step, 1.3 m from the wall
            tf = next(f for b, f in zip(blds, fps) if b["id"] == target)
            k = next((i for i, q in enumerate(xy) if tf.distance(Point(q)) < 1.3), len(xy))
            xy = xy[:k]
            if len(xy) < 10 or tf.distance(Point(xy[-1])) > 1.5:
                continue
            others = fps
        if _alongside(xy, roads) or any(LineString(xy).distance(LineString(o["centerline"])) < 4.0 for o in out):
            continue
        body = LineString(xy).buffer(w / 2 + 0.35)
        if any(body.intersects(f) for f in others):   # final guarantee: a path never touches a house (but its target's door)
            continue
        line = LineString(xy).simplify(0.05)
        pa = {"id": next_id(spec, "paths"), "surface": "dirt", "width": w,
              "centerline": [[round(x, 2), round(y, 2)] for x, y in line.coords]}
        if target is not None:
            pa["serves"] = [target]
        spec["entities"].setdefault("paths", []).append(pa)
        out.append(pa)
    return out


def dangling_paths(spec, tol=1.5):
    """Paths with an end that touches nothing: not on a road, another path or a
    building (within tol m). Must be empty."""
    rc = load_json("configs/roads.json")["classes"]
    bodies = [LineString(_xy(rd)).buffer(rd.get("width", rc.get(rd.get("class"), {}).get("width", 4.0)) / 2)
              for rd in spec["entities"].get("roads", [])]
    bodies += [footprint(b) for b in spec["entities"].get("buildings", [])]
    paths = spec["entities"].get("paths", [])
    bad = []
    for pa in paths:
        c = np.asarray(pa["centerline"])[:, :2]
        others = bodies + [LineString(np.asarray(o["centerline"])[:, :2]) for o in paths if o["id"] != pa["id"]]
        for e in (c[0], c[-1]):
            if not any(g.distance(Point(e)) <= tol for g in others):
                bad.append(pa["id"])
                break
    return bad


def crossings(spec, end_tol=1.5):
    """Pairs of roads that cross or touch without a junction between them
    (neither names the other in connects / connects_end). Must be empty."""
    roads = spec["entities"].get("roads", [])
    lines = {rd["id"]: LineString(_xy(rd)) for rd in roads}
    joined = {rd["id"]: {rd.get("connects"), rd.get("connects_end")} for rd in roads}
    bad = []
    for i, A in enumerate(roads):
        for B in roads[i + 1:]:
            if B["id"] in joined[A["id"]] or A["id"] in joined[B["id"]]:
                continue
            if lines[A["id"]].distance(lines[B["id"]]) >= end_tol:
                continue
            # two branches leaving the same road at the same point form a crossroads: allowed
            # if they touch only there (at both their starts)
            if A.get("connects") and A.get("connects") == B.get("connects"):
                x = lines[A["id"]].intersection(lines[B["id"]].buffer(end_tol))
                sa, sb = Point(_xy(A)[0]), Point(_xy(B)[0])
                if x.is_empty or (x.distance(sa) < 3 * end_tol and x.length < 4 * end_tol and sa.distance(sb) < 3 * end_tol):
                    continue
            bad.append((A["id"], B["id"]))
    return bad


def resolve_crossings(spec):
    """Turn every unjoined crossing into a junction. A street or lane that
    crosses another road is split there into two streets, both joined to it (a
    crossroads, as in any town); a dirt track that runs into a road ends at it.
    The road with the later id is the one changed."""
    n = 0
    for _ in range(60):
        bad = crossings(spec)
        if not bad:
            break
        roads = {rd["id"]: rd for rd in spec["entities"]["roads"]}
        a, b = bad[0]
        keep, cut = (roads[a], roads[b]) if a < b else (roads[b], roads[a])
        L, K = LineString(_xy(cut)), LineString(_xy(keep))
        x = L.intersection(K.buffer(1.5))
        if x.is_empty:
            break
        proj = sorted(L.project(Point(c)) for g in getattr(x, "geoms", [x]) for c in g.coords)
        s0, s1 = proj[0], proj[-1]
        on_k = lambda s_: list(K.interpolate(K.project(L.interpolate(s_))).coords[0])
        before = [c for c in L.coords if L.project(Point(c)) < s0]
        after = [c for c in L.coords if L.project(Point(c)) > s1]
        is_street = cut.get("class") in ("street", "lane")
        if is_street and len(after) >= 2 and L.length - s1 >= 25:
            rest = {"id": next_id(spec, "roads"), "class": cut["class"], "connects": keep["id"],
                    "centerline": [[round(v, 2) for v in on_k(s1)]] + [[round(px, 2), round(py, 2)] for px, py in after]}
            if cut.get("connects_end"):
                rest["connects_end"] = cut["connects_end"]
            if cut.get("end_cap"):
                rest["end_cap"] = cut.pop("end_cap")
            spec["entities"]["roads"].append(rest)
            # roads that ended on the far part of the split street now end on the new part
            for o in spec["entities"]["roads"]:
                if o is not rest and o.get("connects_end") == cut["id"]:
                    if LineString(_xy(o)).distance(LineString(rest["centerline"])) < LineString(_xy(o)).distance(LineString(before + [on_k(s0)] if before else [on_k(s0), on_k(s0)])):
                        o["connects_end"] = rest["id"]
        if s0 < 15 or len(before) < 1:
            if is_street and len(after) >= 2 and L.length - s1 >= 25:
                spec["entities"]["roads"].remove(cut)    # the part before the crossing is negligible
                for o in spec["entities"]["roads"]:
                    for k in ("connects", "connects_end"):
                        if o.get(k) == cut["id"]:
                            o[k] = rest["id"]
            else:
                spec["entities"]["roads"].remove(cut)
        else:
            cut["centerline"] = [[round(px, 2), round(py, 2)] for px, py in before] + [[round(v, 2) for v in on_k(s0)]]
            cut["connects_end"] = keep["id"]
            cut.pop("end_cap", None)
        n += 1
    return n


def unreachable(spec):
    """Roads that cannot be reached from the main road through junctions
    (connects / connects_end in either direction). Must be empty."""
    roads = spec["entities"].get("roads", [])
    if not roads:
        return []
    adj = {rd["id"]: set() for rd in roads}
    for rd in roads:
        for k in ("connects", "connects_end"):
            if rd.get(k) in adj:
                adj[rd["id"]].add(rd[k])
                adj[rd[k]].add(rd["id"])
    seen, todo = {roads[0]["id"]}, [roads[0]["id"]]
    while todo:
        for nb in adj[todo.pop()] - seen:
            seen.add(nb)
            todo.append(nb)
    return [rd["id"] for rd in roads if rd["id"] not in seen]


def enforce_grades(spec, site, log=print, rounds=8, cfg=None):
    """Compute every road's built profile exactly as the build does (heights
    pinned to the roads it joins). Roads over their grade limit are first
    re-routed between the same two junctions (a winding climb); what is still
    too steep is removed - except a road the network depends on that is within
    1.25x of its limit (kept and reported). Returns the removed ids."""
    from synth.spec.to_build import grade_violations, road_records
    rcfg = cfg or _cfg()
    if not hasattr(site.grid, "z_smooth"):
        site.grid.z_smooth = ndimage.gaussian_filter(site.grid.z, 4.0)
    removed, rerouted, router = [], set(), None

    def violations():
        tidy(spec, log=lambda *a: None)
        roads = spec["entities"]["roads"]
        return {k: v for k, v in grade_violations(road_records(roads, site.grid, rcfg)).items() if k != roads[0]["id"]}

    def cuts_off(rid):
        trial = dict(spec, entities=dict(spec["entities"], roads=[rd for rd in spec["entities"]["roads"] if rd["id"] != rid]))
        return bool(unreachable(trial))

    for _ in range(rounds):
        bad = violations()
        if not bad:
            break
        retry = [k for k in bad if k not in rerouted]
        if retry:   # re-route between the same two ends under the grade limit (a winding climb)
            router = router or rplan.Router(site.grid, rcfg["routing_res_m"])
            by = {rd["id"]: rd for rd in spec["entities"]["roads"]}
            for k in retry:
                rd = by[k]
                xy = _xy(rd)
                c = rcfg["classes"][rd["class"]]
                raw = router.route(xy[0], xy[-1], rd.get("max_grade", c["max_grade"]) * 0.6)
                if raw is not None and len(raw) > 3:
                    new = rplan.smooth_polyline(np.vstack([xy[0], raw[1:-1], xy[-1]]), c["smooth_m"], keep_start=True, keep_end=True)
                    rd["centerline"] = [[round(px, 2), round(py, 2)] for px, py in LineString(new).simplify(0.25).coords]
                rerouted.add(k)
            log(f"too steep once built: {bad}; re-routed {retry}")
            if crossings(spec):
                resolve_crossings(spec)
            bad = violations()
            if not bad:
                break
        roads = spec["entities"]["roads"]
        has_branches = {rd.get(k) for rd in roads for k in ("connects", "connects_end")}
        lim = {rd["id"]: rd.get("max_grade", rcfg["classes"][rd["class"]]["max_grade"]) for rd in roads}
        keep = {k for k in bad if (k in has_branches or cuts_off(k)) and bad[k] <= 1.25 * lim[k]}
        if keep:
            log(f"kept (network depends on them) although steep: {sorted(keep)}")
            spec.setdefault("_warnings", []).append({"steep_roads_kept": sorted(keep)})
        drop = set(bad) - keep
        if not drop:
            break
        spec["entities"]["roads"] = [rd for rd in roads if rd["id"] not in drop]
        tidy(spec, log=lambda *a: None)
        removed += sorted(drop)
        log(f"removed (too steep): {sorted(drop)}")
    return removed


def connect_islands(spec, site, field, cls, cfg, log=print, max_gap=140.0, router=None, tries=40):
    """Join road groups that are not connected to the main road's network: a
    connector street from a point of the group to a nearby connected road -
    straight, diagonal, or (on steep ground) a serpentine. A connector is kept
    only if its built profile (heights pinned to both roads) is within 1.2x of
    its grade limit; otherwise the next-closest pair is tried."""
    from synth.spec.to_build import max_grade_10m, road_records
    maxg = cfg["classes"][cls]["max_grade"]
    added, tried = 0, set()
    for _ in range(tries):
        cut = set(unreachable(spec))
        if not cut:
            break
        roads = spec["entities"]["roads"]
        RL = {rd["id"]: LineString(_xy(rd)) for rd in roads if rd["id"] not in cut}
        cands = []
        for rd in roads:
            if rd["id"] not in cut:
                continue
            for k, p in enumerate(_xy(rd)[::4]):
                for rid, L in RL.items():
                    d = L.distance(Point(p))
                    if 12 < d < max_gap and (rd["id"], k, rid) not in tried:
                        cands.append((d, rd, k, p, rid))
        if not cands:
            break
        cands.sort(key=lambda c: c[0])
        d, rd, k, p, rid = cands[0]
        tried.add((rd["id"], k, rid))
        q = np.asarray(RL[rid].interpolate(RL[rid].project(Point(p))).coords[0])
        dirv = (q - p) / d
        options = []
        for deg in (0, 20, -20, 35, -35, 50, -50):
            xy, hit = trace(field, spec, p, _rot(dirv, math.radians(deg)), d * 2.4, follow=0.0, ignore=(rd["id"],), min_before_hit=4.0)
            if hit is not None and hit not in cut and _grade(field, xy, 10.0) <= maxg:
                options.append((xy, hit))
                break
        if router is not None:
            xy = _switchback(router, spec, p, q, rd["id"], cls, cfg)
            if xy is not None:
                options.append((xy, rid))
        for xy, hit in options:
            new = _entity(spec, cls, xy, connects=rd["id"], connects_end=hit)
            rec = [x for x in road_records(spec["entities"]["roads"], site.grid, cfg) if x["id"] == new["id"]][0]
            if max_grade_10m(rec["centerline"]) <= 1.2 * maxg:
                added += 1
                break
            spec["entities"]["roads"].remove(new)
    if added:
        log(f"joined isolated streets with {added} connector(s)")
    return added


def _end_trim(L):
    return min(8.0, L / 4)


def fair_fixed(xy, R, step=1.0, iters=6000, window=10):
    """Relax only the vertices where the curve (measured over `window` steps, like
    min_radius) bends tighter than R; the two end points (junctions) never move."""
    p = rplan.resample_polyline(np.asarray(xy, float), step)[0]
    if len(p) < 2 * window + 3:
        return p
    for it in range(iters):
        a, b, c = p[:-2 * window], p[window:-window], p[2 * window:]
        d1, d2 = b - a, c - b
        ang = np.abs(np.arctan2(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], (d1 * d2).sum(1)))
        bad = step * window / np.maximum(ang, 1e-9) < R
        if not bad.any():
            break
        m = np.zeros(len(p), bool)
        m[window:-window] = bad
        m = ndimage.binary_dilation(m, iterations=window)   # reaches into the end zones too
        m[0] = m[-1] = False                                  # only the junction points stay fixed
        q = p.copy()
        q[1:-1][m[1:-1]] = 0.5 * p[1:-1][m[1:-1]] + 0.25 * (p[:-2][m[1:-1]] + p[2:][m[1:-1]])
        p = q
        if it % 40 == 39:
            p = rplan.resample_polyline(p, step)[0]
    return rplan.resample_polyline(p, step)[0]


def smooth_roads(spec, cfg, step=2.0, site=None):
    """Every road smooth and continuous with differentiable curves: resampled
    densely and smoothed (junction ends fixed) until its tightest curve is at
    least its class `min_radius_m`; stored at `step` metre spacing so the curve
    is kept (no straight-segment simplification, hence no kinks)."""
    def store(rd, xy):
        out = rplan.resample_polyline(np.asarray(xy, float), step)[0]
        rd["centerline"] = [[round(float(x), 2), round(float(y), 2)] for x, y in out]

    def violating():
        if site is None:
            return set()
        from synth.spec.to_build import grade_violations, road_records
        return set(grade_violations(road_records(spec["entities"]["roads"], site.grid, cfg)))

    before = violating()
    orig = {}
    for rd in spec["entities"]["roads"]:
        xy = rplan.resample_polyline(_xy(rd), 1.0)[0]
        orig[rd["id"]] = xy
        L = LineString(xy).length
        R = rd.get("min_radius_m", cfg["classes"][rd["class"]].get("min_radius_m", 12.0))
        if L >= 20 and rd["class"] != "main":   # the main road is already faired
            xy = rplan.smooth_polyline(xy, 3.0, keep_start=True, keep_end=True)   # remove small wiggles
            xy = fair_fixed(xy, R)                                               # then open every tight bend
        store(rd, xy)
    # a road that smoothing made too steep climbs in hairpins: gentler smoothing, tighter radius allowed
    for rid in violating() - before:
        rd = next(x for x in spec["entities"]["roads"] if x["id"] == rid)
        if rd["class"] == "main":
            continue
        R = 0.6 * cfg["classes"][rd["class"]].get("min_radius_m", 12.0)
        rd["min_radius_m"] = R
        store(rd, fair_fixed(rplan.smooth_polyline(orig[rid], 2.0, keep_start=True, keep_end=True), R))


def final_steep_cleanup(spec, site, cfg, log=print, factor=1.25):
    """After the last smoothing: a road far over its grade limit (> factor x) is
    removed if the network does not need it; if it does, its hairpins may be
    tighter (0.45x, then 0.3x the class radius, recorded as its min_radius_m)
    until the grade is within factor x its limit."""
    from synth.spec.to_build import grade_violations, road_records
    roads = spec["entities"]["roads"]
    bad = grade_violations(road_records(roads, site.grid, cfg))
    by = {rd["id"]: rd for rd in roads}
    for rid, g in bad.items():
        rd = by.get(rid)
        if rd is None or rd["class"] == "main":
            continue
        lim = rd.get("max_grade", cfg["classes"][rd["class"]]["max_grade"])
        if g <= factor * lim:
            continue
        trial = dict(spec, entities=dict(spec["entities"], roads=[x for x in spec["entities"]["roads"] if x["id"] != rid]))
        if not unreachable(trial):
            spec["entities"]["roads"] = [x for x in spec["entities"]["roads"] if x["id"] != rid]
            tidy(spec, log=lambda *a: None)
            log(f"removed {rid} ({g:.0%} > {lim:.0%}): not needed by the network")
            continue
        xy0 = _xy(rd)
        shapes = [(f, xy0) for f in (0.45, 0.3)]
        raw = rplan.Router(site.grid, 2.0).route(xy0[0], xy0[-1], lim * 0.55)   # last try: a fresh switchback route
        if raw is not None and len(raw) > 3:
            shapes.append((0.3, np.vstack([xy0[0], raw[1:-1], xy0[-1]])))
        for f, base in shapes:
            R = f * cfg["classes"][rd["class"]].get("min_radius_m", 12.0)
            rd["centerline"] = [[round(float(x), 2), round(float(y), 2)]
                                for x, y in rplan.resample_polyline(fair_fixed(base, R), 2.0)[0]]
            rd["min_radius_m"] = R
            g2 = grade_violations(road_records(spec["entities"]["roads"], site.grid, cfg)).get(rid, 0.0)
            if g2 <= factor * lim:
                log(f"{rid}: tight hairpins (radius {R:.1f} m) keep it at {g2:.0%}")
                break
        else:
            log(f"{rid}: still {g2:.0%} (limit {lim:.0%}); kept because the network depends on it")


def radius_violations(spec, cfg, tol=0.9):
    """{road id: tightest curve radius} for roads tighter than tol x their class minimum."""
    out = {}
    for rd in spec["entities"]["roads"]:
        L = LineString(_xy(rd)).length
        if L < 20:
            continue
        R = rd.get("min_radius_m", cfg["classes"][rd["class"]].get("min_radius_m", 12.0))   # hairpin roads carry their own
        rmin = min_radius(_xy(rd), trim_m=60.0 if rd["class"] == "main" else 0.0)   # main: ends at the site edge
        if rmin < tol * R:
            out[rd["id"]] = round(rmin, 1)
    return out


def tidy(spec, log=print, min_len=15.0):
    """Repair junction references after edits: drop references to roads that no
    longer exist or to the road itself, remove leftover pieces shorter than
    min_len, and break dependency cycles (the build lays parents before the
    roads joined to them) by freeing an end junction - or, failing that, a start."""
    def clean():
        roads = spec["entities"]["roads"]
        ids = {rd["id"] for rd in roads}
        for rd in roads:
            for k in ("connects", "connects_end"):
                if rd.get(k) is not None and (rd[k] not in ids or rd[k] == rd["id"]):
                    rd.pop(k)
    clean()
    roads = spec["entities"]["roads"]
    tiny = {rd["id"] for rd in roads[1:] if LineString(_xy(rd)).length < min_len}
    if tiny:
        spec["entities"]["roads"] = [rd for rd in roads if rd["id"] not in tiny]
        clean()
        log(f"removed {len(tiny)} leftover piece(s) shorter than {min_len:.0f} m")
    for _ in range(200):   # order like the build; where it gets stuck, free one junction
        done, todo = set(), list(spec["entities"]["roads"])
        while todo:
            ready = [rd for rd in todo if all(not rd.get(k) or rd[k] in done for k in ("connects", "connects_end"))]
            if not ready:
                break
            for rd in ready:
                done.add(rd["id"])
                todo.remove(rd)
        if not todo:
            return
        stuck = todo[0]
        stuck.pop("connects_end" if stuck.get("connects_end") else "connects")


def drop_unreachable(spec, log=print):
    """Last resort: remove roads still not joined to the main road's network."""
    removed = []
    cut_off = unreachable(spec)
    if cut_off:
        for rd in spec["entities"]["roads"]:
            for k in ("connects", "connects_end"):
                if rd.get(k) in cut_off:
                    rd.pop(k)
        spec["entities"]["roads"] = [rd for rd in spec["entities"]["roads"] if rd["id"] not in cut_off]
        removed += cut_off
        log(f"no longer connected to the network, removed {cut_off}")
    return removed
