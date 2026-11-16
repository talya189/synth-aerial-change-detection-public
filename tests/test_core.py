"""Fast checks that need no Blender:  ~/synth_aerial_work/venv/bin/python -m pytest tests -q"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.common import camera_math as cm  # noqa: E402
from synth.common.rng import rng  # noqa: E402
from synth.common.sun import solar_position  # noqa: E402


def test_gsd_altitude_roundtrip():
    for gsd in (0.1, 0.25, 0.5, 1.0):
        h = cm.altitude_for_gsd(gsd, 50.0, 36.0, 2048)
        across, along = cm.gsd_at_centre(h, 50.0, 36.0, 2048)
        assert math.isclose(across, gsd, rel_tol=1e-9) and math.isclose(along, gsd, rel_tol=1e-9)
    assert round(cm.altitude_for_gsd(0.25, 50.0, 36.0, 2048)) == 711


def test_oblique_gsd_geometry():
    h = 500.0
    across, along = cm.gsd_at_centre(h, 50.0, 36.0, 2048, tilt_deg=45)
    assert math.isclose(along / across, 1 / math.cos(math.radians(45)), rel_tol=1e-9)


def test_camera_pose_looks_at_target():
    tgt = np.array([10.0, -5.0, 100.0])
    pos, eul = cm.camera_pose(tgt, 300.0, 40.0, 120.0)
    t, h = math.radians(40), math.radians(120)
    look = (tgt - pos) / np.linalg.norm(tgt - pos)
    # Blender camera looks along local -Z: rotate (0,0,-1) by X=tilt then Z=-heading
    v = np.array([0.0, math.sin(t), -math.cos(t)])
    c, s = math.cos(-h), math.sin(-h)
    v = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
    assert np.allclose(look, v, atol=1e-9)


def test_rng_streams_independent_and_stable():
    a1 = rng(42, "building", "b001").random(3)
    a2 = rng(42, "building", "b001").random(3)
    b = rng(42, "building", "b002").random(3)
    assert np.array_equal(a1, a2) and not np.array_equal(a1, b)


def test_solar_position_noon_summer_israel():
    from datetime import datetime, timezone
    el, az = solar_position(32.95, 35.66, datetime(2025, 6, 21, 9, 40, tzinfo=timezone.utc))  # ~local solar noon
    assert 79 < el < 81.5 and 150 < az < 210


def test_planner_deterministic(tmp_path):
    """Same seed -> identical plan.json (the basis of reproducible scenes)."""
    from synth.planner.plan import plan_scene
    p1 = plan_scene(7, "configs/scene/galilee_village.json", out_dir=str(tmp_path / "a"), log=lambda *a: None)
    p2 = plan_scene(7, "configs/scene/galilee_village.json", out_dir=str(tmp_path / "b"), log=lambda *a: None)
    j1, j2 = json.load(open(p1)), json.load(open(p2))
    for k in ("roads", "buildings", "vegetation", "rocks", "cameras", "sun"):
        assert j1[k] == j2[k], k


def test_mask_png_roundtrip(tmp_path):
    """Regression: OIIO's write_image scrambles 2-D arrays; masks must round-trip exactly."""
    import pytest
    oiio = pytest.importorskip("OpenImageIO")
    from PIL import Image
    from synth.post.annotations import write_png
    a = np.zeros((40, 64), np.uint8)
    a[0, 63], a[39, 0], a[10:20, 30:35] = 255, 128, 7
    write_png(str(tmp_path / "s.png"), a, np.uint8)
    assert np.array_equal(np.asarray(Image.open(tmp_path / "s.png")), a)
    b = (np.arange(40 * 64).reshape(40, 64) % 3000).astype(np.uint16)
    write_png(str(tmp_path / "i.png"), b, np.uint16)
    assert np.array_equal(np.asarray(Image.open(tmp_path / "i.png")), b)


def test_colour_calibration_step():
    """update_adjust moves a layer's HSV correction toward its target colour."""
    from synth.steps.step3 import update_adjust
    adj = update_adjust(None, [100, 90, 80], [100, 90, 80])
    assert adj == {"hue": 0.5, "saturation": 1.0, "value": 1.0}
    adj = update_adjust(None, [60, 50, 30], [170, 150, 110])
    assert adj["value"] > 2 and abs(adj["hue"] - 0.5) < 0.05


def test_building_spec_validation():
    import pytest
    from synth.spec.scene_spec import SpecError, load, validate
    spec = load("configs/specs/step4_catalogue.json")
    validate(spec)
    bad = dict(spec["entities"]["buildings"][0], roof="thatched")
    spec["entities"]["buildings"][0] = bad
    with pytest.raises(SpecError):
        validate(spec)


def test_building_footprint_and_completion():
    from synth.spec.buildings import complete, footprint
    b = {"id": "bld_001", "type": "house", "center": [0, 0], "width": 10, "depth": 8, "floors": 2,
         "annex": {"side": "back", "width": 4, "depth": 3, "floors": 1}}
    assert abs(footprint(b).area - (80 + 12)) < 1e-2 and footprint(b).geom_type == "Polygon"
    c1, c2 = complete(b, 7), complete(b, 7)
    assert c1 == c2 and c1["roof"] in ("flat", "tiled") and "annex" in c1


def test_placement_rules_and_growth():
    import numpy as np
    from synth.spec.placement import Site, check_building, grow_village
    n = 401
    y = np.arange(n)[:, None] * np.ones((1, n))
    z = np.where(y > 300, (y - 300) * 0.6, 0.0)          # flat south, steep (31 deg) north strip
    site = Site(z, 200.0)
    spec = {"entities": {"buildings": [], "roads": [{"id": "road_001", "width": 6, "centerline": [[-200, -60], [200, -60]]}]}}
    ok = {"id": "bld_001", "type": "house", "center": [0, 0], "width": 10, "depth": 10, "floors": 1}
    assert check_building(ok, spec, site)[0] == []
    assert any("slope" in p for p in check_building(dict(ok, center=[0, 140]), spec, site)[0])
    assert check_building(dict(ok, center=[0, -60]), spec, site)[0] == ["on a road or path"]
    spec["entities"]["buildings"].append(ok)
    assert any("from bld_001" in p for p in check_building(dict(ok, id="bld_002", center=[12, 0]), spec, site)[0])
    a1, _ = grow_village({"entities": {"buildings": [], "roads": []}}, site, 8, {"house": 1.0}, 3, centre=[0, 0])
    a2, _ = grow_village({"entities": {"buildings": [], "roads": []}}, site, 8, {"house": 1.0}, 3, centre=[0, 0])
    assert len(a1) == 8 and a1 == a2


def test_presets_resolve():
    from synth.spec.scene_spec import load, resolved
    spec = load("configs/specs/step4_catalogue.json")
    assert spec["terrain_material"] == "galilee_summer"
    r = resolved(spec)
    assert r["terrain_material"]["base"]["role"] == "terrain.soil" and "basalt" in r["scatter"]
    assert spec["terrain_material"] == "galilee_summer"   # the original is untouched


def test_building_types_and_options():
    from synth.spec.buildings import catalogue, complete
    cat = catalogue()
    assert len(cat["types"]) >= 12
    for name, t in cat["types"].items():
        assert set(t["roof_p"]) <= set(cat["roofs"]) and set(t["wall_p"]) <= set(cat["walls"]), name
        floors = int(next(iter(t["floors_p"])))
        b = complete({"id": "bld_001", "type": name, "center": [0, 0], "width": 10, "depth": 10, "floors": floors}, 1)
        for k in ("roof", "wall", "top_setback", "pv", "railing", "pergola"):
            assert k in b, (name, k)
        if b["roof"] != "flat":
            assert b["top_setback"] == 0.0 and not b["railing"]


def test_settlement_styles_grow():
    import numpy as np
    from synth.spec.buildings import catalogue
    from synth.spec.placement import Site, grow_village
    site = Site(np.zeros((501, 501)), 250.0)
    for style in ("moshav", "arab_village", "kibbutz", "old_town"):
        spec = {"entities": {"buildings": [], "roads": []}}
        added, _ = grow_village(spec, site, 12, style, 2, centre=[0, 0])
        assert len(added) == 12, style
        allowed = set(catalogue()["settlements"][style]["types"])
        assert {b["type"] for b in added} <= allowed


def test_podium_footprint_and_validation():
    import pytest
    from synth.spec.buildings import footprint
    from synth.spec.scene_spec import SpecError, _check_building
    b = {"id": "bld_001", "type": "tower", "center": [0, 0], "width": 20, "depth": 16, "floors": 14,
         "podium": {"floors": 2, "margin": 3.0}}
    assert abs(footprint(b).area - 26 * 22) < 1e-6
    _check_building(b)
    with pytest.raises(SpecError):
        _check_building(dict(b, podium={"floors": 14, "margin": 3.0}))


def test_coops_come_in_groups():
    import numpy as np
    from synth.spec.placement import Site, grow_village
    site = Site(np.zeros((601, 601)), 300.0)
    spec = {"entities": {"buildings": [{"id": "bld_001", "type": "house", "center": [0, 0], "width": 10, "depth": 10, "floors": 1}],
                         "roads": []}}
    added, _ = grow_village(spec, site, 3, {"coop": 1.0}, 4, centre=[0, 0])
    assert len(added) >= 2
    assert added[0]["yaw_deg"] == added[1]["yaw_deg"]   # the first group is parallel
    assert added[0]["width"] == added[1]["width"]


def test_channels_follow_valleys_not_flat_ground():
    import numpy as np
    from synth.spec.to_build import terrain_wetness
    n = 401
    x = np.arange(n) - n // 2
    y = np.arange(n)[:, None] * np.ones((1, n))
    valley = np.abs(x)[None, :] * 0.15 + y * 0.05          # V-shaped valley draining toward row 0
    w = terrain_wetness(valley)
    assert w[60, n // 2] > 0.5 and w[60, n // 2 + 80] < 0.1
    assert terrain_wetness(np.zeros((n, n))).max() < 0.2     # flat ground is not a channel


def test_vegetation_switch():
    from synth.spec.scene_spec import load, resolved
    spec = load("configs/specs/step4_village_moshav.json")
    spec["vegetation"] = False
    r = resolved(spec)
    assert r["scatter"] and all(not v["kind"].startswith("veg.") for v in r["scatter"].values())   # rocks stay
    assert all(not L.get("vegetation") for L in r["terrain_material"]["layers"])


def test_road_validation():
    import pytest
    from synth.spec.scene_spec import SpecError, _check_roads
    ok = {"entities": {"roads": [{"id": "road_001", "class": "main", "centerline": [[0, 0], [100, 0]]},
                                 {"id": "road_002", "class": "lane", "centerline": [[50, 0], [50, 80]], "connects": "road_001"}]}}
    _check_roads(ok)
    for bad in ({"id": "road_003", "class": "highway", "centerline": [[0, 0], [1, 1]]},
                {"id": "road_003", "class": "street", "centerline": [[0, 0], [1, 1]], "connects": "road_099"}):
        with pytest.raises(SpecError):
            _check_roads({"entities": {"roads": ok["entities"]["roads"] + [bad]}})


def test_network_and_frontage():
    import math
    import numpy as np
    from shapely.geometry import LineString
    from synth.spec.network import plan_network
    from synth.spec.placement import Site, grow_along_streets
    n = 501
    y = np.arange(n)[:, None] * np.ones((1, n))
    site = Site(y * 0.03, 250.0)   # gentle 3 % slope
    spec = {"entities": {"roads": [], "paths": [], "buildings": []}}
    net = plan_network(spec, site, "moshav", 3, [0.0, 0.0], log=lambda *a: None)
    roads = spec["entities"]["roads"]
    from synth.spec.network import crossings, unreachable
    assert roads[0]["class"] == "main" and len(net["streets"]) >= 2
    assert unreachable(spec) == [] and crossings(spec) == []     # one connected network, junctions only
    added, _ = grow_along_streets(spec, site, 10, "moshav", 3, [0.0, 0.0])
    assert len(added) >= 6
    from shapely.geometry import Point
    for b in added:   # the front faces a road within frontage distance (corner plots: not always the nearest)
        yaw = math.radians(b["yaw_deg"])
        front = np.array([math.sin(yaw), -math.cos(yaw)])
        c = np.asarray(b["center"])
        faces = False
        for r in roads:
            L = LineString(r["centerline"])
            p = np.asarray(L.interpolate(L.project(Point(c))).coords[0])
            d = np.linalg.norm(p - c)
            if d < b["depth"] / 2 + 15 and np.dot(front, (p - c) / d) > 0.8:
                faces = True
        assert faces, b["id"]


def test_network_rules_on_real_site():
    """Generated Step 5 specs: no road crosses another without a junction,
    every road is reachable from the main road, the main road is smooth."""
    from synth.spec.network import crossings, min_radius, unreachable
    from synth.spec.scene_spec import load
    for style in ("moshav", "arab_village", "town"):
        spec = load(f"configs/specs/step5_{style}.json")
        roads = spec["entities"]["roads"]
        assert crossings(spec) == [], style
        assert unreachable(spec) == [], style
        assert roads[0]["class"] == "main" and min_radius(roads[0]["centerline"], trim_m=60) >= 45, style
        # built profiles (heights pinned to the joined roads) stay within every class limit
        from synth.common.config import load_json
        from synth.spec.to_build import grade_violations, road_records
        from synth.steps.step5 import site_for
        site = site_for(spec)
        assert grade_violations(road_records(roads, site.grid, load_json("configs/roads.json"))) == {}, style


def test_fair_reaches_radius():
    import numpy as np
    from synth.spec.network import fair, min_radius
    t = np.linspace(0, 600, 301)
    zig = np.column_stack([t, 6 * np.sin(t / 9.0)])          # wiggly line, radius ~13 m
    assert min_radius(zig) < 20 and min_radius(fair(zig, 80), trim_m=60) >= 80


def test_planting_rules():
    """Step 6 specs: trees never on buildings, roads or paths; grove trees inside their grove."""
    from shapely.geometry import Point, Polygon
    from synth.spec.scene_spec import load
    from synth.spec.vegetation import _blocked, grove_trees
    for style in ("moshav", "arab_village", "town"):
        spec = load(f"configs/specs/step6_{style}.json")
        trees, groves = spec["entities"]["trees"], spec["entities"]["groves"]
        assert trees and groves, style
        groves_only = dict(spec, entities=dict(spec["entities"], groves=[]))
        blocked = _blocked(groves_only).buffer(-0.5)
        assert not [t["id"] for t in trees if blocked.contains(Point(t["x"], t["y"]))], style
        for g in groves:
            poly = Polygon(g["polygon"])
            pts = grove_trees(g, spec["seed"])
            assert pts and all(poly.contains(Point(p)) for p in pts), g["id"]


def test_instance_ids_stable_per_entity():
    from synth.blender.labels import IdRegistry
    from synth.common.config import load_json
    ids = IdRegistry(load_json("configs/classes.json"))
    a, b = ids.new("veg.grove.olive", "grove_001"), ids.new("veg.grove.olive", "grove_001")
    c = ids.new("veg.tree.olive", "tree_001")
    assert a == b != c


def test_place_files_complete():
    """Every style of every place defines all parts of the logic."""
    from synth.places import load_place, places, styles
    assert {"arab_village", "arab_city"} <= set(places())
    for p in places():
        for s in styles(p):
            st = load_place(p)["styles"][s]
            for key in ("terrain", "roads", "buildings", "footpaths", "vegetation", "fields", "examples"):
                assert key in st, (p, s, key)


def test_place_scene_obeys_rules():
    """One generated scene (place + style + seed) passes every rule check."""
    from synth.make_scene import checks
    from synth.places import _site, make_spec
    spec = make_spec("arab_village", "hilltop", 2, log=lambda *a: None)
    c = checks(spec, _site(spec["region"])[0])
    assert c["counts"]["buildings"] > 80 and c["building_types"].get("mosque") == 1
    assert c["crossings_without_junction"] == [] and c["unreachable_roads"] == []
    assert c["roads_too_steep"] == {} and c["buildings_breaking_rules"] == []
    assert c["roads_tighter_than_class_radius"] == {}                          # smooth roads, no kinks
    assert c["buildings_without_access"] == [] and c["access_paths_through_buildings"] == []
    assert c["counts"]["walls"] > 30 and len(c["wall_types"]) >= 3                # walled yards, varied
    assert c["walls_through_buildings"] == [] and c["walls_across_roads_paths"] == [] and c["walled_in_buildings"] == []
    assert spec == make_spec("arab_village", "hilltop", 2, log=lambda *a: None)   # same seed, same scene


def test_wall_report_catches_blocking_walls():
    from synth.spec.yards import wall_report
    house = {"id": "bld_001", "center": [0, 0], "width": 8, "depth": 8, "yaw_deg": 0, "floors": 2, "access": {"road": "road_001"}}
    road = {"id": "road_001", "class": "street", "centerline": [[-30, -20], [30, -20]]}
    wall = lambda segs: {"id": "wall_001", "type": "block", "thickness": 0.2, "height": 1.8, "segments": segs}
    spec = {"entities": {"roads": [road], "paths": [], "buildings": [house], "walls": [wall([[[-7, -7], [-2, -7]], [[2, -7], [7, -7]]])]}}
    assert wall_report(spec) == {"walls_through_buildings": [], "walls_across_roads_paths": [], "walled_in_buildings": []}
    spec["entities"]["walls"] = [wall([[[-7, -7], [7, -7]]])]        # no gate in front of the door
    assert wall_report(spec)["walled_in_buildings"] == ["bld_001"]
    spec["entities"]["walls"] = [wall([[[-7, -20], [7, -20]]])]      # on the road
    assert wall_report(spec)["walls_across_roads_paths"] == ["wall_001"]


def test_fair_fixed_keeps_ends_and_opens_bends():
    import numpy as np
    from synth.spec.network import fair_fixed, min_radius
    t = np.linspace(0, 1, 200)
    zig = np.column_stack([t * 120, 8 * np.sign(np.sin(t * 25))])   # sharp corners
    out = fair_fixed(zig, 15.0)
    assert np.allclose(out[0], zig[0]) and np.allclose(out[-1], zig[-1])
    assert min_radius(out) >= 0.9 * 15.0


def test_access_report_follows_path_chains():
    from synth.spec.access import access_report
    spec = {"entities": {"roads": [{"id": "road_001"}],
                         "paths": [{"id": "path_001", "road_join": "road_001"}, {"id": "path_002", "path_join": "path_001"}],
                         "buildings": [{"id": "bld_001", "access": {"road": "road_001"}}, {"id": "bld_002", "access": {"path": "path_002"}},
                                       {"id": "bld_003", "access": {"path": "path_009"}}, {"id": "bld_004"}]}}
    assert access_report(spec) == ["bld_003", "bld_004"]
