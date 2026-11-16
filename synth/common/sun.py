"""Solar position (NOAA General Solar Position algorithm, ~0.1 deg accuracy)."""
import math
from datetime import datetime, timedelta, timezone


def solar_position(lat_deg, lon_deg, when_utc: datetime):
    """Returns (elevation_deg, azimuth_deg) with azimuth clockwise from north."""
    doy = when_utc.timetuple().tm_yday
    hour = when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * lon_deg
    ha = math.radians(tst / 4 - 180)
    lat = math.radians(lat_deg)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(lat) - math.tan(decl) * math.cos(lat))) + 180
    return 90 - math.degrees(zen), az % 360


def sample_sun(r, lat, lon, cfg, year=2025):
    m0, d0 = map(int, cfg["date_range"][0].split("-"))
    m1, d1 = map(int, cfg["date_range"][1].split("-"))
    start = datetime(year, m0, d0)
    end = datetime(year, m1, d1)
    day = start + timedelta(days=int(r.integers(0, (end - start).days + 1)))
    local_h = float(r.uniform(*cfg["local_hours"]))
    local = day + timedelta(hours=local_h)
    utc = (local - timedelta(hours=cfg["utc_offset_h"])).replace(tzinfo=timezone.utc)
    el, az = solar_position(lat, lon, utc)
    return {"datetime_local": local.isoformat(), "datetime_utc": utc.isoformat(), "utc_offset_h": cfg["utc_offset_h"],
            "elevation_deg": el, "azimuth_deg": az}
