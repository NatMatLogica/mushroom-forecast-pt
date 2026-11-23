"""Offline tests for the lagged (weeks-long window) porcini scoring. No network."""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mushroom import scoring as S  # noqa: E402
from mushroom.species_db import SPECIES  # noqa: E402

TODAY = datetime.date(2026, 11, 17)
BE = next(s for s in SPECIES if s.name == "Boletus edulis")


def weather(tmean, rain, past=31, future=7, rh=90.0):
    n = past + 1 + future
    dates = [TODAY + datetime.timedelta(days=k - past) for k in range(n)]
    tm = [tmean(k - past) if callable(tmean) else tmean for k in range(n)]
    pr = [rain(k - past) if callable(rain) else rain for k in range(n)]
    return S.SiteWeather(ok=True, dates=dates, precip=pr, precip_grid=pr,
                         tmin=[t - 4 for t in tm], tmax=[t + 4 for t in tm], rh_mean=[rh] * n)


def test_edulis_uses_lagged_windows():
    assert BE.lagged is not None and BE.lagged.temp_days == 20 and BE.lagged.rain_days == 26


def test_in_band_and_wet_scores_full():
    s = S.score_species(BE, weather(13.0, 3.0), TODAY, 11)
    assert s.score == 100 and s.latency_status == "now"


def test_dry_month_scores_no_rain_points():
    s = S.score_species(BE, weather(13.0, 0.0), TODAY, 11)
    assert s.score == 55 and s.latency_status == "none"   # 45 temperature + 10 humidity


def test_too_warm_gets_no_temperature_points():
    s = S.score_species(BE, weather(21.0, 3.0), TODAY, 11)
    assert s.score == 55 and "too warm" in s.recommendation


def test_one_storm_is_spread_over_26_days():
    # 52 mm on one day = 2 mm/day over 26 days: exactly the full-rain edge
    s = S.score_species(BE, weather(13.0, lambda off: 52.0 if off == -3 else 0.0), TODAY, 11)
    assert s.score == 100


def test_cooling_forecast_gives_a_future_window():
    # 20 °C now (no temperature points), cooling 1.5 °C/day from tomorrow; wet throughout
    s = S.score_species(BE, weather(lambda off: 20.0 if off <= 0 else 20.0 - 1.5 * off, 3.0), TODAY, 11)
    assert s.latency_status in ("soon", "not_yet") and 0 < s.days_until_window <= 7


def test_short_history_scores_nothing():
    s = S.score_species(BE, weather(13.0, 3.0, past=10), TODAY, 11)
    assert s.score == 0 and "history" in s.recommendation


def test_event_species_ignore_rain_older_than_lookback():
    other = next(s for s in SPECIES if s.lagged is None and 11 in s.season)
    w = weather(15.0, lambda off: 100.0 if off == -20 else 0.0)
    s = S.score_species(other, w, TODAY, 11)
    assert not s.rain_met


def test_summer_cep_is_an_oak_bolete_in_season_now():
    from mushroom.species_db import CORK_OAK
    sc = next(s for s in SPECIES if s.name == "Boletus reticulatus")
    assert CORK_OAK in sc.habitats and 11 in sc.season and 7 in sc.season and sc.lagged is None


# ── landcover.py (offline parts) ──

def test_cos_labels_map_to_habitats():
    from mushroom import landcover as L
    from mushroom.species_db import CORK_OAK, PINE, EUCALYPTUS
    assert L.habitats_for_label("Florestas de sobreiro") == [CORK_OAK]
    assert L.habitats_for_label("Superfícies silvopastoris de sobreiro") == [CORK_OAK]
    assert L.habitats_for_label("Florestas de pinheiro manso") == [PINE]
    assert L.habitats_for_label("Florestas de eucalipto") == [EUCALYPTUS]
    assert L.habitats_for_label("Arrozais") == [] and L.habitats_for_label("Oceano") == []


def test_grid_stays_inside_radius():
    from mushroom import landcover as L
    from mushroom.sites import haversine_km
    pts = L.grid_points(38.24, -8.67, 10.0, 2.5)
    assert len(pts) > 40 and all(haversine_km(38.24, -8.67, a, b) <= 10.3 for a, b in pts)


def test_pick_patches_largest_first_and_spaced():
    from mushroom import landcover as L
    polys = {1: {"fid": 1, "ha": 500, "habitats": ["pine"], "lat": 38.0, "lon": -8.6},
             2: {"fid": 2, "ha": 900, "habitats": ["cork_oak"], "lat": 38.001, "lon": -8.6},
             3: {"fid": 3, "ha": 400, "habitats": ["cork_oak"], "lat": 38.2, "lon": -8.6},
             4: {"fid": 4, "ha": 9999, "habitats": [], "lat": 38.4, "lon": -8.6}}
    assert [p["fid"] for p in L.pick_patches(polys)] == [2, 3]


def test_interior_point_lands_inside_a_c_shape():
    from mushroom import landcover as L
    # C-shaped ring whose centroid falls in the notch
    ring = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 2], [3, 2], [3, 3], [0, 3], [0, 0]]
    lat, lon = L.interior_point({"type": "Polygon", "coordinates": [ring]})
    assert L._inside(lon, lat, ring)


def test_do_not_eat_species_are_never_scored_at_a_site():
    from mushroom.sites import ForagingSite
    from mushroom.species_db import PINE
    site = ForagingSite(name="t", lat=38.0, lon=-8.7, habitats=[PINE])
    assert "Tricholoma equestre" not in {s.name for s in site.species}


def test_interior_points_spread_inside():
    from mushroom import landcover as L
    ring = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 2], [3, 2], [3, 3], [0, 3], [0, 0]]
    pts = L.interior_points({"type": "Polygon", "coordinates": [ring]}, n=12)
    assert len(pts) == 12 and all(L._inside(lon, lat, ring) for lat, lon in pts)
