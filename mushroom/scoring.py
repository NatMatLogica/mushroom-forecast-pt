"""
Mushroom fruiting probability scorer.

Fetches weather from Open-Meteo (10 past days + 8 forecast days) and, where an
IPMA surface station sits within 15 km of a site with good hourly coverage,
replaces the grid's past rainfall with that station's OBSERVED rainfall from
an optional local archive, mushroom/ipma_archive.db (table obs: station_id,
time_utc, precip_mm), filled from the IPMA open-data observation API. Without it,
past rain comes from the Open-Meteo grid. Scores each species at each site on rainfall
trigger, days-since-rain latency, temperature and humidity.

Why the station override exists (measured 2026-10-07): the 3 October storm put
29.0 mm on Alcochete / Campo de Tiro and 0.6 mm on Setúbal 17 km away. The
Open-Meteo grid cell between them reported ~12 mm for both. Convective autumn
rain in this region is patchy at exactly the scale the sites are spaced, and a
mushroom trigger is a threshold, so the grid average is the wrong number on
both sides of it.

Scoring (100 pts): rainfall trigger 40 · latency window 30 · temperature 20 ·
humidity 10. Weights unchanged since 2026-06-28.

Corrected 2026-10-07 (all four were demonstrable with a run):
  - the archive URL `api.open-meteo.com/v1/archive` has never existed (404),
    so "recent rain" silently came from the fallback every run;
  - the forecast fetch carried no past days, so `days_since_rain` was the
    hard-coded default of 2 for every species at every site;
  - `Species.rain_window` was declared per species and never read;
  - a failed fetch scored from default 15 °C / 60 % RH and produced phantom
    15-25 % scores. A site with no data now says so and scores nothing.
"""

import json
import os
import math
import sqlite3
import urllib.request
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from mushroom.species_db import Species, get_species_in_season, SPECIES
from mushroom.sites import ForagingSite, SITES, haversine_km

MODEL_VERSION = "0.4.0"

LOOKBACK_DAYS = 10          # past days scanned for a rain event and shown on the page
HISTORY_DAYS = 62           # past days fetched: the fitted timing sums 60 days of lagged weather
LAGGED_GO_SCORE = 70        # lagged species: a day scoring at least this is a fruiting day (fixed — UI threshold)
FORECAST_DAYS = 8           # Open-Meteo forecast_days (day 0 = today, so 7 future days)
IPMA_MAX_KM = 15.0          # a station further than this does not override the grid
IPMA_MIN_COVERAGE = 0.80    # fraction of hours with a valid reading over the lookback
SIGNIFICANT_RAIN_MM = 5.0   # a day with at least this is shown with its value on the chart

HERE = Path(__file__).resolve().parent
IPMA_DB = HERE / "ipma_archive.db"          # optional local archive of IPMA hourly observations; absent = grid only
IPMA_STATIONS = HERE / "ipma_stations.json"  # IPMA station list (api.ipma.pt open data)


# ── Weather container ────────────────────────────────────────────────

@dataclass
class SiteWeather:
    """Daily weather for one site: LOOKBACK_DAYS past, today, FORECAST_DAYS-1 ahead."""
    ok: bool
    dates: List[datetime.date] = field(default_factory=list)
    precip: List[float] = field(default_factory=list)          # mm/day, past days from rain_source
    precip_grid: List[float] = field(default_factory=list)     # mm/day, Open-Meteo for every day
    tmin: List[float] = field(default_factory=list)
    tmax: List[float] = field(default_factory=list)
    rh_mean: List[float] = field(default_factory=list)
    rain_source: str = ""                                      # human-readable provenance
    ipma_station: Optional[dict] = None                        # {id, name, km, coverage}
    ipma_last_obs: str = ""                                    # last observation hour in the archive
    fetched_utc: str = ""
    error: str = ""

    def offset(self, d: datetime.date, today: datetime.date) -> int:
        return (d - today).days


# ── Fetchers ─────────────────────────────────────────────────────────

def fetch_open_meteo(lat: float, lon: float) -> dict:
    """Open-Meteo forecast with past days. Returns {} on failure (caller handles)."""
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean"
           f"&past_days={HISTORY_DAYS}&forecast_days={FORECAST_DAYS}"
           f"&timezone=UTC")
    try:
        return json.loads(urllib.request.urlopen(url, timeout=15).read())
    except Exception as e:  # noqa: BLE001 — reported on the site card, never scored
        print(f"  [warn] Open-Meteo failed for ({lat},{lon}): {e}")
        return {"_error": str(e)}


_IPMA_CACHE: Dict[str, object] = {}


def _ipma_stations() -> List[dict]:
    if "stations" not in _IPMA_CACHE:
        try:
            _IPMA_CACHE["stations"] = json.loads(IPMA_STATIONS.read_text())
        except Exception:  # noqa: BLE001
            _IPMA_CACHE["stations"] = []
    return _IPMA_CACHE["stations"]  # type: ignore[return-value]


def ipma_daily_rain(lat: float, lon: float, start: datetime.date,
                    today: datetime.date) -> Optional[Tuple[dict, Dict[datetime.date, float], str]]:
    """
    Observed daily rainfall from the nearest IPMA station within IPMA_MAX_KM
    whose hourly coverage over [start, now] is at least IPMA_MIN_COVERAGE.
    Returns (station_info, {date: mm}, last_obs_hour) or None.
    Stations are tried nearest-first; a dead station (e.g. 'Setúbal, Areias',
    0 rows) is skipped rather than returning zero rain.
    """
    if not IPMA_DB.exists():
        return None
    stations = _ipma_stations()
    if not stations:
        return None
    near = sorted(
        ((haversine_km(lat, lon, s["lat"], s["lon"]), s) for s in stations),
        key=lambda t: t[0],
    )
    try:
        con = sqlite3.connect(f"file:{IPMA_DB}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        last_obs = con.execute("SELECT MAX(time_utc) FROM obs").fetchone()[0] or ""
        if not last_obs:
            return None
        last_dt = datetime.datetime.fromisoformat(last_obs)
        start_dt = datetime.datetime.combine(start, datetime.time(0, 0))
        expected_hours = max(1, int((last_dt - start_dt).total_seconds() // 3600) + 1)
        for km, s in near:
            if km > IPMA_MAX_KM:
                break
            rows = con.execute(
                """SELECT substr(time_utc,1,10) AS d,
                          SUM(CASE WHEN precip_mm IS NOT NULL AND precip_mm > -1 THEN precip_mm ELSE 0 END),
                          SUM(CASE WHEN precip_mm IS NOT NULL AND precip_mm > -1 THEN 1 ELSE 0 END)
                   FROM obs WHERE station_id=? AND time_utc>=? GROUP BY d ORDER BY d""",
                (s["id"], start.isoformat()),
            ).fetchall()
            valid_hours = sum(r[2] for r in rows)
            coverage = valid_hours / expected_hours
            if coverage < IPMA_MIN_COVERAGE:
                continue
            daily = {datetime.date.fromisoformat(r[0]): float(r[1]) for r in rows}
            info = {"id": s["id"], "name": s["name"], "km": round(km, 1),
                    "coverage": round(coverage, 3)}
            return info, daily, last_obs
    finally:
        con.close()
    return None


WEATHER_SOURCE = os.environ.get("MUSHROOM_WEATHER", "openmeteo")   # "ecmwf" = ECMWF open data (CC BY 4.0, commercial use allowed)
_ECMWF: Dict[str, object] = {}


import threading as _threading
_ECMWF_LOCK = _threading.Lock()


def ecmwf_grids(today: datetime.date, wait: bool = True):
    """Daily ECMWF grids for HISTORY_DAYS back and FORECAST_DAYS-1 ahead, built once per process and day.
    One build at a time. wait=False: return the last complete grids (possibly yesterday's) or raise if none yet."""
    from mushroom import weather_ecmwf
    if _ECMWF.get("today") == today:
        return _ECMWF["dates"], _ECMWF["grids"]
    if not wait:
        if "grids" in _ECMWF:
            return _ECMWF["dates"], _ECMWF["grids"]
        raise RuntimeError("weather is being prepared; try again in a few minutes")
    with _ECMWF_LOCK:
        if _ECMWF.get("today") != today:
            dates, g = weather_ecmwf.daily_grids(today, HISTORY_DAYS, FORECAST_DAYS - 1)
            _ECMWF.update(today=today, dates=dates, grids=g)
    return _ECMWF["dates"], _ECMWF["grids"]


def site_weather_ecmwf(lat: float, lon: float, today: datetime.date, wait: bool = True) -> SiteWeather:
    from mushroom import weather_ecmwf
    fetched = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        dates, g = ecmwf_grids(today, wait=wait)
    except Exception as e:  # noqa: BLE001 — reported, never scored
        return SiteWeather(ok=False, fetched_utc=fetched, error=f"ECMWF open data: {e}")
    col = {k: [float(v) for v in weather_ecmwf.at_point(lat, lon, g[k])] for k in g}
    return SiteWeather(ok=True, dates=list(dates), precip=col["rain"], precip_grid=list(col["rain"]),
                       tmin=col["tmin"], tmax=col["tmax"], rh_mean=col["rh"],
                       rain_source=weather_ecmwf.ATTRIBUTION, fetched_utc=fetched)


def fetch_site_weather(site: ForagingSite, today: datetime.date) -> SiteWeather:
    """Daily series for one site: ECMWF open data, or Open-Meteo with an IPMA station override."""
    if WEATHER_SOURCE == "ecmwf":
        return site_weather_ecmwf(site.lat, site.lon, today)
    raw = fetch_open_meteo(site.lat, site.lon)
    fetched = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    daily = raw.get("daily", {}) if raw else {}
    if not daily.get("time"):
        return SiteWeather(ok=False, fetched_utc=fetched,
                           error=raw.get("_error", "Open-Meteo returned no daily data"))

    dates = [datetime.date.fromisoformat(t) for t in daily["time"]]
    n = len(dates)

    def col(key: str, default: float) -> List[float]:
        vals = daily.get(key, [])
        return [float(v) if v is not None else default for v in vals] + [default] * (n - len(vals))

    precip_grid = col("precipitation_sum", 0.0)
    tmin = col("temperature_2m_min", math.nan)
    tmax = col("temperature_2m_max", math.nan)
    rh = col("relative_humidity_2m_mean", math.nan)

    precip = list(precip_grid)
    source = "Open-Meteo grid (past and forecast)"
    station = None
    last_obs = ""
    obs = ipma_daily_rain(site.lat, site.lon, dates[0], today)
    if obs is not None:
        station, obs_daily, last_obs = obs
        for i, d in enumerate(dates):
            if d <= today:
                precip[i] = obs_daily.get(d, 0.0)
        source = (f"IPMA station {station['name']} ({station['km']} km, "
                  f"{station['coverage']*100:.0f}% of hours) for past days; Open-Meteo grid for forecast")

    return SiteWeather(ok=True, dates=dates, precip=precip, precip_grid=precip_grid,
                       tmin=tmin, tmax=tmax, rh_mean=rh, rain_source=source,
                       ipma_station=station, ipma_last_obs=last_obs, fetched_utc=fetched)


# ── Scoring ──────────────────────────────────────────────────────────

@dataclass
class SpeciesScore:
    """Score for one species at one site."""
    species: Species
    score: int               # 0-100
    rain_met: bool           # Was the rainfall trigger met in the lookback?
    rain_forecast: bool      # Is forecast rain enough to trigger a future flush?
    temp_ok: bool
    humidity_ok: bool
    latency_status: str      # "now" / "soon" / "not_yet" / "past_window" / "forecast" / "none" / "off_season"
    days_since_rain: Optional[int]   # days since the triggering window ended (None if none)
    event_mm: float          # rainfall in the triggering window
    days_until_window: int   # days until optimal foraging (0 = go now; 99 = unknown)
    recommendation: str
    avg_temp: float = math.nan
    humidity: float = math.nan


@dataclass
class SiteResult:
    """Scored results for one site."""
    site: ForagingSite
    weather: SiteWeather
    species_scores: List[SpeciesScore] = field(default_factory=list)
    max_score: int = 0
    recommendation: str = ""
    rain_past_mm: float = 0.0      # sum over the lookback (chosen source)
    rain_forecast_mm: float = 0.0  # sum over the forecast days


def last_rain_event(w: SiteWeather, today: datetime.date, rain_min: float,
                    rain_window: int) -> Optional[Tuple[int, float]]:
    """
    Find the most recent rain_window-day window (ending on a past day or today)
    whose rainfall reaches rain_min, then anchor "days since rain" on the LAST
    day inside that window that actually rained (>= 1 mm), because the latency
    clock starts when the soil was wetted, not when a rolling sum last happened
    to clear the bar. Falls back to the window's end day if no single day in it
    reached 1 mm. Returns (days_since_rain, mm_in_window) or None.
    """
    best = None
    # the fetch holds HISTORY_DAYS; event-driven species look back LOOKBACK_DAYS only (as in v0.2.0)
    first = next((k for k, d in enumerate(w.dates) if (today - d).days <= LOOKBACK_DAYS), 0)
    for i, d in enumerate(w.dates):
        if d > today:
            break
        if i < first:
            continue
        lo = max(first, i - rain_window + 1)
        mm = sum(w.precip[lo:i + 1])
        if mm >= rain_min:
            anchor = i
            for j in range(i, lo - 1, -1):
                if w.precip[j] >= 1.0:
                    anchor = j
                    break
            best = ((today - w.dates[anchor]).days, round(mm, 1))
    return best


def window_means(w: SiteWeather, today: datetime.date, day_start: int,
                 day_end: int) -> Tuple[float, float]:
    """(mean of daily mid-range temperature, mean daily RH) over offsets [day_start, day_end]."""
    temps, rhs = [], []
    for i, d in enumerate(w.dates):
        off = (d - today).days
        if day_start <= off <= day_end:
            if not math.isnan(w.tmin[i]) and not math.isnan(w.tmax[i]):
                temps.append(0.5 * (w.tmin[i] + w.tmax[i]))
            if not math.isnan(w.rh_mean[i]):
                rhs.append(w.rh_mean[i])
    t = sum(temps) / len(temps) if temps else math.nan
    h = sum(rhs) / len(rhs) if rhs else math.nan
    return t, h


def _day_mean_temp(w: SiteWeather, i: int) -> float:
    return 0.5 * (w.tmin[i] + w.tmax[i])   # mid-range; the source study used the 24 h mean


def lagged_points(lp, w: SiteWeather, i: int) -> Optional[Tuple[float, float, float, float]]:
    """
    (temperature points 0-45, rain points 0-45, T mean over temp_days, rain mm/day over
    rain_days) for the day at index i, or None if the series does not reach back far enough.
    """
    if i + 1 < max(lp.temp_days, lp.rain_days):
        return None
    temps = [_day_mean_temp(w, j) for j in range(i - lp.temp_days + 1, i + 1)]
    if any(math.isnan(t) for t in temps):
        return None
    t = sum(temps) / len(temps)
    r = sum(w.precip[i - lp.rain_days + 1:i + 1]) / lp.rain_days
    lo, hi = lp.temp_band
    zlo, zhi = lp.temp_zero
    if lo <= t <= hi:
        tf = 1.0
    elif t < lo:
        tf = max(0.0, (t - zlo) / (lo - zlo))
    else:
        tf = max(0.0, (zhi - t) / (zhi - hi))
    rf = min(1.0, r / lp.rain_full_mm_day)
    return 45.0 * tf, 45.0 * rf, t, r


def score_lagged(species: Species, w: SiteWeather, today: datetime.date) -> SpeciesScore:
    """
    Weeks-long windows instead of one rain event (Species.lagged). 100 pts:
    temperature 45 · rain 45 · humidity 10. Scored for today and each forecast day;
    the reported score is today's, and days_until_window is the first day that clears
    LAGGED_GO_SCORE (the best day if none does).
    """
    lp = species.lagged
    _, humidity = window_means(w, today, -4, 0)
    hum_pts, humidity_ok = 0, False
    if not math.isnan(humidity):
        if humidity >= species.humidity_min:
            hum_pts, humidity_ok = 10, True
        elif humidity >= species.humidity_min - 10:
            hum_pts, humidity_ok = 5, True

    days = []   # (offset, score, T, rain mm/day)
    for i, d in enumerate(w.dates):
        off = (d - today).days
        if off < 0:
            continue
        pts = lagged_points(lp, w, i)
        if pts is None:
            continue
        tp, rp, t, r = pts
        days.append((off, int(round(tp + rp + hum_pts)), t, r))
    if not days or days[0][0] != 0:
        return SpeciesScore(species=species, score=0, rain_met=False, rain_forecast=False,
                            temp_ok=False, humidity_ok=False, latency_status="none",
                            days_since_rain=None, event_mm=0.0, days_until_window=99,
                            recommendation="Not enough weather history for the %d-day windows" % lp.rain_days)

    _, s0, t0, r0 = days[0]
    rain_met = r0 >= lp.rain_full_mm_day
    temp_ok = lp.temp_zero[0] < t0 < lp.temp_zero[1]
    basis = "%d-day mean %.1f °C, %d-day rain %.1f mm/day" % (lp.temp_days, t0, lp.rain_days, r0)
    first_go = next((d for d in days if d[1] >= LAGGED_GO_SCORE), None)
    if first_go is not None and first_go[0] == 0:
        status, until, rec = "now", 0, "Fruiting conditions now (%s)" % basis
    elif first_go is not None:
        off = first_go[0]
        status = "soon" if off <= 2 else "not_yet"
        until = off
        rec = "Fruiting conditions in about %d day%s; today %s" % (off, "" if off == 1 else "s", basis)
    else:
        best = max(days, key=lambda d: d[1])
        status, until = "none", 99
        if t0 >= lp.temp_band[1] and r0 >= lp.rain_full_mm_day:
            why = "wet enough, still too warm"
        elif t0 <= lp.temp_band[0] and r0 >= lp.rain_full_mm_day:
            why = "wet enough, too cold"
        elif r0 < lp.rain_full_mm_day:
            why = "not enough rain over the last %d days" % lp.rain_days
        else:
            why = "below the fruiting score"
        rec = "No fruiting window in the next 7 days: %s (%s; best day +%d at %d%%)" % (why, basis, best[0], best[1])

    return SpeciesScore(
        species=species, score=s0, rain_met=rain_met, rain_forecast=False,
        temp_ok=temp_ok, humidity_ok=humidity_ok, latency_status=status,
        days_since_rain=None, event_mm=round(r0 * lp.rain_days, 1), days_until_window=until,
        recommendation=rec, avg_temp=t0, humidity=humidity,
    )


FITTED_FILE = HERE / "fitted_params.json"
_FITTED: Dict[str, dict] = {}


def fitted_params() -> Dict[str, dict]:
    if not _FITTED and FITTED_FILE.exists():
        _FITTED.update(json.loads(FITTED_FILE.read_text()))
    return _FITTED


def fitted_score_raw(fp: dict, w: SiteWeather, i: int) -> Optional[float]:
    """
    s(t) of calibrate_aadc.py for the day at index i, weekend term excluded (it is observer
    behaviour, not weather). Plain floats: inference only; the parameters were fitted on the
    AADC tape, and test_calibrate_aadc.py checks this function against that tape.
    """
    p, J = fp["params"], fp["J"]
    if i + 1 < J:
        return None

    def weights(log_tau):
        q = math.exp(-math.exp(-log_tau))
        raw = [q ** j for j in range(J)]
        tot = sum(raw)
        return [r / tot for r in raw]
    aR, aT = weights(p["log_tau_r"]), weights(p["log_tau_t"])
    R = Tm = 0.0
    for j in range(J):
        t = _day_mean_temp(w, i - j)
        if math.isnan(t):
            return None
        R += aR[j] * w.precip[i - j]
        Tm += aT[j] * t
    d = Tm - p["T_opt"]
    return p["w_r"] * R + p["w_t"] * math.exp(-(d * d) / (2.0 * math.exp(2.0 * p["log_sd"])))


def percentile(q: List[float], v: float) -> float:
    """Where v falls among the in-season days the fit was trained on, 0-100."""
    if v <= q[0]:
        return 0.0
    if v >= q[-1]:
        return 100.0
    for k in range(1, len(q)):
        if v <= q[k]:
            span = q[k] - q[k - 1]
            return (k - 1) + ((v - q[k - 1]) / span if span > 0 else 0.0)
    return 100.0


def score_fitted(species: Species, w: SiteWeather, today: datetime.date) -> SpeciesScore:
    """AADC-calibrated timing. Score = percentile of today's weather among in-season days."""
    fp = fitted_params().get(species.fitted)
    if fp is None:
        return SpeciesScore(species=species, score=0, rain_met=False, rain_forecast=False, temp_ok=False,
                            humidity_ok=False, latency_status="none", days_since_rain=None, event_mm=0.0,
                            days_until_window=99, recommendation="Fitted timing missing (fitted_params.json)")
    days = []
    for i, d in enumerate(w.dates):
        off = (d - today).days
        if off < 0:
            continue
        raw = fitted_score_raw(fp, w, i)
        if raw is not None:
            days.append((off, int(round(percentile(fp["quantiles"], raw)))))
    if not days or days[0][0] != 0:
        return SpeciesScore(species=species, score=0, rain_met=False, rain_forecast=False, temp_ok=False,
                            humidity_ok=False, latency_status="none", days_since_rain=None, event_mm=0.0,
                            days_until_window=99, recommendation="Not enough weather history (60 days needed)")
    s0 = days[0][1]
    first = next((d for d in days if d[1] >= LAGGED_GO_SCORE), None)
    basis = "better than %d%% of in-season days in the fitted record" % s0
    if first is not None and first[0] == 0:
        status, until, rec = "now", 0, "Good fruiting weather now (%s)" % basis
    elif first is not None:
        status, until = ("soon" if first[0] <= 2 else "not_yet"), first[0]
        rec = "Good fruiting weather in about %d day%s; today %s" % (until, "" if until == 1 else "s", basis)
    else:
        best = max(days, key=lambda d: d[1])
        status, until = "none", 99
        rec = "No strong fruiting weather in the next 7 days (best day +%d at %d%%)" % best
    return SpeciesScore(species=species, score=s0, rain_met=s0 >= LAGGED_GO_SCORE, rain_forecast=False,
                        temp_ok=True, humidity_ok=True, latency_status=status, days_since_rain=None,
                        event_mm=0.0, days_until_window=until, recommendation=rec)


def score_species(species: Species, w: SiteWeather, today: datetime.date,
                  current_month: int) -> SpeciesScore:
    """Score one species against one site's weather. 100 pts: 40 / 30 / 20 / 10."""
    off = SpeciesScore(species=species, score=0, rain_met=False, rain_forecast=False,
                       temp_ok=False, humidity_ok=False, latency_status="off_season",
                       days_since_rain=None, event_mm=0.0, days_until_window=99,
                       recommendation="Off season")
    if current_month not in species.season:
        return off
    if species.fitted:
        return score_fitted(species, w, today)
    if species.lagged is not None:
        return score_lagged(species, w, today)

    score = 0
    rain_met = False
    rain_forecast = False
    latency_status = "none"
    days_since: Optional[int] = None
    event_mm = 0.0
    days_until_window = 99
    recommendation = "No triggering rain in the last %d days or the next 7" % LOOKBACK_DAYS

    # ── Rainfall trigger (40) + latency window (30) ──
    event = last_rain_event(w, today, species.rain_min, species.rain_window)
    if event is not None:
        days_since, event_mm = event
        rain_met = True
        score += 40
        if days_since >= species.latency_max:
            if days_since <= species.latency_max + species.harvest_days:
                score += 20
                latency_status = "now"
                days_until_window = 0
                recommendation = "Harvest window open (past peak)"
            else:
                score += 5
                latency_status = "past_window"
                recommendation = "Flush from %d days ago is likely over; wait for the next rain" % days_since
        elif days_since >= species.latency_min:
            score += 30
            latency_status = "now"
            days_until_window = 0
            recommendation = "Prime fruiting time"
        elif days_since + 2 >= species.latency_min:
            score += 20
            latency_status = "soon"
            days_until_window = species.latency_min - days_since
            recommendation = "Check back in %d day%s" % (days_until_window, "" if days_until_window == 1 else "s")
        else:
            score += 10
            latency_status = "not_yet"
            days_until_window = species.latency_min - days_since
            recommendation = "Too soon after the rain; wait about %d days" % days_until_window
    else:
        # Forecast rain: cumulative from tomorrow
        cumulative = 0.0
        for i, d in enumerate(w.dates):
            offset = (d - today).days
            if offset <= 0:
                continue
            cumulative += w.precip[i]
            if cumulative >= species.rain_min:
                rain_forecast = True
                score += 20
                score += int(round(10 * max(0.0, 1 - offset / 7)))
                days_until_window = offset + species.latency_min
                latency_status = "forecast"
                recommendation = "Rain due in %dd; check about %dd from now" % (offset, days_until_window)
                break

    # ── Temperature (20) and humidity (10), over the window that matters ──
    if rain_met:
        avg_temp, humidity = window_means(w, today, 0, 5)
    else:
        avg_temp, humidity = window_means(w, today, 1, 7)

    temp_ok = False
    if not math.isnan(avg_temp):
        if species.temp_min <= avg_temp <= species.temp_max:
            score += 20
            temp_ok = True
        elif species.temp_min - 5 <= avg_temp <= species.temp_max + 5:
            score += 10
            temp_ok = True  # partial

    humidity_ok = False
    if not math.isnan(humidity):
        if humidity >= species.humidity_min:
            score += 10
            humidity_ok = True
        elif humidity >= species.humidity_min - 10:
            score += 5
            humidity_ok = True  # partial

    score = max(0, min(100, int(score)))

    return SpeciesScore(
        species=species, score=score, rain_met=rain_met, rain_forecast=rain_forecast,
        temp_ok=temp_ok, humidity_ok=humidity_ok, latency_status=latency_status,
        days_since_rain=days_since, event_mm=event_mm, days_until_window=days_until_window,
        recommendation=recommendation, avg_temp=avg_temp, humidity=humidity,
    )


def score_site(site: ForagingSite, today: datetime.date, current_month: int) -> SiteResult:
    """Score all species at one site."""
    print(f"  Fetching weather for {site.name}...")
    w = fetch_site_weather(site, today)
    import time
    time.sleep(0.3)  # Open-Meteo rate courtesy

    if not w.ok:
        return SiteResult(site=site, weather=w, species_scores=[], max_score=0,
                          recommendation="No weather data for this site; nothing scored")

    scores = [score_species(sp, w, today, current_month) for sp in site.species]
    max_score = max((s.score for s in scores), default=0)

    high = [s for s in scores if s.score >= 50]
    if high:
        best = max(high, key=lambda s: s.score)
        rec = f"{best.species.pt_name} {best.score}%: {best.recommendation}"
    else:
        rec = "No strong signal this week"

    past = sum(p for d, p in zip(w.dates, w.precip) if d <= today and (today - d).days <= LOOKBACK_DAYS)
    fut = sum(p for d, p in zip(w.dates, w.precip) if d > today)
    return SiteResult(site=site, weather=w, species_scores=scores, max_score=max_score,
                      recommendation=rec, rain_past_mm=round(past, 1), rain_forecast_mm=round(fut, 1))


def score_all_sites(current_month: Optional[int] = None,
                    today: Optional[datetime.date] = None) -> List[SiteResult]:
    """Score all sites. Returns sorted by max score (best first); no-data sites last."""
    today = today or datetime.date.today()
    if current_month is None:
        current_month = today.month
    results = [score_site(site, today, current_month) for site in SITES]
    results.sort(key=lambda r: (r.weather.ok, r.max_score), reverse=True)
    return results
