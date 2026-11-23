"""
Daily weather for the mushroom forecast from ECMWF open data.

Source: ECMWF real-time IFS forecasts, 0.25 degree, published free of charge under
Creative Commons CC-BY-4.0 and the ECMWF Terms of Use, which allow commercial use with
attribution ("Contains modified ECMWF open data"). The archive of past runs is the
public AWS mirror, bucket ecmwf-forecasts (eu-central-1).

How a daily series is built (all days are UTC days, as in scoring.py):
  * past days: each day d is taken from the 00z run of day d, steps 3..24 h (the
    first-day forecast, the closest open product to an analysis);
  * today and the next days: from the latest 00z run available, steps up to 168 h.
Per day: rain = tp(24k) - tp(24k-24); max/min temperature from the 3-hourly
mx2t3/mn2t3 fields (6-hourly mx2t6/mn2t6 beyond 144 h); relative humidity as the
mean of the 6-hourly 2t/2d pairs (Magnus formula).

Only the needed fields are downloaded (HTTP range requests on the run's .index),
decoded with ecCodes, cropped to Iberia and cached under mushroom/ecmwf_cache/.
"""

import datetime
import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecmwf_cache")
LAT_N, LAT_S, LON_W, LON_E = 44.0, 35.5, -10.0, 4.0     # Iberia crop (mainland PT + ES)
ATTRIBUTION = "Contains modified ECMWF open data (CC BY 4.0)"


def _get(url: str, rng: Optional[Tuple[int, int]] = None, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mushroom-forecast-pt"})
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[0] + rng[1] - 1}")
    for attempt in range(6):                     # S3 answers 503 SlowDown under load: back off and retry
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
            time.sleep(5 * 2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError("unreachable")


def _paths(date: datetime.date, step: int) -> Tuple[str, str]:
    d = date.strftime("%Y%m%d")
    stem = f"{BASE}/{d}/00z/ifs/0p25/oper/{d}000000-{step}h-oper-fc"
    return stem + ".index", stem + ".grib2"


def run_available(date: datetime.date) -> bool:
    try:
        _get(_paths(date, 168)[0], timeout=20)
        return True
    except urllib.error.HTTPError:
        return False


def fields(date: datetime.date, step: int, params: List[str]) -> Dict[str, np.ndarray]:
    """Iberia crops of the requested surface params for one run/step (cached as .npz)."""
    import eccodes
    os.makedirs(CACHE, exist_ok=True)
    out: Dict[str, np.ndarray] = {}
    missing = []
    for p in params:
        f = os.path.join(CACHE, f"{date:%Y%m%d}_{step:03d}_{p}.npy")
        if os.path.exists(f):
            out[p] = np.load(f)
        else:
            missing.append(p)
    if not missing:
        return out
    idx_url, grib_url = _paths(date, step)
    entries = [json.loads(l) for l in _get(idx_url).decode().splitlines() if l.strip()]
    for p in missing:
        e = next((x for x in entries if x.get("param") == p and x.get("levtype") == "sfc"), None)
        if e is None:
            raise KeyError(f"{p} not in {idx_url}")
        msg = eccodes.codes_new_from_message(_get(grib_url, (e["_offset"], e["_length"])))
        try:
            ni, nj = eccodes.codes_get(msg, "Ni"), eccodes.codes_get(msg, "Nj")
            lat0 = eccodes.codes_get(msg, "latitudeOfFirstGridPointInDegrees")
            lon0 = eccodes.codes_get(msg, "longitudeOfFirstGridPointInDegrees")
            dlat = eccodes.codes_get(msg, "jDirectionIncrementInDegrees")
            dlon = eccodes.codes_get(msg, "iDirectionIncrementInDegrees")
            vals = eccodes.codes_get_values(msg).reshape(nj, ni)
        finally:
            eccodes.codes_release(msg)
        lats = lat0 - dlat * np.arange(nj)                    # north to south
        lons = ((lon0 + dlon * np.arange(ni)) + 180.0) % 360.0 - 180.0
        order = np.argsort(lons)
        lons, vals = lons[order], vals[:, order]
        jm = (lats <= LAT_N + 1e-9) & (lats >= LAT_S - 1e-9)
        im = (lons >= LON_W - 1e-9) & (lons <= LON_E + 1e-9)
        crop = vals[np.ix_(jm, im)].astype(np.float32)
        np.save(os.path.join(CACHE, f"{date:%Y%m%d}_{step:03d}_{p}.npy"), crop)
        out[p] = crop
    return out


def grid() -> Tuple[np.ndarray, np.ndarray]:
    """Latitudes (north to south) and longitudes of the Iberia crop."""
    lats = np.round(np.arange(LAT_N, LAT_S - 1e-9, -0.25), 4)
    lons = np.round(np.arange(LON_W, LON_E + 1e-9, 0.25), 4)
    return lats, lons


def _rh(t_k: np.ndarray, td_k: np.ndarray) -> np.ndarray:
    t, td = t_k - 273.15, td_k - 273.15
    es = np.exp(17.625 * t / (243.04 + t))
    e = np.exp(17.625 * td / (243.04 + td))
    return np.clip(100.0 * e / es, 0.0, 100.0)


def day_from_run(run: datetime.date, k: int) -> Dict[str, np.ndarray]:
    """Forecast day k (1 = first 24 h) of the 00z run: rain mm, tmax/tmin C, rh %."""
    hi, lo = 24 * k, 24 * k - 24
    tp_hi = fields(run, hi, ["tp"])["tp"]
    tp_lo = fields(run, lo, ["tp"])["tp"] if lo > 0 else np.zeros_like(tp_hi)
    rain = np.maximum(tp_hi - tp_lo, 0.0) * 1000.0
    three = hi <= 144
    steps = range(lo + 3, hi + 1, 3) if three else range(lo + 6, hi + 1, 6)
    mx, mn = ("mx2t3", "mn2t3") if three else ("mx2t6", "mn2t6")
    tmax = np.max([fields(run, s, [mx])[mx] for s in steps], axis=0) - 273.15
    tmin = np.min([fields(run, s, [mn])[mn] for s in steps], axis=0) - 273.15
    syn = range(lo + 6, hi + 1, 6)
    rh = np.mean([_rh(*(lambda f: (f["2t"], f["2d"]))(fields(run, s, ["2t", "2d"]))) for s in syn], axis=0)
    return {"rain": rain, "tmax": tmax, "tmin": tmin, "rh": rh}


def latest_run(today: datetime.date) -> datetime.date:
    for back in range(0, 4):
        d = today - datetime.timedelta(days=back)
        if run_available(d):
            return d
    raise RuntimeError("no ECMWF open-data run found in the last 4 days")


def daily_grids(today: datetime.date, history_days: int, future_days: int) -> Tuple[List[datetime.date], Dict[str, np.ndarray]]:
    """Stack of daily grids from today-history_days to today+future_days."""
    run = latest_run(today)
    dates, stacks = [], {k: [] for k in ("rain", "tmax", "tmin", "rh")}
    first = today - datetime.timedelta(days=history_days)
    d = first
    while d <= today + datetime.timedelta(days=future_days):
        if d < run:
            day = day_from_run(d, 1)                       # past: that day's own first-day forecast
        else:
            day = day_from_run(run, (d - run).days + 1)    # today/future: latest run
        for k in stacks:
            stacks[k].append(day[k])
        dates.append(d)
        d += datetime.timedelta(days=1)
    return dates, {k: np.stack(v) for k, v in stacks.items()}


def at_point(lat: float, lon: float, arr: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a (days, lat, lon) stack at one point."""
    lats, lons = grid()
    y = (LAT_N - lat) / 0.25
    x = (lon - LON_W) / 0.25
    j0, i0 = int(math.floor(y)), int(math.floor(x))
    j0 = min(max(j0, 0), len(lats) - 2)
    i0 = min(max(i0, 0), len(lons) - 2)
    fy, fx = y - j0, x - i0
    return ((1 - fy) * (1 - fx) * arr[:, j0, i0] + (1 - fy) * fx * arr[:, j0, i0 + 1]
            + fy * (1 - fx) * arr[:, j0 + 1, i0] + fy * fx * arr[:, j0 + 1, i0 + 1])
