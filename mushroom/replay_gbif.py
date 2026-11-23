#!/usr/bin/env python3
"""
Replay test: does the model say "go" more on days people actually found a species?

For every dated, located GBIF record in mainland Portugal (2015 onward, coordinate
uncertainty <= 10 km, day-level date), fetch the Open-Meteo historical archive
(ERA5-based, free) around the record and score the species with the CURRENT model
on the find day and on CONTROL days at the same place: every 3rd day from 45 days
before to 45 days after, excluding +/-7 days around the find and any day outside
the species' season months. Controls are therefore in season, same place, same
year: the test isolates the weather timing, not the season calendar or the habitat.

Metric per species: AUC = P(find-day score > control-day score), ties count half.
0.5 = no better than picking a random in-season day; 1.0 = perfect.

Limits, stated: GBIF records are observations, not a survey (people go out on
weekends and good days, which biases find days toward pleasant weather); the
archive grid is coarse and has no station override; the find day may be days
after emergence. Nothing here is a fitted result; it is an out-of-sample check
because no parameter was ever set from these records.

Usage: python3 mushroom/replay_gbif.py [--species "Boletus edulis,..."] [--max N]
Writes runs/replay_gbif_<today>.json (every record, every score).
"""

import argparse
import dataclasses
import datetime
import json
import math
import os
import sys
import time
import urllib.request

MUSHROOM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MUSHROOM_DIR))

from mushroom.scoring import SiteWeather, score_species, MODEL_VERSION  # noqa: E402
from mushroom.species_db import SPECIES  # noqa: E402

DEFAULT_SPECIES = ["Boletus edulis", "Boletus reticulatus", "Boletus aereus",
                   "Armillaria mellea", "Amanita caesarea"]
BOXES = {"PT": (36.9, 42.2, -9.6, -6.1),   # mainland Portugal lat/lon box (Azores/Madeira excluded)
         "ES": (35.9, 43.9, -9.4, 3.4)}    # mainland Spain (Canaries/Balearics excluded)
PRE, POST = 90, 55                      # archive days fetched around each record
CTRL_SPAN, CTRL_STEP, CTRL_GAP = 45, 3, 7


def get(url, timeout=60):
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read())


def gbif_records(name: str, country: str = "PT"):
    key = get(f"https://api.gbif.org/v1/species/match?name={urllib.request.quote(name)}")["usageKey"]
    out, off = [], 0
    while True:
        d = get(f"https://api.gbif.org/v1/occurrence/search?taxonKey={key}&country={country}&hasCoordinate=true"
                f"&hasGeospatialIssue=false&year=2015,2026&limit=300&offset={off}")
        for r in d["results"]:
            la, lo = r.get("decimalLatitude"), r.get("decimalLongitude")
            box = BOXES[country]
            if la is None or not (box[0] <= la <= box[1] and box[2] <= lo <= box[3]):
                continue
            if (r.get("coordinateUncertaintyInMeters") or 0) > 10000:
                continue
            if not (r.get("year") and r.get("month") and r.get("day")):
                continue
            out.append({"gbif": r["key"], "country": country, "lat": round(la, 4), "lon": round(lo, 4),
                        "date": datetime.date(r["year"], r["month"], r["day"]).isoformat(),
                        "basis": r.get("basisOfRecord"), "dataset": r.get("datasetName", "")})
        off += 300
        if d.get("endOfRecords") or off >= d.get("count", 0):
            return out


def archive(lat, lon, d0, d1):
    u = ("https://archive-api.open-meteo.com/v1/archive?"
         f"latitude={lat}&longitude={lon}&start_date={d0}&end_date={d1}"
         "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean"
         "&timezone=UTC")
    dl = get(u)["daily"]
    dates = [datetime.date.fromisoformat(t) for t in dl["time"]]

    def col(k, dflt):
        return [float(v) if v is not None else dflt for v in dl[k]]
    p = col("precipitation_sum", 0.0)
    return SiteWeather(ok=True, dates=dates, precip=p, precip_grid=p,
                       tmin=col("temperature_2m_min", math.nan), tmax=col("temperature_2m_max", math.nan),
                       rh_mean=col("relative_humidity_2m_mean", math.nan), rain_source="Open-Meteo archive")


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else (0.5 if p == n else 0.0)
    return s / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default=",".join(DEFAULT_SPECIES))
    ap.add_argument("--max", type=int, default=0, help="cap records per species (0 = all)")
    a = ap.parse_args()
    today = datetime.date.today()
    last_ok = today - datetime.timedelta(days=7)          # archive lags about 5 days
    names = [s.strip() for s in a.species.split(",") if s.strip()]
    spmap = {s.name: s for s in SPECIES}
    variants = {}
    for n in names:
        variants[n] = spmap[n]
        if spmap[n].lagged is not None:                   # also test the old event rule
            variants[n + " [v0.2 event rule]"] = dataclasses.replace(spmap[n], lagged=None)

    results, summary = [], {}
    for n in names:
        recs = gbif_records(n)
        recs = [r for r in recs if datetime.date.fromisoformat(r["date"]) + datetime.timedelta(days=POST) <= last_ok]
        if a.max:
            recs = recs[:a.max]
        print(f"{n}: {len(recs)} usable records")
        for r in recs:
            d = datetime.date.fromisoformat(r["date"])
            try:
                w = archive(r["lat"], r["lon"], d - datetime.timedelta(days=PRE), d + datetime.timedelta(days=POST))
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] archive failed for {r['gbif']}: {e}")
                continue
            time.sleep(0.25)
            for vname, sp in variants.items():
                if not vname.startswith(n):
                    continue
                find = score_species(sp, w, d, d.month).score
                ctrls = []
                for k in range(-CTRL_SPAN, CTRL_SPAN + 1, CTRL_STEP):
                    if abs(k) <= CTRL_GAP:
                        continue
                    c = d + datetime.timedelta(days=k)
                    if c.month in sp.season:
                        ctrls.append(score_species(sp, w, c, c.month).score)
                results.append({**r, "variant": vname, "find_score": find, "control_scores": ctrls})

    for vname in variants:
        rows = [x for x in results if x["variant"] == vname]
        pos = [x["find_score"] for x in rows]
        neg = [c for x in rows for c in x["control_scores"]]
        go = lambda xs: (sum(1 for v in xs if v >= 70) / len(xs)) if xs else None  # noqa: E731
        summary[vname] = {"records": len(rows), "auc": auc(pos, neg),
                          "mean_find": (sum(pos) / len(pos)) if pos else None,
                          "mean_control": (sum(neg) / len(neg)) if neg else None,
                          "go_rate_find": go(pos), "go_rate_control": go(neg)}
        s = summary[vname]
        if s["records"]:
            print(f"{vname:40s} n={s['records']:4d}  AUC={s['auc']:.3f}  mean find {s['mean_find']:.0f} vs "
                  f"control {s['mean_control']:.0f}  'go' {s['go_rate_find']:.0%} vs {s['go_rate_control']:.0%}")

    out = os.path.join(MUSHROOM_DIR, "runs", f"replay_gbif_{today}.json")
    with open(out, "w") as f:
        json.dump({"model_version": MODEL_VERSION, "generated": today.isoformat(),
                   "design": {"pre_days": PRE, "post_days": POST, "control_span": CTRL_SPAN,
                              "control_step": CTRL_STEP, "control_gap": CTRL_GAP,
                              "source": "GBIF occurrence API + Open-Meteo archive"},
                   "summary": summary, "records": results}, f, ensure_ascii=False)
    print(f"Record: {out}")


if __name__ == "__main__":
    main()
