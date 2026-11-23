#!/usr/bin/env python3
"""
Where to go: rank the oak / pine / eucalyptus patches around a point for chosen days.

Finds the trees from the official land-cover map (landcover.py), fetches the weather
for each patch (scoring.fetch_site_weather: Open-Meteo, IPMA station where one is
within 15 km), and scores every in-season species whose habitat the patch has.
Days must be within the 7-day forecast.

Usage:
  python3 mushroom/spots.py 38.24036 -8.66606 --days 2026-11-21,2026-11-22
  python3 mushroom/spots.py 38.24036 -8.66606 --species "Boletus edulis,Armillaria mellea"
Writes runs/spots_<today>_<lat>_<lon>.json with every input and score.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time

MUSHROOM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MUSHROOM_DIR))

from mushroom import landcover  # noqa: E402
from mushroom.scoring import fetch_site_weather, score_species, MODEL_VERSION, FORECAST_DAYS  # noqa: E402
from mushroom.sites import ForagingSite, haversine_km  # noqa: E402
from mushroom.species_db import SPECIES, HABITAT_LABELS  # noqa: E402


def bearing(lat0, lon0, lat, lon) -> str:
    b = math.degrees(math.atan2((lon - lon0) * math.cos(math.radians(lat0)), lat - lat0)) % 360
    return "N NE E SE S SW W NW".split()[int((b + 22.5) // 45) % 8]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("lat", type=float)
    ap.add_argument("lon", type=float)
    ap.add_argument("--radius", type=float, default=20.0, help="km (default 20)")
    ap.add_argument("--step", type=float, default=2.5, help="grid spacing km (default 2.5)")
    ap.add_argument("--days", default="", help="comma-separated YYYY-MM-DD; default the next weekend")
    ap.add_argument("--species", default="", help="comma-separated scientific names; default all in season")
    ap.add_argument("--limit", type=int, default=10, help="patches to score (default 10)")
    ap.add_argument("--spacing", type=float, default=5.0, help="min km between scored patches (default 5)")
    ap.add_argument("--min-ha", type=float, default=150.0, help="smallest patch in hectares (default 150)")
    ap.add_argument("--habitats", default="",
                    help="only patches with these habitats, e.g. cork_oak,holm_oak,chestnut")
    ap.add_argument("--public", action="store_true",
                    help="score only public/community forests (ICNF Regime Florestal) instead of the map grid")
    a = ap.parse_args()

    today = datetime.date.today()
    if a.days:
        days = [datetime.date.fromisoformat(s) for s in a.days.split(",")]
    else:
        sat = today + datetime.timedelta(days=(5 - today.weekday()) % 7)
        days = [sat, sat + datetime.timedelta(days=1)]
    horizon = today + datetime.timedelta(days=FORECAST_DAYS - 1)
    late = [d for d in days if d > horizon or d < today]
    if late:
        print(f"Days outside today..{horizon} (the forecast): {', '.join(map(str, late))}")
        return 2
    wanted = {s.strip() for s in a.species.split(",") if s.strip()}

    if a.public:
        print(f"Public and community forests within {a.radius:.0f} km of {a.lat},{a.lon} (ICNF REFLOA)...")
        forests = [f for f in landcover.public_forests()
                   if haversine_km(a.lat, a.lon, f["lat"], f["lon"]) <= a.radius and f["ha"] >= a.min_ha]
        polys, patches = {}, []
        for f in sorted(forests, key=lambda f: -f["ha"])[:a.limit]:
            found = landcover.forest_habitats(f.pop("geometry"))
            if not found:
                print(f"  {f['name']}: no mushroom habitat at any sampled point; skipped")
                continue
            # weather at the largest tree patch (not scrub) where there is one
            trees = {h: v for h, v in found.items() if h != "grassland"} or found
            main = max(trees.values(), key=lambda v: v["ha"])
            patches.append({"label": main["label"], "code": "", "lat": main["lat"], "lon": main["lon"],
                            "habitats": list(found), "public": {**f, "patches": found}, "ha": f["ha"]})
    else:
        print(f"Land cover within {a.radius:.0f} km of {a.lat},{a.lon} (COS2025, every {a.step} km)...")
        polys = landcover.scan(a.lat, a.lon, a.radius, a.step)
        if a.habitats:
            keep = set(a.habitats.split(","))
            polys = {k: v for k, v in polys.items() if keep & set(v["habitats"])}
        patches = landcover.pick_patches(polys, min_ha=a.min_ha, spacing_km=a.spacing, limit=a.limit)
        here = landcover.cos_at(a.lat, a.lon)
        if here and here["habitats"] and all(p["fid"] != here["fid"] for p in patches):
            patches.insert(0, {**here, "lat": a.lat, "lon": a.lon})

    rows = []
    for p in patches:
        site = ForagingSite(name=p["label"], lat=p["lat"], lon=p["lon"], habitats=p["habitats"])
        w = fetch_site_weather(site, today)
        time.sleep(0.3)
        scores = {}
        if w.ok:
            for sp in site.species:
                if wanted and sp.name not in wanted:
                    continue
                if not any(d.month in sp.season for d in days):
                    continue
                best = max((score_species(sp, w, d, d.month) for d in days), key=lambda s: s.score)
                scores[sp.name] = {"score": best.score, "status": best.latency_status,
                                   "why": best.recommendation}
        rows.append({"lat": p["lat"], "lon": p["lon"], "cos_class": p["label"], "cos_code": p["code"],
                     "ha": round(p["ha"]), "habitats": p["habitats"],
                     "km": round(haversine_km(a.lat, a.lon, p["lat"], p["lon"]), 1),
                     "dir": bearing(a.lat, a.lon, p["lat"], p["lon"]),
                     "public": p.get("public"),
                     "weather_ok": w.ok, "rain_source": w.rain_source, "scores": scores,
                     "best": max((v["score"] for v in scores.values()), default=0)})

    rows.sort(key=lambda r: (-r["best"], r["km"]))
    print(f"\nBest patches for {', '.join(d.strftime('%a %d %b') for d in days)} "
          f"(model {MODEL_VERSION}; a point inside each patch):\n")
    for r in rows:
        print(f"{r['lat']:.4f}, {r['lon']:.4f}  {r['km']:4.1f} km {r['dir']:2s}  {r['ha']:5d} ha  "
              f"{r['cos_class']}  [{', '.join(HABITAT_LABELS[h] for h in r['habitats'])}]")
        if r["public"]:
            pf = r["public"]
            print(f"      {pf['name']} — {pf['type']}; owner: {pf['owner']}; managed by {pf['manager']}")
            for h, v in pf.get("patches", {}).items():
                print(f"        {HABITAT_LABELS[h]:26s} {v['lat']:.4f}, {v['lon']:.4f}  ({v['label']}, {v['ha']:.0f} ha)")
        if not r["weather_ok"]:
            print("      no weather data; nothing scored")
        for name, v in sorted(r["scores"].items(), key=lambda kv: -kv[1]["score"])[:6]:
            print(f"      {v['score']:3d}  {name:24s} {v['why']}")

    tag = ("public_" if a.public else "") + (a.habitats.replace(",", "-") + "_" if a.habitats else "")
    out = os.path.join(MUSHROOM_DIR, "runs", f"spots_{tag}{today}_{a.lat:.4f}_{a.lon:.4f}.json")
    with open(out, "w") as f:
        json.dump({"model_version": MODEL_VERSION, "generated": today.isoformat(),
                   "point": [a.lat, a.lon], "radius_km": a.radius, "step_km": a.step,
                   "days": [d.isoformat() for d in days], "source": "DGT COS2025v1 WMS " + landcover.COS_LAYER,
                   "polygons_seen": len(polys), "patches": rows}, f, ensure_ascii=False, indent=1)
    print(f"\nRecord: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
