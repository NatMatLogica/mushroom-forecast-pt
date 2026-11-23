#!/usr/bin/env python3
"""
Discovery mode: likely-accessible, never-recorded spots with the right trees and the best weather.

A candidate is a sample point of the official tree map (COS 2025) with a mushroom habitat. It is kept when ALL hold:
  * access: inside a public or community forest (ICNF REFLOA), or within TRACK_M of a public track or path
    (OpenStreetMap highway=track|path|unclassified|service|bridleway, excluding access=private|no);
  * no fire: not inside an area burnt in the last BURN_YEARS years (ICNF burnt areas, yearly layers to 2025);
  * new: no GBIF record of any target species within NOVEL_KM;
then ranked by the best forecast score among in-season species that grow with its trees.

What it cannot see: fences and gates that OpenStreetMap does not map (rural fences in Portugal are mapped patchily),
local rules (protected areas, baldios), slope, soil and tree age. A track on the map is not permission to pick.

Usage: python3 mushroom/discovery.py LAT LON --radius 10 --date 2026-11-23 [--species "Amanita caesarea,..."]
Writes runs/discovery_<date>_<lat>_<lon>.json.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

MUSHROOM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MUSHROOM_DIR))

from mushroom import landcover as L  # noqa: E402
from mushroom import scoring as S  # noqa: E402
from mushroom.replay_gbif import gbif_records  # noqa: E402
from mushroom.sites import ForagingSite, haversine_km  # noqa: E402
from mushroom.species_db import SPECIES, HABITAT_LABELS  # noqa: E402

TRACK_M = 200.0        # fixed — "next to a public track": a short walk in
BURN_YEARS = 10        # fixed — fire damage to the underground fungus lasts years; 10 is a cautious cut
NOVEL_KM = 1.0         # fixed — "no recorded find nearby"
SPACING_KM = 1.5       # fixed — one spot per wood section in the list
UA = "mushroom-forecast-pt/1.0"
BURN_LAYERS = {2025: 20, 2024: 19, 2023: 18, 2022: 17, 2021: 15, 2020: 0, 2019: 1, 2018: 2, 2017: 3, 2016: 4}
BURN_URL = "https://sigservices.icnf.pt/server/rest/services/BDG/areas_ardidas/FeatureServer/{layer}/query"
OVERPASS = "https://overpass-api.de/api/interpreter"
DEFAULT_SPECIES = ["Boletus sect. Boletus", "Boletus edulis", "Boletus reticulatus", "Boletus aereus",
                   "Armillaria mellea", "Amanita caesarea"]


def _get_json(url, data=None, timeout=90):
    """GET/POST JSON with backoff on busy servers (Overpass and ICNF answer 429/502/503/504 under load)."""
    import urllib.error
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
        time.sleep(10 * 2 ** attempt)


def bbox(lat, lon, km):
    dlat = km / 111.0
    dlon = km / (111.0 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def burnt_polygons(box, since_year):
    """ICNF burnt-area rings (lon, lat) intersecting box, for every year >= since_year."""
    s, w, n, e = box
    rings = []
    for year, layer in BURN_LAYERS.items():
        if year < since_year:
            continue
        q = urllib.parse.urlencode({"where": "1=1", "geometry": f"{w},{s},{e},{n}",
                                    "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                                    "spatialRel": "esriSpatialRelIntersects", "outFields": "Ano",
                                    "returnGeometry": "true", "outSR": 4326, "f": "geojson"})
        d = _get_json(BURN_URL.format(layer=layer) + "?" + q)
        for f in d.get("features", []):
            g = f.get("geometry") or {}
            polys = g.get("coordinates", []) if g.get("type") == "MultiPolygon" else [g.get("coordinates", [])]
            for p in polys:
                if p:
                    rings.append((year, p[0]))
        time.sleep(0.2)
    return rings


def public_tracks(box):
    """Node coordinates (lat, lon) of public tracks and paths in box (OpenStreetMap via Overpass)."""
    s, w, n, e = box
    q = (f'[out:json][timeout:80];way["highway"~"^(track|path|unclassified|service|bridleway)$"]'
         f'["access"!~"^(private|no)$"]({s},{w},{n},{e});out geom;')
    d = _get_json(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode())
    return [(p["lat"], p["lon"]) for el in d.get("elements", []) for p in el.get("geometry", [])]


def nearest_m(lat, lon, pts):
    if not pts:
        return float("inf")
    return 1000.0 * min(haversine_km(lat, lon, a, b) for a, b in pts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("lat", type=float)
    ap.add_argument("lon", type=float)
    ap.add_argument("--radius", type=float, default=10.0, help="km, max 20")
    ap.add_argument("--step", type=float, default=1.0, help="tree-map sampling, km")
    ap.add_argument("--date", default="", help="YYYY-MM-DD within the 7-day forecast (default today)")
    ap.add_argument("--species", default=",".join(DEFAULT_SPECIES))
    ap.add_argument("--top", type=int, default=4, help="spots per species")
    a = ap.parse_args()
    radius = min(20.0, a.radius)
    today = datetime.date.today()
    day = datetime.date.fromisoformat(a.date) if a.date else today
    wanted = {s.strip() for s in a.species.split(",") if s.strip()}
    box = bbox(a.lat, a.lon, radius)

    print(f"Tree map within {radius:.0f} km (every {a.step} km)...", flush=True)
    pts = []
    for la, lo in L.grid_points(a.lat, a.lon, radius, a.step):
        try:
            c = L.cos_at(la, lo)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {la},{lo}: {e}")
            continue
        if c and c["habitats"]:
            pts.append({"lat": la, "lon": lo, **c})
        time.sleep(0.12)
    print(f"  {len(pts)} habitat points; fires since {today.year - BURN_YEARS + 1}...", flush=True)
    rings = burnt_polygons(box, today.year - BURN_YEARS + 1)
    print(f"  {len(rings)} burnt areas; public tracks...", flush=True)
    tracks = public_tracks(box)
    print(f"  {len(tracks)} track points; public forests and recorded finds...", flush=True)
    forests = [f for f in L.public_forests()
               if haversine_km(a.lat, a.lon, f["lat"], f["lon"]) <= radius + 15]
    finds = [r for sp in wanted if sp in {s.name for s in SPECIES} and not sp.startswith("Boletus sect")
             for r in gbif_records(sp, "PT")
             if box[0] <= r["lat"] <= box[2] and box[1] <= r["lon"] <= box[3]]

    weather_cache = {}
    rows = []
    for p in pts:
        la, lo = p["lat"], p["lon"]
        burnt = next((y for y, ring in rings if L._inside(lo, la, ring)), None)
        if burnt:
            continue
        forest = next((f["name"] for f in forests if any(
            L._inside(lo, la, poly[0]) for poly in (f["geometry"]["coordinates"]
                                                    if f["geometry"]["type"] == "MultiPolygon"
                                                    else [f["geometry"]["coordinates"]]))), None)
        track_m = nearest_m(la, lo, tracks)
        if not forest and track_m > TRACK_M:
            continue
        near_find = min((haversine_km(la, lo, r["lat"], r["lon"]) for r in finds), default=float("inf"))
        if near_find < NOVEL_KM:
            continue
        key = (round(la, 1), round(lo, 1))
        if key not in weather_cache:
            weather_cache[key] = S.fetch_site_weather(ForagingSite(name="c", lat=la, lon=lo, habitats=p["habitats"]), today)
        w = weather_cache[key]
        if not w.ok:
            continue
        site = ForagingSite(name="c", lat=la, lon=lo, habitats=p["habitats"])
        scores = sorted(((S.score_species(sp, w, day, day.month).score, sp.name) for sp in site.species
                         if sp.name in wanted and day.month in sp.season), reverse=True)
        if not scores:
            continue
        rows.append({"lat": la, "lon": lo, "trees": [HABITAT_LABELS[h] for h in p["habitats"]], "map_class": p["label"],
                     "access": f"public forest: {forest}" if forest else f"public track {track_m:.0f} m away",
                     "nearest_recorded_find_km": None if near_find == float("inf") else round(near_find, 1),
                     "best": scores[:4], "top": scores[0][0]})
    rows.sort(key=lambda r: (-r["top"], r["access"].startswith("public track")))
    # per species: best spots by that species' own score, at least SPACING_KM apart, public forests first on ties
    per_species = {}
    for name in sorted(wanted):
        cand = [(next((sc for sc, n in r["best"] if n == name), None), r)
                for r in rows]
        cand = sorted([(sc, r) for sc, r in cand if sc is not None],
                      key=lambda t: (-t[0], t[1]["access"].startswith("public track")))
        picked = []
        for sc, r in cand:
            if all(haversine_km(r["lat"], r["lon"], q["lat"], q["lon"]) >= SPACING_KM for _, q in picked):
                picked.append((sc, r))
            if len(picked) == a.top:
                break
        if picked:
            per_species[name] = [{"score": sc, "lat": r["lat"], "lon": r["lon"], "trees": r["trees"],
                                  "access": r["access"], "nearest_recorded_find_km": r["nearest_recorded_find_km"]}
                                 for sc, r in picked]
    print(f"\nDiscovery spots for {day}: {len(rows)} of {len(pts)} habitat points pass all filters")
    for name, lst in per_species.items():
        print(f"\n{name}:")
        for x in lst:
            print(f"  {x['score']:3d}  {x['lat']:.4f}, {x['lon']:.4f}  {', '.join(x['trees'])} | {x['access']}")
    out = os.path.join(MUSHROOM_DIR, "runs", f"discovery_{day}_{a.lat:.4f}_{a.lon:.4f}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"date": str(day), "point": [a.lat, a.lon], "radius_km": radius,
                   "filters": {"track_m": TRACK_M, "burn_years": BURN_YEARS, "novel_km": NOVEL_KM},
                   "counts": {"habitat_points": len(pts), "burnt_areas": len(rings), "track_points": len(tracks),
                              "recorded_finds_in_box": len(finds), "passing": len(rows)},
                   "spots": rows, "per_species": per_species}, fh, ensure_ascii=False, indent=1)
    print(f"Record: {out}")


if __name__ == "__main__":
    main()
