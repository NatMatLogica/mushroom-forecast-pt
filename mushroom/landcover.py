"""
Tree cover for any point in mainland Portugal, from the official land-cover map.

Source: Carta de Uso e Ocupação do Solo 2025 (COS2025v1, Série 2), Direção-Geral do
Território. Free, no key ("Fees: No conditions apply"). Queried through the DGT WMS
GetFeatureInfo endpoint, layer `cos2025v1-s2`; the WFS and the OGC API listed on
dados.gov.pt were disabled / unreachable when this was written (2026-11-17).
Dataset page: https://dados.gov.pt/datasets/carta-de-uso-e-ocupacao-do-solo-cos-serie-2-2018v4-2025v1

COS level-4 classes are mapped onto the habitat keys in species_db by their
Portuguese label. Classes with no mushroom habitat in the model (rice, orchards,
built-up land, sea...) map to nothing, and so does "outras folhosas", which
mixes riparian and non-riparian broadleaves.
"""

import json
import math
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

from mushroom.species_db import CORK_OAK, HOLM_OAK, PINE, EUCALYPTUS, GRASSLAND, CHESTNUT
from mushroom.sites import ForagingSite, haversine_km

COS_WMS = "https://geo2.dgterritorio.gov.pt/geoserver/COS-S2/wms"
COS_LAYER = "cos2025v1-s2"
COS_PREFIX = "cos25v1"      # attribute prefix of this layer's level-1..4 code/label fields

# Substring of the lower-cased level-4 label -> habitat. First match wins per rule; a
# label can hit more than one rule (e.g. a mixed stand) and then gets every habitat.
LABEL_RULES: List[Tuple[str, str]] = [
    ("sobreiro", CORK_OAK),
    ("azinheira", HOLM_OAK),
    ("carvalh", HOLM_OAK),     # the model's HOLM_OAK is "Carvalhal": oak woodland in general
    ("castanheiro", CHESTNUT),
    ("pinheiro", PINE),
    ("resinosas", PINE),
    ("eucalipto", EUCALYPTUS),
    ("matos", GRASSLAND),      # the model's GRASSLAND is "Grassland / Maquis"
    ("pastagens espont", GRASSLAND),
]


def habitats_for_label(label: str) -> List[str]:
    lab = (label or "").lower()
    out: List[str] = []
    for key, hab in LABEL_RULES:
        if key in lab and hab not in out:
            out.append(hab)
    return out


def cos_at(lat: float, lon: float, timeout: float = 30.0) -> Optional[dict]:
    """The COS2025 polygon under (lat, lon): {fid, code, label, ha, habitats} or None."""
    h = 0.0004
    url = (f"{COS_WMS}?service=WMS&version=1.1.1&request=GetFeatureInfo"
           f"&layers={COS_LAYER}&query_layers={COS_LAYER}&srs=EPSG:4326"
           f"&bbox={lon - h},{lat - h},{lon + h},{lat + h}"
           f"&width=3&height=3&x=1&y=1&info_format=application/json")
    d = json.loads(urllib.request.urlopen(url, timeout=timeout).read())
    feats = d.get("features") or []
    if not feats:
        return None
    p = feats[0]["properties"]
    label = p.get(f"{COS_PREFIX}_n4_l", "")
    return {"fid": p.get("fid"), "code": p.get(f"{COS_PREFIX}_n4_c", ""), "label": label,
            "ha": float(p.get("Area_ha") or 0.0), "habitats": habitats_for_label(label)}


def site_at(name: str, lat: float, lon: float) -> ForagingSite:
    """A ForagingSite whose habitats come from the land-cover map. Raises if there are none."""
    c = cos_at(lat, lon)
    if c is None or not c["habitats"]:
        raise ValueError(f"No mushroom habitat at {lat},{lon}: COS2025 says "
                         f"{c['label'] if c else 'nothing'}")
    return ForagingSite(name=name, lat=lat, lon=lon, habitats=c["habitats"],
                        notes=f"COS2025: {c['label']} ({c['ha']:.0f} ha)")


def grid_points(lat0: float, lon0: float, radius_km: float, step_km: float) -> List[Tuple[float, float]]:
    """Square grid of points step_km apart inside a circle of radius_km."""
    dlat = step_km / 111.0
    dlon = step_km / (111.0 * math.cos(math.radians(lat0)))
    n = int(radius_km // step_km)
    pts = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            if math.hypot(i * step_km, j * step_km) <= radius_km:
                pts.append((round(lat0 + i * dlat, 5), round(lon0 + j * dlon, 5)))
    return pts


def scan(lat0: float, lon0: float, radius_km: float = 20.0, step_km: float = 2.5,
         pause_s: float = 0.15) -> Dict[int, dict]:
    """
    Sample the map on a grid; one entry per distinct polygon, keyed by fid, holding the
    first grid point that fell inside it. Sequential and paced: one public server.
    """
    polys: Dict[int, dict] = {}
    for la, lo in grid_points(lat0, lon0, radius_km, step_km):
        try:
            c = cos_at(la, lo)
        except Exception as e:  # noqa: BLE001 — a failed point is reported, not invented
            print(f"  [warn] COS query failed at {la},{lo}: {e}")
            continue
        if c is not None and c["fid"] not in polys:
            polys[c["fid"]] = {**c, "lat": la, "lon": lo}
        time.sleep(pause_s)
    return polys


def pick_patches(polys: Dict[int, dict], min_ha: float = 150.0, spacing_km: float = 5.0,
                 limit: int = 10) -> List[dict]:
    """Largest habitat polygons first, at most one per spacing_km (weather is the same inside)."""
    cand = sorted((p for p in polys.values() if p["habitats"] and p["ha"] >= min_ha),
                  key=lambda p: -p["ha"])
    picked: List[dict] = []
    for p in cand:
        if all(haversine_km(p["lat"], p["lon"], q["lat"], q["lon"]) > spacing_km for q in picked):
            picked.append(p)
        if len(picked) == limit:
            break
    return picked


# ── Public and community forest (fences) ─────────────────────────────
# ICNF "Regime Florestal e outras áreas" (REFLOA): national forests (MN), forest
# perimeters (PF, mostly baldios = community common land, or municipal) and other areas.
# Free WFS, no key. Dataset page: https://dados.gov.pt/datasets/regime-florestal-e-outras-areas-refloa
# Being in the Regime Florestal says who owns and manages the land, NOT that anyone may
# pick there: baldios belong to their local community and managers can set rules.
REFLOA_WFS = ("https://si.icnf.pt/wfs/refloa_rf?service=wfs&version=2.0.0&request=GetFeature"
              "&typeNames=BDG:refloa_rf&outputFormat=application/json&srsName=EPSG:4326")
TIPOLOGIA = {"MN": "Mata Nacional (state forest)", "PF": "Perímetro Florestal",
             "OAF": "other area under forest regime"}


def _ring_area_centroid(ring):
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        c = x0 * y1 - x1 * y0
        a += c
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    if abs(a) < 1e-15:
        return 0.0, ring[0][0], ring[0][1]
    return abs(a) / 2, cx / (3 * a), cy / (3 * a)


def _inside(x, y, ring) -> bool:
    ins = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            ins = not ins
    return ins


def interior_point(geom: dict) -> Optional[Tuple[float, float]]:
    """(lat, lon) inside the largest outer ring: its centroid if inside, else a grid search."""
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    rings = [p[0] for p in polys if p and len(p[0]) >= 4]
    if not rings:
        return None
    ring = max(rings, key=lambda r: _ring_area_centroid(r)[0])
    _, cx, cy = _ring_area_centroid(ring)
    if _inside(cx, cy, ring):
        return cy, cx
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    best = None
    for i in range(1, 20):
        for j in range(1, 20):
            x = min(xs) + (max(xs) - min(xs)) * i / 20
            y = min(ys) + (max(ys) - min(ys)) * j / 20
            if _inside(x, y, ring):
                dd = (x - cx) ** 2 + (y - cy) ** 2
                if best is None or dd < best[0]:
                    best = (dd, y, x)
    return (best[1], best[2]) if best else None


def public_forests(timeout: float = 120.0) -> List[dict]:
    """Every REFLOA area: {name, type, owner, manager, ha, lat, lon} (lat/lon = an interior point)."""
    d = json.loads(urllib.request.urlopen(REFLOA_WFS, timeout=timeout).read())
    out = []
    for f in d.get("features", []):
        p = f["properties"]
        pt = interior_point(f["geometry"]) if f.get("geometry") else None
        if pt is None:
            continue
        out.append({"geometry": f["geometry"], "name": p.get("nome"), "type": TIPOLOGIA.get(p.get("tipologia"), p.get("tipologia")),
                    "owner": p.get("proprieda"), "manager": p.get("ent_ges"),
                    "ha": float(p.get("area_ha") or 0.0), "lat": round(pt[0], 5), "lon": round(pt[1], 5)})
    return out


def interior_points(geom: dict, n: int = 12) -> List[Tuple[float, float]]:
    """Up to n (lat, lon) points spread on a grid inside the forest's outer rings."""
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    rings = [p[0] for p in polys if p and len(p[0]) >= 4]
    pts: List[Tuple[float, float]] = []
    for k in (4, 6, 9, 13, 19, 27):           # refine the grid until enough points fall inside
        pts = []
        for ring in rings:
            xs, ys = [p[0] for p in ring], [p[1] for p in ring]
            for i in range(1, k):
                for j in range(1, k):
                    x = min(xs) + (max(xs) - min(xs)) * i / k
                    y = min(ys) + (max(ys) - min(ys)) * j / k
                    if _inside(x, y, ring):
                        pts.append((round(y, 5), round(x, 5)))
        if len(pts) >= n:
            break
    if len(pts) > n:                          # keep an even spread
        step = len(pts) / n
        pts = [pts[int(i * step)] for i in range(n)]
    return pts


def forest_habitats(geom: dict, n: int = 12, pause_s: float = 0.15) -> Dict[str, dict]:
    """
    Sample the land-cover map at up to n points inside a forest. Returns, per habitat,
    the largest COS polygon found for it: {habitat: {label, ha, lat, lon}}.
    """
    found: Dict[str, dict] = {}
    for la, lo in interior_points(geom, n):
        try:
            c = cos_at(la, lo)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] COS query failed at {la},{lo}: {e}")
            continue
        time.sleep(pause_s)
        if c is None:
            continue
        for h in c["habitats"]:
            if h not in found or c["ha"] > found[h]["ha"]:
                found[h] = {"label": c["label"], "ha": c["ha"], "lat": la, "lon": lo}
    return found
