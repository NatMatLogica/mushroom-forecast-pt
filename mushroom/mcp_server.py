"""Private (key-only) MCP server: mushroom forecast for mainland Portugal.

Tools: species, forecast (one point), public_forests (state, municipal and community forests near a point).
Weather is ECMWF open data only (CC BY 4.0, commercial use allowed), refreshed in-process once a day; trees come
from the official COS 2025 map (one live lookup per forecast call); public forests from a table shipped with the
server (build_public_forests.py). The model code stays on the server; answers carry scores, never code.

Run locally:  MUSHROOM_WEATHER=ecmwf MUSHROOM_FORESTS=data/public_forests.json \\
              python -I mushroom/mcp_server.py
Env: MUSHROOM_KEYS (required, comma-separated access keys), MUSHROOM_FORESTS (required), MCP_HOST (127.0.0.1), MCP_PORT (3003), MCP_RATE_LIMIT (requests/min per client, 20),
     MUSHROOM_REFRESH_UTC (hour of the daily weather refresh, 9)
"""
import datetime
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict, deque

os.environ.setdefault("MUSHROOM_WEATHER", "ecmwf")
if os.environ["MUSHROOM_WEATHER"] != "ecmwf":
    raise SystemExit("the public server uses ECMWF open data only (MUSHROOM_WEATHER=ecmwf)")
if not os.environ.get("MUSHROOM_FORESTS"):
    raise SystemExit("MUSHROOM_FORESTS must point at the shipped public-forest table")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mushroom import scoring as S, landcover as L  # noqa: E402
from mushroom.species_db import SPECIES, HABITAT_LABELS  # noqa: E402
from mushroom.sites import ForagingSite, haversine_km  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.transport_security import TransportSecuritySettings  # noqa: E402

SITE = "https://fairvalueof.com/mushrooms/"
PT = (36.9, 42.2, -9.6, -6.1)
with open(os.environ["MUSHROOM_FORESTS"]) as fh:
    FORESTS = json.load(fh)["forests"]

SAFETY = ("A forecast is not an identification: never eat a wild mushroom on the strength of a forecast; have every "
          "find checked by someone who knows. Deadly species (death cap, Amanita phalloides) share the same woods. "
          "Most forest in Portugal is private: ask the owner. Public or community land is not automatically open for "
          "picking. Poison line (CIAV): 800 250 250.")
SOURCES = ("Weather: contains modified ECMWF open data (CC BY 4.0). Trees: COS 2025, Direção-Geral do Território "
           "(CC BY 4.0). Public forests: ICNF REFLOA. Timing fitted to GBIF occurrence records with MatLogica AADC.")

server = MCPServer("mushroom-forecast-portugal", website_url=SITE, instructions=(
    "Mushroom forecast for mainland Portugal by Raganele Consulting. For a point it says which trees grow there "
    "(official COS 2025 map) and how good today's and the next 7 days' weather is for each edible species that grows "
    "with those trees, as a 0-100 score. For porcini, Caesar's mushroom and honey mushroom the timing is fitted to "
    "thousands of dated public finds with MatLogica AADC. Start with species; forecast takes a latitude/longitude; "
    "public_forests lists state, municipal and community forests nearby with their trees and scores. "
    f"{SAFETY} More: {SITE}"))


def _today() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()


def _check_point(lat: float, lon: float):
    if not (PT[0] <= lat <= PT[1] and PT[2] <= lon <= PT[3]):
        raise ValueError("point must be in mainland Portugal (lat 36.9-42.2, lon -9.6 to -6.1)")


def _day(date: str | None) -> datetime.date:
    today = _today()
    if not date:
        return today
    d = datetime.date.fromisoformat(date)
    if not today <= d <= today + datetime.timedelta(days=S.FORECAST_DAYS - 1):
        raise ValueError(f"date must be between {today} and {today + datetime.timedelta(days=S.FORECAST_DAYS - 1)}")
    return d


def _scores(habitats, lat, lon, day):
    w = S.site_weather_ecmwf(lat, lon, _today(), wait=False)   # never builds on a request; the refresher does
    if not w.ok:
        raise RuntimeError("the daily weather is being prepared; try again in a few minutes")
    site = ForagingSite(name="point", lat=lat, lon=lon, habitats=habitats)
    out = []
    for sp in site.species:
        if day.month not in sp.season:
            continue
        r = S.score_species(sp, w, day, day.month)
        out.append({"species": sp.name, "common_name": sp.pt_name, "score": r.score, "outlook": r.recommendation,
                    "timing": "fitted to finds (AADC)" if sp.fitted else
                              ("published study rule" if sp.lagged else "rain-event rule")})
    return sorted(out, key=lambda x: -x["score"])


@server.tool()
def species() -> list:
    """Every edible species the forecast covers: name, common name, the trees it grows with, the months it is scored,
    how its timing is set, and safety notes. Species marked do-not-eat are listed for identification only."""
    rows = []
    for sp in SPECIES:
        rows.append({"species": sp.name, "common_name": sp.pt_name,
                     "trees": [HABITAT_LABELS[h] for h in sp.habitats], "months": sp.season,
                     "timing": "fitted to finds (AADC)" if sp.fitted else
                               ("published study rule" if sp.lagged else "rain-event rule"),
                     "edible": sp.edible, "notes": sp.notes})
    return rows


@server.tool()
def forecast(latitude: float, longitude: float, date: str | None = None) -> dict:
    """Forecast for one point in mainland Portugal: the trees there (COS 2025), the nearest public or community forest,
    and a 0-100 score for each in-season species that grows with those trees, for date (YYYY-MM-DD, today to 7 days
    ahead; default today). A score is how the day's weather compares with typical in-season days for that species."""
    lat, lon = float(latitude), float(longitude)
    _check_point(lat, lon)
    day = _day(date)
    c = L.cos_at(lat, lon)
    habitats = c["habitats"] if c else []
    near = min(FORESTS, key=lambda f: haversine_km(lat, lon, f["lat"], f["lon"]))
    km = haversine_km(lat, lon, near["lat"], near["lon"])
    return {"date": str(day), "point": [round(lat, 4), round(lon, 4)],
            "land_cover": c["label"] if c else "unknown", "trees": [HABITAT_LABELS[h] for h in habitats],
            "scores": _scores(habitats, lat, lon, day) if habitats else [],
            "note": None if habitats else "no mushroom habitat mapped at this point; try a nearby wood",
            "nearest_public_forest": {"name": near["name"], "type": near["type"], "owner": near["owner"],
                                      "km": round(km, 1)},
            "safety": SAFETY, "sources": SOURCES}


@server.tool()
def public_forests(latitude: float, longitude: float, radius_km: float = 50.0, date: str | None = None) -> dict:
    """State, municipal and community forests (ICNF Regime Florestal) within radius_km (max 100) of a point in
    mainland Portugal, each with its main tree patches (with coordinates) and the best species scores there for date."""
    lat, lon = float(latitude), float(longitude)
    _check_point(lat, lon)
    radius = max(1.0, min(100.0, float(radius_km)))
    day = _day(date)
    rows = []
    for f in FORESTS:
        km = haversine_km(lat, lon, f["lat"], f["lon"])
        if km > radius or not f["patches"]:
            continue
        trees = {h: v for h, v in f["patches"].items() if h != "grassland"} or f["patches"]
        main = max(trees.values(), key=lambda v: v["ha"])
        rows.append({"name": f["name"], "type": f["type"], "owner": f["owner"], "km": round(km, 1),
                     "area_ha": round(f["ha"]),
                     "patches": [{"trees": HABITAT_LABELS[h], "point": [v["lat"], v["lon"]], "map_class": v["label"]}
                                 for h, v in f["patches"].items()],
                     "best": _scores(list(f["patches"]), main["lat"], main["lon"], day)[:5]})
    rows.sort(key=lambda r: (-(r["best"][0]["score"] if r["best"] else 0), r["km"]))
    return {"date": str(day), "forests": rows[:25], "safety": SAFETY, "sources": SOURCES}


def _refresher():
    hour = int(os.environ.get("MUSHROOM_REFRESH_UTC", "9"))
    while True:
        try:
            S.ecmwf_grids(_today())
        except Exception as e:  # noqa: BLE001
            print(f"weather refresh failed: {e}", flush=True)
        now = datetime.datetime.now(datetime.timezone.utc)
        nxt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if nxt <= now:
            nxt += datetime.timedelta(days=1)
        time.sleep(max(60.0, (nxt - now).total_seconds()))


class KeyAuth:
    """Private service (key-only). A request needs one of the keys in MUSHROOM_KEYS,
    as "Authorization: Bearer <key>" or "?key=<key>" on the URL (for clients that cannot set headers)."""

    def __init__(self, app, keys):
        self.app, self.keys = app, {k.strip() for k in keys if k.strip()}
        if not self.keys:
            raise SystemExit("MUSHROOM_KEYS is empty: refusing to start an open server")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            import hmac
            from urllib.parse import parse_qs
            h = dict(scope.get("headers") or [])
            auth = h.get(b"authorization", b"").decode(errors="ignore")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            if not given:
                given = (parse_qs(scope.get("query_string", b"").decode(errors="ignore")).get("key") or [""])[0]
            if not any(hmac.compare_digest(given, k) for k in self.keys):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"text/plain"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b"This forecast is private: an access key is needed.\n"})
                return
        await self.app(scope, receive, send)


class RateLimit:
    """Per-client sliding window. The client is the first X-Forwarded-For address set by Caddy (the only way in)."""

    def __init__(self, app, per_min):
        self.app, self.per_min, self.hits = app, per_min, defaultdict(deque)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            h = dict(scope.get("headers") or [])
            ip = (h.get(b"x-forwarded-for", b"").split(b",")[0].strip() or (scope.get("client") or ("?",))[0])
            now, q = time.monotonic(), self.hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_min:
                await send({"type": "http.response.start", "status": 429,
                            "headers": [(b"content-type", b"text/plain"), (b"retry-after", b"60")]})
                await send({"type": "http.response.body", "body": b"Too many requests; try again in a minute.\n"})
                return
            q.append(now)
            if len(self.hits) > 50000:
                self.hits.clear()
        await self.app(scope, receive, send)


def app():
    sec = TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                    allowed_hosts=["fairvalueof.com", "www.fairvalueof.com", "127.0.0.1:*", "localhost:*",
                                                   "mushroom-mcp:*"],
                                    allowed_origins=["https://fairvalueof.com", "https://www.fairvalueof.com"])
    a = server.streamable_http_app(streamable_http_path="/mushrooms/mcp", stateless_http=True, json_response=True,
                                   max_request_body_size=65536, transport_security=sec)
    a = KeyAuth(a, os.environ.get("MUSHROOM_KEYS", "").split(","))
    return RateLimit(a, int(os.environ.get("MCP_RATE_LIMIT", "20")))


if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=_refresher, daemon=True).start()
    uvicorn.run(app(), host=os.environ.get("MCP_HOST", "127.0.0.1"), port=int(os.environ.get("MCP_PORT", "3003")),
                proxy_headers=False, log_level="warning")
