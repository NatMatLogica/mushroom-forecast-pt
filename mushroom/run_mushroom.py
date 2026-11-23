#!/usr/bin/env python3
"""
Mushroom foraging forecast — Setúbal region.

Fetches weather (Open-Meteo grid + nearby IPMA station observations), scores
every species at every site, and writes:

  mushroom-dashboard.html          the page (complete document, works offline)
  mushroom/runs/forecast_<date>.json   every input and output of the run (provenance)

Usage:
  python3 mushroom/run_mushroom.py                  # page + run record
  python3 mushroom/run_mushroom.py --json           # run record to stdout
  python3 mushroom/run_mushroom.py --status         # text summary
  python3 mushroom/run_mushroom.py --fragment PATH  # also write the page body
                                                    # without <html>/<head>/<body>
                                                    # (for hosts that wrap it)
"""

import dataclasses
import datetime
import html
import json
import math
import os
import sys
from typing import List

MUSHROOM_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPO_DIR = os.path.dirname(MUSHROOM_DIR)
sys.path.insert(0, SNAPO_DIR)

from mushroom.species_db import SPECIES, HABITAT_LABELS, get_species_in_season  # noqa: E402
from mushroom.sites import SITES, SETUBAL_LAT, SETUBAL_LON  # noqa: E402
from mushroom.scoring import (  # noqa: E402
    score_all_sites, SiteResult, SpeciesScore, MODEL_VERSION,
    LOOKBACK_DAYS, SIGNIFICANT_RAIN_MM,
)

esc = html.escape


def month_label(m: int) -> str:
    return datetime.date(2000, m, 1).strftime("%b")


def season_status(month: int) -> str:
    if month in (10, 11):
        return "Peak bolete season"
    if month in (9, 12, 1):
        return "Good season: boletes and chanterelles"
    if month in (2, 3):
        return "Late season: Lepista, horn of plenty"
    if month in (4, 5):
        return "Spring: morels possible, mostly too early"
    if month in (6, 7, 8):
        return "Off-season: summer is too dry. Watch for the first autumn rains."
    return f"{len(get_species_in_season(month))} species in season"


# ── Status semantics (one place) ───────────────────────────────────────

STATUS = {
    # key: (label, css class)
    "now":         ("GO",        "go"),
    "soon":        ("SOON",      "soon"),
    "forecast":    ("RAIN DUE",  "due"),
    "not_yet":     ("WAIT",      "wait"),
    "past_window": ("OVER",      "quiet"),
    "none":        ("QUIET",     "quiet"),
    "off_season":  ("OFF",       "quiet"),
    "nodata":      ("NO DATA",   "nodata"),
}


def species_status(ss: SpeciesScore) -> str:
    return ss.latency_status if ss.latency_status in STATUS else "none"


def site_status(r: SiteResult) -> str:
    """The site's chip is its best species' timing, with GO outranking everything."""
    if not r.weather.ok:
        return "nodata"
    order = ["now", "soon", "forecast", "not_yet", "past_window", "none"]
    present = {species_status(s) for s in r.species_scores if s.score > 0}
    for k in order:
        if k in present:
            return k
    return "none"


def chip(key: str) -> str:
    label, cls = STATUS[key]
    return f'<span class="chip chip-{cls}">{label}</span>'


# ── SVG: site map ──────────────────────────────────────────────────────

MAP_W, MAP_H = 420, 600
LON_MIN, LON_MAX = -9.50, -8.02
LAT_MIN, LAT_MAX = 37.82, 39.42
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * math.cos(math.radians(38.6))

LABEL_POS = {  # (dx, dy, anchor) relative to the dot
    "Serra da Arrábida NP":        (8, 16, "start"),
    "Palmela Cork Oak":            (10, 4, "start"),
    "Montijo / Alcochete":         (-10, -6, "end"),
    "Vale do Sorraia":             (10, 4, "start"),
    "Alcácer do Sal":              (10, 4, "start"),
    "Sines / Costa Norte":         (10, 4, "start"),
    "Sintra-Cascais NP":           (10, 4, "start"),
    "Vila Franca de Xira (west)":  (10, 4, "start"),
    "Torres Vedras Pine":          (10, 4, "start"),
    "Santarém Riparian":           (10, 4, "start"),
    "Montargil (Ponte de Sor)":    (-10, 4, "end"),
}


def project(lat: float, lon: float):
    pad = 28
    x = pad + (lon - LON_MIN) / (LON_MAX - LON_MIN) * (MAP_W - 2 * pad)
    y = pad + (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * (MAP_H - 2 * pad)
    return x, y


def site_map_svg(results: List[SiteResult]) -> str:
    parts = [f'<svg class="sitemap" viewBox="0 0 {MAP_W} {MAP_H}" role="img" '
             f'aria-label="Foraging sites around Setúbal, coloured by this week\'s status">']
    parts.append(f'<rect x="0" y="0" width="{MAP_W}" height="{MAP_H}" class="map-bg"/>')
    # graticule: whole and half degrees
    lat = 38.0
    while lat <= 39.5:
        _, y = project(lat, LON_MIN)
        if 0 < y < MAP_H:
            parts.append(f'<line x1="0" x2="{MAP_W}" y1="{y:.1f}" y2="{y:.1f}" class="grat"/>')
            parts.append(f'<text x="4" y="{y - 3:.1f}" class="grat-label">{lat:.1f}°N</text>')
        lat += 0.5
    lon = -9.5
    while lon <= -8.0:
        x, _ = project(LAT_MIN, lon)
        if 0 < x < MAP_W:
            parts.append(f'<line y1="0" y2="{MAP_H}" x1="{x:.1f}" x2="{x:.1f}" class="grat"/>')
            parts.append(f'<text x="{x + 3:.1f}" y="{MAP_H - 6}" class="grat-label">{abs(lon):.1f}°W</text>')
        lon += 0.5
    # 25 km scale bar
    km_px = (MAP_W - 56) / ((LON_MAX - LON_MIN) * KM_PER_DEG_LON)
    x0, y0 = MAP_W - 28 - 25 * km_px, MAP_H - 24
    parts.append(f'<line x1="{x0:.1f}" x2="{x0 + 25 * km_px:.1f}" y1="{y0}" y2="{y0}" class="scalebar"/>')
    parts.append(f'<text x="{x0 + 12.5 * km_px:.1f}" y="{y0 - 5}" class="grat-label" text-anchor="middle">25 km</text>')
    # Setúbal reference
    sx, sy = project(SETUBAL_LAT, SETUBAL_LON)
    parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" class="ref-dot"/>')
    parts.append(f'<text x="{sx - 9:.1f}" y="{sy + 4:.1f}" class="ref-label" text-anchor="end">Setúbal</text>')
    # sites
    for r in results:
        x, y = project(r.site.lat, r.site.lon)
        key = site_status(r)
        _, cls = STATUS[key]
        dx, dy, anchor = LABEL_POS.get(r.site.name, (10, 4, "start"))
        best = max((s.score for s in r.species_scores), default=0)
        parts.append(f'<g class="site-pin st-{cls}">'
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" class="pin"/>'
                     f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" class="pin-label">'
                     f'{esc(r.site.name)} <tspan class="pin-score">{best}%</tspan></text>'
                     f'<title>{esc(r.site.name)}: {STATUS[key][0]}, best score {best}%</title></g>')
    parts.append('</svg>')
    return "\n".join(parts)


# ── SVG: rain strip ────────────────────────────────────────────────────

def rain_strip_svg(r: SiteResult, today: datetime.date) -> str:
    w = r.weather
    # the fetch reaches HISTORY_DAYS back for the lagged species; the strip shows LOOKBACK_DAYS
    k0 = next((k for k, d in enumerate(w.dates) if (today - d).days <= LOOKBACK_DAYS), 0)
    w = dataclasses.replace(w, dates=w.dates[k0:], precip=w.precip[k0:], precip_grid=w.precip_grid[k0:],
                            tmin=w.tmin[k0:], tmax=w.tmax[k0:], rh_mean=w.rh_mean[k0:])
    n = len(w.dates)
    W, H = 360, 84
    top, base, lab = 14, 60, 76
    slot = W / n
    bar_w = slot - 2
    ymax = max(10.0, max(w.precip + w.precip_grid, default=0.0))
    ymax = math.ceil(ymax / 5.0) * 5.0
    obs_label = "observed" if w.ipma_station else "grid (past)"
    parts = [f'<svg class="rain" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Daily rainfall, {LOOKBACK_DAYS} days back and 7 days ahead">']
    # faint grid at ymax/2 and ymax
    for frac in (0.5, 1.0):
        y = base - frac * (base - top)
        parts.append(f'<line x1="0" x2="{W}" y1="{y:.1f}" y2="{y:.1f}" class="rain-grid"/>')
    parts.append(f'<text x="{W}" y="{top - 3}" class="rain-ymax" text-anchor="end">{ymax:.0f} mm</text>')
    for i, d in enumerate(w.dates):
        off = (d - today).days
        mm = w.precip[i]
        grid_mm = w.precip_grid[i]
        h = (base - top) * (mm / ymax) if ymax > 0 else 0
        x = i * slot + 1
        y = base - h
        cls = "bar past" if off < 0 else ("bar today" if off == 0 else "bar future")
        src = obs_label if off <= 0 else "forecast"
        if off <= 0 and w.ipma_station and abs(grid_mm - mm) >= 1.0:
            src += f"; grid said {grid_mm:.1f} mm"
        title = f"{d.strftime('%a %d %b')}: {mm:.1f} mm ({src})"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(h, 0):.1f}" '
                     f'rx="2" class="{cls}"><title>{esc(title)}</title></rect>')
        if mm >= SIGNIFICANT_RAIN_MM:
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 3:.1f}" class="rain-val" '
                         f'text-anchor="middle">{mm:.0f}</text>')
        if off == 0:
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{lab}" class="rain-day today-label" '
                         f'text-anchor="middle">today</text>')
        elif i % 2 == 0 or n <= 12:
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{lab}" class="rain-day" '
                         f'text-anchor="middle">{d.day}</text>')
    parts.append(f'<line x1="0" x2="{W}" y1="{base}" y2="{base}" class="rain-base"/>')
    parts.append('</svg>')
    return "\n".join(parts)


# ── Page ───────────────────────────────────────────────────────────────

CSS = """
/* Layout: one column, summary first; site cards ranked; reference tables last.
   Palette: lichen ground, humus dark, cap-ochre accent, moss for "go". */
:root {
  --bg: #eff1e8;
  --surface: #ffffff;
  --surface-2: #e4e8d9;
  --ink: #1f261c;
  --ink-2: #4f5a4c;
  --muted: #7a857a;
  --line: #d3d9c7;
  --accent: #a9661a;
  --accent-ink: #7d4a0f;
  --moss: #4c6b2f;
  --go: #2e7d3a;      --go-bg: #dff0dc;
  --soon: #a36400;    --soon-bg: #fbe9c7;
  --due: #2b6a9c;     --due-bg: #dbe9f5;
  --wait: #6b6f8a;    --wait-bg: #e6e7f0;
  --quiet: #7a857a;   --quiet-bg: #eceee6;
  --nodata: #8a4b4b;  --nodata-bg: #f3dede;
  --rain: #3b7dad;
  --rain-future: #a8c8e2;
  --map-bg: #e9ecdf;
  --font-display: "Hepta Slab", "Rockwell", "Roboto Slab", Georgia, serif;
  --font-body: "Atkinson Hyperlegible", "Verdana", "Helvetica Neue", Arial, sans-serif;
  --font-mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #151a13; --surface: #1e241b; --surface-2: #28301f;
    --ink: #e7eadf; --ink-2: #b9c1b2; --muted: #8e988b; --line: #353e30;
    --accent: #d99a3a; --accent-ink: #e8b665; --moss: #8fb15f;
    --go: #7ccb86;     --go-bg: #1f3a24;
    --soon: #e5a63a;   --soon-bg: #3d2f12;
    --due: #78b6e3;    --due-bg: #17304a;
    --wait: #a9aec9;   --wait-bg: #2b2d3f;
    --quiet: #9aa497;  --quiet-bg: #262c24;
    --nodata: #e09a9a; --nodata-bg: #3d2323;
    --rain: #7cb6e2; --rain-future: #3f5e78; --map-bg: #1a2018;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #151a13; --surface: #1e241b; --surface-2: #28301f;
  --ink: #e7eadf; --ink-2: #b9c1b2; --muted: #8e988b; --line: #353e30;
  --accent: #d99a3a; --accent-ink: #e8b665; --moss: #8fb15f;
  --go: #7ccb86;     --go-bg: #1f3a24;
  --soon: #e5a63a;   --soon-bg: #3d2f12;
  --due: #78b6e3;    --due-bg: #17304a;
  --wait: #a9aec9;   --wait-bg: #2b2d3f;
  --quiet: #9aa497;  --quiet-bg: #262c24;
  --nodata: #e09a9a; --nodata-bg: #3d2323;
  --rain: #7cb6e2; --rain-future: #3f5e78; --map-bg: #1a2018;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: var(--font-body); font-size: 16px; line-height: 1.5;
  padding-block: 20px 48px; padding-inline: 16px;
}
.wrap { max-width: 760px; margin: 0 auto; display: grid; gap: 28px; }
h1, h2, h3 { font-family: var(--font-display); text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: 1.9rem; font-weight: 800; }
h2 { font-size: 1.25rem; font-weight: 700; margin-bottom: 10px; }
h3 { font-size: 1.1rem; font-weight: 700; }
p { margin: 0; }
.eyebrow { font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.masthead { display: grid; gap: 6px; }
.season { color: var(--accent-ink); font-weight: 700; font-family: var(--font-display); font-size: 1.05rem; }
.asof { color: var(--ink-2); font-size: 0.92rem; }
.asof code { font-family: var(--font-mono); font-size: 0.85em; }

.verdict { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; display: grid; gap: 10px; }
.verdict .lead { font-size: 1.12rem; }
.verdict ul { margin: 0; padding-left: 1.1em; display: grid; gap: 4px; }

.chip { display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
  padding: 2px 8px; border-radius: 999px; line-height: 1.5; white-space: nowrap; vertical-align: middle; }
.chip-go     { color: var(--go);     background: var(--go-bg); }
.chip-soon   { color: var(--soon);   background: var(--soon-bg); }
.chip-due    { color: var(--due);    background: var(--due-bg); }
.chip-wait   { color: var(--wait);   background: var(--wait-bg); }
.chip-quiet  { color: var(--quiet);  background: var(--quiet-bg); }
.chip-nodata { color: var(--nodata); background: var(--nodata-bg); }

.map-wrap { max-width: 480px; }
.sitemap { width: 100%; height: auto; display: block; border-radius: 10px; border: 1px solid var(--line); }
.map-bg { fill: var(--map-bg); }
.grat { stroke: var(--line); stroke-width: 1; }
.grat-label { font-family: var(--font-mono); font-size: 9px; fill: var(--muted); }
.scalebar { stroke: var(--ink-2); stroke-width: 2; }
.ref-dot { fill: none; stroke: var(--ink); stroke-width: 2; }
.ref-label { font-size: 11px; fill: var(--ink-2); font-weight: 700; }
.pin { stroke: var(--surface); stroke-width: 2; }
.st-go .pin { fill: var(--go); }  .st-soon .pin { fill: var(--soon); } .st-due .pin { fill: var(--due); }
.st-wait .pin { fill: var(--wait); } .st-quiet .pin { fill: var(--quiet); } .st-nodata .pin { fill: var(--nodata); }
.pin-label { font-size: 11.5px; fill: var(--ink); paint-order: stroke; stroke: var(--map-bg); stroke-width: 3px; stroke-linejoin: round; }
.pin-score { font-family: var(--font-mono); fill: var(--ink-2); }
.map-key { display: flex; flex-wrap: wrap; gap: 8px 14px; margin-top: 8px; font-size: 0.85rem; color: var(--ink-2); }
.map-key span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
.map-key .k-go::before { background: var(--go); } .map-key .k-soon::before { background: var(--soon); }
.map-key .k-due::before { background: var(--due); } .map-key .k-wait::before { background: var(--wait); }
.map-key .k-quiet::before { background: var(--quiet); } .map-key .k-nodata::before { background: var(--nodata); }

.sites { display: grid; gap: 14px; }
.site { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; display: grid; gap: 10px; }
.site-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 10px; }
.site-head .km { font-family: var(--font-mono); font-size: 0.85rem; color: var(--muted); }
.site-head .chip { margin-left: auto; }
.habitat { color: var(--ink-2); font-size: 0.9rem; }
.site-rec { font-weight: 700; }
.rainfacts { display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: 0.9rem; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.rainfacts b { color: var(--ink); font-family: var(--font-mono); font-weight: 700; }
.source { font-size: 0.8rem; color: var(--muted); }
.rain { width: 100%; height: auto; display: block; }
.rain-grid { stroke: var(--line); stroke-width: 1; stroke-dasharray: 2 3; }
.rain-base { stroke: var(--line); stroke-width: 1; }
.rain-ymax, .rain-day, .rain-val { font-family: var(--font-mono); font-size: 9px; fill: var(--muted); }
.rain-val { fill: var(--ink-2); font-weight: 700; }
.today-label { fill: var(--accent-ink); font-weight: 700; }
.bar.past { fill: var(--rain); }
.bar.today { fill: var(--rain); stroke: var(--accent); stroke-width: 1.5; }
.bar.future { fill: var(--rain-future); }
.rain-legend { display: flex; gap: 14px; font-size: 0.78rem; color: var(--muted); }
.rain-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.rain-legend .l-obs i { background: var(--rain); } .rain-legend .l-fc i { background: var(--rain-future); }

.species-list { display: grid; gap: 6px; border-top: 1px solid var(--line); padding-top: 8px; }
.sp { display: grid; grid-template-columns: minmax(130px, 1.2fr) auto minmax(60px, 1fr) 42px; gap: 8px; align-items: center; font-size: 0.92rem; }
.sp .n { line-height: 1.2; } .sp .n i { display: block; color: var(--muted); font-size: 0.78rem; font-style: italic; }
.sp .why { grid-column: 1 / -1; color: var(--ink-2); font-size: 0.85rem; margin-top: -4px; }
.track { height: 8px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
.track > i { display: block; height: 100%; border-radius: 4px; background: var(--quiet); }
.sp.st-go .track > i { background: var(--go); } .sp.st-soon .track > i { background: var(--soon); }
.sp.st-due .track > i { background: var(--due); } .sp.st-wait .track > i { background: var(--wait); }
.pct { font-family: var(--font-mono); font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; }
.notes { color: var(--muted); font-size: 0.82rem; }
.nodata-box { color: var(--nodata); background: var(--nodata-bg); border-radius: 6px; padding: 8px 10px; font-size: 0.9rem; }

.tablewrap { overflow-x: auto; }
table.cal { border-collapse: collapse; font-size: 0.8rem; width: 100%; min-width: 520px; }
.cal th, .cal td { padding: 4px 5px; text-align: center; border-bottom: 1px solid var(--line); }
.cal th { font-weight: 700; color: var(--ink-2); font-family: var(--font-mono); font-size: 0.75rem; }
.cal td.name { text-align: left; white-space: nowrap; }
.cal td.name i { color: var(--muted); font-size: 0.75rem; }
.cal td.in { background: var(--surface-2); color: var(--moss); font-weight: 700; }
.cal th.cur, .cal td.cur { box-shadow: inset 2px 0 0 var(--accent), inset -2px 0 0 var(--accent); }
.cal td.in.cur { background: var(--go-bg); color: var(--go); }

.guide { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; display: grid; gap: 4px; font-size: 0.88rem; }
.card h3 { font-size: 1rem; } .card .latin { color: var(--muted); font-style: italic; font-size: 0.85rem; }
.card dl { margin: 6px 0 0; display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; font-variant-numeric: tabular-nums; }
.card dt { color: var(--muted); } .card dd { margin: 0; font-family: var(--font-mono); font-size: 0.82rem; }
.card .note { color: var(--ink-2); margin-top: 4px; }
.star { color: var(--accent); }

.footer { border-top: 1px solid var(--line); padding-top: 14px; color: var(--ink-2); font-size: 0.88rem; display: grid; gap: 8px; }
.footer strong { color: var(--nodata); }
.method { font-size: 0.82rem; color: var(--muted); }
a { color: var(--accent-ink); }
@media (max-width: 480px) {
  h1 { font-size: 1.55rem; }
  .sp { grid-template-columns: minmax(110px, 1fr) auto 36px; }
  .sp .track { grid-column: 1 / -1; }
}
@media (prefers-reduced-motion: no-preference) { .track > i { transition: width .4s ease; } }
"""


def build_page(results: List[SiteResult], today: datetime.date, generated_utc: str):
    month = today.month
    ok = [r for r in results if r.weather.ok]

    # ── Verdict ──
    go = [(r, s) for r in ok for s in r.species_scores if species_status(s) == "now" and s.score > 0]
    soon = [(r, s) for r in ok for s in r.species_scores if species_status(s) == "soon" and s.score > 0]
    due = [(r, s) for r in ok for s in r.species_scores if species_status(s) == "forecast" and s.score > 0]
    go.sort(key=lambda t: t[1].score, reverse=True)
    soon.sort(key=lambda t: (t[1].days_until_window, -t[1].score))
    due.sort(key=lambda t: (t[1].days_until_window, -t[1].score))

    best_site = ok[0] if ok else None
    lines = []
    if go:
        r, s = go[0]
        obs = "observed at the nearest station" if r.weather.ipma_station else "grid estimate, no station within 15 km"
        lines.append(f"<p class=\"lead\">Best bet: <b>{esc(s.species.pt_name)}</b> at <b>{esc(r.site.name)}</b> "
                     f"({r.site.distance_km:.0f} km), score {s.score}%. {esc(s.recommendation)}; "
                     + (f"{s.event_mm:.0f} mm fell {s.days_since_rain} days ago ({obs}).</p>"
                        if s.days_since_rain is not None else f"Rain {obs}.</p>"))
        others = []
        seen = {r.site.name}
        for r2, s2 in go[1:]:
            if r2.site.name in seen:
                continue
            seen.add(r2.site.name)
            tag = "" if r2.weather.ipma_station else ", grid estimate, no station nearby"
            others.append(f"{esc(s2.species.pt_name)} at {esc(r2.site.name)} ({s2.score}%{tag})")
            if len(others) == 3:
                break
        if others:
            lines.append("<p>Also open now: " + "; ".join(others) + ".</p>")
    elif best_site is not None:
        lines.append(f"<p class=\"lead\">Nothing is in its harvest window today. "
                     f"Highest score is {best_site.max_score}% at <b>{esc(best_site.site.name)}</b>.</p>")
    if soon:
        r, s = soon[0]
        lines.append(f"<p>Next up: <b>{esc(s.species.pt_name)}</b> at {esc(r.site.name)} in "
                     f"{s.days_until_window} day{'s' if s.days_until_window != 1 else ''}.</p>")
    if due:
        r, s = due[0]
        lines.append(f"<p>{esc(s.recommendation)} for {esc(s.species.pt_name)} at {esc(r.site.name)}.</p>")
    fc_rain = max((r.rain_forecast_mm for r in ok), default=0.0)
    if ok and fc_rain < 1.0:
        lines.append("<p>No rain in the 7-day forecast at any site. Anything not already triggered will not be before the next front.</p>")
    elif ok:
        wettest = max(ok, key=lambda r: r.rain_forecast_mm)
        lines.append(f"<p>Wettest forecast: {wettest.rain_forecast_mm:.0f} mm at {esc(wettest.site.name)} over the next 7 days.</p>")
    nodata = [r for r in results if not r.weather.ok]
    if nodata:
        lines.append("<p>No weather data for: " + ", ".join(esc(r.site.name) for r in nodata) + ". Those sites are unscored.</p>")

    stations = sorted({(r.weather.ipma_station["name"]) for r in ok if r.weather.ipma_station})
    last_obs = max((r.weather.ipma_last_obs for r in ok if r.weather.ipma_last_obs), default="")
    src_line = (f"Past rainfall from IPMA stations ({', '.join(esc(s) for s in stations)}; last observation "
                f"{esc(last_obs.replace('T', ' '))} UTC) where one sits within 15 km of a site, "
                f"otherwise the Open-Meteo grid. Forecast from Open-Meteo.") if stations else \
               "All rainfall from the Open-Meteo grid (no IPMA station within 15 km of any site)."

    # ── Site cards ──
    cards = []
    for r in results:
        site = r.site
        key = site_status(r)
        head = (f'<div class="site-head"><h3>{esc(site.name)}</h3>'
                f'<span class="km">{site.distance_km:.0f} km</span>{chip(key)}</div>'
                f'<div class="habitat">{esc(site.habitat_labels)}</div>')
        if not r.weather.ok:
            cards.append(f'<article class="site">{head}<div class="nodata-box">No weather data: '
                         f'{esc(r.weather.error)}</div><div class="notes">{esc(site.notes)}</div></article>')
            continue
        w = r.weather
        facts = (f'<div class="rainfacts"><span>Last {LOOKBACK_DAYS} days <b>{r.rain_past_mm:.1f} mm</b></span>'
                 f'<span>Next 7 days <b>{r.rain_forecast_mm:.1f} mm</b></span></div>'
                 f'<div class="source">{esc(w.rain_source)}</div>')
        strip = rain_strip_svg(r, today)
        legend = ('<div class="rain-legend"><span class="l-obs"><i></i>past (observed or grid, see source)</span>'
                  '<span class="l-fc"><i></i>forecast</span></div>')
        rows = []
        for ss in sorted(r.species_scores, key=lambda s: s.score, reverse=True):
            if ss.score <= 0:
                continue
            if len(rows) == 6:
                break
            k = species_status(ss)
            _, cls = STATUS[k]
            why = ss.recommendation
            if ss.rain_met and ss.days_since_rain is not None:
                why += f" ({ss.event_mm:.0f} mm, {ss.days_since_rain} d ago"
                why += ")" if r.weather.ipma_station else ", grid estimate)"
            if not math.isnan(ss.avg_temp):
                why += f". Temp {ss.avg_temp:.0f} °C" + (" ok" if ss.temp_ok else " off-range")
            if not math.isnan(ss.humidity):
                why += f", RH {ss.humidity:.0f}%" + (" ok" if ss.humidity_ok else " low")
            rows.append(f'<div class="sp st-{cls}"><span class="n">{esc(ss.species.pt_name)}'
                        f'{" <span class=star>&#9733;</span>" if ss.species.premium else ""}'
                        f'<i>{esc(ss.species.name)}</i></span>{chip(k)}'
                        f'<span class="track"><i style="width:{ss.score}%"></i></span>'
                        f'<span class="pct">{ss.score}%</span>'
                        f'<span class="why">{esc(why)}</span></div>')
        splist = ('<div class="species-list">' + "".join(rows) + '</div>') if rows else \
                 '<div class="notes">No species scores above zero this week.</div>'
        cards.append(f'<article class="site">{head}<div class="site-rec">{esc(r.recommendation)}</div>'
                     f'{facts}{strip}{legend}{splist}<div class="notes">{esc(site.notes)}</div></article>')

    # ── Calendar ──
    cal_head = "".join(f'<th class="{"cur" if m == month else ""}">{month_label(m)}</th>' for m in range(1, 13))
    cal_rows = []
    for sp in SPECIES:
        cells = "".join(
            f'<td class="{"in " if m in sp.season else ""}{"cur" if m == month else ""}">'
            f'{"&#9679;" if m in sp.season else ""}</td>' for m in range(1, 13))
        cal_rows.append(f'<tr><td class="name">{esc(sp.pt_name)}'
                        f'{" <span class=star>&#9733;</span>" if sp.premium else ""} <i>{esc(sp.name)}</i></td>{cells}</tr>')

    # ── Species guide ──
    guide = []
    for sp in SPECIES:
        months = sp.season
        if len(months) == 12:
            season = "all year"
        else:
            season = f"{month_label(months[0])}–{month_label(months[-1])}"
        guide.append(
            f'<div class="card"><h3>{esc(sp.pt_name)}{" <span class=star>&#9733;</span>" if sp.premium else ""}</h3>'
            f'<div class="latin">{esc(sp.name)}</div>'
            f'<dl><dt>Season</dt><dd>{season}</dd>'
            f'<dt>Trigger</dt><dd>&ge;{sp.rain_min:g} mm / {sp.rain_window} d</dd>'
            f'<dt>Fruits</dt><dd>{sp.latency_min}–{sp.latency_max} d after, {sp.harvest_days} d window</dd>'
            f'<dt>Temp</dt><dd>{sp.temp_min:g}–{sp.temp_max:g} °C</dd>'
            f'<dt>Humidity</dt><dd>&ge;{sp.humidity_min:g}%</dd>'
            f'<dt>Habitat</dt><dd>{esc(", ".join(HABITAT_LABELS.get(h, h).split(" (")[0] for h in sp.habitats))}</dd></dl>'
            f'<div class="note">{esc(sp.notes)}</div></div>')

    head_frag = f"""<title>Setúbal Mushroom Forecast</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=Hepta+Slab:wght@700;800&display=swap">
<style>{CSS}</style>"""

    body_frag = f"""<main class="wrap">
<header class="masthead">
  <div class="eyebrow">Setúbal · Arrábida · Sado · Alentejo montado</div>
  <h1>Setúbal Mushroom Forecast</h1>
  <div class="season">{esc(season_status(month))}</div>
  <div class="asof">Week of {today.strftime('%A %d %B %Y')}. Run {esc(generated_utc)} · model v{MODEL_VERSION}</div>
</header>

<section class="verdict" aria-label="This week">
  <div class="eyebrow">This week</div>
  {"".join(lines)}
  <p class="source">{src_line}</p>
</section>

<section aria-label="Map">
  <h2>Where</h2>
  <div class="map-wrap">{site_map_svg(results)}
  <div class="map-key"><span class="k-go">Go</span><span class="k-soon">Soon</span><span class="k-due">Rain due</span>
  <span class="k-wait">Wait</span><span class="k-quiet">Quiet</span><span class="k-nodata">No data</span></div></div>
</section>

<section aria-label="Sites">
  <h2>Sites, best first</h2>
  <div class="sites">{"".join(cards)}</div>
</section>

<section aria-label="Calendar">
  <h2>Season calendar</h2>
  <div class="tablewrap"><table class="cal"><thead><tr><th class="name"></th>{cal_head}</tr></thead>
  <tbody>{"".join(cal_rows)}</tbody></table></div>
</section>

<section aria-label="Species guide">
  <h2>Species and their triggers</h2>
  <div class="guide">{"".join(guide)}</div>
</section>

<footer class="footer">
  <p><strong>Safety:</strong> never eat a mushroom you cannot identify with certainty. Poisoning: call 112, or CIAV 800 250 250.</p>
  <p>Personal picking limit in Portugal is generally 2 kg per day. Most woodland is private: do not cross fences or ignore signs. Carry finds in a breathable basket.</p>
  <p class="method">Scores are weather-based, not observations: rainfall trigger 40 pts, days-since-rain window 30, temperature 20, humidity 10,
  per species and per site. A score says the weather fits; it does not say the mushrooms are there. Species thresholds come from
  First Nature (Algarve), FungiAtlas and the cogumelos.folkup.fit Setúbal guide.</p>
</footer>
</main>"""
    return head_frag, body_frag


def run_record(results: List[SiteResult], today: datetime.date, generated_utc: str) -> dict:
    sites = []
    for r in results:
        w = r.weather
        sites.append({
            "site": r.site.name, "lat": r.site.lat, "lon": r.site.lon,
            "distance_km": round(r.site.distance_km, 1),
            "data_ok": w.ok, "error": w.error,
            "rain_source": w.rain_source, "ipma_station": w.ipma_station,
            "ipma_last_obs_utc": w.ipma_last_obs, "fetched_utc": w.fetched_utc,
            "dates": [d.isoformat() for d in w.dates],
            "precip_mm": w.precip, "precip_grid_mm": w.precip_grid,
            "tmin_c": w.tmin, "tmax_c": w.tmax, "rh_mean_pct": w.rh_mean,
            "rain_past_mm": r.rain_past_mm, "rain_forecast_mm": r.rain_forecast_mm,
            "max_score": r.max_score, "status": site_status(r), "recommendation": r.recommendation,
            "species": [{
                "name": s.species.name, "pt_name": s.species.pt_name, "score": s.score,
                "status": species_status(s), "rain_met": s.rain_met, "rain_forecast": s.rain_forecast,
                "days_since_rain": s.days_since_rain, "event_mm": s.event_mm,
                "days_until_window": s.days_until_window, "temp_ok": s.temp_ok,
                "humidity_ok": s.humidity_ok, "avg_temp_c": None if math.isnan(s.avg_temp) else round(s.avg_temp, 1),
                "rh_pct": None if math.isnan(s.humidity) else round(s.humidity, 1),
                "recommendation": s.recommendation,
            } for s in sorted(r.species_scores, key=lambda x: x.score, reverse=True)],
        })
    return {"generated_utc": generated_utc, "model_version": MODEL_VERSION,
            "today": today.isoformat(), "lookback_days": LOOKBACK_DAYS, "sites": sites}


def print_text_summary(results: List[SiteResult], today: datetime.date):
    print(f"\nSetúbal Mushroom Forecast, {today.isoformat()}: {season_status(today.month)}\n")
    for r in results:
        site = r.site
        print("=" * 64)
        print(f"  {site.name} ({site.distance_km:.0f} km) [{STATUS[site_status(r)][0]}] {site.habitat_labels}")
        if not r.weather.ok:
            print(f"  NO DATA: {r.weather.error}")
            continue
        print(f"  rain past {LOOKBACK_DAYS}d {r.rain_past_mm:.1f} mm | next 7d {r.rain_forecast_mm:.1f} mm | {r.weather.rain_source}")
        print(f"  {r.recommendation}")
        for ss in sorted(r.species_scores, key=lambda s: s.score, reverse=True)[:5]:
            if ss.score > 0:
                print(f"    {ss.species.pt_name:28s} {ss.score:3d}%  {STATUS[species_status(ss)][0]:8s} {ss.recommendation}")
        print()


def main():
    mode = "html"
    if "--json" in sys.argv:
        mode = "json"
    elif "--status" in sys.argv:
        mode = "text"
    fragment = None
    if "--fragment" in sys.argv:
        fragment = sys.argv[sys.argv.index("--fragment") + 1]

    today = datetime.date.today()
    generated_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"Scoring mushroom conditions for {today.isoformat()}...")
    results = score_all_sites(today.month, today)
    record = run_record(results, today, generated_utc)

    if mode == "json":
        print(json.dumps(record, indent=2))
        return
    if mode == "text":
        print_text_summary(results, today)
        return

    runs_dir = os.path.join(MUSHROOM_DIR, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    rec_path = os.path.join(runs_dir, f"forecast_{today.isoformat()}.json")
    with open(rec_path, "w") as f:
        json.dump(record, f, indent=1)

    head_frag, body_frag = build_page(results, today, generated_utc)
    out_path = os.path.join(SNAPO_DIR, "mushroom-dashboard.html")
    with open(out_path, "w") as f:
        f.write('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                + head_frag + '\n</head>\n<body>\n' + body_frag + '\n</body>\n</html>\n')
    print(f"\nDashboard written to {out_path}")
    print(f"Run record written to {rec_path}")
    if fragment:
        with open(fragment, "w") as f:
            f.write(head_frag + "\n" + body_frag + "\n")
        print(f"Fragment written to {fragment}")


if __name__ == "__main__":
    main()
