# Mushroom forecast for Portugal

Where and when to look for porcini, Caesar's mushroom and honey mushroom in Portugal, from the
weather of the last few weeks, the official tree map and public finds.

For any point in mainland Portugal it answers three questions:

1. **What trees grow there?** From the official land-cover map (COS 2025), down to the species:
   cork oak, holm and other oaks, chestnut, maritime and stone pine, eucalyptus.
2. **Is the weather right for each mushroom, today and over the next 7 days?** From the rain and
   temperature of the last 60 days and the forecast.
3. **Can I go in?** It can limit the search to public and community forests (national forests,
   municipal forests, *baldios*) from the forestry agency's map, or to woods next to a public track.

It can also look for **new spots** (discovery mode): the right trees, on public land or within 200 m of
a public track, not burnt in the last ten years, with the best weather, and more than 1 km from any
recorded find.

## How the timing was fitted

For five forecasts, the timing (how many weeks of rain matter, which temperature suits the species)
is **fitted to about 2,600 dated public finds** in Portugal and Spain (GBIF, mostly iNaturalist
research-grade records) and the historical weather at each find. The fit uses
**[AADC](https://matlogica.com)**, MatLogica's automatic adjoint differentiation engine: it gives
exact gradients through the whole model, including the lengths of the rain and temperature windows,
which are fitted as continuous numbers rather than picked from a grid.

The design compares each find day with other in-season days **at the same place**, so the habitat,
the observer and the place cancel out and only the weather timing is tested. The fit learns from
finds up to 2022 and is scored on Portuguese finds from 2023 on, which it never saw.

| Forecast | Held-out Portuguese finds | Fitted timing | Hand-set starting rule |
|---|---|---|---|
| Caesar's mushroom (*Amanita caesarea*) | 26 | **0.75** | 0.58 |
| Honey mushroom (*Armillaria mellea*) | 54 | **0.73** | 0.55 |
| Dark porcino (*Boletus aereus*) | 29 | **0.69** | 0.54 |
| Any porcino (*B. edulis*, *B. reticulatus*, *B. aereus* together) | 146 | **0.65** | 0.59 |
| Summer cep (*Boletus reticulatus*) | 73 | **0.63** | 0.56 |

The score is the chance that the forecast rates a real find day above a random in-season day at the
same place: 0.5 is a coin toss, 1.0 is perfect. All five fitted forecasts also beat chance on
held-out log-likelihood. Full numbers: `results/`.

**Porcino on its own** (*Boletus edulis*) uses a rule from a ten-year daily study of porcini
fruiting (Brejon Lamartinière & Hoffman 2025, bioRxiv, doi:10.64898/2025.12.12.693895): best
fruiting at a 20-day mean temperature near 13.7 °C, rising with 26 days of rain. Its independent fit
here lands on 14.0 °C with 20- and 18-day windows, matching the study, and scores 0.65 on held-out
finds against the rule's 0.67, so the rule stays.

What the fits learned, in plain words: Caesar's mushroom and the dark porcino follow about six weeks
of rain and warm days (about 17 °C); the summer cep and "any porcino" follow three to four weeks of
rain at about 16 °C; the honey mushroom follows the autumn cooling (about 15 °C over six weeks)
rather than any particular rain.

Each species' season (the months it is scored in) comes from the months holding at least 2% of its
Iberian finds.

## Quick start

```bash
# the forecast needs only Python 3.10+ and its standard library
python3 mushroom/spots.py 38.52 -8.89 --radius 20 --days 2026-11-21       # best patches near a point
python3 mushroom/spots.py 41.15 -8.61 --radius 100 --public               # public/community forests only
python3 mushroom/discovery.py 38.24 -8.67 --radius 8                      # never-recorded, reachable spots
python3 mushroom/run_mushroom.py                                          # offline dashboard page for 11 sites
python3 -m pytest -q mushroom/test_scoring.py
```

**Weather source.** By default the forecast reads the [Open-Meteo](https://open-meteo.com/) API, whose free
tier is for non-commercial use. Set `MUSHROOM_WEATHER=ecmwf` to use ECMWF open data instead (CC BY 4.0,
commercial use allowed); this needs `pip install eccodes numpy` and downloads only the fields it needs.

**As an MCP server.** `mushroom/mcp_server.py` serves three tools (species, forecast at a point, public
forests near a point) to AI assistants over streamable HTTP, with access keys and a rate limit. It needs
`pip install mcp uvicorn eccodes numpy` and `MUSHROOM_KEYS`, `MUSHROOM_FORESTS=data/public_forests.json`.

Refitting the timing (`mushroom/calibrate_aadc.py`) and its gradient tests
(`mushroom/test_calibrate_aadc.py`) need AADC: see [matlogica.com](https://matlogica.com).
The fitted settings are in `mushroom/fitted_params.json`, so the forecast runs without it.

## Files

| File | What it does |
|---|---|
| `mushroom/species_db.py` | species, habitats, seasons, safety notes |
| `mushroom/scoring.py` | weather fetch and scoring (fitted timing, research rule, simple rain rule) |
| `mushroom/landcover.py` | trees at a point (COS 2025), public and community forests (ICNF) |
| `mushroom/spots.py` | ranks the best patches around a point for chosen days |
| `mushroom/discovery.py` | never-recorded spots on public land or next to a public track, not recently burnt |
| `mushroom/weather_ecmwf.py` | daily weather from ECMWF open data (past runs and the latest forecast) |
| `mushroom/mcp_server.py` | the forecast as an MCP server for AI assistants (key-only) |
| `mushroom/build_public_forests.py` | builds `data/public_forests.json`: trees inside each public forest |
| `mushroom/run_mushroom.py` | offline dashboard page for a fixed list of sites |
| `mushroom/calibrate_aadc.py` | fits the timing to GBIF finds with AADC gradients |
| `mushroom/replay_gbif.py` | scores any version of the forecast against GBIF finds |
| `mushroom/fitted_params.json` | the fitted settings used by the forecast |
| `results/` | fit and replay summaries behind the table above |
| `data/public_forests.json` | public and community forests with their sampled tree patches |

## Safety

- **A forecast is not an identification.** Never eat a wild mushroom on the strength of this or any
  forecast. Have every find checked by someone who knows. Deadly species (death cap,
  *Amanita phalloides*) grow in the same oak and chestnut woods.
- **Yellow knight** (*Tricholoma equestre*) is marked do-not-eat and never recommended: repeated meals
  are linked to rhabdomyolysis, including deaths (PubMed 28356809, 14505933).
- **Honey mushroom** must be cooked thoroughly. **Caesar's mushroom** can be confused with deadly
  *Amanita* species at the egg stage.
- **Land and rules.** Most Portuguese forest is private: ask the owner. Public or community land is
  not automatically open for picking; protected areas and *baldios* can set their own rules. A track on
  the map is not permission to pick, and fences are mapped only patchily.
- In Portugal, the poison information line (CIAV) is 800 250 250.

## Data sources and attribution

- **Weather:** [Open-Meteo](https://open-meteo.com/) forecast and historical archive, CC BY 4.0 (free tier
  for non-commercial use); or ECMWF open data, CC BY 4.0: *contains modified ECMWF open data*.
- **Finds:** [GBIF.org](https://www.gbif.org/) occurrence data via the GBIF API; each record keeps its
  own dataset licence and credit (mostly iNaturalist research-grade observations).
- **Trees:** Carta de Uso e Ocupação do Solo 2025 (COS 2025), Direção-Geral do Território, CC BY 4.0.
  *Informação geográfica cedida pela Direção-Geral do Território.*
- **Public and community forests:** ICNF, Regime Florestal e outras áreas (REFLOA), open WFS.
- **Burnt areas:** ICNF, Áreas Ardidas (yearly maps), open feature service.
- **Tracks and paths:** © OpenStreetMap contributors, ODbL, via the Overpass API.
- **Optional station rain:** IPMA open-data observations (`mushroom/ipma_archive.db`, not included).
- Species notes: First Nature, FungiAtlas Portugal, cogumelos.folkup.fit (links in `species_db.py`).
- Caesar's mushroom habitats: checked against all 58 Portuguese GBIF records (pine 13, cork oak 9, other
  broadleaves 6, holm oak 2, chestnut 1, eucalyptus 1).

## Licence

PolyForm Noncommercial 1.0.0, see `LICENSE.md`: free for personal, research and other non-commercial use;
commercial use needs a separate licence.
