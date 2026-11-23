"""
Mushroom species database — Setúbal, Portugal region.

Fruiting conditions sourced from:
- First Nature Algarve: https://mail.first-nature.com/algarve/species-fungi.php
- FungiAtlas Portugal: https://fungiatlas.com/mushrooms-in-portugal/
- cogumelos.folkup.fit Setúbal guide: https://cogumelos.folkup.fit/en/docs/setubal/species-guide/
- FungiAtlas growth rates: https://fungiatlas.com/how-fast-do-mushrooms-grow-after-rain/
- Arrábida Guide: https://guidetoarrabida.pt/explore/flora/amanita-caesarea
- SearXNG boletus fruiting research (2026-06-28)

Each species has fruiting conditions (rainfall trigger, temperature range,
humidity preference, latency after rain, harvest window, habitat associations,
season months). Used by scoring.py to compute fruiting probability.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# Habitat types (used by both species and sites)
CORK_OAK = "cork_oak"        # Montado de sobro
HOLM_OAK = "holm_oak"        # Carvalhal
PINE = "pine"                # Pinhal
EUCALYPTUS = "eucalyptus"    # Eucaliptal
RIPARIAN = "riparian"        # Mata ribeirinha
GRASSLAND = "grassland"      # Pastizal / matagal
DECAYING_WOOD = "decaying_wood"  # Madeira em decomposição
CHESTNUT = "chestnut"        # Souto / castanheiro (added 2026-11-17 for the north)

HABITAT_LABELS = {
    CORK_OAK: "Cork Oak (Montado)",
    HOLM_OAK: "Holm Oak (Carvalhal)",
    PINE: "Pine (Pinhal)",
    EUCALYPTUS: "Eucalyptus (Eucaliptal)",
    RIPARIAN: "Riparian (Mata ribeirinha)",
    GRASSLAND: "Grassland / Maquis",
    DECAYING_WOOD: "Decaying Wood",
    CHESTNUT: "Chestnut (Souto)",
}

MONTHS = list(range(1, 13))  # 1=Jan, 12=Dec


@dataclass(frozen=True)
class LaggedPhenology:
    """
    Fruiting driven by weather integrated over weeks, not by one rain event.

    Scored by scoring.score_lagged(): the mean daily temperature over the last
    temp_days and the mean daily rainfall over the last rain_days, evaluated for
    today and for every forecast day, so the score also says WHEN.
    """
    temp_days: int             # window for the mean daily temperature
    rain_days: int             # window for the mean daily rainfall
    temp_band: tuple           # (lo, hi) °C, full temperature points inside
    temp_zero: tuple           # (lo, hi) °C, temperature points taper linearly to 0 at these
    rain_full_mm_day: float    # mean daily rain at which the rain points are full
    source: str = ""


# Brejon Lamartinière & Hoffman, bioRxiv 10.64898/2025.12.12.693895 v2 (DOI resolved
# 2026-11-17): 10 years of daily sporocarp counts, B. edulis, beech/oak woodland,
# Bielefeld, Germany. Best-fitting lags: 20-day mean temperature, 26-day mean
# precipitation. Temperature optimum 13.7 °C (all sites); fruiting days concentrated
# at 10-15 °C and 2-4 mm/day; fruiting largely absent at 5-10 °C; observed 20-day
# means spanned 5-20 °C; precipitation effect linear, no upper threshold found.
# CLASS TRANSFER: German beech/oak -> Portuguese cork oak / pine on sand. Not yet
# checked here first; calibrate_aadc.py then fits the timing to Iberian finds.
BOLETUS_EDULIS_LAGS = LaggedPhenology(
    temp_days=20, rain_days=26,
    temp_band=(10.0, 15.0),
    temp_zero=(5.0, 20.0),     # 5: "largely absent 5-10 °C"; 20: top of the observed range,
                               # beyond it the study has no data, so no points (fixed — no evidence)
    rain_full_mm_day=2.0,      # lower edge of the 2-4 mm/day fruiting band (fixed — linear ramp below it is a modelling choice)
    source="Brejon Lamartiniere & Hoffman 2025, bioRxiv 10.64898/2025.12.12.693895 v2",
)


@dataclass(frozen=True)
class Species:
    """One mushroom species with fruiting conditions."""
    name: str                  # Scientific name
    pt_name: str               # Portuguese common name
    habitats: List[str]        # Habitat types where this species fruits
    season: List[int]          # Months (1-12) when this species fruits
    rain_min: float            # Minimum rainfall (mm) over rain_window days to trigger fruiting
    rain_window: int           # Days over which rain_min is accumulated
    temp_min: float            # Minimum temperature (°C) for fruiting
    temp_max: float            # Maximum temperature (°C) for fruiting
    humidity_min: float        # Minimum relative humidity (%) preference (0-100)
    latency_min: int           # Minimum days after rain before first fruiting bodies appear
    latency_max: int           # Maximum days after rain for peak fruiting
    harvest_days: int          # How many days the harvest window lasts after latency
    premium: bool = False      # Is this a highly prized species? (for UI emphasis)
    notes: str = ""            # Additional notes for the dashboard
    image_url: str = ""        # External image URL (loaded by client browser, not server)
    lagged: Optional[LaggedPhenology] = None  # if set, scored on weeks-long windows, not a rain event
    edible: bool = True        # False = never recommended as a pick; kept only for identification
    fitted: str = ""           # key in fitted_params.json: scored by the AADC-calibrated timing (calibrate_aadc.py)


# ── Species database ──
# Season months for the five species fitted in calibrate_aadc.py come from GBIF finds in Portugal
# and Spain, 2015 onward: every month holding at least 2% of that species' finds (fixed rule).────────────────────────────────────────────────
# Ordered by premium status (prized first), then by typical season start.
# Images: cogumelos.folkup.fit (webp, Portugal-focused) preferred,
#   First Nature (jpg) as fallback, FungiAtlas (jpg) for gaps.

COG = "https://cogumelos.folkup.fit/images/species/"  # Portugal-focused, webp
FN = "https://www.first-nature.com/fungi/images/"      # First Nature, jpg
FA = "https://fungiatlas.com/wp-content/uploads/"       # FungiAtlas, jpg

SPECIES: List[Species] = [
    # ── Premium boletes (the kings) ──
    Species(
        name="Boletus edulis",
        pt_name="Porcino / Carrilhão",
        habitats=[CORK_OAK, HOLM_OAK, PINE, CHESTNUT],  # chestnut: FungiAtlas Portugal, read 2026-11-17
        season=[6, 7, 8, 9, 10, 11, 12],  # GBIF PT+ES 2015+: months with >=2% of finds (2026-11-17); was [9, 10, 11, 12, 1, 2]
        rain_min=20, rain_window=3,
        temp_min=15, temp_max=25,
        humidity_min=70,
        latency_min=3, latency_max=6,
        harvest_days=7,
        premium=True,
        notes="Thick bulbous stem, brown cap, spongy underside. Most prized edible in Portugal.",
        image_url=COG + "boletus-edulis.webp",
        lagged=BOLETUS_EDULIS_LAGS,   # rain_min/latency above are unused for this species since v0.3.0
    ),
    Species(
        name="Boletus aereus",
        fitted="Boletus aereus",        # held-out AUC 0.687 vs 0.545 for the rule (calibration 2026-11-17)
        pt_name="Porcino escuro / Rainha",
        habitats=[CORK_OAK, HOLM_OAK, CHESTNUT],  # chestnut: FungiAtlas Portugal, read 2026-11-17
        season=[4, 5, 6, 7, 8, 9, 10, 11, 12],  # GBIF PT+ES 2015+: months with >=2% of finds (2026-11-17); was [10, 11, 12, 1, 2]
        rain_min=20, rain_window=3,
        temp_min=12, temp_max=22,
        humidity_min=70,
        latency_min=3, latency_max=6,
        harvest_days=7,
        premium=True,
        notes="Dark brown to blackish cap, pinkish-orange tubes when young. Cork oak specialist.",
        image_url=COG + "boletus-aereus.webp",
    ),
    # ── Any porcino: the four ceps pooled (added 2026-11-17). Foragers rarely separate them and
    # photo IDs often cannot; fitted on all their Iberian finds together (held-out AUC 0.655, 146 finds).
    Species(
        name="Boletus sect. Boletus",
        pt_name="Any porcino (all ceps)",
        habitats=[CORK_OAK, HOLM_OAK, PINE, CHESTNUT],
        season=[4, 5, 6, 7, 8, 9, 10, 11, 12],   # union of the members' GBIF-derived seasons
        rain_min=20, rain_window=3, temp_min=12, temp_max=25, humidity_min=70,   # unused: fitted
        latency_min=3, latency_max=6, harvest_days=7,                            # unused: fitted
        premium=True,
        fitted="Any porcino (Boletus sect. Boletus)",
        notes="Porcino, summer cep or dark porcino: one forecast for any of them (fitted on their finds together).",
    ),
    # ── Summer cep (added 2026-11-17) ──
    # Habitat, season, range, "fruits a little earlier than B. edulis" and the ID notes:
    # First Nature, https://www.first-nature.com/fungi/boletus-reticulatus.php (read 2026-11-17).
    # No source gives quantitative fruiting triggers for this species. The rain/latency/
    # temperature/humidity values are a DECLARED CLASS TRANSFER of B. edulis's v0.2.0 event
    # parameters (same section Boletus, same guide-level sourcing as B. aereus above), chosen
    # over B. aereus's 12-22 °C because the source says it fruits earlier, i.e. warmer.
    Species(
        name="Boletus reticulatus",
        fitted="Boletus reticulatus",   # held-out AUC 0.626 vs 0.559 for the rule (calibration 2026-11-17)
        pt_name="Summer cep (B. reticulatus)",   # [TODO: verify Portuguese common name]
        habitats=[CORK_OAK, HOLM_OAK],
        season=[5, 6, 7, 8, 9, 10, 11],  # GBIF PT+ES 2015+: months with >=2% of finds (2026-11-17); was [6, 7, 8, 9, 10, 11, 12, 1, 2]
        rain_min=20, rain_window=3,              # class transfer: B. edulis v0.2.0
        temp_min=15, temp_max=25,                # class transfer: B. edulis v0.2.0
        humidity_min=70,                         # class transfer: B. edulis v0.2.0
        latency_min=3, latency_max=6,            # class transfer: B. edulis v0.2.0
        harvest_days=7,                          # class transfer: B. edulis v0.2.0
        premium=True,
        notes=("Pale tan to hazel matte cap, often finely cracked, no whitish margin. Brown net over "
               "most of the stem (B. edulis: white net near the top, whitish cap margin). Oaks; "
               "fruits a little earlier than B. edulis. Look-alike: bitter bolete, pinkish pores."),
    ),
    Species(
        name="Butyriboletus regius",
        pt_name="Carrilhão branco / Real",
        habitats=[CORK_OAK, HOLM_OAK],
        season=[10, 11, 12, 1, 2],
        rain_min=20, rain_window=3,
        temp_min=15, temp_max=25,
        humidity_min=70,
        latency_min=3, latency_max=6,
        harvest_days=7,
        premium=True,
        notes="Pure white cap and stem, turns pinkish with age. Does NOT blue when cut. Premium edible.",
        image_url=FN + "boletaceae/butyriboletus-regius1.jpg",
    ),
    # ── Amanita caesarea (Caesar's Mushroom) ──
    Species(
        name="Amanita caesarea",
        fitted="Amanita caesarea",   # held-out AUC 0.748 vs 0.577 (26 finds) (calibration 2026-11-18)
        pt_name="Amanita cesárea",
        habitats=[HOLM_OAK, CHESTNUT, CORK_OAK, PINE],  # GBIF PT 2015+ (58 finds) on COS2025, checked 2026-11-22: pine 13, cork oak 9, other broadleaves 6, holm oak 2, chestnut 1, eucalyptus 1 (eucalyptus dropped)
        season=[6, 7, 8, 9, 10, 11, 12],  # GBIF PT+ES 2015+: months with >=2% of finds (2026-11-17); was [8, 9, 10, 11, 12]
        rain_min=25, rain_window=5,
        temp_min=18, temp_max=28,
        humidity_min=70,
        latency_min=7, latency_max=14,
        harvest_days=10,
        premium=True,
        notes="Bright orange cap, yellow gills. Collect when cap first emerges from white egg-like veil.",
        image_url=COG + "amanita-caesarea.webp",
    ),
    # ── Chanterelles ──
    Species(
        name="Cantharellus cibarius",
        pt_name="Gongelim",
        habitats=[HOLM_OAK, PINE, EUCALYPTUS],
        season=[9, 10, 11, 12, 1, 2],
        rain_min=25, rain_window=5,
        temp_min=20, temp_max=30,
        humidity_min=70,
        latency_min=2, latency_max=5,
        harvest_days=14,
        premium=True,
        notes="Maggot-resistant. Wrinkled cap underside. Remains viable for weeks once fruiting.",
        image_url=COG + "cantharellus-cibarius.webp",
    ),
    # ── Honey mushrooms ──
    Species(
        name="Armillaria mellea",
        fitted="Armillaria mellea",   # held-out AUC 0.733 vs 0.554 (54 finds); rain weight ~0: in effect a temperature-season forecast (calibration 2026-11-18)
        pt_name="Cabeleira",
        habitats=[HOLM_OAK, CORK_OAK, PINE, EUCALYPTUS, RIPARIAN],
        season=[10, 11, 12],  # GBIF PT+ES 2015+: months with >=2% of finds (2026-11-17); was [10, 11, 12, 1, 2, 3]
        rain_min=15, rain_window=3,
        temp_min=8, temp_max=18,
        humidity_min=70,
        latency_min=3, latency_max=7,
        harvest_days=10,
        premium=True,
        notes="Grows in clusters on/around trees. MUST be boiled first (mildly toxic raw). Very common in Portugal.",
        image_url=COG + "armillaria-mellea.webp",
    ),
    # ── Yellow Knight (replaces Urala — birch species not in Portugal) ──
    Species(
        name="Tricholoma equestre",
        pt_name="Cavaleiro amarelo",
        habitats=[PINE, HOLM_OAK],
        season=[10, 11, 12],
        rain_min=15, rain_window=5,
        temp_min=12, temp_max=20,
        humidity_min=70,
        latency_min=7, latency_max=14,
        harvest_days=7,
        premium=False,
        # Corrected 2026-11-17: was "Conditionally edible — cook well". Repeated meals are
        # linked to rhabdomyolysis, including deaths: Laubner & Mikulevičienė 2016, PMID 28356809;
        # Karlson-Stiber & Persson 2003 (Toxicon), PMID 14505933.
        edible=False,
        notes="Bright yellow. Pine forest specialist. DO NOT EAT — linked to rhabdomyolysis (muscle breakdown), including deaths.",
        image_url=COG + "tricholoma-equestre.webp",
    ),
    # ── Other prized edibles ──
    Species(
        name="Macrolepiota procera",
        pt_name="Parasol",
        habitats=[GRASSLAND, HOLM_OAK, RIPARIAN],
        season=[9, 10, 11, 12, 1],
        rain_min=15, rain_window=3,
        temp_min=18, temp_max=28,
        humidity_min=60,
        latency_min=5, latency_max=7,
        harvest_days=5,
        premium=False,
        notes="Tall stems with snakeskin pattern. Open woodland and grassland edges.",
        image_url=COG + "macrolepiota-procera.webp",
    ),
    Species(
        name="Lepista nuda",
        pt_name="Oronja",
        habitats=[HOLM_OAK, RIPARIAN, GRASSLAND],
        season=[11, 12, 1, 2, 3],  # Late season — cool weather species
        rain_min=20, rain_window=5,
        temp_min=8, temp_max=15,
        humidity_min=70,
        latency_min=10, latency_max=14,
        harvest_days=10,
        premium=False,
        notes="Violet/blue cap, gills, stem that brown with age. Cool-weather species.",
        image_url=FN + "tricholomataceae/lepista-nuda1.jpg",
    ),
    Species(
        name="Craterellus cornucopioides",
        pt_name="Trombeta da morte",
        habitats=[HOLM_OAK, PINE],
        season=[10, 11, 12, 1, 2],
        rain_min=20, rain_window=5,
        temp_min=10, temp_max=18,
        humidity_min=70,
        latency_min=5, latency_max=7,
        harvest_days=7,
        premium=False,
        notes="Tough-skinned, maggot-proof. Edible when properly cooked. Horn-shaped.",
        image_url=COG + "craterellus-cornucopioides.webp",
    ),
    Species(
        name="Hydnum rufescens",
        pt_name="Ouriço",
        habitats=[HOLM_OAK, CORK_OAK, PINE],
        season=[10, 11, 12, 1, 2],
        rain_min=20, rain_window=5,
        temp_min=12, temp_max=20,
        humidity_min=70,
        latency_min=5, latency_max=7,
        harvest_days=7,
        premium=False,
        notes="Terracotta color. Underside bears long, stalactite-like spines. Fine edible.",
        image_url=FN + "hydnaceae/hydnum-rufescens1.jpg",
    ),
    Species(
        name="Russula aurea",
        pt_name="Rússula dourada",
        habitats=[HOLM_OAK, CORK_OAK],
        season=[10, 11, 12, 1, 2],
        rain_min=15, rain_window=3,
        temp_min=12, temp_max=20,
        humidity_min=70,
        latency_min=5, latency_max=7,
        harvest_days=5,
        premium=False,
        notes="Delicious. Frequently damaged by slugs and bugs. Golden-yellow cap.",
        image_url=FN + "russulaceae/russula-aurea1.jpg",
    ),
    Species(
        name="Suillus collinitus",
        pt_name="Enguia",
        habitats=[PINE, EUCALYPTUS],
        season=[9, 10, 11, 12, 1],
        rain_min=20, rain_window=3,
        temp_min=15, temp_max=25,
        humidity_min=65,
        latency_min=2, latency_max=5,
        harvest_days=5,
        premium=False,
        notes="Pine specialist. Pink mycelial threads at stem base. Slippery cap when wet.",
        image_url=FN + "suillaceae/suillus-collinitus7.jpg",
    ),
    Species(
        name="Clitopilus prunulus",
        pt_name="Moinho",
        habitats=[GRASSLAND, HOLM_OAK, RIPARIAN],
        season=[10, 11, 12, 1, 2],
        rain_min=15, rain_window=3,
        temp_min=10, temp_max=18,
        humidity_min=65,
        latency_min=3, latency_max=5,
        harvest_days=5,
        premium=False,
        notes="Floury texture and mealy scent. Grassy fields and woodland edges.",
        image_url=FA + "2025/11/the-miller-clitopilus-prunulus-1.jpg",
    ),
    # ── Fast responders (hours to 1-2 days) ──
    Species(
        name="Coprinopsis atramentaria",
        pt_name="Tinta",
        habitats=[GRASSLAND, RIPARIAN],
        season=[10, 11, 12, 1, 2, 3],
        rain_min=5, rain_window=1,  # Any rain triggers it
        temp_min=10, temp_max=20,
        humidity_min=60,
        latency_min=0, latency_max=1,
        harvest_days=1,
        premium=False,
        notes="Ink cap. Dissolves into black ink. Appears hours after rain. Eat immediately.",
        image_url=FN + "psathyrellaceae/coprinopsis-atramentaria9.jpg",
    ),
    Species(
        name="Pleurotus ostreatus",
        pt_name="Ostra",
        habitats=[DECAYING_WOOD, RIPARIAN, HOLM_OAK],
        season=MONTHS,  # Year-round in mild wet weather
        rain_min=10, rain_window=2,
        temp_min=10, temp_max=24,
        humidity_min=70,
        latency_min=2, latency_max=4,
        harvest_days=3,
        premium=False,
        notes="Oyster mushroom. Decaying wood. Erupts and matures in 2-4 days after storms.",
        image_url=COG + "pleurotus-ostreatus.webp",
    ),
]


def get_species_by_habitat(habitat: str) -> List[Species]:
    """Return EDIBLE species that fruit in the given habitat (sites only score what may be picked)."""
    return [s for s in SPECIES if habitat in s.habitats and s.edible]


def get_species_in_season(month: int) -> List[Species]:
    """Return species that fruit in the given month (1-12)."""
    return [s for s in SPECIES if month in s.season]


def find_by_name(name: str) -> Species | None:
    """Find species by scientific or Portuguese name (case-insensitive)."""
    low = name.lower()
    for s in SPECIES:
        if low in s.name.lower() or low in s.pt_name.lower():
            return s
    return None
