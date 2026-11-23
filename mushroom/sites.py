"""
Foraging sites database — Setúbal, Portugal region (within 150km).

Sites defined by name, coordinates, forest types, associated species,
and accessibility notes. Used by scoring.py to compute per-site predictions.

Source: cogumelos.folkup.fit Setúbal guide, First Nature Algarve,
Arrábida Guide, local knowledge.
"""

from dataclasses import dataclass, field
from typing import List

from mushroom.species_db import (
    CORK_OAK, HOLM_OAK, PINE, EUCALYPTUS, RIPARIAN, GRASSLAND,
    HABITAT_LABELS, get_species_by_habitat, Species, SPECIES,
)

# Reference point for distance calculations
SETUBAL_LAT, SETUBAL_LON = 38.559, -8.927


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two lat/lon points."""
    import math
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class ForagingSite:
    """One foraging location."""
    name: str
    lat: float
    lon: float
    habitats: List[str]                    # Forest types at this site
    notes: str = ""                        # Accessibility / legal notes
    species: List[Species] = field(default_factory=list)

    def __post_init__(self):
        # Auto-populate species from habitats
        seen = set()
        self.species = []
        for h in self.habitats:
            for s in get_species_by_habitat(h):
                if s.name not in seen:
                    seen.add(s.name)
                    self.species.append(s)

    @property
    def distance_km(self) -> float:
        return haversine_km(SETUBAL_LAT, SETUBAL_LON, self.lat, self.lon)

    @property
    def habitat_labels(self) -> str:
        return ", ".join(HABITAT_LABELS.get(h, h) for h in self.habitats)


# ── Site definitions ─────────────────────────────────────────────────

SITES: List[ForagingSite] = [
    ForagingSite(
        name="Serra da Arrábida NP",
        lat=38.55, lon=-8.80,
        habitats=[HOLM_OAK, GRASSLAND],
        notes="Protected national park — check foraging regulations. Granite soils, maquis understory. Excellent Amanita caesarea habitat.",
    ),
    ForagingSite(
        name="Palmela Cork Oak",
        lat=38.65, lon=-8.75,
        habitats=[CORK_OAK, HOLM_OAK],
        notes="Classic montado. Private land — ask permission before entering. Best bolete country near Setúbal.",
    ),
    ForagingSite(
        name="Montijo / Alcochete",
        lat=38.70, lon=-8.85,
        habitats=[RIPARIAN, EUCALYPTUS],
        notes="Easy access from Setúbal. Riparian corridors along Sado river. Eucalyptus patches for Suillus and Amanita caesarea.",
    ),
    ForagingSite(
        name="Vale do Sorraia",
        lat=38.70, lon=-8.45,
        habitats=[RIPARIAN, GRASSLAND],
        notes="Access via IC2. Poplar, willow, oak riparian forest. Good for Macrolepiota and Lepista.",
    ),
    ForagingSite(
        name="Alcácer do Sal",
        lat=38.35, lon=-8.53,
        habitats=[CORK_OAK, HOLM_OAK],
        notes="Montado country. Both cork and holm oak. Boletus and Amanita caesarea territory.",
    ),
    ForagingSite(
        name="Sines / Costa Norte",
        lat=37.95, lon=-8.85,
        habitats=[HOLM_OAK, PINE, GRASSLAND],
        notes="Coastal dune pine (Costa Norte / Santo André) + holm oak inland. Diverse habitat mix. ~70km south.",
    ),
    ForagingSite(
        name="Sintra-Cascais NP",
        lat=38.80, lon=-9.35,
        habitats=[HOLM_OAK, GRASSLAND],
        notes="Protected national park. Stone oak, laurel forest. Humid microclimate. ~45km west.",
    ),
    ForagingSite(
        # Renamed 2026-10-07: this entry was labelled "Parque Natural de Montargil" but its
        # coordinates and its own note are Vila Franca de Xira, 75 km west of Montargil.
        name="Vila Franca de Xira (west)",
        lat=38.95, lon=-9.05,
        habitats=[RIPARIAN, EUCALYPTUS],
        notes="Near Vila Franca de Xira. Riparian oak + eucalyptus. Suillus and Macrolepiota. ~45km north.",
    ),
    ForagingSite(
        # Coordinates: Montargil village, Ponte de Sor (Barragem de Montargil to the south).
        # Habitat from the regional land cover (Ponte de Sor cork-oak montado with
        # maritime-pine stands), not from a site visit. [TODO: verify on the ground]
        name="Montargil (Ponte de Sor)",
        lat=39.077, lon=-8.171,
        habitats=[CORK_OAK, PINE],
        notes="Cork-oak montado around the Montargil reservoir, pine stands by the water. Private montado: ask before entering. ~88km north-east.",
    ),
    ForagingSite(
        name="Torres Vedras Pine",
        lat=39.30, lon=-9.30,
        habitats=[PINE],
        notes="Pinus pinaster forest. Cortinarius and Chanterelles. ~80km north.",
    ),
    ForagingSite(
        name="Santarém Riparian",
        lat=39.25, lon=-8.65,
        habitats=[RIPARIAN, HOLM_OAK],
        notes="Poplar, willow, oak along Tagus. Lepista and Macrolepiota. ~80km north.",
    ),
]


def get_sites_sorted_by_distance() -> List[ForagingSite]:
    """Return sites sorted by distance from Setúbal."""
    return sorted(SITES, key=lambda s: s.distance_km)


def get_sites_in_radius(radius_km: float) -> List[ForagingSite]:
    """Return sites within a given radius of Setúbal."""
    return [s for s in SITES if s.distance_km <= radius_km]
