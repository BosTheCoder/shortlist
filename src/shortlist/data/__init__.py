"""The three bundled demo datasets.

Every record is invented. There are no real businesses, people or addresses in
here; the numbers are made up to give the rankers something to disagree about.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shortlist.loaders import load_options
from shortlist.models import Option, Profile, SoftRule

DATA_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Control:
    """One numeric slider in the demo, wired to a hard rule."""

    field: str
    label: str
    bound: str  # "max" or "min"
    default: float
    minimum: float
    maximum: float
    step: float


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    blurb: str
    query: str
    soft: list[SoftRule]
    controls: list[Control]
    tag_field: str
    tags: list[str]

    @property
    def options(self) -> list[Option]:
        return load_options(DATA_DIR / f"{self.key}.json")

    @property
    def profile(self) -> Profile:
        return Profile(name=self.key, query=self.query, soft=list(self.soft))

    def vectors(self) -> dict[str, list[float]]:
        """Option text to vector, precomputed by `just embed`."""
        path = DATA_DIR / f"{self.key}.npz"
        if not path.exists():
            return {}
        with np.load(path, allow_pickle=False) as archive:
            return {
                str(text): [float(value) for value in vector]
                for text, vector in zip(archive["texts"], archive["vectors"], strict=True)
            }


DATASETS: dict[str, Dataset] = {
    "london-restaurants": Dataset(
        key="london-restaurants",
        title="London restaurants",
        blurb="24 invented places to eat, with price, rating, distance and how loud they are.",
        query="relaxed dinner spot, good vegetarian and fish, not too loud",
        soft=[
            SoftRule(field="price_gbp_pp", prefer="low", weight=1.0),
            SoftRule(field="rating", prefer="high", weight=1.5),
            SoftRule(field="distance_km", prefer="low", weight=1.0),
            SoftRule(field="noise_level", prefer="low", weight=0.5),
        ],
        controls=[
            Control("price_gbp_pp", "Max price per head (£)", "max", 60, 10, 150, 5),
            Control("distance_km", "Max distance (km)", "max", 8, 1, 12, 0.5),
            Control("rating", "Min rating", "min", 4.0, 3.5, 5.0, 0.1),
        ],
        tag_field="cuisine_tags",
        tags=["shellfish", "meat", "vegetarian", "fish", "vegan", "casual", "tasting"],
    ),
    "travel-backpacks": Dataset(
        key="travel-backpacks",
        title="Travel backpacks",
        blurb="20 invented packs, with price, volume, weight and whether they fit a cabin locker.",
        query="one bag carry-on for a week, laptop sleeve, light enough to walk with",
        soft=[
            SoftRule(field="price_gbp", prefer="low", weight=1.0),
            SoftRule(field="rating", prefer="high", weight=1.5),
            SoftRule(field="weight_kg", prefer="low", weight=1.0),
            SoftRule(field="litres", prefer="high", weight=0.5),
        ],
        controls=[
            Control("price_gbp", "Max price (£)", "max", 200, 30, 300, 5),
            Control("weight_kg", "Max weight (kg)", "max", 2.0, 0.2, 3.0, 0.05),
            Control("litres", "Min volume (litres)", "min", 30, 15, 65, 1),
        ],
        tag_field="tags",
        tags=["carry-on", "laptop", "waterproof", "trekking", "ultralight", "clamshell"],
    ),
    "city-break-hotels": Dataset(
        key="city-break-hotels",
        title="City-break hotels",
        blurb="20 invented hotels, with nightly price, rating, distance from the centre and noise.",
        query="quiet room near the centre, breakfast included, good value for three nights",
        soft=[
            SoftRule(field="price_gbp_night", prefer="low", weight=1.0),
            SoftRule(field="rating", prefer="high", weight=1.5),
            SoftRule(field="distance_km", prefer="low", weight=1.0),
            SoftRule(field="noise_level", prefer="low", weight=1.0),
        ],
        controls=[
            Control("price_gbp_night", "Max per night (£)", "max", 180, 40, 300, 10),
            Control("distance_km", "Max distance from centre (km)", "max", 4, 0.5, 12, 0.5),
            Control("rating", "Min rating", "min", 4.0, 3.5, 5.0, 0.1),
        ],
        tag_field="tags",
        tags=["quiet", "breakfast", "central", "pool", "kitchen", "budget", "parking"],
    ),
}
