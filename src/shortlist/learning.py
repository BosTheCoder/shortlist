"""The personalisation loop: a pairwise logistic model fitted on picks.

Every recorded pick says "this option beat those options in that run". Each such
pair becomes one training row: the feature delta between the picked option and
one it beat. The model learns a weight per feature; the ranker then scores an
option as the dot product of those weights with its standardised features.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from shortlist.models import Option

ITERATIONS = 400
LEARNING_RATE = 0.5
L2 = 0.01


@dataclass(frozen=True)
class LearnedWeights:
    features: list[str]
    mean: list[float]
    scale: list[float]
    coefficients: list[float]
    pairs: int

    def score(self, values: dict[str, float]) -> float:
        total = 0.0
        for name, mean, scale, coefficient in zip(
            self.features, self.mean, self.scale, self.coefficients, strict=True
        ):
            value = values.get(name)
            if value is None:
                continue
            total += coefficient * (value - mean) / scale
        return total

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @staticmethod
    def read(path: Path) -> LearnedWeights | None:
        if not path.exists():
            return None
        return LearnedWeights(**json.loads(path.read_text()))


def numeric_features(option: Option) -> dict[str, float]:
    """The numeric fields of an option, which is all the model can see."""
    return {
        name: float(value)
        for name, value in option.fields.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def fit(pairs: Sequence[tuple[dict[str, float], dict[str, float]]]) -> LearnedWeights | None:
    """Fit weights from (picked, rejected) feature pairs. Returns None with nothing to learn."""
    names = sorted({name for picked, rejected in pairs for name in (*picked, *rejected)})
    if not pairs or not names:
        return None

    picked_matrix = np.asarray([[p.get(n, 0.0) for n in names] for p, _ in pairs], dtype=float)
    rejected_matrix = np.asarray([[r.get(n, 0.0) for n in names] for _, r in pairs], dtype=float)
    seen = np.vstack([picked_matrix, rejected_matrix])

    mean = seen.mean(axis=0)
    scale = seen.std(axis=0)
    scale[scale == 0] = 1.0

    deltas = (picked_matrix - rejected_matrix) / scale
    coefficients = np.zeros(len(names))
    for _ in range(ITERATIONS):
        probability = 1.0 / (1.0 + np.exp(-deltas @ coefficients))
        gradient = deltas.T @ (1.0 - probability) / len(deltas) - L2 * coefficients
        coefficients += LEARNING_RATE * gradient

    return LearnedWeights(
        features=names,
        mean=[float(value) for value in mean],
        scale=[float(value) for value in scale],
        coefficients=[float(value) for value in coefficients],
        pairs=len(pairs),
    )


def weights_path(profile_path: Path) -> Path:
    """Weights live next to the profile they personalise."""
    return profile_path.with_suffix(".weights.json")
