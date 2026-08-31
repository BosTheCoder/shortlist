"""Scores options with weights fitted from past picks."""

from __future__ import annotations

from collections.abc import Sequence

from shortlist.learning import LearnedWeights, numeric_features
from shortlist.models import Option, Profile, ScoredOption


class LearnedRanker:
    """Abstains until there is a fitted model, so a cold profile is not guessed at."""

    name = "learned"

    def __init__(self, weights: LearnedWeights | None = None) -> None:
        self.weights = weights

    def rank(self, options: Sequence[Option], profile: Profile) -> list[ScoredOption] | None:
        if self.weights is None or not options:
            return None
        known = set(self.weights.features)
        scored = [
            ScoredOption(option_id=option.id, score=self.weights.score(numeric_features(option)))
            for option in options
        ]
        if not any(known & set(numeric_features(option)) for option in options):
            return None
        return scored
