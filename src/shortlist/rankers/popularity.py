"""Bayesian-shrunk rating: a perfect score from two reviews is not evidence."""

from __future__ import annotations

from collections.abc import Sequence

from shortlist.models import Option, Profile, ScoredOption

PRIOR_REVIEWS = 25.0


class PopularityRanker:
    """`(v*R + m*C) / (v + m)` — pulls thinly reviewed options towards the set mean."""

    name = "popularity"

    def __init__(
        self,
        rating_field: str = "rating",
        count_field: str = "review_count",
        prior_reviews: float = PRIOR_REVIEWS,
    ) -> None:
        self.rating_field = rating_field
        self.count_field = count_field
        self.prior_reviews = prior_reviews

    def rank(self, options: Sequence[Option], profile: Profile) -> list[ScoredOption] | None:
        ratings = {
            option.id: float(value)
            for option in options
            if isinstance(value := option.fields.get(self.rating_field), (int, float))
        }
        if not ratings:
            return None

        mean_rating = sum(ratings.values()) / len(ratings)
        scored: list[ScoredOption] = []
        for option in options:
            rating = ratings.get(option.id, mean_rating)
            count = option.fields.get(self.count_field, 0)
            reviews = float(count) if isinstance(count, (int, float)) else 0.0
            shrunk = (reviews * rating + self.prior_reviews * mean_rating) / (
                reviews + self.prior_reviews
            )
            scored.append(ScoredOption(option_id=option.id, score=shrunk))
        return scored
