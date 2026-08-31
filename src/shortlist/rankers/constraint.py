"""Scores the profile's soft preferences. The hard rules are applied before any ranking."""

from __future__ import annotations

from collections.abc import Sequence

from shortlist.constraints import soft_scores
from shortlist.models import Option, Profile, ScoredOption


class ConstraintRanker:
    name = "constraint"

    def rank(self, options: Sequence[Option], profile: Profile) -> list[ScoredOption] | None:
        if not profile.soft or not options:
            return None
        scores = soft_scores(options, profile.soft)
        return [ScoredOption(option_id=oid, score=score) for oid, score in scores.items()]
