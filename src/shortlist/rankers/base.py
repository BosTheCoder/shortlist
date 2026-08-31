"""The ranker contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from shortlist.models import Option, Profile, ScoredOption


class Ranker(Protocol):
    """One opinion about how to order options.

    `rank` returns a score for every option it was given, or `None` to abstain.
    Abstaining is not the same as scoring everything equally: an abstaining
    ranker is dropped from the fusion entirely rather than adding noise.
    """

    name: str

    def rank(self, options: Sequence[Option], profile: Profile) -> list[ScoredOption] | None: ...


def order(scored: Sequence[ScoredOption]) -> list[ScoredOption]:
    """Best first. Ties break on option id so the same input gives the same output."""
    return sorted(scored, key=lambda s: (-s.score, s.option_id))
