"""The ranker registry.

Adding a ranker is adding a class and one line here. The README works through an
LLM-judge ranker as the example.
"""

from __future__ import annotations

from collections.abc import Callable

from shortlist.embedding import Embedder, NullEmbedder
from shortlist.learning import LearnedWeights
from shortlist.rankers.base import Ranker, order
from shortlist.rankers.constraint import ConstraintRanker
from shortlist.rankers.learned import LearnedRanker
from shortlist.rankers.lexical import LexicalRanker
from shortlist.rankers.popularity import PopularityRanker
from shortlist.rankers.semantic import SemanticRanker

REGISTRY: dict[str, Callable[..., Ranker]] = {
    "constraint": lambda **_: ConstraintRanker(),
    "lexical": lambda **_: LexicalRanker(),
    "semantic": lambda embedder=None, **_: SemanticRanker(embedder or NullEmbedder()),
    "popularity": lambda **_: PopularityRanker(),
    "learned": lambda weights=None, **_: LearnedRanker(weights),
}

DEFAULT_RANKERS = ["constraint", "lexical", "semantic", "popularity", "learned"]


def build(
    name: str,
    *,
    embedder: Embedder | None = None,
    weights: LearnedWeights | None = None,
) -> Ranker:
    if name not in REGISTRY:
        raise KeyError(f"unknown ranker {name!r}; known rankers are {sorted(REGISTRY)}")
    return REGISTRY[name](embedder=embedder, weights=weights)


__all__ = ["DEFAULT_RANKERS", "REGISTRY", "Ranker", "build", "order"]
