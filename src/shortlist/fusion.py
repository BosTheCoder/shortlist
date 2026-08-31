"""Reciprocal Rank Fusion.

The rankers produce scores on incomparable scales (a BM25 score, a cosine, a
shrunk rating). RRF throws the magnitudes away and fuses the positions instead,
so no ranker's units can dominate the others.
"""

from collections.abc import Mapping, Sequence

DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    weights: Mapping[str, float],
    k: int = DEFAULT_K,
) -> dict[str, float]:
    """Fuse per-ranker orderings into one score per option id.

    `rankings` maps a ranker name to its ordering, best first. An option absent
    from a ranking simply contributes nothing for that ranker.
    """
    scores: dict[str, float] = {}
    for name, ordering in rankings.items():
        weight = weights.get(name, 1.0)
        for position, option_id in enumerate(ordering, start=1):
            scores[option_id] = scores.get(option_id, 0.0) + weight / (k + position)
    return scores


def contribution(weight: float, position: int, k: int = DEFAULT_K) -> float:
    """One ranker's share of an option's fused score."""
    return weight / (k + position)
