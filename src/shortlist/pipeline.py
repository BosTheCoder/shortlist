"""`rank()` — the one entry point everything else goes through."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict

from shortlist.constraints import checks, failed_rules
from shortlist.embedding import Embedder
from shortlist.fusion import DEFAULT_K, contribution, reciprocal_rank_fusion
from shortlist.learning import LearnedWeights
from shortlist.models import Excluded, Option, Profile, RankerView, Result, Shortlist
from shortlist.rankers import DEFAULT_RANKERS, build, order


def run_id(options: Sequence[Option], profile: Profile, ranker_names: Sequence[str], k: int) -> str:
    """A stable name for one (options, profile, rankers) combination."""
    payload = {
        "options": sorted(
            ({"id": o.id, "title": o.title, "text": o.text, "fields": o.fields} for o in options),
            key=lambda record: record["id"],
        ),
        "profile": asdict(profile),
        "rankers": list(ranker_names),
        "k": k,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _spread(ranks: Sequence[int]) -> float:
    """How much the rankers disagreed: the population standard deviation of their ranks."""
    if len(ranks) < 2:
        return 0.0
    mean = sum(ranks) / len(ranks)
    return math.sqrt(sum((rank - mean) ** 2 for rank in ranks) / len(ranks))


def rank(
    options: Sequence[Option],
    profile: Profile,
    *,
    rankers: Sequence[str] | None = None,
    embedder: Embedder | None = None,
    weights: LearnedWeights | None = None,
    k: int = DEFAULT_K,
    top: int | None = None,
) -> Shortlist:
    """Filter on the hard rules, run each ranker, fuse the orderings, explain the result."""
    names = list(rankers if rankers is not None else DEFAULT_RANKERS)

    survivors: list[Option] = []
    excluded: list[Excluded] = []
    for option in options:
        broken = failed_rules(option, profile.hard)
        if broken:
            excluded.append(Excluded(option=option, failed=broken))
        else:
            survivors.append(option)

    orderings: dict[str, list[str]] = {}
    positions: dict[str, dict[str, int]] = {}
    scores_by_ranker: dict[str, dict[str, float]] = {}
    abstained: list[str] = []
    for name in names:
        scored = build(name, embedder=embedder, weights=weights).rank(survivors, profile)
        if scored is None:
            abstained.append(name)
            continue
        ordered = order(scored)
        orderings[name] = [s.option_id for s in ordered]
        positions[name] = {s.option_id: index for index, s in enumerate(ordered, start=1)}
        scores_by_ranker[name] = {s.option_id: s.score for s in ordered}

    fused = reciprocal_rank_fusion(orderings, profile.weights, k=k)
    ordered_ids = sorted(survivors, key=lambda o: (-fused.get(o.id, 0.0), o.id))

    results: list[Result] = []
    for position, option in enumerate(ordered_ids, start=1):
        views = [
            RankerView(
                ranker=name,
                rank=places[option.id],
                score=scores_by_ranker[name][option.id],
                contribution=contribution(profile.weights.get(name, 1.0), places[option.id], k),
            )
            for name, places in positions.items()
        ]
        ranks = [view.rank for view in views]
        results.append(
            Result(
                option=option,
                rank=position,
                score=fused.get(option.id, 0.0),
                per_ranker=views,
                disagreement=_spread(ranks),
                hard_checks=checks(option, profile.hard),
            )
        )

    return Shortlist(
        run_id=run_id(options, profile, names, k),
        profile=profile.name,
        results=results[:top] if top else results,
        rankers=list(orderings),
        abstained=abstained,
        excluded=excluded,
    )
