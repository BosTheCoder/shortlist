"""One JSON shape, shared by the CLI, the HTTP endpoint and the determinism test."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from shortlist.models import Shortlist


def to_dict(shortlist: Shortlist) -> dict[str, Any]:
    return {
        "run_id": shortlist.run_id,
        "profile": shortlist.profile,
        "rankers": shortlist.rankers,
        "abstained": shortlist.abstained,
        "results": [
            {
                "rank": result.rank,
                "id": result.option.id,
                "title": result.option.title,
                "score": result.score,
                "disagreement": result.disagreement,
                "per_ranker": [asdict(view) for view in result.per_ranker],
                "fields": result.option.fields,
            }
            for result in shortlist.results
        ],
        "excluded": [
            {"id": excluded.option.id, "failed": [asdict(rule) for rule in excluded.failed]}
            for excluded in shortlist.excluded
        ],
    }
