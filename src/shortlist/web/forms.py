"""Turning the demo's form fields into a Profile.

Form values arrive as `{field: [value, ...]}` so that repeated checkboxes and
single inputs are handled the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

from shortlist.data import DATASETS, Dataset
from shortlist.models import HardRule, Profile
from shortlist.rankers import DEFAULT_RANKERS

Values = dict[str, list[str]]


@dataclass(frozen=True)
class DemoRequest:
    dataset: Dataset
    profile: Profile
    rankers: list[str]
    excluded_tags: list[str]
    focus: str
    compare: tuple[str, str]


def _one(values: Values, key: str, default: str = "") -> str:
    found = values.get(key) or []
    return found[0] if found else default


def _number(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def starting_values(dataset: Dataset) -> Values:
    """Where the controls sit before anybody touches them."""
    return {
        "dataset": [dataset.key],
        "query": [dataset.query],
        "rankers": list(DEFAULT_RANKERS),
        **{f"{c.bound}__{c.field}": [str(c.default)] for c in dataset.controls},
    }


def parse(values: Values, focus_override: str | None = None) -> DemoRequest:
    dataset = DATASETS.get(_one(values, "dataset"), DATASETS["london-restaurants"])

    hard = [
        HardRule(field=control.field, max=bound)
        if control.bound == "max"
        else HardRule(field=control.field, min=bound)
        for control in dataset.controls
        if (bound := _number(_one(values, f"{control.bound}__{control.field}"))) is not None
    ]
    excluded = [tag for tag in values.get("exclude", []) if tag in dataset.tags]
    if excluded:
        hard.append(HardRule(field=dataset.tag_field, contains_none=excluded))

    chosen = [name for name in values.get("rankers", []) if name in DEFAULT_RANKERS]
    return DemoRequest(
        dataset=dataset,
        profile=Profile(
            name=dataset.key,
            query=_one(values, "query", dataset.query),
            hard=hard,
            soft=list(dataset.soft),
        ),
        rankers=chosen,
        excluded_tags=excluded,
        focus=focus_override or _one(values, "focus"),
        compare=(_one(values, "compare_a"), _one(values, "compare_b")),
    )
