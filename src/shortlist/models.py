"""The data the whole pipeline moves around."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Option:
    """One candidate: a title, free text, and whatever fields the dataset carries."""

    id: str
    title: str
    text: str = ""
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HardRule:
    """A pass/fail filter. An option failing any hard rule is removed before fusion."""

    field: str
    min: float | None = None
    max: float | None = None
    equals: Any | None = None
    contains_any: list[str] = field(default_factory=list)
    contains_none: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SoftRule:
    """A preference on a numeric field, weighted against the other soft rules."""

    field: str
    prefer: str  # "low" or "high"
    weight: float = 1.0


@dataclass(frozen=True)
class Profile:
    name: str
    query: str = ""
    hard: list[HardRule] = field(default_factory=list)
    soft: list[SoftRule] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredOption:
    """A ranker's opinion of one option. Higher score is better."""

    option_id: str
    score: float


@dataclass(frozen=True)
class RuleCheck:
    rule: HardRule
    passed: bool
    value: Any


@dataclass(frozen=True)
class RankerView:
    """What one ranker contributed to one option's final placement."""

    ranker: str
    rank: int
    score: float
    contribution: float


@dataclass(frozen=True)
class Result:
    option: Option
    rank: int
    score: float
    per_ranker: list[RankerView]
    disagreement: float
    hard_checks: list[RuleCheck]


@dataclass(frozen=True)
class Excluded:
    option: Option
    failed: list[HardRule]


@dataclass(frozen=True)
class Shortlist:
    run_id: str
    profile: str
    results: list[Result]
    rankers: list[str]
    abstained: list[str]
    excluded: list[Excluded]
