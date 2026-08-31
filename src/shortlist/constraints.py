"""Evaluating a profile's hard filters and soft preferences against options."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from shortlist.models import HardRule, Option, RuleCheck, SoftRule


def _as_tokens(value: Any) -> list[str]:
    """Read a field as a lowercased bag of strings, whether it is a list or a string."""
    if isinstance(value, str):
        return value.lower().split()
    if isinstance(value, (list, tuple, set)):
        return [str(item).lower() for item in value]
    return [str(value).lower()]


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def check(option: Option, rule: HardRule) -> RuleCheck:
    """Decide whether one option satisfies one hard rule."""
    value = option.fields.get(rule.field)
    return RuleCheck(rule=rule, passed=_passes(value, rule), value=value)


def _passes(value: Any, rule: HardRule) -> bool:
    if value is None:
        return False
    if rule.min is not None or rule.max is not None:
        number = _as_number(value)
        if number is None:
            return False
        if rule.min is not None and number < rule.min:
            return False
        if rule.max is not None and number > rule.max:
            return False
    if rule.equals is not None and value != rule.equals:
        return False
    if rule.contains_any or rule.contains_none:
        tokens = set(_as_tokens(value))
        if rule.contains_any and tokens.isdisjoint({t.lower() for t in rule.contains_any}):
            return False
        if rule.contains_none and not tokens.isdisjoint({t.lower() for t in rule.contains_none}):
            return False
    return True


def checks(option: Option, rules: Sequence[HardRule]) -> list[RuleCheck]:
    return [check(option, rule) for rule in rules]


def failed_rules(option: Option, rules: Sequence[HardRule]) -> list[HardRule]:
    return [c.rule for c in checks(option, rules) if not c.passed]


def soft_scores(options: Sequence[Option], rules: Sequence[SoftRule]) -> dict[str, float]:
    """Score every option on the soft rules, in [-1, 0], best is 0.

    Each rule's field is min-max normalised across the candidate set, so the
    penalty means "how far from the best option here", not "how big is this
    number". An option missing the field is scored as the worst case.
    """
    if not rules:
        return {option.id: 0.0 for option in options}

    total_weight = sum(abs(rule.weight) for rule in rules)
    if total_weight == 0:
        return {option.id: 0.0 for option in options}

    penalties = {option.id: 0.0 for option in options}
    for rule in rules:
        values = {
            option.id: number
            for option in options
            if (number := _as_number(option.fields.get(rule.field))) is not None
        }
        low, high = (min(values.values()), max(values.values())) if values else (0.0, 0.0)
        spread = high - low
        for option in options:
            value = values.get(option.id)
            if value is None:
                deviation = 1.0
            elif spread == 0:
                deviation = 0.0
            else:
                fraction = (value - low) / spread
                deviation = fraction if rule.prefer == "low" else 1.0 - fraction
            penalties[option.id] += rule.weight * deviation

    return {oid: -penalty / total_weight for oid, penalty in penalties.items()}
