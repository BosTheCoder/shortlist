"""Hard rules gate what can be ranked at all; soft rules are scored across the set."""

import pytest

from shortlist.constraints import failed_rules, soft_scores
from shortlist.models import HardRule, Option, SoftRule


def opt(oid: str, **fields: object) -> Option:
    return Option(id=oid, title=oid, text="", fields=dict(fields))


@pytest.mark.parametrize(
    ("rule", "fields", "expected_pass"),
    [
        (HardRule(field="price", max=60), {"price": 59.9}, True),
        (HardRule(field="price", max=60), {"price": 60}, True),
        (HardRule(field="price", max=60), {"price": 60.01}, False),
        (HardRule(field="price", min=10), {"price": 9}, False),
        (HardRule(field="price", max=60), {}, False),
        (HardRule(field="tags", contains_none=["shellfish"]), {"tags": ["fish", "veg"]}, True),
        (HardRule(field="tags", contains_none=["shellfish"]), {"tags": ["shellfish"]}, False),
        (HardRule(field="tags", contains_none=["shellfish"]), {"tags": ["SHELLFISH"]}, False),
        (HardRule(field="tags", contains_any=["veg"]), {"tags": ["fish"]}, False),
        (HardRule(field="tags", contains_any=["veg"]), {"tags": ["fish", "veg"]}, True),
        (HardRule(field="tags", contains_none=["shellfish"]), {"tags": "shellfish platter"}, False),
        (HardRule(field="area", equals="east"), {"area": "east"}, True),
        (HardRule(field="area", equals="east"), {"area": "west"}, False),
    ],
)
def test_hard_rule_verdicts(rule: HardRule, fields: dict[str, object], expected_pass: bool) -> None:
    assert (failed_rules(opt("x", **fields), [rule]) == []) is expected_pass


def test_a_missing_field_never_silently_passes() -> None:
    # An option with no `price` cannot be shown to satisfy "price under 60".
    assert failed_rules(opt("x"), [HardRule(field="price", max=60)]) != []


def test_soft_scores_are_normalised_across_the_candidate_set() -> None:
    options = [opt("cheap", price=10), opt("mid", price=20), opt("dear", price=30)]
    rules = [SoftRule(field="price", prefer="low", weight=1.0)]

    scores = soft_scores(options, rules)

    # Best possible is 0 penalty, worst is -1; the middle sits proportionally between.
    assert scores["cheap"] == pytest.approx(0.0)
    assert scores["mid"] == pytest.approx(-0.5)
    assert scores["dear"] == pytest.approx(-1.0)


def test_soft_weights_trade_the_rules_off_against_each_other() -> None:
    options = [opt("a", price=10, rating=3.0), opt("b", price=30, rating=5.0)]
    price_first = soft_scores(
        options,
        [
            SoftRule(field="price", prefer="low", weight=3.0),
            SoftRule(field="rating", prefer="high", weight=1.0),
        ],
    )
    rating_first = soft_scores(
        options,
        [
            SoftRule(field="price", prefer="low", weight=1.0),
            SoftRule(field="rating", prefer="high", weight=3.0),
        ],
    )

    assert price_first["a"] > price_first["b"]
    assert rating_first["b"] > rating_first["a"]


def test_a_field_with_no_spread_penalises_nobody() -> None:
    options = [opt("a", price=10), opt("b", price=10)]

    scores = soft_scores(options, [SoftRule(field="price", prefer="low", weight=1.0)])

    assert scores["a"] == scores["b"] == pytest.approx(0.0)


def test_an_option_missing_a_soft_field_takes_the_worst_penalty() -> None:
    # Not neutral, not free: no price means it is scored as if it were the priciest.
    options = [opt("cheap", price=10), opt("dear", price=20), opt("unpriced")]

    scores = soft_scores(options, [SoftRule(field="price", prefer="low", weight=1.0)])

    assert scores["unpriced"] == pytest.approx(scores["dear"])
    assert scores["unpriced"] < scores["cheap"]
