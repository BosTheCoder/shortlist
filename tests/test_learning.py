"""The personalisation loop, end to end: picks -> weights -> a different order."""

from pathlib import Path

import pytest

from shortlist import rank
from shortlist.learning import LearnedWeights, fit
from shortlist.models import Option, Profile
from shortlist.store import FeedbackStore

RANKERS = ["popularity", "learned"]

CANDIDATES = [
    Option(id="p1", title="p1", fields={"price_gbp_pp": 20, "rating": 4.1, "review_count": 400}),
    Option(id="p2", title="p2", fields={"price_gbp_pp": 35, "rating": 4.9, "review_count": 900}),
    Option(id="p3", title="p3", fields={"price_gbp_pp": 50, "rating": 4.3, "review_count": 800}),
    Option(id="p4", title="p4", fields={"price_gbp_pp": 65, "rating": 4.6, "review_count": 700}),
]
PROFILE = Profile(name="thrifty")


def rank_of(option_id: str, shortlist) -> int:
    return next(r.rank for r in shortlist.results if r.option.id == option_id)


def log_cheap_picks(store: FeedbackStore, runs: int = 5) -> None:
    """Five past runs where the cheapest option on the table was the one chosen."""
    for index in range(runs):
        options = [
            Option(
                id=f"r{index}-{price}",
                title=f"r{index}-{price}",
                fields={"price_gbp_pp": price, "rating": 4.5, "review_count": 300},
            )
            for price in (15 + index, 45 + index, 75 + index)
        ]
        shortlist = rank(options, PROFILE, rankers=["popularity"])
        store.record_run(shortlist)
        store.record_pick(shortlist.run_id, f"r{index}-{15 + index}")


def test_training_on_cheap_picks_moves_a_cheap_option_up(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path)
    log_cheap_picks(store)

    weights = fit(store.training_pairs(PROFILE.name))
    assert weights is not None

    before = rank(CANDIDATES, PROFILE, rankers=RANKERS)
    after = rank(CANDIDATES, PROFILE, rankers=RANKERS, weights=weights)

    assert rank_of("p1", before) == 4
    assert rank_of("p1", after) < rank_of("p1", before)
    assert "learned" in before.abstained
    assert "learned" in after.rankers


def test_an_empty_feedback_log_leaves_the_order_exactly_as_it_was(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path)

    weights = fit(store.training_pairs(PROFILE.name))
    assert weights is None

    before = rank(CANDIDATES, PROFILE, rankers=RANKERS)
    after = rank(CANDIDATES, PROFILE, rankers=RANKERS, weights=weights)

    assert [r.option.id for r in after.results] == [r.option.id for r in before.results]


def test_the_fitted_weight_has_the_sign_the_feedback_implies(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path)
    log_cheap_picks(store)

    weights = fit(store.training_pairs(PROFILE.name))
    assert weights is not None
    coefficient = dict(zip(weights.features, weights.coefficients, strict=True))

    assert coefficient["price_gbp_pp"] < 0  # cheaper scored higher
    assert coefficient["rating"] == pytest.approx(0.0)  # rating never varied, so nothing learned


def test_picks_from_another_profile_are_not_trained_on(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path)
    log_cheap_picks(store)

    assert store.training_pairs("someone-else") == []
    assert store.training_pairs(PROFILE.name) != []


def test_recording_a_pick_against_an_unknown_run_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        FeedbackStore(tmp_path).record_pick("nope", "p1")


def test_weights_survive_a_round_trip_to_disk(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path)
    log_cheap_picks(store)
    weights = fit(store.training_pairs(PROFILE.name))
    assert weights is not None

    weights.write(tmp_path / "thrifty.weights.json")
    reloaded = LearnedWeights.read(tmp_path / "thrifty.weights.json")

    assert reloaded == weights
