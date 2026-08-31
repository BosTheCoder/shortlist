"""The pipeline: filter, rank, fuse, explain."""

import json

import pytest

from shortlist import rank
from shortlist.embedding import HashingEmbedder, NullEmbedder
from shortlist.fusion import reciprocal_rank_fusion
from shortlist.models import HardRule, Option, Profile, Shortlist, SoftRule
from shortlist.serialisation import to_dict

THREE = ["constraint", "lexical", "popularity"]


def ids(shortlist: Shortlist) -> list[str]:
    return [result.option.id for result in shortlist.results]


def test_end_to_end_top_three_matches_the_hand_computation(dining_options, dining_profile) -> None:
    # Derivation committed at tests/fixtures/expected.md.
    result = rank(dining_options, dining_profile)

    assert ids(result)[:3] == ["alpha", "bravo", "charlie"]
    assert ids(result) == ["alpha", "bravo", "charlie", "echo", "delta"]
    assert result.rankers == THREE
    assert sorted(result.abstained) == ["learned", "semantic"]
    assert pytest.approx(result.results[0].score, abs=1e-8) == 0.04891591750396615


def test_per_ranker_positions_match_the_hand_computation(dining_options, dining_profile) -> None:
    result = rank(dining_options, dining_profile)
    positions = {
        r.option.id: {view.ranker: view.rank for view in r.per_ranker} for r in result.results
    }

    assert positions["alpha"] == {"constraint": 1, "lexical": 2, "popularity": 1}
    assert positions["bravo"] == {"constraint": 2, "lexical": 1, "popularity": 2}
    assert positions["delta"] == {"constraint": 5, "lexical": 5, "popularity": 4}


class TestHardConstraintsAreAbsolute:
    def test_violating_options_never_appear_at_any_ranker_weighting(
        self,
        dining_options: list[Option],
        dining_profile,
    ) -> None:
        # foxtrot is over budget and golf has shellfish; both would otherwise rank highly.
        weightings = [
            {},
            {"constraint": 0.0},
            {"constraint": 0.0, "lexical": 1000.0},
            {"constraint": 0.0, "popularity": 1000.0},
            {"lexical": 0.0, "popularity": 0.0, "constraint": 0.0},
        ]
        for weights in weightings:
            profile = Profile(
                name=dining_profile.name,
                query=dining_profile.query,
                hard=dining_profile.hard,
                soft=dining_profile.soft,
                weights=weights,
            )
            assert "foxtrot" not in ids(rank(dining_options, profile))
            assert "golf" not in ids(rank(dining_options, profile))

    def test_violating_options_never_appear_with_the_constraint_ranker_switched_off(
        self,
        dining_options: list[Option],
        dining_profile,
    ) -> None:
        result = rank(dining_options, dining_profile, rankers=["lexical", "popularity"])

        assert "foxtrot" not in ids(result)
        assert "golf" not in ids(result)

    def test_the_excluded_options_are_reported_with_the_rule_they_broke(
        self,
        dining_options: list[Option],
        dining_profile,
    ) -> None:
        result = rank(dining_options, dining_profile)
        broken = {e.option.id: [rule.field for rule in e.failed] for e in result.excluded}

        assert broken == {"foxtrot": ["price_gbp_pp"], "golf": ["tags"]}


class TestAbstention:
    def test_the_null_embedder_drops_semantic_and_leaves_the_rest_untouched(
        self,
        dining_options: list[Option],
        dining_profile,
    ) -> None:
        with_null = rank(dining_options, dining_profile, embedder=NullEmbedder())
        without_semantic = rank(dining_options, dining_profile, rankers=THREE)

        assert "semantic" in with_null.abstained
        assert to_dict(with_null)["results"] == to_dict(without_semantic)["results"]

    def test_a_working_embedder_puts_semantic_back_into_the_fusion(
        self,
        dining_options: list[Option],
        dining_profile,
    ) -> None:
        result = rank(dining_options, dining_profile, embedder=HashingEmbedder())

        assert "semantic" in result.rankers
        assert "semantic" not in result.abstained

    def test_the_fused_score_only_counts_the_rankers_that_took_part(
        self,
        dining_options: list[Option],
        dining_profile,
    ) -> None:
        result = rank(dining_options, dining_profile, rankers=["lexical", "popularity"])
        rankings = {
            "lexical": ["bravo", "alpha", "charlie", "echo", "delta"],
            "popularity": ["alpha", "bravo", "charlie", "delta", "echo"],
        }
        expected = reciprocal_rank_fusion(rankings, {"lexical": 1.0, "popularity": 1.0})

        assert {r.option.id: r.score for r in result.results} == pytest.approx(expected)


def test_disagreement_is_zero_when_the_rankers_agree_and_positive_when_they_do_not() -> None:
    options = [
        Option(id="a", title="a", text="quiet", fields={"rating": 3.0, "review_count": 500}),
        Option(id="b", title="b", text="loud", fields={"rating": 5.0, "review_count": 500}),
    ]
    profile = Profile(name="p", query="quiet")

    result = rank(options, profile, rankers=["lexical", "popularity"])
    disagreement = {r.option.id: r.disagreement for r in result.results}

    assert disagreement["a"] > 0  # lexical says a, popularity says b
    assert result.results[0].disagreement == result.results[1].disagreement

    agreeing = rank(options, profile, rankers=["lexical"])
    assert all(r.disagreement == 0 for r in agreeing.results)


class TestDeterminism:
    def test_the_same_inputs_serialise_byte_identically(
        self, dining_options, dining_profile
    ) -> None:
        first = json.dumps(to_dict(rank(dining_options, dining_profile)), sort_keys=True)
        second = json.dumps(to_dict(rank(dining_options, dining_profile)), sort_keys=True)

        assert first == second

    def test_the_run_id_changes_when_an_input_changes(self, dining_options, dining_profile) -> None:
        baseline = rank(dining_options, dining_profile).run_id
        reordered = rank(list(reversed(dining_options)), dining_profile).run_id
        requeried = rank(
            dining_options,
            Profile(
                name=dining_profile.name,
                query="loud steakhouse",
                hard=dining_profile.hard,
                soft=dining_profile.soft,
            ),
        ).run_id

        assert baseline == reordered  # option order is not an input to the ranking
        assert baseline != requeried


def test_top_truncates_without_changing_the_order(dining_options, dining_profile) -> None:
    full = rank(dining_options, dining_profile)
    trimmed = rank(dining_options, dining_profile, top=2)

    assert ids(trimmed) == ids(full)[:2]


def test_an_option_set_wiped_out_by_the_hard_rules_ranks_to_nothing() -> None:
    options = [Option(id="a", title="a", fields={"price": 100})]
    profile = Profile(
        name="p",
        query="anything",
        hard=[HardRule(field="price", max=10)],
        soft=[SoftRule(field="price", prefer="low")],
    )

    result = rank(options, profile)

    assert result.results == []
    assert [e.option.id for e in result.excluded] == ["a"]
