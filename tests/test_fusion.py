"""Reciprocal Rank Fusion is the one place scores from different rankers meet."""

from shortlist.fusion import reciprocal_rank_fusion


def test_rrf_matches_hand_computation_to_five_dp() -> None:
    # Four options, three rankers. Ranks are 1-based.
    #   lexical:    a b c d
    #   popularity: b a d c
    #   constraint: c a b d
    # weights: lexical 1.0, popularity 2.0, constraint 0.5; k = 60.
    #
    # a: 1/61 + 2/62 + 0.5/62      = 0.0163934 + 0.0322581 + 0.0080645 = 0.0567160
    # b: 1/62 + 2/61 + 0.5/63      = 0.0161290 + 0.0327869 + 0.0079365 = 0.0568524
    # c: 1/63 + 2/64 + 0.5/61      = 0.0158730 + 0.0312500 + 0.0081967 = 0.0553197
    # d: 1/64 + 2/63 + 0.5/64      = 0.0156250 + 0.0317460 + 0.0078125 = 0.0551835
    rankings = {
        "lexical": ["a", "b", "c", "d"],
        "popularity": ["b", "a", "d", "c"],
        "constraint": ["c", "a", "b", "d"],
    }
    weights = {"lexical": 1.0, "popularity": 2.0, "constraint": 0.5}

    scores = reciprocal_rank_fusion(rankings, weights, k=60)

    assert round(scores["a"], 5) == 0.05672
    assert round(scores["b"], 5) == 0.05685
    assert round(scores["c"], 5) == 0.05532
    assert round(scores["d"], 5) == 0.05518


def test_k_damps_the_gap_between_first_and_second_place() -> None:
    rankings = {"only": ["a", "b"]}
    weights = {"only": 1.0}

    tight = reciprocal_rank_fusion(rankings, weights, k=1)
    loose = reciprocal_rank_fusion(rankings, weights, k=1000)

    assert tight["a"] - tight["b"] > loose["a"] - loose["b"]


def test_an_option_missing_from_a_ranking_scores_only_where_it_appears() -> None:
    rankings = {"one": ["a", "b"], "two": ["a"]}
    weights = {"one": 1.0, "two": 1.0}

    scores = reciprocal_rank_fusion(rankings, weights, k=60)

    assert scores["b"] == 1 / 62
