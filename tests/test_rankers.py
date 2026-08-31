"""Each ranker is a different opinion. These pin the opinion, not the numbers."""

import pytest

from shortlist.embedding import HashingEmbedder, NullEmbedder
from shortlist.models import Option, Profile, SoftRule
from shortlist.rankers import build
from shortlist.rankers.base import order


def opt(oid: str, text: str = "", **fields: object) -> Option:
    return Option(id=oid, title=oid, text=text, fields=dict(fields))


def ordering(ranker_name: str, options: list[Option], profile: Profile, **kw: object) -> list[str]:
    ranker = build(ranker_name, **kw)  # type: ignore[arg-type]
    scored = ranker.rank(options, profile)
    assert scored is not None
    return [s.option_id for s in order(scored)]


class TestLexical:
    def test_a_rare_query_word_outweighs_a_common_one(self) -> None:
        # "restaurant" is in every document, so it carries almost no information;
        # "izakaya" is in one. Drop the IDF term and this flips.
        options = [
            opt("common", "restaurant restaurant restaurant restaurant restaurant"),
            opt("rare", "restaurant izakaya"),
            opt("filler1", "restaurant bistro"),
            opt("filler2", "restaurant canteen"),
        ]
        profile = Profile(name="p", query="restaurant izakaya")

        assert ordering("lexical", options, profile)[0] == "rare"

    def test_padding_a_document_does_not_buy_relevance(self) -> None:
        # Same two hits either way; length normalisation should keep the short one on top.
        options = [
            opt("short", "quiet vegetarian"),
            opt("padded", "quiet vegetarian " + "menu terrace parking wifi " * 20),
            opt("other", "loud steakhouse"),
        ]
        profile = Profile(name="p", query="quiet vegetarian")

        assert ordering("lexical", options, profile)[0] == "short"

    def test_it_abstains_when_the_profile_has_no_query(self) -> None:
        assert build("lexical").rank([opt("a", "x")], Profile(name="p", query="  ")) is None


class TestPopularity:
    def test_shrinkage_protects_against_a_perfect_score_from_two_reviews(self) -> None:
        options = [
            opt("thin", rating=5.0, review_count=2),
            opt("proven", rating=4.6, review_count=900),
            opt("weak", rating=3.2, review_count=400),
        ]

        assert ordering("popularity", options, Profile(name="p"))[0] == "proven"

    def test_among_equally_reviewed_options_the_better_rated_wins(self) -> None:
        options = [
            opt("good", rating=4.8, review_count=500),
            opt("bad", rating=3.1, review_count=500),
        ]

        assert ordering("popularity", options, Profile(name="p"))[0] == "good"

    def test_it_abstains_when_no_option_carries_a_rating(self) -> None:
        assert build("popularity").rank([opt("a", price=10)], Profile(name="p")) is None


class TestSemantic:
    def test_it_abstains_on_the_null_embedder(self) -> None:
        options = [opt("a", "quiet vegetarian dinner"), opt("b", "loud steakhouse")]
        profile = Profile(name="p", query="quiet vegetarian")

        assert build("semantic", embedder=NullEmbedder()).rank(options, profile) is None

    def test_it_prefers_the_option_whose_text_overlaps_the_query(self) -> None:
        options = [
            opt("match", "a quiet vegetarian dining room with a short seasonal menu"),
            opt("miss", "a loud sports bar showing football with wings and beer"),
        ]
        profile = Profile(name="p", query="quiet vegetarian dining")

        assert ordering("semantic", options, profile, embedder=HashingEmbedder())[0] == "match"


class TestConstraint:
    def test_it_ranks_by_the_soft_rules(self) -> None:
        options = [opt("a", price=50, rating=4.0), opt("b", price=20, rating=4.9)]
        profile = Profile(
            name="p",
            soft=[
                SoftRule(field="price", prefer="low", weight=1.0),
                SoftRule(field="rating", prefer="high", weight=1.0),
            ],
        )

        assert ordering("constraint", options, profile) == ["b", "a"]

    def test_it_abstains_when_the_profile_has_no_soft_rules(self) -> None:
        assert build("constraint").rank([opt("a", price=1)], Profile(name="p")) is None


def test_ties_break_on_option_id_so_orderings_are_stable() -> None:
    options = [opt("z", rating=4.0, review_count=100), opt("a", rating=4.0, review_count=100)]

    assert ordering("popularity", options, Profile(name="p")) == ["a", "z"]


def test_asking_for_an_unknown_ranker_is_an_error() -> None:
    with pytest.raises(KeyError):
        build("magic")
