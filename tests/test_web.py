"""The HTTP surface: a JSON endpoint and the HTMX demo."""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shortlist import rank
from shortlist.data import DATASETS
from shortlist.embedding import HashingEmbedder, PrecomputedEmbedder
from shortlist.models import Option, Profile
from shortlist.serialisation import to_dict
from shortlist.web.app import app

FIXTURES = Path(__file__).parent / "fixtures"
ROW = re.compile(r'data-option-id="([^"]+)"')


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def rows(html: str) -> list[str]:
    return ROW.findall(html)


def test_post_rank_returns_the_same_ordering_as_the_library(
    client: TestClient,
    dining_options: list[Option],
    dining_profile: Profile,
) -> None:
    payload = {
        "options": json.loads((FIXTURES / "dining.json").read_text()),
        "profile": {
            "name": dining_profile.name,
            "query": dining_profile.query,
            "hard": [
                {"field": "price_gbp_pp", "max": 60},
                {"field": "tags", "contains_none": ["shellfish"]},
            ],
            "soft": [
                {"field": "price_gbp_pp", "prefer": "low", "weight": 1.0},
                {"field": "rating", "prefer": "high", "weight": 1.5},
            ],
        },
    }

    response = client.post("/rank", json=payload)

    assert response.status_code == 200
    assert response.json() == to_dict(
        rank(dining_options, dining_profile, embedder=HashingEmbedder())
    )


def test_post_rank_honours_the_requested_rankers(client: TestClient) -> None:
    payload = {
        "options": json.loads((FIXTURES / "dining.json").read_text()),
        "profile": {"name": "d", "query": "quiet vegetarian dinner"},
        "rankers": ["lexical"],
    }

    body = client.post("/rank", json=payload).json()

    assert body["rankers"] == ["lexical"]


def test_post_rank_rejects_an_unknown_ranker(client: TestClient) -> None:
    payload = {
        "options": [{"id": "a", "title": "a"}],
        "profile": {"name": "d"},
        "rankers": ["vibes"],
    }

    assert client.post("/rank", json=payload).status_code == 422


class TestDemo:
    def test_the_first_paint_carries_the_cold_start_banner(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        # Server-rendered, so it is on screen while the machine is still waking up.
        assert 'id="cold-start"' in response.text
        assert rows(response.text)  # and a ranking, not an empty form

    def test_turning_a_ranker_off_changes_the_order(self, client: TestClient) -> None:
        dataset = DATASETS["london-restaurants"]
        form = {
            "dataset": dataset.key,
            "query": dataset.query,
            "rankers": ["constraint", "lexical", "semantic", "popularity"],
            "max__price_gbp_pp": "150",
            "max__distance_km": "12",
            "min__rating": "3.5",
        }

        with_all = client.post("/demo/rank", data=form)
        without_popularity = client.post(
            "/demo/rank", data={**form, "rankers": ["constraint", "lexical", "semantic"]}
        )

        assert with_all.status_code == 200
        assert rows(with_all.text) != rows(without_popularity.text)

    def test_the_sliders_actually_filter(self, client: TestClient) -> None:
        dataset = DATASETS["london-restaurants"]
        form = {
            "dataset": dataset.key,
            "query": dataset.query,
            "rankers": ["constraint", "popularity"],
            "max__price_gbp_pp": "20",
            "max__distance_km": "12",
            "min__rating": "3.5",
        }

        shown = rows(client.post("/demo/rank", data=form).text)
        options = {option.id: option for option in dataset.options}

        assert shown
        assert all(options[oid].fields["price_gbp_pp"] <= 20 for oid in shown)

    def test_excluding_a_tag_removes_every_option_carrying_it(self, client: TestClient) -> None:
        dataset = DATASETS["london-restaurants"]
        form = {
            "dataset": dataset.key,
            "query": dataset.query,
            "rankers": ["constraint", "lexical"],
            "max__price_gbp_pp": "150",
            "max__distance_km": "12",
            "min__rating": "3.5",
            "exclude": ["shellfish"],
        }

        shown = rows(client.post("/demo/rank", data=form).text)
        options = {option.id: option for option in dataset.options}

        assert shown
        assert all("shellfish" not in options[oid].fields["cuisine_tags"] for oid in shown)

    def test_every_bundled_dataset_ranks(self, client: TestClient) -> None:
        for key, dataset in DATASETS.items():
            form = {"dataset": key, "query": dataset.query, "rankers": ["constraint", "lexical"]}
            for control in dataset.controls:
                form[f"{control.bound}__{control.field}"] = str(
                    control.maximum if control.bound == "max" else control.minimum
                )
            assert rows(client.post("/demo/rank", data=form).text)


def test_the_committed_vectors_match_the_committed_datasets() -> None:
    # Edit a dataset's text without running `just embed` and this fails.
    embedder = HashingEmbedder()
    for dataset in DATASETS.values():
        cached = dataset.vectors()
        texts = [option.text or option.title for option in dataset.options]
        assert set(cached) == set(texts)
        assert cached == dict(zip(texts, embedder.embed(texts), strict=True))


def test_the_precomputed_embedder_only_computes_what_it_is_missing() -> None:
    calls: list[list[str]] = []

    class Counting:
        def embed(self, texts: list[str]) -> list[list[float]]:
            calls.append(texts)
            return HashingEmbedder().embed(texts)

    dataset = DATASETS["london-restaurants"]
    embedder = PrecomputedEmbedder(dataset.vectors(), Counting())
    texts = [option.text for option in dataset.options]

    embedder.embed([*texts, "a query nobody precomputed"])

    assert calls == [["a query nobody precomputed"]]
