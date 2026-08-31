"""The CLI is a thin skin over the library. These check it stays thin."""

import json
from pathlib import Path

from typer.testing import CliRunner

from shortlist import rank
from shortlist.cli import app
from shortlist.embedding import HashingEmbedder
from shortlist.learning import LearnedWeights
from shortlist.serialisation import to_dict

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def test_rank_json_output_matches_the_library(
    tmp_path: Path, dining_options, dining_profile
) -> None:
    result = runner.invoke(
        app,
        [
            "rank",
            str(FIXTURES / "dining.json"),
            "--profile",
            str(FIXTURES / "dining.yaml"),
            "--json",
            "--log-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    expected = to_dict(rank(dining_options, dining_profile, embedder=HashingEmbedder()))
    assert json.loads(result.stdout) == expected


def test_only_restricts_the_rankers_that_run(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "rank",
            str(FIXTURES / "dining.json"),
            "--profile",
            str(FIXTURES / "dining.yaml"),
            "--only",
            "lexical,popularity",
            "--json",
            "--log-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["rankers"] == ["lexical", "popularity"]


def test_top_limits_the_rows(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "rank",
            str(FIXTURES / "dining.json"),
            "--profile",
            str(FIXTURES / "dining.yaml"),
            "--top",
            "2",
            "--json",
            "--log-dir",
            str(tmp_path),
        ],
    )

    assert len(json.loads(result.stdout)["results"]) == 2


def test_an_unknown_ranker_name_fails_the_command(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "rank",
            str(FIXTURES / "dining.json"),
            "--profile",
            str(FIXTURES / "dining.yaml"),
            "--only",
            "vibes",
            "--log-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0


def test_rank_then_feedback_then_train_produces_usable_weights(tmp_path: Path) -> None:
    profile_path = tmp_path / "dining.yaml"
    profile_path.write_text((FIXTURES / "dining.yaml").read_text())
    common = ["--profile", str(profile_path), "--log-dir", str(tmp_path)]

    ranked = runner.invoke(app, ["rank", str(FIXTURES / "dining.json"), *common, "--json"])
    assert ranked.exit_code == 0, ranked.output
    run_id = json.loads(ranked.stdout)["run_id"]

    picked = runner.invoke(
        app, ["feedback", run_id, "--picked", "echo", "--log-dir", str(tmp_path)]
    )
    assert picked.exit_code == 0, picked.output

    trained = runner.invoke(app, ["train", str(profile_path), "--log-dir", str(tmp_path)])
    assert trained.exit_code == 0, trained.output

    weights = LearnedWeights.read(tmp_path / "dining.weights.json")
    assert weights is not None
    assert weights.pairs == 4  # echo beat the four other survivors

    # The trained model is picked up on the next rank without being asked for.
    after = runner.invoke(app, ["rank", str(FIXTURES / "dining.json"), *common, "--json"])
    assert "learned" in json.loads(after.stdout)["rankers"]


def test_training_with_no_feedback_exits_nonzero(tmp_path: Path) -> None:
    profile_path = tmp_path / "dining.yaml"
    profile_path.write_text((FIXTURES / "dining.yaml").read_text())

    result = runner.invoke(app, ["train", str(profile_path), "--log-dir", str(tmp_path)])

    assert result.exit_code != 0
    assert not (tmp_path / "dining.weights.json").exists()


def test_feedback_on_an_unrecorded_run_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["feedback", "deadbeef", "--picked", "echo", "--log-dir", str(tmp_path)]
    )

    assert result.exit_code != 0
