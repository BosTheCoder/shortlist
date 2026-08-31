"""`shortlist` — rank, record a pick, retrain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from shortlist.embedding import Embedder, HashingEmbedder, NullEmbedder, OpenAICompatibleEmbedder
from shortlist.learning import LearnedWeights, fit, weights_path
from shortlist.loaders import load_options, load_profile
from shortlist.pipeline import rank as rank_options
from shortlist.serialisation import to_dict
from shortlist.store import DEFAULT_ROOT, FeedbackStore

app = typer.Typer(add_completion=False, help="Rank options against a preference profile.")
console = Console()

EMBEDDERS: dict[str, type[Embedder]] = {
    "hashing": HashingEmbedder,
    "none": NullEmbedder,
    "openai": OpenAICompatibleEmbedder,
}

LogDir = Annotated[Path, typer.Option("--log-dir", help="Where runs and feedback are kept.")]


def _results_table(shortlist, explain: bool) -> Table:
    table = Table(title=f"{shortlist.profile}  ({shortlist.run_id})")
    table.add_column("#", justify="right")
    table.add_column("option")
    table.add_column("score", justify="right")
    table.add_column("spread", justify="right")
    if explain:
        for name in shortlist.rankers:
            table.add_column(name, justify="right")
    for result in shortlist.results:
        positions = {view.ranker: view.rank for view in result.per_ranker}
        row = [
            str(result.rank),
            result.option.title,
            f"{result.score:.5f}",
            f"{result.disagreement:.2f}",
        ]
        if explain:
            row += [str(positions[name]) for name in shortlist.rankers]
        table.add_row(*row)
    return table


@app.command()
def rank(
    options_path: Annotated[Path, typer.Argument(help="Options as JSON, CSV or YAML.")],
    profile: Annotated[Path, typer.Option("--profile", "-p", help="Profile YAML.")],
    top: Annotated[int, typer.Option("--top", help="How many to show. 0 for all.")] = 10,
    explain: Annotated[bool, typer.Option("--explain", help="Show each ranker's rank.")] = False,
    only: Annotated[str, typer.Option("--only", help="Comma-separated ranker names.")] = "",
    embedder: Annotated[
        str, typer.Option("--embedder", help="hashing, none or openai.")
    ] = "hashing",
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table.")] = False,
    log_dir: LogDir = DEFAULT_ROOT,
) -> None:
    """Rank OPTIONS_PATH against a profile."""
    loaded = load_options(options_path)
    prefs = load_profile(profile)
    if embedder not in EMBEDDERS:
        raise typer.BadParameter(f"unknown embedder {embedder!r}; try {sorted(EMBEDDERS)}")

    try:
        shortlist = rank_options(
            loaded,
            prefs,
            rankers=[name.strip() for name in only.split(",") if name.strip()] or None,
            embedder=EMBEDDERS[embedder](),
            weights=LearnedWeights.read(weights_path(profile)),
            top=top or None,
        )
    except KeyError as error:
        raise typer.BadParameter(str(error)) from error

    FeedbackStore(log_dir).record_run(shortlist)

    if as_json:
        typer.echo(json.dumps(to_dict(shortlist), indent=2))
        return
    console.print(_results_table(shortlist, explain))
    if shortlist.abstained:
        console.print(f"[dim]abstained: {', '.join(shortlist.abstained)}[/dim]")
    if shortlist.excluded:
        console.print(f"[dim]{len(shortlist.excluded)} filtered out by hard rules[/dim]")


@app.command()
def feedback(
    run_id: Annotated[str, typer.Argument(help="The run id printed by `shortlist rank`.")],
    picked: Annotated[str, typer.Option("--picked", help="The option id you chose.")],
    log_dir: LogDir = DEFAULT_ROOT,
) -> None:
    """Record which option you actually chose from a past run."""
    try:
        FeedbackStore(log_dir).record_pick(run_id, picked)
    except KeyError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    console.print(f"recorded {picked} for run {run_id}")


@app.command()
def train(
    profile: Annotated[Path, typer.Argument(help="The profile to personalise.")],
    log_dir: LogDir = DEFAULT_ROOT,
) -> None:
    """Fit the learned ranker's weights from recorded picks."""
    prefs = load_profile(profile)
    pairs = FeedbackStore(log_dir).training_pairs(prefs.name)
    weights = fit(pairs)
    if weights is None:
        console.print(f"[red]no feedback recorded for profile {prefs.name!r}[/red]")
        raise typer.Exit(1)
    destination = weights_path(profile)
    weights.write(destination)
    console.print(f"fitted {len(weights.features)} features from {weights.pairs} pairs")
    console.print(f"wrote {destination}")
