"""Where runs and picks are kept between commands.

`shortlist rank` writes the run; `shortlist feedback` points at it by id. The run
file carries each shown option's numeric features so that training does not need
the original dataset back.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from shortlist.learning import numeric_features
from shortlist.models import Shortlist

DEFAULT_ROOT = Path(".shortlist")


class FeedbackStore:
    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.feedback_path = self.root / "feedback.jsonl"

    def record_run(self, shortlist: Shortlist) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"{shortlist.run_id}.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": shortlist.run_id,
                    "profile": shortlist.profile,
                    "features": {
                        result.option.id: numeric_features(result.option)
                        for result in shortlist.results
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return path

    def read_run(self, run_id: str) -> dict[str, object]:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            raise KeyError(f"no run {run_id!r} recorded under {self.root}")
        return json.loads(path.read_text())

    def record_pick(self, run_id: str, picked: str) -> None:
        run = self.read_run(run_id)
        features = run["features"]
        assert isinstance(features, dict)
        if picked not in features:
            raise KeyError(f"option {picked!r} was not shown in run {run_id!r}")
        self.root.mkdir(parents=True, exist_ok=True)
        row = {
            "run_id": run_id,
            "picked": picked,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with self.feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def picks(self) -> list[dict[str, str]]:
        if not self.feedback_path.exists():
            return []
        return [
            json.loads(line) for line in self.feedback_path.read_text().splitlines() if line.strip()
        ]

    def training_pairs(self, profile: str) -> list[tuple[dict[str, float], dict[str, float]]]:
        """Every pick becomes one pair per option it was chosen over."""
        pairs: list[tuple[dict[str, float], dict[str, float]]] = []
        for pick in self.picks():
            run = self.read_run(pick["run_id"])
            if run["profile"] != profile:
                continue
            features = run["features"]
            assert isinstance(features, dict)
            chosen = features.get(pick["picked"])
            if chosen is None:
                continue
            pairs.extend(
                (chosen, other) for oid, other in features.items() if oid != pick["picked"]
            )
        return pairs
