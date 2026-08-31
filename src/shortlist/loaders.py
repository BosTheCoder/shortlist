"""Reading options and profiles off disk. JSON, CSV or YAML for options; YAML for profiles."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from shortlist.models import HardRule, Option, Profile, SoftRule

RESERVED = {"id", "title", "text", "fields"}


def option_from_record(record: dict[str, Any]) -> Option:
    """Anything that is not id/title/text becomes a field."""
    fields = {key: value for key, value in record.items() if key not in RESERVED}
    fields.update(record.get("fields") or {})
    return Option(
        id=str(record["id"]),
        title=str(record.get("title", record["id"])),
        text=str(record.get("text", "")),
        fields=fields,
    )


def _coerce(value: str) -> Any:
    """CSV has no types. Numbers become numbers, `a;b` becomes a list."""
    if ";" in value:
        return [part.strip() for part in value.split(";") if part.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_options(path: str | Path) -> list[Option]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [
                {key: _coerce(value) for key, value in row.items() if value != ""}
                for row in csv.DictReader(handle)
            ]
    elif suffix in {".yaml", ".yml"}:
        rows = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows["options"]
    return [option_from_record(row) for row in rows]


def profile_from_dict(data: dict[str, Any]) -> Profile:
    return Profile(
        name=str(data.get("name", "profile")),
        query=str(data.get("query", "")),
        hard=[HardRule(**rule) for rule in data.get("hard") or []],
        soft=[SoftRule(**rule) for rule in data.get("soft") or []],
        weights={str(k): float(v) for k, v in (data.get("weights") or {}).items()},
    )


def load_profile(path: str | Path) -> Profile:
    return profile_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
