from pathlib import Path

import pytest

from shortlist.loaders import load_options, load_profile
from shortlist.models import Option, Profile

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def dining_options() -> list[Option]:
    return load_options(FIXTURES / "dining.json")


@pytest.fixture
def dining_profile() -> Profile:
    return load_profile(FIXTURES / "dining.yaml")
