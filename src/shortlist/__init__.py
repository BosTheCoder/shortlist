"""Rank options against a written preference profile, and show why."""

from shortlist.embedding import (
    Embedder,
    HashingEmbedder,
    NullEmbedder,
    OpenAICompatibleEmbedder,
)
from shortlist.loaders import load_options, load_profile
from shortlist.models import (
    Excluded,
    HardRule,
    Option,
    Profile,
    RankerView,
    Result,
    ScoredOption,
    Shortlist,
    SoftRule,
)
from shortlist.pipeline import rank
from shortlist.serialisation import to_dict

__all__ = [
    "Embedder",
    "Excluded",
    "HardRule",
    "HashingEmbedder",
    "NullEmbedder",
    "OpenAICompatibleEmbedder",
    "Option",
    "Profile",
    "RankerView",
    "Result",
    "ScoredOption",
    "Shortlist",
    "SoftRule",
    "load_options",
    "load_profile",
    "rank",
    "to_dict",
]
