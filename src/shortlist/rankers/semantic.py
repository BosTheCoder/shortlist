"""Cosine similarity between the profile query and each option's text."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shortlist.embedding import Embedder
from shortlist.models import Option, Profile, ScoredOption


class SemanticRanker:
    """Abstains when the embedder gives no signal, so `NullEmbedder` drops it from fusion."""

    name = "semantic"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def rank(self, options: Sequence[Option], profile: Profile) -> list[ScoredOption] | None:
        if not profile.query.strip() or not options:
            return None
        query = np.asarray(self.embedder.embed([profile.query])[0], dtype=float)
        if not np.any(query):
            return None
        texts = [option.text or option.title for option in options]
        matrix = np.asarray(self.embedder.embed(texts), dtype=float)
        if not np.any(matrix):
            return None

        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        similarity = (matrix @ query) / (norms * np.linalg.norm(query))
        return [
            ScoredOption(option_id=option.id, score=float(value))
            for option, value in zip(options, similarity, strict=True)
        ]
