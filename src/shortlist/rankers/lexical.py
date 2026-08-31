"""BM25 over the option text."""

from __future__ import annotations

import math
from collections.abc import Sequence

from shortlist.embedding import tokenize
from shortlist.models import Option, Profile, ScoredOption

K1 = 1.5
B = 0.75


class LexicalRanker:
    name = "lexical"

    def rank(self, options: Sequence[Option], profile: Profile) -> list[ScoredOption] | None:
        query = tokenize(profile.query)
        docs = {option.id: tokenize(f"{option.title} {option.text}") for option in options}
        if not query or not options or not any(docs.values()):
            return None

        lengths = {oid: len(tokens) for oid, tokens in docs.items()}
        average_length = sum(lengths.values()) / len(lengths)
        total = len(docs)

        document_frequency = {
            term: sum(1 for tokens in docs.values() if term in tokens) for term in set(query)
        }

        scored: list[ScoredOption] = []
        for oid, tokens in docs.items():
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            length_penalty = (
                K1 * (1 - B + B * lengths[oid] / average_length) if average_length else K1
            )
            score = 0.0
            for term in set(query):
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                score += idf * frequency * (K1 + 1) / (frequency + length_penalty)
            scored.append(ScoredOption(option_id=oid, score=score))
        return scored
