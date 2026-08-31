"""Embedders for the semantic ranker.

Three of them, on purpose. `NullEmbedder` is the "no signal" case that makes the
semantic ranker abstain. `HashingEmbedder` is a dependency-free signed hashing
projection: it needs no model and no network, so the hosted demo can embed a
query typed a second ago. `OpenAICompatibleEmbedder` is what you use when you
want real semantics.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")

DEFAULT_DIM = 256


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class NullEmbedder:
    """Returns zero vectors. The semantic ranker treats that as an abstention."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * DEFAULT_DIM for _ in texts]


class HashingEmbedder:
    """Signed hashing trick with sublinear term frequency, L2 normalised.

    Not a learned model: two texts are close when they share vocabulary, not
    when they share meaning. It is deterministic across processes and machines,
    which is what makes the bundled vectors reproducible.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim

    def _index_and_sign(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if value >> 63 & 1 else -1.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            counts: dict[str, int] = {}
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
            vector = [0.0] * self.dim
            for token, count in counts.items():
                index, sign = self._index_and_sign(token)
                vector[index] += sign * (1.0 + math.log(count))
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


class OpenAICompatibleEmbedder:
    """Any `/embeddings` endpoint that speaks the OpenAI shape (OpenAI, OpenRouter, ...)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("EMBEDDING_BASE_URL") or "").rstrip(
            "/"
        ) or "https://api.openai.com/v1"
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY", "")
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [item["embedding"] for item in payload["data"]]
