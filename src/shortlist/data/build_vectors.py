"""Regenerate the committed option vectors. Run with `just embed`."""

from __future__ import annotations

import numpy as np

from shortlist.data import DATA_DIR, DATASETS
from shortlist.embedding import HashingEmbedder


def main() -> None:
    embedder = HashingEmbedder()
    for dataset in DATASETS.values():
        texts = [option.text or option.title for option in dataset.options]
        vectors = embedder.embed(texts)
        path = DATA_DIR / f"{dataset.key}.npz"
        np.savez_compressed(path, texts=np.asarray(texts), vectors=np.asarray(vectors))
        print(f"{path.name}: {len(texts)} vectors of {len(vectors[0])} dimensions")


if __name__ == "__main__":
    main()
