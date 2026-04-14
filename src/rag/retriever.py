"""
rag/retriever.py
────────────────
Wraps the FAISS index to provide semantic search over review chunks.

Exports
-------
    Retriever.retrieve(query, k=5) -> list[dict]
"""

from __future__ import annotations

import logging
from typing import List, Dict

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    """Semantic retriever backed by a FAISS flat index."""

    def __init__(self, index, chunks: List[Dict]):
        self.index  = index
        self.chunks = chunks
        self._model = None

    # ── lazy model load ────────────────────────────────────────────────────────
    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    # ── public API ─────────────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve the top-k most semantically similar review chunks.

        Parameters
        ----------
        query : str
            Natural language question.
        k : int
            Number of chunks to return.

        Returns
        -------
        list[dict]
            Each dict has keys: order_id, text, start_word, score.
        """
        model = self._get_model()
        q_emb = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        logger.info(f"Retrieved {len(results)} chunks for query: {query[:60]}…")
        return results
