

from __future__ import annotations

import pickle
import sqlite3
import logging
import textwrap
from pathlib import Path
from typing import List, Dict
from src.config import CHUNK_WORDS,CHUNKS_PATH,OVERLAP_WORDS,MODEL_NAME,METADATA_PATH,INDEX_PATH,FAISS_INDEX_PATH,VECTORSTORE_DIR
import numpy as np

logger = logging.getLogger(__name__)



# ── text chunker ───────────────────────────────────────────────────────────────

def _chunk_text(text: str, order_id: str, chunk_words: int = CHUNK_WORDS,
                overlap: int = OVERLAP_WORDS) -> List[Dict]:
    words = text.split()
    chunked_data = []
    metadata = []
    step = chunk_words - overlap
    for i in range(0, max(1, len(words)), step):
        segment = " ".join(words[i: i + chunk_words])
        if len(segment.strip()) < 20:         
            continue
        chunked_data.append({"order_id": order_id, "text": segment, "start_word": i})
        metadata.append({
                "order_id": order_id,
                "text": segment[:200],
                "source": "Brazilian E-Commerce Public Dataset by Olist"
        })
    return chunked_data,metadata


def _load_reviews(db_path: Path) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT
            order_id,
            review_score,
            COALESCE(sentiment_label, 'neutral') AS sentiment,
            review_comment_message
        FROM order_reviews
        WHERE review_comment_message IS NOT NULL
          AND TRIM(review_comment_message) != ''
          AND LENGTH(review_comment_message) > 20
    """).fetchall()
    conn.close()
    logger.info(f"Loaded {len(rows):,} reviews with usable text from DB.")
    return rows


# ── embedding ──────────────────────────────────────────────────────────────────

def _get_model():
    """Lazy-load the sentence-transformer model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def _embed_chunks(chunks: List[Dict], model) -> np.ndarray:
    """Return (N, dim) float32 array of embeddings."""
    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts):,} chunks …")
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine ≡ dot product when normalised
    )
    return embeddings.astype("float32")


# ── FAISS index ────────────────────────────────────────────────────────────────

def _build_index(embeddings: np.ndarray):
    """Create a flat cosine-similarity FAISS index."""
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product on normalised vecs = cosine
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


# ── public API ─────────────────────────────────────────────────────────────────

def build_faiss_index(db_path: Path, force_rebuild: bool = False):

    import faiss

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists() and CHUNKS_PATH.exists() and not force_rebuild:
        logger.info("Loading FAISS index from cache …")
        index  = faiss.read_index(str(INDEX_PATH))
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        logger.info(f"Cache loaded: {index.ntotal} vectors, {len(chunks)} chunks.")
        return index, chunks

    # ── Build from scratch ────────────────────────────────────────────────────
    rows   = _load_reviews(db_path)
    print(f"\n✅ Loaded {len(rows):,} reviews from DB." )
    chunks = []
    all_metadata = []
    for order_id, review_score, sentiment, review_text in rows:
        chunked_data,metadata = _chunk_text(str(review_text), order_id)
        chunks.extend(chunked_data)
        all_metadata.extend(metadata)
    logger.info(f"Total chunks after splitting: {len(chunks):,}")

    model      = _get_model()
    embeddings = _embed_chunks(chunks, model)
    index      = _build_index(embeddings)

    # ── Persist ───────────────────────────────────────────────────────────────
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(all_metadata, f)
    logger.info("FAISS index and chunks saved.")

    return index, chunks
