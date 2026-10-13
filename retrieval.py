"""
retrieval.py

Runtime RAG retrieval layer for CoopIQ.

Loads the pre-computed embeddings in embeddings.json (built offline by
build_embeddings.py from poultry_data.py) and, given a user's question,
returns the most relevant knowledge passages by cosine similarity between
the query embedding and each cached document embedding.

Kept dependency-light (no faiss/annoy) since the knowledge base is small
(~100 passages) -- a brute-force numpy cosine search is fast enough and
simple enough to run inside a serverless function.

IMPORTANT: EMBEDDING_MODEL and EMBEDDING_DIM here must match
build_embeddings.py, and embeddings.json must be rebuilt after any change.
"""

import json
import logging
import os
from typing import Dict, List

import numpy as np
import google.generativeai as genai

logger = logging.getLogger("coopiq")

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768  # must match build_embeddings.py
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.json")

# Score thresholds differ between embedding models. While tuning min_score,
# keep this on to log the top similarity scores for each query; then set
# RETRIEVAL_DEBUG=0 (or delete the log line) once you've settled on a value.
RETRIEVAL_DEBUG = os.environ.get("RETRIEVAL_DEBUG", "1") == "1"

_cache: Dict[str, np.ndarray] = {}
_records: List[dict] = []


def _load_embeddings() -> None:
    """Load embeddings.json into memory once per process (cold start)."""
    global _records
    if _records:
        return
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"{EMBEDDINGS_PATH} not found. Run build_embeddings.py first "
            "to generate it from poultry_data.py."
        )
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        _records = json.load(f)
    _cache["matrix"] = np.array([r["embedding"] for r in _records], dtype=np.float32)
    # Pre-normalize for fast cosine similarity via dot product.
    norms = np.linalg.norm(_cache["matrix"], axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    _cache["normalized"] = _cache["matrix"] / norms


def embed_query(text: str) -> np.ndarray:
    """Embed a user query with Gemini, using the query-optimized task type."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=EMBEDDING_DIM,
    )
    vec = np.array(result["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def retrieve(query: str, top_k: int = 4, min_score: float = 0.55) -> List[dict]:
    """
    Return the top_k most relevant knowledge passages for `query`.

    Each returned dict has: id, text, source, section, topic, score.
    Passages scoring below `min_score` (cosine similarity, 0-1) are dropped
    so that off-topic questions don't get force-fed irrelevant context.

    NOTE: 0.55 was tuned for text-embedding-004. Re-tune it for
    gemini-embedding-001 using the RETRIEVAL_DEBUG score log.
    """
    _load_embeddings()

    query_vec = embed_query(query)

    if query_vec.shape[0] != _cache["normalized"].shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: query={query_vec.shape[0]}, "
            f"index={_cache['normalized'].shape[1]}. Rebuild embeddings.json."
        )

    scores = _cache["normalized"] @ query_vec  # cosine similarity, since both sides are unit vectors

    if RETRIEVAL_DEBUG:
        logger.info("Retrieval top scores: %s", np.sort(scores)[::-1][:5].round(3).tolist())

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < min_score:
            continue
        doc = _records[idx]
        results.append(
            {
                "id": doc["id"],
                "text": doc["text"],
                "source": doc["source"],
                "section": doc["section"],
                "topic": doc["topic"],
                "score": round(score, 4),
            }
        )
    return results


def format_context(passages: List[dict]) -> str:
    """Render retrieved passages into a context block for the generation prompt."""
    if not passages:
        return ""
    blocks = []
    for p in passages:
        blocks.append(f"[{p['topic']} — {p['source']}]\n{p['text']}")
    return "\n\n".join(blocks)
