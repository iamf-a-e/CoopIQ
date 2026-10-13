"""
build_embeddings.py

Offline/build-time script for CoopIQ.

Computes a Gemini embedding for every passage in poultry_data.py and caches
the result to embeddings.json. Run this once locally (and again whenever
poultry_data.py changes) so that the live webhook in main.py never has to
call the embeddings API on the hot path -- it just loads embeddings.json and
does an in-memory cosine-similarity search.

Usage (macOS / Linux, zsh or bash):
    export GEN_API="your-key-here"
    python build_embeddings.py

Output:
    embeddings.json -- a list of
        {"id": ..., "text": ..., "source": ..., "section": ...,
         "topic": ..., "embedding": [float, ...]}
    aligned 1:1 with POULTRY_KNOWLEDGE_BASE, in the same order.

IMPORTANT: EMBEDDING_MODEL and EMBEDDING_DIM here must match retrieval.py.
"""

import json
import os
import sys
import time

import numpy as np
import google.generativeai as genai
from google.api_core import exceptions as gexc

from poultry_data import POULTRY_KNOWLEDGE_BASE

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768  # must match retrieval.py
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "embeddings.json")

# Embedding one passage at a time is simple and reliable for a knowledge
# base of this size (~100 docs).
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list:
    """Call the Gemini embeddings API for a single passage, with retries."""
    last_error = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIM,
            )
            # Vectors below 3072 dims aren't unit-length, so normalize.
            v = np.array(result["embedding"], dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            return v.tolist()
        except gexc.NotFound:
            # Wrong or retired model name: retrying won't help.
            raise
        except Exception as exc:  # noqa: BLE001 - want to retry on anything else
            last_error = exc
            print(f"  attempt {attempt} failed: {exc}", file=sys.stderr)
            time.sleep(RETRY_DELAY_SECONDS * attempt)  # simple backoff for 429s
    raise RuntimeError(f"Failed to embed text after {RETRY_ATTEMPTS} attempts") from last_error


def main() -> None:
    api_key = os.environ.get("GEN_API")
    if not api_key:
        print("ERROR: set the GEN_API environment variable to your Gemini API key.", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=api_key)

    print(
        f"Embedding {len(POULTRY_KNOWLEDGE_BASE)} passages with "
        f"{EMBEDDING_MODEL} ({EMBEDDING_DIM} dims) ..."
    )
    records = []
    for i, doc in enumerate(POULTRY_KNOWLEDGE_BASE, start=1):
        embedding = embed_text(doc["text"], task_type="RETRIEVAL_DOCUMENT")
        records.append(
            {
                "id": doc["id"],
                "text": doc["text"],
                "source": doc["source"],
                "section": doc["section"],
                "topic": doc["topic"],
                "embedding": embedding,
            }
        )
        print(f"  [{i}/{len(POULTRY_KNOWLEDGE_BASE)}] {doc['id']}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f)

    print(f"Wrote {len(records)} embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
