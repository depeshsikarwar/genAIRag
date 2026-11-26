"""
Vector store — ChromaDB persistence layer.

Wraps ChromaDB's persistent client so the rest of the pipeline only
talks to plain Python functions.  ChromaDB's built‑in
``all-MiniLM-L6-v2`` Sentence‑Transformer embedding is used by default
(no API key needed; runs fully offline).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import chromadb
from chromadb.config import Settings

from src.chunker import Chunk
from src.config import CHROMADB_DIR, COLLECTION_NAME, DEFAULT_TOP_K

logger = logging.getLogger(__name__)


def _get_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client rooted at ``CHROMADB_DIR``."""
    return chromadb.PersistentClient(
        path=str(CHROMADB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(client: chromadb.ClientAPI | None = None) -> chromadb.Collection:
    """Return (or create) the default collection."""
    client = client or _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Write path ───────────────────────────────────────────────────────────

def add_chunks(chunks: list[Chunk], source: str = "") -> int:
    """Upsert a batch of chunks into ChromaDB.

    Each chunk receives a deterministic ID derived from its content hash
    so that re‑ingesting the same document is idempotent.

    Parameters
    ----------
    chunks:
        Chunk objects produced by any chunking strategy.
    source:
        An optional label (e.g. the filename) stored in metadata.

    Returns
    -------
    int
        Number of chunks upserted.
    """
    if not chunks:
        return 0

    collection = _get_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in chunks:
        content_hash = hashlib.sha256(chunk.text.encode()).hexdigest()[:16]
        ids.append(f"{source}_{content_hash}" if source else content_hash)
        documents.append(chunk.text)

        meta = dict(chunk.metadata)
        if source:
            meta["source"] = source
        metadatas.append(meta)

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Upserted %d chunks (source=%s).", len(ids), source or "<none>")
    return len(ids)


# ── Read path ────────────────────────────────────────────────────────────

def query(
    text: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Semantic similarity search.

    Parameters
    ----------
    text:
        Natural‑language query string.
    top_k:
        Number of results to return.

    Returns
    -------
    list[dict]
        Each dict has keys ``id``, ``document``, ``metadata``, and
        ``distance``.
    """
    collection = _get_collection()

    results = collection.query(query_texts=[text], n_results=top_k)

    hits: list[dict[str, Any]] = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return hits


def collection_count() -> int:
    """Return the number of documents currently stored."""
    return _get_collection().count()
