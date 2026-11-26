"""
Centralised configuration for the RAG pipeline.

All tuneable knobs live here so they are easy to override from env‑vars,
a YAML file, or CLI flags without touching business logic.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
CHROMADB_DIR: Path = PROJECT_ROOT / "data" / "chromadb"

# ── Chunking ─────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))

# ── ChromaDB ─────────────────────────────────────────────────────────────
COLLECTION_NAME: str = os.getenv("RAG_COLLECTION", "rag_docs")

# ── Retrieval defaults ───────────────────────────────────────────────────
DEFAULT_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
