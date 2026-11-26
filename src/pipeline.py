"""
End‑to‑end ingestion pipeline.

Orchestrates:  **Load → Clean → Chunk → Store**

The caller picks the chunking strategy (token‑window or heading‑based).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from src.chunker import Chunk, by_headings, by_tokens
from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.loader import load_pdf
from src.preprocessor import clean
from src.vector_store import add_chunks, query as vs_query

logger = logging.getLogger(__name__)

ChunkStrategy = Literal["tokens", "headings"]


def ingest(
    pdf_path: str | Path,
    strategy: ChunkStrategy = "tokens",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> int:
    """Run the full ingestion pipeline on a single PDF.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    strategy:
        ``"tokens"`` for fixed‑window chunking, ``"headings"`` for
        Markdown‑heading‑based chunking.
    chunk_size:
        Token‑window size (ignored for heading strategy).
    chunk_overlap:
        Token overlap (ignored for heading strategy).

    Returns
    -------
    int
        Number of chunks stored.
    """
    pdf_path = Path(pdf_path)
    logger.info("Pipeline START — file=%s, strategy=%s", pdf_path.name, strategy)

    # 1. Load
    raw_text: str = load_pdf(pdf_path)

    # 2. Clean
    cleaned: str = clean(raw_text)
    logger.info("Cleaned text: %d → %d chars.", len(raw_text), len(cleaned))

    # 3. Chunk
    chunks: list[Chunk]
    if strategy == "headings":
        chunks = by_headings(cleaned)
    else:
        chunks = by_tokens(cleaned, chunk_size=chunk_size, overlap=chunk_overlap)
    logger.info("Produced %d chunks.", len(chunks))

    # 4. Store
    stored = add_chunks(chunks, source=pdf_path.stem)
    logger.info("Pipeline DONE  — %d chunks stored.", stored)
    return stored


def retrieve(query_text: str, top_k: int = 5) -> list[dict]:
    """Thin wrapper around vector‑store query for external callers."""
    return vs_query(query_text, top_k=top_k)
