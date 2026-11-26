"""
Document loader — extracts raw text from PDF files.

Uses *pymupdf* (the ``fitz`` package) for fast, layout‑aware text
extraction.  Falls back gracefully with a clear error when a file
cannot be read.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import fitz  # pymupdf

logger = logging.getLogger(__name__)


def load_pdf(path: str | Path) -> str:
    """Return the full plain‑text content of a PDF file.

    Parameters
    ----------
    path:
        Filesystem path to a ``.pdf`` file.

    Returns
    -------
    str
        Concatenated text of every page, separated by newlines.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    RuntimeError
        If *pymupdf* cannot open or parse the file.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    try:
        doc = fitz.open(str(filepath))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{filepath}': {exc}") from exc

    pages: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        text: Optional[str] = page.get_text("text")
        if text:
            pages.append(text)
        else:
            logger.warning("Page %d of '%s' yielded no text.", page_num, filepath.name)

    doc.close()
    full_text = "\n".join(pages)
    logger.info(
        "Loaded %d pages (%d chars) from '%s'.",
        len(pages),
        len(full_text),
        filepath.name,
    )
    return full_text
