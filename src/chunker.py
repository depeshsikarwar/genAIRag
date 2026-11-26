"""
Text chunking strategies.

Two strategies are provided:

* :func:`by_tokens` — fixed‑size token‑window chunking with configurable
  overlap (good default for dense prose).
* :func:`by_headings` — hierarchical Markdown‑heading chunker that
  preserves structural context via breadcrumb metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


# ── Data container ───────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk with optional heading‑breadcrumb metadata."""

    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        preview = self.text[:60].replace("\n", "\\n")
        return f"Chunk(len={len(self.text)}, meta={self.metadata}, text='{preview}…')"


# ── Strategy 1: fixed‑size token windows ─────────────────────────────────

def by_tokens(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split *text* into overlapping token‑window chunks.

    Tokens are defined as whitespace‑delimited words.  Each chunk
    contains at most *chunk_size* tokens, and successive chunks share
    *overlap* tokens for context continuity.

    Parameters
    ----------
    text:
        The cleaned document text.
    chunk_size:
        Maximum number of tokens per chunk.
    overlap:
        Number of tokens shared between consecutive chunks.

    Returns
    -------
    list[Chunk]
        Ordered list of chunks.
    """
    tokens = text.split()
    if not tokens:
        return []

    step = max(chunk_size - overlap, 1)
    chunks: list[Chunk] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        chunks.append(Chunk(text=" ".join(window)))
    return chunks


# ── Strategy 2: Markdown heading hierarchy ───────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)

# Canonical key order for breadcrumb levels
_LEVEL_KEYS: dict[int, str] = {1: "h1", 2: "h2", 3: "h3", 4: "h4", 5: "h5", 6: "h6"}


def by_headings(text: str) -> list[Chunk]:
    """Split *text* by Markdown headings and attach breadcrumb metadata.

    The function recognises ``#`` through ``######`` headings.  For every
    section between two headings (or between a heading and EOF) it emits
    a :class:`Chunk` whose ``metadata`` dict carries the **full heading
    path** (breadcrumb) at the point of that section.

    When a higher‑level heading is encountered, all deeper keys are
    pruned from the breadcrumb so that the path always reflects the
    current nesting accurately.

    Example
    -------
    Given::

        # Intro
        Some intro text.
        ## Details
        Detail text.
        # Outro
        Bye.

    Returns three chunks with metadata::

        {"h1": "Intro"}
        {"h1": "Intro", "h2": "Details"}
        {"h1": "Outro"}

    Parameters
    ----------
    text:
        Cleaned Markdown‑formatted text.

    Returns
    -------
    list[Chunk]
        One chunk per headed section, in document order.
    """
    chunks: list[Chunk] = []
    breadcrumb: dict[str, str] = {}

    # Find every heading and its position
    headings: list[tuple[int, int, str]] = []  # (pos, level, title)
    for m in _HEADING_RE.finditer(text):
        level = len(m.group(1))
        title = m.group(2).strip()
        headings.append((m.start(), level, title))

    if not headings:
        # No headings at all — return the whole text as one chunk.
        stripped = text.strip()
        if stripped:
            return [Chunk(text=stripped)]
        return []

    # Text before the first heading (preamble)
    preamble = text[: headings[0][0]].strip()
    if preamble:
        chunks.append(Chunk(text=preamble, metadata=dict(breadcrumb)))

    for idx, (pos, level, title) in enumerate(headings):
        # 1. Update breadcrumb: set current level and prune deeper keys
        key = _LEVEL_KEYS[level]
        breadcrumb[key] = title

        deeper_keys = [_LEVEL_KEYS[lv] for lv in range(level + 1, 7)]
        for dk in deeper_keys:
            breadcrumb.pop(dk, None)

        # 2. Extract body text between this heading and the next
        heading_line_end = text.index("\n", pos) + 1 if "\n" in text[pos:] else len(text)
        next_pos = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        body = text[heading_line_end:next_pos].strip()

        if body:
            chunks.append(Chunk(text=body, metadata=dict(breadcrumb)))

    return chunks
