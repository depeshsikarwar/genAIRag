"""
Text preprocessor — cleans raw text extracted from PDFs.

The single public entry‑point is :func:`clean`.  It applies a strict
sequence of transformations so that downstream chunking and embedding
receive normalised, noise‑free text.
"""

from __future__ import annotations

import re
import unicodedata


def clean(text: str) -> str:
    """Clean raw PDF‑extracted text for downstream NLP.

    Processing steps (in order):

    1. **De‑hyphenation** — recombine words broken across line‑breaks
       (e.g. ``"multi-\\n developer"`` → ``"multideveloper"``).
    2. **Unicode normalisation** — NFKD decomposition followed by
       stripping non‑ASCII combining marks, bullets (``•·●○■□▪▸►``),
       and miscellaneous symbols.
    3. **Whitespace normalisation** — collapse runs of spaces / tabs
       into a single space; collapse 3+ consecutive newlines into 2;
       strip leading/trailing whitespace.

    Parameters
    ----------
    text:
        Raw text, typically from :func:`loader.load_pdf`.

    Returns
    -------
    str
        Cleaned text ready for chunking.
    """
    if not text:
        return ""

    # 1. De‑hyphenation: word fragment + hyphen + newline + optional
    #    spaces + continuation → joined word.
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # 2. Unicode normalisation
    text = unicodedata.normalize("NFKD", text)

    # Remove combining marks (accents stripped by NFKD)
    text = "".join(
        ch for ch in text if not unicodedata.combining(ch)
    )

    # Remove common bullet / ornament glyphs
    text = re.sub(r"[•·●○■□▪▸►▻◦‣⁃※†‡§¶]", "", text)

    # Remove leftover non‑printable control chars (except \n \r \t)
    text = re.sub(r"[^\S\n\r\t]", " ", text)  # non‑standard spaces → ASCII space
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # 3. Whitespace normalisation
    text = re.sub(r"[ \t]+", " ", text)          # horizontal runs → single space
    text = re.sub(r"\n{3,}", "\n\n", text)        # 3+ newlines → 2
    text = text.strip()

    return text
