"""Unit tests for src.chunker (by_tokens & by_headings)."""

from __future__ import annotations

import pytest

from src.chunker import Chunk, by_headings, by_tokens


# ═══════════════════════════════════════════════════════════════════════
#  by_tokens
# ═══════════════════════════════════════════════════════════════════════

class TestByTokens:
    """Fixed‑size token‑window chunking."""

    def test_basic_split(self) -> None:
        text = " ".join(f"w{i}" for i in range(20))
        chunks = by_tokens(text, chunk_size=10, overlap=0)
        assert len(chunks) == 2
        assert chunks[0].text.startswith("w0")

    def test_overlap(self) -> None:
        text = " ".join(f"w{i}" for i in range(15))
        chunks = by_tokens(text, chunk_size=10, overlap=5)
        # With 15 tokens, size 10, step 5 → starts at 0, 5, 10 → 3 chunks
        assert len(chunks) == 3
        # Second chunk should start at w5
        assert chunks[1].text.startswith("w5")

    def test_empty_input(self) -> None:
        assert by_tokens("") == []
        assert by_tokens("   ") == []

    def test_short_text_single_chunk(self) -> None:
        chunks = by_tokens("hello world", chunk_size=100, overlap=0)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"

    def test_returns_chunk_objects(self) -> None:
        chunks = by_tokens("a b c", chunk_size=2, overlap=0)
        assert all(isinstance(c, Chunk) for c in chunks)


# ═══════════════════════════════════════════════════════════════════════
#  by_headings
# ═══════════════════════════════════════════════════════════════════════

class TestByHeadings:
    """Hierarchical Markdown heading chunker with breadcrumbs."""

    SAMPLE = (
        "# Intro\n"
        "Some intro text.\n"
        "## Details\n"
        "Detail text here.\n"
        "### Sub‑detail\n"
        "Deep detail.\n"
        "# Outro\n"
        "Goodbye.\n"
    )

    def test_chunk_count(self) -> None:
        chunks = by_headings(self.SAMPLE)
        # Intro body, Details body, Sub‑detail body, Outro body → 4
        assert len(chunks) == 4

    def test_h1_breadcrumb(self) -> None:
        chunks = by_headings(self.SAMPLE)
        # First chunk under "# Intro"
        assert chunks[0].metadata == {"h1": "Intro"}

    def test_h2_breadcrumb(self) -> None:
        chunks = by_headings(self.SAMPLE)
        assert chunks[1].metadata == {"h1": "Intro", "h2": "Details"}

    def test_h3_breadcrumb(self) -> None:
        chunks = by_headings(self.SAMPLE)
        assert chunks[2].metadata == {
            "h1": "Intro",
            "h2": "Details",
            "h3": "Sub\u2011detail",
        }

    def test_pruning_on_new_h1(self) -> None:
        """When a new h1 appears, h2/h3 must be pruned."""
        chunks = by_headings(self.SAMPLE)
        outro = chunks[-1]
        assert outro.metadata == {"h1": "Outro"}
        assert "h2" not in outro.metadata
        assert "h3" not in outro.metadata

    def test_no_headings_returns_full_text(self) -> None:
        chunks = by_headings("Just plain text, no headings.")
        assert len(chunks) == 1
        assert chunks[0].text == "Just plain text, no headings."
        assert chunks[0].metadata == {}

    def test_empty_input(self) -> None:
        assert by_headings("") == []
        assert by_headings("   ") == []

    def test_heading_without_body(self) -> None:
        """A heading followed immediately by another heading → no chunk for the bodyless one."""
        text = "# A\n# B\nBody of B."
        chunks = by_headings(text)
        # A has no body → skipped; only B's body
        assert len(chunks) == 1
        assert chunks[0].metadata["h1"] == "B"

    def test_preamble_before_first_heading(self) -> None:
        text = "Some preamble.\n# Chapter 1\nContent."
        chunks = by_headings(text)
        assert chunks[0].text == "Some preamble."
        assert chunks[0].metadata == {}
        assert chunks[1].metadata == {"h1": "Chapter 1"}

    def test_deeper_prune_on_mid_level(self) -> None:
        """Encountering ## should prune ###+ but keep #."""
        text = "# Top\n## Mid\n### Deep\ndeep body\n## Mid2\nmid2 body"
        chunks = by_headings(text)
        mid2 = [c for c in chunks if c.metadata.get("h2") == "Mid2"][0]
        assert "h3" not in mid2.metadata
        assert mid2.metadata["h1"] == "Top"
