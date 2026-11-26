"""Unit tests for src.preprocessor.clean()."""

from __future__ import annotations

import pytest

from src.preprocessor import clean


class TestDehyphenation:
    """PDF de‑hyphenation — words broken across line‑breaks."""

    def test_simple_dehyphenation(self) -> None:
        assert clean("multi-\n developer") == "multideveloper"

    def test_dehyphenation_with_extra_spaces(self) -> None:
        assert clean("multi-\n   developer") == "multideveloper"

    def test_dehyphenation_preserves_real_hyphens(self) -> None:
        # A hyphen NOT followed by a newline should survive.
        result = clean("well-known fact")
        assert "well-known" in result

    def test_multiple_dehyphenations(self) -> None:
        text = "micro-\nservices and multi-\nthreaded"
        result = clean(text)
        assert "microservices" in result
        assert "multithreaded" in result


class TestUnicodeAndBullets:
    """Unicode glyph / bullet removal."""

    def test_bullets_removed(self) -> None:
        assert clean("• item one") == "item one"
        assert clean("■ item two") == "item two"

    def test_multiple_bullet_styles(self) -> None:
        result = clean("● first\n○ second\n▸ third")
        assert "●" not in result
        assert "○" not in result
        assert "▸" not in result

    def test_control_characters_stripped(self) -> None:
        result = clean("hello\x00world\x07!")
        assert "\x00" not in result
        assert "\x07" not in result


class TestWhitespace:
    """Whitespace normalisation."""

    def test_collapse_spaces(self) -> None:
        assert clean("hello    world") == "hello world"

    def test_collapse_tabs(self) -> None:
        assert clean("hello\t\tworld") == "hello world"

    def test_collapse_excessive_newlines(self) -> None:
        result = clean("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_strip_leading_trailing(self) -> None:
        assert clean("   hello   ") == "hello"


class TestEdgeCases:
    """Edge / degenerate inputs."""

    def test_empty_string(self) -> None:
        assert clean("") == ""

    def test_only_whitespace(self) -> None:
        assert clean("   \n\n  \t  ") == ""

    def test_already_clean(self) -> None:
        assert clean("Hello world.") == "Hello world."
