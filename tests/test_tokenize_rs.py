"""
Correctness tests for the Rust encode_and_spans_rs / init_vocab_rs /
encode_and_spans_positions_rs functions.

Each test compares the Rust output against the reference Python implementation
in Tokenizer.encode() and Tokenizer.get_span_start_positions().
"""
from __future__ import annotations

import numpy as np
import pytest
from softmatcha_rs import encode_and_spans_rs, encode_and_spans_positions_rs, init_vocab_rs
from softmatcha.tokenizers.base import Tokenizer


# ---------------------------------------------------------------------------
# Helpers that replicate the Python reference implementations
# ---------------------------------------------------------------------------

def py_encode(tokens: list[str], dictionary: dict[str, int], unk_idx: int) -> np.ndarray:
    """Reference: Tokenizer.encode"""
    n = len(tokens)
    out = np.empty(n, dtype=np.uint32)
    for i, tok in enumerate(tokens):
        out[i] = dictionary.get(tok, unk_idx)
    return out


def py_get_span_start_positions(line: str, tokens: list[str]) -> np.ndarray:
    """Reference: Tokenizer.get_span_start_positions"""
    span_starts = np.empty(len(tokens), dtype=np.uint32)
    start_position = 0
    prev_pos = 0
    cumsum = 0
    for i, token in enumerate(tokens):
        start_position = line.find(token, start_position)
        cumsum += len(line[prev_pos:start_position].encode("utf-8"))
        span_starts[i] = max(0, cumsum)
        prev_pos = start_position
        if start_position > 0:
            start_position += len(token)
    return span_starts


def run_both(line: str, raw_tokens: list[str], dictionary: dict[str, int], unk_idx: int):
    """Run Python reference and Rust, return both results."""
    lower_tokens = [t.lower() for t in raw_tokens]
    py_ids = py_encode(lower_tokens, dictionary, unk_idx)
    py_spans = py_get_span_start_positions(line, raw_tokens)

    rs_ids, rs_spans = encode_and_spans_rs(line, raw_tokens)

    return py_ids, py_spans, rs_ids, rs_spans


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_vocab() -> tuple[dict[str, int], int]:
    """A small ASCII vocabulary."""
    words = ["hello", "world", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    d = {w: i for i, w in enumerate(words)}
    unk = len(words)
    return d, unk


@pytest.fixture(scope="module", autouse=True)
def init_rust_vocab(simple_vocab):
    """Initialise the Rust thread-local vocab before any test in this module."""
    d, unk = simple_vocab
    init_vocab_rs(list(d.keys()), [np.uint32(v) for v in d.values()], np.uint32(unk))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAscii:
    def test_basic_sentence(self, simple_vocab):
        d, unk = simple_vocab
        line = "the quick brown fox"
        tokens = ["the", "quick", "brown", "fox"]
        py_ids, py_spans, rs_ids, rs_spans = run_both(line, tokens, d, unk)
        np.testing.assert_array_equal(rs_ids, py_ids)
        np.testing.assert_array_equal(rs_spans, py_spans)

    def test_unknown_tokens(self, simple_vocab):
        d, unk = simple_vocab
        line = "hello unknown world"
        tokens = ["hello", "unknown", "world"]
        py_ids, py_spans, rs_ids, rs_spans = run_both(line, tokens, d, unk)
        np.testing.assert_array_equal(rs_ids, py_ids)
        assert rs_ids[1] == unk  # "unknown" not in vocab
        np.testing.assert_array_equal(rs_spans, py_spans)

    def test_single_token(self, simple_vocab):
        d, unk = simple_vocab
        line = "hello"
        tokens = ["hello"]
        py_ids, py_spans, rs_ids, rs_spans = run_both(line, tokens, d, unk)
        np.testing.assert_array_equal(rs_ids, py_ids)
        np.testing.assert_array_equal(rs_spans, py_spans)

    def test_empty_tokens(self, simple_vocab):
        d, unk = simple_vocab
        line = ""
        tokens: list[str] = []
        rs_ids, rs_spans = encode_and_spans_rs(line, tokens)
        assert len(rs_ids) == 0
        assert len(rs_spans) == 0

    def test_leading_whitespace(self, simple_vocab):
        """Tokens can be preceded by whitespace; spans should still be correct."""
        d, unk = simple_vocab
        line = "  hello world"
        tokens = ["hello", "world"]
        py_ids, py_spans, rs_ids, rs_spans = run_both(line, tokens, d, unk)
        np.testing.assert_array_equal(rs_ids, py_ids)
        np.testing.assert_array_equal(rs_spans, py_spans)

    def test_mixed_case_tokens(self, simple_vocab):
        """Raw tokens may be mixed-case; encoding must use their lowercase form."""
        d, unk = simple_vocab
        # Reinit vocab with lowercase keys (standard for ICU tokenizer)
        line = "Hello World"
        # tokenize_raw returns original-case tokens; encode uses lowercased
        raw_tokens = ["Hello", "World"]
        py_ids, py_spans, rs_ids, rs_spans = run_both(line, raw_tokens, d, unk)
        np.testing.assert_array_equal(rs_ids, py_ids)
        np.testing.assert_array_equal(rs_spans, py_spans)
        # Both should map to "hello" and "world" in vocab
        assert rs_ids[0] == d["hello"]
        assert rs_ids[1] == d["world"]

    def test_repeated_token(self, simple_vocab):
        """Same token appearing multiple times — offsets must be sequential."""
        d, unk = simple_vocab
        line = "the dog the fox"
        tokens = ["the", "dog", "the", "fox"]
        py_ids, py_spans, rs_ids, rs_spans = run_both(line, tokens, d, unk)
        np.testing.assert_array_equal(rs_ids, py_ids)
        np.testing.assert_array_equal(rs_spans, py_spans)
        # Spans must be strictly increasing
        for i in range(1, len(rs_spans)):
            assert rs_spans[i] > rs_spans[i - 1]

    def test_output_dtype(self, simple_vocab):
        d, unk = simple_vocab
        line = "the quick fox"
        tokens = ["the", "quick", "fox"]
        rs_ids, rs_spans = encode_and_spans_rs(line, tokens)
        assert rs_ids.dtype == np.uint32
        assert rs_spans.dtype == np.uint32


class TestUtf8:
    """Tests with multi-byte UTF-8 characters."""

    @pytest.fixture(scope="class")
    def utf8_vocab(self):
        words = ["héllo", "wörld", "日本語", "café", "naïve", "über"]
        d = {w: i for i, w in enumerate(words)}
        unk = len(words)
        init_vocab_rs(list(d.keys()), [np.uint32(v) for v in d.values()], np.uint32(unk))
        return d, unk

    def test_two_byte_chars(self, utf8_vocab):
        d, unk = utf8_vocab
        line = "héllo wörld"
        tokens = ["héllo", "wörld"]
        py_ids, py_spans, rs_ids, rs_spans = run_both(line, tokens, d, unk)
        np.testing.assert_array_equal(rs_ids, py_ids)
        np.testing.assert_array_equal(rs_spans, py_spans)

    def test_three_byte_chars(self, utf8_vocab):
        d, unk = utf8_vocab
        line = "日本語 café"
        tokens = ["日本語", "café"]
        py_ids, py_spans, rs_ids, rs_spans = run_both(line, tokens, d, unk)
        np.testing.assert_array_equal(rs_ids, py_ids)
        np.testing.assert_array_equal(rs_spans, py_spans)

    def test_span_is_byte_offset_not_char_offset(self, utf8_vocab):
        """The span of 'wörld' must be the byte offset, not the character offset."""
        d, unk = utf8_vocab
        line = "héllo wörld"
        tokens = ["héllo", "wörld"]
        rs_ids, rs_spans = encode_and_spans_rs(line, tokens)
        # "héllo " in UTF-8: h(1)+é(2)+l(1)+l(1)+o(1)+' '(1) = 7 bytes
        assert rs_spans[0] == 0
        assert rs_spans[1] == 7

    def test_mixed_ascii_and_unicode(self, utf8_vocab):
        d, unk = utf8_vocab
        line = "über café naïve"
        tokens = ["über", "café", "naïve"]
        py_ids, py_spans, rs_ids, rs_spans = run_both(line, tokens, d, unk)
        np.testing.assert_array_equal(rs_ids, py_ids)
        np.testing.assert_array_equal(rs_spans, py_spans)


class TestVocabInit:
    def test_reinit_replaces_vocab(self):
        """Calling init_vocab_rs again completely replaces the vocabulary."""
        d1 = {"alpha": np.uint32(0), "beta": np.uint32(1)}
        init_vocab_rs(list(d1.keys()), list(d1.values()), np.uint32(99))
        ids1, _ = encode_and_spans_rs("alpha beta", ["alpha", "beta"])
        assert ids1[0] == 0
        assert ids1[1] == 1

        d2 = {"gamma": np.uint32(10), "delta": np.uint32(11)}
        init_vocab_rs(list(d2.keys()), list(d2.values()), np.uint32(99))
        ids2, _ = encode_and_spans_rs("gamma delta", ["gamma", "delta"])
        assert ids2[0] == 10
        assert ids2[1] == 11

        # Old vocab tokens should now be unknown
        ids3, _ = encode_and_spans_rs("alpha beta", ["alpha", "beta"])
        assert ids3[0] == 99
        assert ids3[1] == 99

        # Restore the simple_vocab for remaining tests in module
        words = ["hello", "world", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        d = {w: i for i, w in enumerate(words)}
        init_vocab_rs(list(d.keys()), [np.uint32(v) for v in d.values()], np.uint32(len(words)))


# ---------------------------------------------------------------------------
# encode_and_spans_positions_rs — the positions-based fast path
# ---------------------------------------------------------------------------

def _icu_char_ends(line: str) -> list[int]:
    """Return raw BreakIterator char-end positions for line (no strip, no filtering)."""
    from icu import BreakIterator, Locale
    bi = BreakIterator.createWordInstance(Locale("en"))
    bi.setText(line)
    return list(bi)


def _icu_tokens(line: str) -> list[str]:
    """Return the whitespace-filtered token strings (mirrors apply_break_iterator)."""
    from icu import BreakIterator, Locale
    bi = BreakIterator.createWordInstance(Locale("en"))
    bi.setText(line)
    parts, p0 = [], 0
    for p1 in bi:
        part = line[p0:p1].strip()
        if part:
            parts.append(part)
        p0 = p1
    return parts


class TestPositionsAPI:
    """Tests for encode_and_spans_positions_rs: the positions-based ICU fast path.

    The key invariant: for any line, encode_and_spans_positions_rs(line, list(bi))
    must produce the same (token_ids, span_starts) as encode_and_spans_rs(line, tokens)
    where tokens = apply_break_iterator(bi, line).
    """

    def test_empty_line(self, simple_vocab):
        ids, spans = encode_and_spans_positions_rs("", [])
        assert len(ids) == 0
        assert len(spans) == 0

    def test_empty_char_ends(self, simple_vocab):
        """No boundaries at all → empty output."""
        ids, spans = encode_and_spans_positions_rs("hello world", [])
        assert len(ids) == 0

    def test_whitespace_only_spans(self, simple_vocab):
        """Whitespace-only spans must be filtered; no tokens returned for pure spaces."""
        # All boundaries are within whitespace — no word tokens
        ids, spans = encode_and_spans_positions_rs("   ", _icu_char_ends("   "))
        assert len(ids) == 0

    def test_ascii_equivalence(self, simple_vocab):
        """Positions path matches string path on a basic ASCII sentence."""
        d, unk = simple_vocab
        line = "the quick brown fox"
        char_ends = _icu_char_ends(line)
        tokens = _icu_tokens(line)

        rs_ids, rs_spans = encode_and_spans_rs(line, tokens)
        pos_ids, pos_spans = encode_and_spans_positions_rs(line, char_ends)

        np.testing.assert_array_equal(pos_ids, rs_ids)
        np.testing.assert_array_equal(pos_spans, rs_spans)

    def test_single_word(self, simple_vocab):
        line = "hello"
        char_ends = _icu_char_ends(line)
        tokens = _icu_tokens(line)
        rs_ids, rs_spans = encode_and_spans_rs(line, tokens)
        pos_ids, pos_spans = encode_and_spans_positions_rs(line, char_ends)
        np.testing.assert_array_equal(pos_ids, rs_ids)
        np.testing.assert_array_equal(pos_spans, rs_spans)

    def test_repeated_word(self, simple_vocab):
        """Repeated tokens — positions must advance, not reuse first occurrence."""
        line = "the dog the fox"
        char_ends = _icu_char_ends(line)
        tokens = _icu_tokens(line)
        rs_ids, rs_spans = encode_and_spans_rs(line, tokens)
        pos_ids, pos_spans = encode_and_spans_positions_rs(line, char_ends)
        np.testing.assert_array_equal(pos_ids, rs_ids)
        np.testing.assert_array_equal(pos_spans, rs_spans)
        # Spans must be strictly increasing
        for i in range(1, len(pos_spans)):
            assert pos_spans[i] > pos_spans[i - 1]

    def test_unknown_tokens(self, simple_vocab):
        """Tokens not in vocab get unk_idx."""
        d, unk = simple_vocab
        line = "hello unknown world"
        char_ends = _icu_char_ends(line)
        pos_ids, pos_spans = encode_and_spans_positions_rs(line, char_ends)
        tokens = _icu_tokens(line)
        assert pos_ids[tokens.index("hello")] == d["hello"]
        assert pos_ids[tokens.index("unknown")] == unk
        assert pos_ids[tokens.index("world")] == d["world"]

    def test_leading_whitespace_line(self, simple_vocab):
        """Leading whitespace in the line: spans relative to original (not stripped) line."""
        line = "  the quick"
        char_ends = _icu_char_ends(line)
        tokens = _icu_tokens(line)
        rs_ids, rs_spans = encode_and_spans_rs(line, tokens)
        pos_ids, pos_spans = encode_and_spans_positions_rs(line, char_ends)
        np.testing.assert_array_equal(pos_ids, rs_ids)
        np.testing.assert_array_equal(pos_spans, rs_spans)
        # "the" must NOT start at byte 0 — it starts after the two leading spaces
        assert pos_spans[0] == 2

    def test_output_dtype(self, simple_vocab):
        ids, spans = encode_and_spans_positions_rs("the quick", _icu_char_ends("the quick"))
        assert ids.dtype == np.uint32
        assert spans.dtype == np.uint32

    def test_utf8_equivalence(self):
        """Positions path matches string path on UTF-8 multi-byte input."""
        words = ["héllo", "wörld", "café", "naïve"]
        d = {w: i for i, w in enumerate(words)}
        unk = len(words)
        init_vocab_rs(list(d.keys()), [np.uint32(v) for v in d.values()], np.uint32(unk))

        line = "héllo wörld café naïve"
        char_ends = _icu_char_ends(line)
        tokens = _icu_tokens(line)
        rs_ids, rs_spans = encode_and_spans_rs(line, tokens)
        pos_ids, pos_spans = encode_and_spans_positions_rs(line, char_ends)
        np.testing.assert_array_equal(pos_ids, rs_ids)
        np.testing.assert_array_equal(pos_spans, rs_spans)

        # Restore simple vocab
        simple_words = ["hello", "world", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        sd = {w: i for i, w in enumerate(simple_words)}
        init_vocab_rs(list(sd.keys()), [np.uint32(v) for v in sd.values()], np.uint32(len(simple_words)))

    def test_byte_offset_not_char_offset(self):
        """Span starts must be UTF-8 byte offsets, not character offsets."""
        words = ["héllo", "wörld"]
        d = {w: i for i, w in enumerate(words)}
        init_vocab_rs(list(d.keys()), [np.uint32(v) for v in d.values()], np.uint32(len(words)))

        line = "héllo wörld"
        char_ends = _icu_char_ends(line)
        pos_ids, pos_spans = encode_and_spans_positions_rs(line, char_ends)
        # "héllo " in UTF-8: h(1)+é(2)+l(1)+l(1)+o(1)+' '(1) = 7 bytes
        assert pos_spans[0] == 0   # "héllo" starts at byte 0
        assert pos_spans[1] == 7   # "wörld" starts at byte 7 (not char 6)

        # Restore simple vocab
        simple_words = ["hello", "world", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        sd = {w: i for i, w in enumerate(simple_words)}
        init_vocab_rs(list(sd.keys()), [np.uint32(v) for v in sd.values()], np.uint32(len(simple_words)))
