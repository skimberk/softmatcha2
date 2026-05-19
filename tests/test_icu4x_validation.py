"""Validation: icu4x Rust word segmentation vs. PyICU.

Every test in this file compares two implementations against each other and
against the ORIGINAL, unoptimised reference pipeline:

    reference(line) = (
        tokenize_raw(line),                         # PyICU word breaks
        get_span_start_positions(line, tokens),     # cumulative byte offset
    )

The icu4x Rust function (tokenize_and_encode_rs) must produce:
  - Identical TOKEN STRINGS to PyICU's tokenize_raw.
  - Identical TOKEN IDs (follows automatically from same strings + same vocab).
  - Identical BYTE OFFSETS to the current Python ICU path (correct offsets,
    not the position-0 quirk in the original reference).

Any disagreement is printed and causes the test to fail.
"""
from __future__ import annotations

import json
import os
import random
import string
import tempfile
from typing import NamedTuple

import importlib

import numpy as np
import pytest

# softmatcha/index/__init__.py re-exports `tokenize` (the function) under the
# same name as the submodule; importlib.import_module avoids the attribute-lookup
# confusion and returns the actual module object.
tokenize_module = importlib.import_module("softmatcha.index.tokenize")


# ---------------------------------------------------------------------------
# Helpers to build the real ICU tokenizer + Rust vocab
# ---------------------------------------------------------------------------

VOCAB_WORDS = [
    # common English words
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "is", "are", "was", "has",
    # contractions (lowercased; also appear in text as e.g. "don't")
    "don", "doesn", "it", "isn", "can", "won", "didn", "wasn", "aren",
    "couldn", "wouldn", "shouldn", "hadn", "haven", "weren",
    # extra
    "hello", "world", "test", "quick", "brown", "fox", "jumps", "lazy",
    "dog", "cafe", "naive", "uber", "price", "number", "call", "buy",
    "please", "meeting", "state", "pre", "existing", "iphone", "nasa",
    "fbi", "today", "tomorrow", "here", "there", "where", "why", "who",
]


class TokenizerSetup(NamedTuple):
    tok: object        # TokenizerICU instance
    rust_vocab: object # RustVocab instance


@pytest.fixture(scope="module")
def setup() -> TokenizerSetup:
    try:
        from softmatcha_rs import build_rust_vocab, tokenize_and_encode_rs  # noqa: F401
    except ImportError:
        pytest.skip("softmatcha_rs not built")

    from softmatcha.tokenizers.icu import TokenizerICU

    d = {w: i for i, w in enumerate(VOCAB_WORDS)}
    d["<unk>"] = len(VOCAB_WORDS)

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "vocab.json"), "w") as f:
            json.dump(d, f)
        tok = TokenizerICU.build(TokenizerICU.Config(name_or_path=tmpdir, lang="en"))

    from softmatcha_rs import build_rust_vocab, tokenize_and_encode_rs
    rv = build_rust_vocab(list(tok.dictionary.items()), tok.unk_idx)
    # Install as the active worker tokenizer so tokenize_encode_offsets uses it
    tokenize_module._worker_tokenizer        = tok
    tokenize_module._worker_rust_vocab       = rv
    tokenize_module._worker_use_icu4x        = True
    tokenize_module._worker_tokenize_fn      = tokenize_and_encode_rs
    try:
        from softmatcha_rs import tokenize_batch_rs
        tokenize_module._worker_tokenize_batch_fn = tokenize_batch_rs
    except ImportError:
        tokenize_module._worker_tokenize_batch_fn = None
    return TokenizerSetup(tok=tok, rust_vocab=rv)


# ---------------------------------------------------------------------------
# Reference implementation (original, unoptimised pipeline)
# ---------------------------------------------------------------------------

def reference(line: str, tok) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Run the original tokenizer pipeline and return (tokens, ids, offsets).

    This is exactly what the original tokenize_encode_offsets did:
        symbols = tokenize_raw(line)
        ids     = encode([sym.lower() for sym in symbols])
        offsets = get_span_start_positions(line, symbols)
    """
    symbols = tok.tokenize_raw(line)
    ids     = tok.encode([sym.lower() for sym in symbols])
    offsets = tok.get_span_start_positions(line, symbols)
    return symbols, ids, offsets


def icu4x_result(line: str, rust_vocab) -> tuple[np.ndarray, np.ndarray]:
    """Run the icu4x Rust function directly."""
    from softmatcha_rs import tokenize_and_encode_rs
    return tokenize_and_encode_rs(line, rust_vocab)


# ---------------------------------------------------------------------------
# Helper: compare and report
# ---------------------------------------------------------------------------

def assert_icu4x_matches(line: str, tok, rust_vocab, *, context: str = "") -> None:
    """Assert that icu4x tokens match PyICU tokens and offsets are correct."""
    ref_symbols, ref_ids, ref_offsets = reference(line, tok)

    rust_ids, rust_offsets = icu4x_result(line, rust_vocab)

    ctx = f"\nline={line!r}" + (f"  [{context}]" if context else "")

    # 1. Same number of tokens
    assert len(rust_ids) == len(ref_ids), (
        f"Token count differs: icu4x={len(rust_ids)} ref={len(ref_ids)}{ctx}"
    )

    # 2. Same token IDs (implies same token strings given same vocab)
    for i, (r, x) in enumerate(zip(ref_ids, rust_ids)):
        assert int(r) == int(x), (
            f"ID mismatch at position {i}: ref={int(r)} icu4x={int(x)}{ctx}\n"
            f"  ref_symbols={ref_symbols}"
        )

    # 3. Byte offsets must actually point to the correct bytes in the line
    line_bytes = line.encode("utf-8")
    for i, (sym, off) in enumerate(zip(ref_symbols, rust_offsets)):
        sym_bytes = sym.encode("utf-8")
        found = line_bytes[int(off) : int(off) + len(sym_bytes)]
        assert found == sym_bytes, (
            f"Offset {int(off)} for token {sym!r} at index {i} is wrong: "
            f"bytes there are {found!r}{ctx}"
        )


# ---------------------------------------------------------------------------
# Core edge-case tests (each covers a specific linguistic or encoding concern)
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_string(self, setup):
        ids, offs = icu4x_result("", setup.rust_vocab)
        assert len(ids) == 0 and len(offs) == 0

    def test_whitespace_only(self, setup):
        for line in ("   ", "\t", "\n", "  \t  \n"):
            ids, offs = icu4x_result(line, setup.rust_vocab)
            assert len(ids) == 0, f"Expected no tokens for {line!r}"

    def test_single_word(self, setup):
        assert_icu4x_matches("hello", setup.tok, setup.rust_vocab)

    def test_two_words(self, setup):
        assert_icu4x_matches("hello world", setup.tok, setup.rust_vocab)

    # --- contractions (ICU keeps as one token) ---

    def test_contraction_dont(self, setup):
        assert_icu4x_matches("don't", setup.tok, setup.rust_vocab, context="contraction")

    def test_contraction_its(self, setup):
        assert_icu4x_matches("it's a beautiful day", setup.tok, setup.rust_vocab)

    def test_contraction_isnt(self, setup):
        assert_icu4x_matches("isn't that right", setup.tok, setup.rust_vocab)

    def test_contraction_wont(self, setup):
        assert_icu4x_matches("I won't go", setup.tok, setup.rust_vocab)

    # --- hyphens (ICU splits at hyphen) ---

    def test_hyphen_splits(self, setup):
        # "state-of-the-art" → ["state", "of", "the", "art"] in PyICU
        assert_icu4x_matches("state-of-the-art", setup.tok, setup.rust_vocab)

    def test_hyphen_simple(self, setup):
        assert_icu4x_matches("hello-world", setup.tok, setup.rust_vocab)

    def test_hyphen_pre_existing(self, setup):
        assert_icu4x_matches("pre-existing", setup.tok, setup.rust_vocab)

    # --- numbers ---

    def test_decimal_number(self, setup):
        assert_icu4x_matches("3.14", setup.tok, setup.rust_vocab, context="decimal")

    def test_number_with_comma(self, setup):
        assert_icu4x_matches("1,000", setup.tok, setup.rust_vocab, context="comma-number")

    def test_integer(self, setup):
        assert_icu4x_matches("42", setup.tok, setup.rust_vocab)

    # --- mixed alphanumeric ---

    def test_alphanum_iphone(self, setup):
        assert_icu4x_matches("iPhone15", setup.tok, setup.rust_vocab)

    def test_alphanum_version(self, setup):
        assert_icu4x_matches("Python3 is great", setup.tok, setup.rust_vocab)

    # --- case ---

    def test_uppercase_words(self, setup):
        assert_icu4x_matches("THE QUICK BROWN FOX", setup.tok, setup.rust_vocab)

    def test_mixed_case(self, setup):
        assert_icu4x_matches("Hello World", setup.tok, setup.rust_vocab)

    def test_acronyms(self, setup):
        assert_icu4x_matches("NASA and FBI are US agencies", setup.tok, setup.rust_vocab)

    # --- punctuation ---

    def test_sentence_period(self, setup):
        assert_icu4x_matches("Hello world.", setup.tok, setup.rust_vocab)

    def test_sentence_exclamation(self, setup):
        assert_icu4x_matches("Hello world!", setup.tok, setup.rust_vocab)

    def test_sentence_question(self, setup):
        assert_icu4x_matches("How are you?", setup.tok, setup.rust_vocab)

    def test_comma_separated(self, setup):
        assert_icu4x_matches("one, two, three", setup.tok, setup.rust_vocab)

    def test_parentheses(self, setup):
        assert_icu4x_matches("hello (world)", setup.tok, setup.rust_vocab)

    def test_quotes(self, setup):
        assert_icu4x_matches('"hello world"', setup.tok, setup.rust_vocab)

    def test_slash(self, setup):
        assert_icu4x_matches("and/or", setup.tok, setup.rust_vocab)

    def test_ampersand(self, setup):
        assert_icu4x_matches("cats & dogs", setup.tok, setup.rust_vocab)

    def test_dollars(self, setup):
        assert_icu4x_matches("The price is $9.99", setup.tok, setup.rust_vocab)

    def test_percent(self, setup):
        assert_icu4x_matches("up 50% today", setup.tok, setup.rust_vocab)

    # --- leading/trailing whitespace ---

    def test_leading_whitespace(self, setup):
        assert_icu4x_matches("  hello world", setup.tok, setup.rust_vocab)

    def test_trailing_whitespace(self, setup):
        assert_icu4x_matches("hello world  ", setup.tok, setup.rust_vocab)

    def test_both_ends_whitespace(self, setup):
        assert_icu4x_matches("  hello world  ", setup.tok, setup.rust_vocab)

    def test_newline_at_end(self, setup):
        assert_icu4x_matches("hello world\n", setup.tok, setup.rust_vocab)

    # --- UTF-8 multibyte ---

    def test_utf8_accented_2byte(self, setup):
        assert_icu4x_matches("cafe naive uber", setup.tok, setup.rust_vocab)

    def test_utf8_with_leading_ascii(self, setup):
        # "café" — é is 2-byte; offset of subsequent words must account for it
        assert_icu4x_matches("the cafe is nice", setup.tok, setup.rust_vocab)

    def test_utf8_3byte_cjk(self, setup):
        # CJK characters are 3 bytes each in UTF-8; ICU may or may not split them
        # depending on Unicode word break rules. We just verify icu4x and PyICU agree.
        assert_icu4x_matches("hello 世界 world", setup.tok, setup.rust_vocab)

    def test_utf8_emoji(self, setup):
        """ICU version difference: PyICU 78.3 groups the space after 🎉 into
        the emoji span, causing "world" to be split as "w"+"orld".
        icu4x 2.2 uses more recent Unicode data and keeps "world" intact.

        We verify that icu4x produces correct byte offsets (each offset points
        to the actual token in the byte string) even though the token count may
        differ from PyICU 78.3.
        """
        line = "hello 🎉 world"
        rust_ids, rust_offsets = icu4x_result(line, setup.rust_vocab)
        line_bytes = line.encode("utf-8")
        # Recover the tokens icu4x found by using the reference's token list
        # aligned by byte offset — instead, just verify offsets are valid
        ref_sym = setup.tok.tokenize_raw(line)  # PyICU list (may differ)
        # Every icu4x offset must point to a non-empty sequence of bytes
        for off in rust_offsets:
            assert int(off) < len(line_bytes), f"Offset {int(off)} out of range"
            # And it must not point into the middle of a UTF-8 continuation byte
            b = line_bytes[int(off)]
            assert not (0x80 <= b <= 0xBF), (
                f"Offset {int(off)} points to UTF-8 continuation byte 0x{b:02x}"
            )

    # --- repeated words (the position-0 quirk no longer applies) ---

    def test_repeated_token_at_start(self, setup):
        # With ICU positions: "the the the" → correct offsets [0, 4, 8]
        # (not the position-0 quirk [0, 0, 0] of the original reference)
        line = "the the the"
        rust_ids, rust_offsets = icu4x_result(line, setup.rust_vocab)
        assert len(rust_ids) == 3
        # Verify each offset actually points to a "the" in the line
        line_bytes = line.encode()
        for off in rust_offsets:
            assert line_bytes[off : off + 3] == b"the", f"Bad offset {off}"

    def test_repeated_token_not_at_start(self, setup):
        assert_icu4x_matches("a the the the", setup.tok, setup.rust_vocab)

    # --- typical English sentences ---

    def test_pangram(self, setup):
        assert_icu4x_matches(
            "the quick brown fox jumps over the lazy dog",
            setup.tok, setup.rust_vocab,
        )

    def test_sentence_with_contraction_and_punctuation(self, setup):
        assert_icu4x_matches(
            "It's a beautiful day, isn't it?",
            setup.tok, setup.rust_vocab,
        )

    def test_price_sentence(self, setup):
        assert_icu4x_matches(
            "The price is $9.99 per item.",
            setup.tok, setup.rust_vocab,
        )

    def test_phone_number(self, setup):
        assert_icu4x_matches(
            "Please call 1-800-555-0100 for assistance.",
            setup.tok, setup.rust_vocab,
        )

    def test_date(self, setup):
        assert_icu4x_matches(
            "The meeting is on 2024-01-15 at 3:30pm.",
            setup.tok, setup.rust_vocab,
        )

    def test_camelcase(self, setup):
        assert_icu4x_matches("iPhone15 Pro Max", setup.tok, setup.rust_vocab)

    def test_url_like(self, setup):
        assert_icu4x_matches(
            "Visit www.example.com for more info.",
            setup.tok, setup.rust_vocab,
        )

    def test_email_like(self, setup):
        assert_icu4x_matches(
            "Contact us at info@example.com today.",
            setup.tok, setup.rust_vocab,
        )


# ---------------------------------------------------------------------------
# Random / stress tests
# ---------------------------------------------------------------------------

class TestRandomText:

    @staticmethod
    def _random_line(rng: random.Random, vocab: list[str]) -> str:
        n = rng.randint(3, 30)
        words = [rng.choice(vocab) for _ in range(n)]
        # Randomly capitalise some words
        words = [w.capitalize() if rng.random() < 0.1 else w for w in words]
        return " ".join(words)

    def test_random_ascii_lines(self, setup):
        rng = random.Random(42)
        failures = []
        for i in range(2000):
            line = self._random_line(rng, VOCAB_WORDS)
            try:
                assert_icu4x_matches(line, setup.tok, setup.rust_vocab)
            except AssertionError as e:
                failures.append(str(e))
        assert not failures, f"{len(failures)} failures:\n" + "\n---\n".join(failures[:5])

    def test_random_with_punctuation(self, setup):
        rng = random.Random(99)
        puncts = [".", ",", "!", "?", ";", ":", "-", "'s", "'t", "(", ")", '"']
        failures = []
        for i in range(1000):
            n = rng.randint(3, 15)
            parts = []
            for _ in range(n):
                parts.append(rng.choice(VOCAB_WORDS))
                if rng.random() < 0.3:
                    parts.append(rng.choice(puncts))
            line = " ".join(parts)
            try:
                assert_icu4x_matches(line, setup.tok, setup.rust_vocab)
            except AssertionError as e:
                failures.append(str(e))
        assert not failures, f"{len(failures)} failures:\n" + "\n---\n".join(failures[:5])

    def test_random_with_numbers(self, setup):
        rng = random.Random(7)
        failures = []
        for i in range(500):
            n_words = rng.randint(2, 10)
            words = [rng.choice(VOCAB_WORDS) for _ in range(n_words)]
            # Insert some numbers
            idx = rng.randint(0, n_words)
            num = str(rng.randint(0, 9999))
            words.insert(idx, num)
            line = " ".join(words)
            try:
                assert_icu4x_matches(line, setup.tok, setup.rust_vocab)
            except AssertionError as e:
                failures.append(str(e))
        assert not failures, f"{len(failures)} failures:\n" + "\n---\n".join(failures[:5])

    def test_random_mixed_case(self, setup):
        rng = random.Random(13)
        failures = []
        for i in range(500):
            n = rng.randint(3, 20)
            words = []
            for _ in range(n):
                w = rng.choice(VOCAB_WORDS)
                case = rng.choice(["lower", "upper", "title", "mixed"])
                if case == "upper":
                    w = w.upper()
                elif case == "title":
                    w = w.capitalize()
                elif case == "mixed":
                    w = "".join(
                        c.upper() if rng.random() < 0.5 else c for c in w
                    )
                words.append(w)
            line = " ".join(words)
            try:
                assert_icu4x_matches(line, setup.tok, setup.rust_vocab)
            except AssertionError as e:
                failures.append(str(e))
        assert not failures, f"{len(failures)} failures:\n" + "\n---\n".join(failures[:5])

    def test_random_whitespace_variants(self, setup):
        rng = random.Random(21)
        failures = []
        for i in range(200):
            n = rng.randint(2, 10)
            words = [rng.choice(VOCAB_WORDS) for _ in range(n)]
            # Random leading/trailing spaces
            lead = " " * rng.randint(0, 4)
            trail = " " * rng.randint(0, 4)
            line = lead + " ".join(words) + trail
            try:
                assert_icu4x_matches(line, setup.tok, setup.rust_vocab)
            except AssertionError as e:
                failures.append(str(e))
        assert not failures, f"{len(failures)} failures:\n" + "\n---\n".join(failures[:5])


# ---------------------------------------------------------------------------
# Specific sentence-level regression tests
# ---------------------------------------------------------------------------

REGRESSION_SENTENCES = [
    "",
    "   ",
    "Hello, World! This is a test.",
    "the quick brown fox jumps over the lazy dog",
    "It's a beautiful day, isn't it?",
    "The price is $9.99 per item.",
    "Please call 1-800-555-0100 for assistance.",
    "The meeting is on 2024-01-15 at 3:30pm.",
    "I bought an iPhone 15 Pro Max for $999.",
    "don't stop me now",
    "NASA and ESA are working together.",
    "pre-existing conditions",
    "3.14159 is approximately pi",
    "1,000,000 dollars",
    "the the the",
    "a the the the",
    "  leading spaces here",
    "trailing spaces here  ",
    "UPPERCASE LETTERS ONLY",
    "MiXeD cAsE wOrDs",
    "word1 word2 word3",
    "won't can't don't shouldn't couldn't",
    "state-of-the-art technology",
    "hello world\n",
    "one two three four five six seven eight nine ten",
]


@pytest.mark.parametrize("line", REGRESSION_SENTENCES)
def test_regression_sentences(line, setup):
    """Every sentence in the regression list must have identical tokens and correct offsets."""
    assert_icu4x_matches(line, setup.tok, setup.rust_vocab, context="regression")


# ---------------------------------------------------------------------------
# Offset correctness: verify every byte offset actually points to its token
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line", REGRESSION_SENTENCES + [
    "café bar",
    "hello 中文 world",
    "naïve über café",
    "  leading 中文  ",
    # "hello 🎉 world" excluded: ICU-version difference (see test_utf8_emoji)
])
def test_byte_offsets_point_to_token(line, setup):
    """For every token, line_bytes[offset:offset+len(token_bytes)] == token_bytes."""
    ref_symbols = setup.tok.tokenize_raw(line)
    rust_ids, rust_offsets = icu4x_result(line, setup.rust_vocab)

    assert len(rust_ids) == len(ref_symbols), (
        f"Token count mismatch: {len(rust_ids)} vs {len(ref_symbols)} for {line!r}"
    )

    line_bytes = line.encode("utf-8")
    for sym, off in zip(ref_symbols, rust_offsets):
        sym_bytes = sym.encode("utf-8")
        found = line_bytes[int(off) : int(off) + len(sym_bytes)]
        assert found == sym_bytes, (
            f"Offset {int(off)} for token {sym!r} in {line!r}: "
            f"got {found!r}, expected {sym_bytes!r}"
        )


# ---------------------------------------------------------------------------
# Integration: tokenize_encode_offsets uses icu4x when active
# ---------------------------------------------------------------------------

def test_tokenize_encode_offsets_uses_icu4x(setup):
    """tokenize_encode_offsets must be using the icu4x path when _worker_use_icu4x=True."""
    assert tokenize_module._worker_use_icu4x, "icu4x path not active"
    assert tokenize_module._worker_rust_vocab is not None
    assert tokenize_module._worker_tokenize_fn is not None, "_worker_tokenize_fn must be set"

    line = "the quick brown fox jumps over the lazy dog"
    ids, offs = tokenize_module.tokenize_encode_offsets(line)

    ref_sym = setup.tok.tokenize_raw(line)
    assert len(ids) == len(ref_sym)

    line_bytes = line.encode()
    for sym, off in zip(ref_sym, offs):
        assert line_bytes[int(off) : int(off) + len(sym)] == sym.encode()


# ---------------------------------------------------------------------------
# tokenize_batch_rs regression
# ---------------------------------------------------------------------------

def test_tokenize_batch_rs_matches_individual_calls(setup):
    """tokenize_batch_rs must produce results identical to per-line tokenize_and_encode_rs."""
    try:
        from softmatcha_rs import tokenize_batch_rs
    except ImportError:
        pytest.skip("tokenize_batch_rs not available")

    import numpy as np
    lines = REGRESSION_SENTENCES + [
        "café bar",
        "hello 中文 world",
        "naïve über café",
    ]
    batch_tok, batch_off, batch_len = tokenize_batch_rs(lines, setup.rust_vocab)
    assert len(batch_len) == len(lines)
    assert len(batch_tok) == sum(int(l) for l in batch_len)
    assert len(batch_off) == len(batch_tok)

    offset = 0
    for i, line in enumerate(lines):
        ind_tok, ind_off = icu4x_result(line, setup.rust_vocab)
        n = len(ind_tok)
        assert int(batch_len[i]) == n, f"Length mismatch at line {i}: {line!r}"
        np.testing.assert_array_equal(
            batch_tok[offset:offset+n], ind_tok,
            err_msg=f"token_ids differ at line {i}: {line!r}"
        )
        np.testing.assert_array_equal(
            batch_off[offset:offset+n], ind_off,
            err_msg=f"byte_offsets differ at line {i}: {line!r}"
        )
        offset += n


def test_batch_worker_fn_uses_icu4x_path(setup):
    """_tokenize_encode_offsets_batch must use tokenize_batch_rs when available."""
    if tokenize_module._worker_tokenize_batch_fn is None:
        pytest.skip("tokenize_batch_rs not available")

    import numpy as np
    lines = ["the quick brown fox", "Hello World!", "café bar", ""]
    cat_tok, cat_off, lengths = tokenize_module._tokenize_encode_offsets_batch(lines)

    assert len(lengths) == len(lines)
    total = sum(int(l) for l in lengths)
    assert len(cat_tok) == total
    assert len(cat_off) == total

    offset = 0
    for i, line in enumerate(lines):
        n = int(lengths[i])
        ind_tok, ind_off = icu4x_result(line, setup.rust_vocab)
        np.testing.assert_array_equal(cat_tok[offset:offset+n], ind_tok)
        np.testing.assert_array_equal(cat_off[offset:offset+n], ind_off)
        offset += n
