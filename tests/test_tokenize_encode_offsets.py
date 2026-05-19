"""Tests for tokenize_encode_offsets in softmatcha.index.tokenize.

Strategy:
- Set the module-level _worker_tokenizer global directly (no subprocess needed).
- Use SimpleTestTokenizer, which splits on whitespace and inherits the real
  encode() and get_span_start_positions() implementations from the base class.
- Restore _worker_tokenizer after every test via an autouse fixture.

Notable behavior documented in these tests:
- get_span_start_positions has a position-0 quirk: when the first token is
  found at character position 0, start_position is not advanced past it.
  Consecutive identical tokens that all find themselves at position 0 will
  receive offset 0.  See test_repeated_token_at_start_position_zero_quirk.
"""
from __future__ import annotations

import importlib
import numpy as np
import pytest

# softmatcha/index/__init__.py re-exports the `tokenize` *function* under the
# same name as the submodule, so `import softmatcha.index.tokenize as x`
# resolves via attribute lookup and binds x to the function, not the module.
# importlib.import_module looks up sys.modules directly and returns the module.
tokenize_module = importlib.import_module("softmatcha.index.tokenize")
tokenize_encode_offsets = tokenize_module.tokenize_encode_offsets

from softmatcha.tokenizers.base import Tokenizer


# ---------------------------------------------------------------------------
# Test tokenizer
# ---------------------------------------------------------------------------

class SimpleTestTokenizer(Tokenizer):
    """Whitespace-splitting tokenizer for unit tests.

    tokenize_raw(line) -> line.strip().split()   (case-preserving)
    tokenize(line)     -> line.strip().lower().split()
    encode() and get_span_start_positions() are inherited from Tokenizer.
    """

    @property
    def unk_idx(self) -> int:
        return self.dictionary[self.UNK_TOKEN]

    @classmethod
    def build(cls, cfg: Tokenizer.Config) -> "SimpleTestTokenizer":
        raise NotImplementedError("Use make_test_tokenizer() instead")

    def tokenize(self, line: str) -> list[str]:
        return line.strip().lower().split()

    def tokenize_raw(self, line: str) -> list[str]:
        return line.strip().split()


def make_test_tokenizer(vocab: list[str]) -> SimpleTestTokenizer:
    """Build a SimpleTestTokenizer from a vocabulary list.

    vocab words get IDs 0..len(vocab)-1; <unk> is added at index len(vocab).
    """
    dictionary: dict[str, int] = {word: i for i, word in enumerate(vocab)}
    dictionary[Tokenizer.UNK_TOKEN] = len(vocab)
    cfg = Tokenizer.Config(name_or_path="test")
    return SimpleTestTokenizer(cfg, tokenizer=None, dictionary=dictionary)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_worker_tokenizer():
    """Restore all worker globals after every test."""
    orig_tok       = tokenize_module._worker_tokenizer
    orig_vocab     = tokenize_module._worker_rust_vocab
    orig_use_icu4x = tokenize_module._worker_use_icu4x
    yield
    tokenize_module._worker_tokenizer  = orig_tok
    tokenize_module._worker_rust_vocab = orig_vocab
    tokenize_module._worker_use_icu4x  = orig_use_icu4x


def set_tokenizer(vocab: list[str]) -> SimpleTestTokenizer:
    """Create and install a test tokenizer; return it for further inspection."""
    tok = make_test_tokenizer(vocab)
    tokenize_module._worker_tokenizer = tok
    # The mock tokenizer is not ICU-based; force the Python fallback path so
    # that tests are not contaminated by icu4x globals left by other test files.
    tokenize_module._worker_use_icu4x = False
    tokenize_module._worker_rust_vocab = None
    return tok


def tokenize_encode_offsets_reference(line: str, tokenizer: Tokenizer):
    """Reference implementation identical to the current tokenize_encode_offsets.

    Used in comparison tests: when an optimised replacement is written, call
    it alongside this function and assert the arrays are equal.
    """
    symbols = tokenizer.tokenize_raw(line)
    token_ids = tokenizer.encode([sym.lower() for sym in symbols])
    offsets = tokenizer.get_span_start_positions(line, symbols)
    return token_ids, offsets


# ---------------------------------------------------------------------------
# Empty / blank input
# ---------------------------------------------------------------------------

def test_empty_string():
    set_tokenizer(["hello", "world"])
    token_ids, offsets = tokenize_encode_offsets("")
    assert token_ids.dtype == np.uint32
    assert offsets.dtype == np.uint32
    assert len(token_ids) == 0
    assert len(offsets) == 0


def test_whitespace_only():
    set_tokenizer(["hello"])
    token_ids, offsets = tokenize_encode_offsets("   \t  ")
    assert len(token_ids) == 0
    assert len(offsets) == 0


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_single_token_no_leading_whitespace():
    set_tokenizer(["hello"])
    token_ids, offsets = tokenize_encode_offsets("hello")
    assert token_ids.tolist() == [0]
    assert offsets.tolist() == [0]


def test_single_token_with_leading_whitespace():
    """Leading whitespace is stripped before tokenizing but the offset is
    computed against the original line, so the leading bytes are counted."""
    set_tokenizer(["hello"])
    # tokenize_raw("  hello") -> ["hello"]
    # get_span_start_positions("  hello", ["hello"]):
    #   find("hello", 0) = 2; cumsum += len("  ") = 2; span_starts[0] = 2
    token_ids, offsets = tokenize_encode_offsets("  hello")
    assert token_ids.tolist() == [0]
    assert offsets.tolist() == [2]


def test_two_tokens_ascii():
    set_tokenizer(["hello", "world"])
    token_ids, offsets = tokenize_encode_offsets("hello world")
    assert token_ids.tolist() == [0, 1]
    # "world" starts at byte 6 ("hello " = 6 bytes)
    assert offsets.tolist() == [0, 6]


def test_sentence_ascii():
    """Known offsets for a standard ASCII sentence."""
    vocab = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    set_tokenizer(vocab)
    line = "the quick brown fox jumps over the lazy dog"
    token_ids, offsets = tokenize_encode_offsets(line)
    # All ASCII: byte offset == character position
    assert offsets.tolist() == [0, 4, 10, 16, 20, 26, 31, 35, 40]
    assert token_ids[0] == vocab.index("the")
    assert token_ids[1] == vocab.index("quick")
    assert token_ids[6] == vocab.index("the")


def test_many_tokens_sequential():
    """100-token vocabulary; every ID and length is correct."""
    vocab = [f"word{i}" for i in range(100)]
    set_tokenizer(vocab)
    line = " ".join(vocab)
    token_ids, offsets = tokenize_encode_offsets(line)
    assert len(token_ids) == 100
    assert len(offsets) == 100
    assert token_ids.tolist() == list(range(100))


# ---------------------------------------------------------------------------
# Output types and shape consistency
# ---------------------------------------------------------------------------

def test_output_dtypes():
    set_tokenizer(["a", "b", "c"])
    token_ids, offsets = tokenize_encode_offsets("a b c")
    assert token_ids.dtype == np.uint32
    assert offsets.dtype == np.uint32


def test_lengths_always_match():
    set_tokenizer(["x", "y", "z"])
    for line in ["x", "x y", "x y z", "", "  "]:
        token_ids, offsets = tokenize_encode_offsets(line)
        assert len(token_ids) == len(offsets), f"Length mismatch for {line!r}"


# ---------------------------------------------------------------------------
# Encoding: lowercasing and OOV handling
# ---------------------------------------------------------------------------

def test_lowercasing_for_ids():
    """IDs are looked up using the lowercased token even when raw is uppercase."""
    set_tokenizer(["hello", "world"])
    # tokenize_raw("Hello World") -> ["Hello", "World"]
    # encode(["hello", "world"]) -> [0, 1]
    token_ids, offsets = tokenize_encode_offsets("Hello World")
    assert token_ids.tolist() == [0, 1]
    assert offsets.tolist() == [0, 6]


def test_mixed_case_offsets_use_raw_tokens():
    """get_span_start_positions receives the case-preserving raw tokens,
    so line.find() is case-sensitive and locates tokens correctly."""
    set_tokenizer(["the", "fox"])
    # tokenize_raw("The Fox") -> ["The", "Fox"]
    # find("The", 0) = 0; find("Fox", ...) = 4
    token_ids, offsets = tokenize_encode_offsets("The Fox")
    assert token_ids.tolist() == [0, 1]
    assert offsets.tolist() == [0, 4]


def test_oov_token_maps_to_unk():
    tok = set_tokenizer(["hello"])
    unk = tok.unk_idx
    token_ids, offsets = tokenize_encode_offsets("hello unknown_word")
    assert token_ids[0] == 0
    assert token_ids[1] == unk


def test_all_oov_tokens():
    tok = set_tokenizer([])
    unk = tok.unk_idx
    token_ids, offsets = tokenize_encode_offsets("foo bar baz")
    assert len(token_ids) == 3
    assert all(t == unk for t in token_ids.tolist())


def test_partial_oov_offsets():
    """OOV tokens still have correct offsets."""
    tok = set_tokenizer(["word1", "word2"])
    unk = tok.unk_idx
    token_ids, offsets = tokenize_encode_offsets("word1 unknown_word word2")
    assert token_ids.tolist() == [0, unk, 1]
    # "word1"(0-4) " "(5) "unknown_word"(6-17) " "(18) "word2"(19-23)
    assert offsets.tolist() == [0, 6, 19]


# ---------------------------------------------------------------------------
# ASCII offset correctness
# ---------------------------------------------------------------------------

def test_ascii_offsets_slice_verification():
    """Each offset should point exactly to the start of the raw token."""
    vocab = ["the", "quick", "brown", "fox"]
    set_tokenizer(vocab)
    line = "the quick brown fox"
    token_ids, offsets = tokenize_encode_offsets(line)
    line_bytes = line.encode("utf-8")
    for sym, off in zip(line.split(), offsets):
        n = len(sym.encode("utf-8"))
        assert line_bytes[off : off + n] == sym.encode("utf-8")


def test_ascii_multiple_spaces_between_tokens():
    """Multiple spaces between tokens are included in the byte offset."""
    set_tokenizer(["a", "b"])
    line = "a   b"
    token_ids, offsets = tokenize_encode_offsets(line)
    assert offsets.tolist() == [0, 4]
    line_bytes = line.encode("utf-8")
    assert line_bytes[0:1] == b"a"
    assert line_bytes[4:5] == b"b"


def test_single_character_tokens():
    set_tokenizer(["a", "b", "c"])
    token_ids, offsets = tokenize_encode_offsets("a b c")
    assert token_ids.tolist() == [0, 1, 2]
    assert offsets.tolist() == [0, 2, 4]


# ---------------------------------------------------------------------------
# UTF-8 multibyte offset correctness
# ---------------------------------------------------------------------------

def test_utf8_2byte_accented_offsets():
    """é is 2 bytes in UTF-8; the byte offset of the following token is shifted."""
    set_tokenizer(["café", "bar"])
    line = "café bar"
    token_ids, offsets = tokenize_encode_offsets(line)
    # "café" = c(1)+a(1)+f(1)+é(2) = 5 bytes; " "(1); "bar" at byte 6
    assert offsets[0] == 0
    assert offsets[1] == len("café ".encode("utf-8"))  # 6
    line_bytes = line.encode("utf-8")
    assert line_bytes[offsets[1] : offsets[1] + 3] == b"bar"


def test_utf8_3byte_cjk_offsets():
    """CJK characters are 3 bytes each in UTF-8."""
    set_tokenizer(["hello", "中文", "world"])
    line = "hello 中文 world"
    token_ids, offsets = tokenize_encode_offsets(line)
    # "hello "(6) + "中文 "(7) = 13; "world" at byte 13
    assert offsets[0] == 0
    assert offsets[1] == len("hello ".encode("utf-8"))        # 6
    assert offsets[2] == len("hello 中文 ".encode("utf-8"))   # 13
    line_bytes = line.encode("utf-8")
    assert line_bytes[offsets[2] : offsets[2] + 5] == b"world"


def test_utf8_4byte_emoji_offsets():
    """Emoji are 4 bytes each in UTF-8."""
    set_tokenizer(["hello", "🎉", "world"])
    line = "hello 🎉 world"
    token_ids, offsets = tokenize_encode_offsets(line)
    # "hello "(6) + "🎉 "(5) = 11; "world" at byte 11
    assert offsets[0] == 0
    assert offsets[1] == len("hello ".encode("utf-8"))       # 6
    assert offsets[2] == len("hello 🎉 ".encode("utf-8"))    # 11
    line_bytes = line.encode("utf-8")
    assert line_bytes[offsets[2] : offsets[2] + 5] == b"world"


def test_utf8_mixed_multibyte_slice_verification():
    """For mixed multibyte input, every offset correctly points into the byte string."""
    set_tokenizer(["café", "中文", "🎉🎉🎉", "end"])
    line = "café 中文 🎉🎉🎉 end"
    token_ids, offsets = tokenize_encode_offsets(line)
    line_bytes = line.encode("utf-8")
    for sym, off in zip(line.strip().split(), offsets):
        sym_bytes = sym.encode("utf-8")
        assert line_bytes[off : off + len(sym_bytes)] == sym_bytes


# ---------------------------------------------------------------------------
# Leading / trailing whitespace edge cases
# ---------------------------------------------------------------------------

def test_trailing_whitespace_no_extra_tokens():
    set_tokenizer(["hello"])
    token_ids, offsets = tokenize_encode_offsets("hello   ")
    assert len(token_ids) == 1
    assert token_ids[0] == 0


def test_newline_at_end():
    set_tokenizer(["hello"])
    token_ids, offsets = tokenize_encode_offsets("hello\n")
    assert len(token_ids) == 1
    assert token_ids[0] == 0


def test_leading_whitespace_offsets():
    """Leading bytes are counted in the offset even though tokenize_raw strips."""
    set_tokenizer(["leading", "spaces"])
    line = "  leading spaces"
    token_ids, offsets = tokenize_encode_offsets(line)
    # " "(0)" "(1)"leading"(2-8)" "(9)"spaces"(10-15)
    assert offsets.tolist() == [2, 10]
    line_bytes = line.encode("utf-8")
    assert line_bytes[2:9] == b"leading"
    assert line_bytes[10:16] == b"spaces"


# ---------------------------------------------------------------------------
# Repeated token edge cases (documents position-0 behaviour)
# ---------------------------------------------------------------------------

def test_repeated_token_at_start_position_zero_quirk():
    """When the first token lands at char position 0, get_span_start_positions
    does NOT advance start_position past it (the `if start_position > 0` guard).
    Subsequent searches for the same string restart from 0, so all repeated
    occurrences receive offset 0.  This is the existing behaviour."""
    set_tokenizer(["the"])
    token_ids, offsets = tokenize_encode_offsets("the the the")
    assert offsets.tolist() == [0, 0, 0]


def test_repeated_token_not_at_start_correct_offsets():
    """Repeated tokens that are not the first token get correct offsets."""
    set_tokenizer(["a", "the"])
    line = "a the the the"
    token_ids, offsets = tokenize_encode_offsets(line)
    # "a"(0) " "(1) "the"(2-4) " "(5) "the"(6-8) " "(9) "the"(10-12)
    assert offsets.tolist() == [0, 2, 6, 10]


def test_repeated_different_case_tokens():
    """Case-sensitive find() distinguishes 'The' (raw) from 'the' in the line."""
    set_tokenizer(["the"])
    line = "The the"
    # tokenize_raw -> ["The", "the"]
    # find("The", 0) = 0; NOT advanced (position 0 quirk)
    # find("the", 0) = 4 (finds lowercase "the", not "The")
    token_ids, offsets = tokenize_encode_offsets(line)
    assert offsets.tolist() == [0, 4]


# ---------------------------------------------------------------------------
# Comparison tests: function output vs. inline reference implementation
# ---------------------------------------------------------------------------

_COMPARISON_VOCAB = [
    "hello", "world", "the", "quick", "brown", "fox", "jumps", "over",
    "lazy", "dog", "café", "bar", "leading", "spaces", "trailing",
    "uppercase", "text", "a", "b", "c", "d", "e", "f", "g",
    "single", "word1", "word2", "中文", "🎉",
]

_COMPARISON_LINES = [
    "",
    "   ",
    "hello",
    "hello world",
    "Hello World",
    "the quick brown fox jumps over the lazy dog",
    "the the the",
    "a the the the",
    "café bar",
    "hello 中文 world",
    "hello 🎉 world",
    "  leading spaces",
    "trailing spaces  ",
    "UPPERCASE TEXT",
    "a b c d e f g",
    "word1 unknown_word word2",
]

@pytest.mark.parametrize("line", _COMPARISON_LINES)
def test_comparison_with_reference(line):
    """tokenize_encode_offsets must produce byte-for-byte identical output to
    the reference implementation across a range of inputs."""
    tok = set_tokenizer(_COMPARISON_VOCAB)
    ref_ids, ref_offsets = tokenize_encode_offsets_reference(line, tok)
    actual_ids, actual_offsets = tokenize_encode_offsets(line)
    np.testing.assert_array_equal(actual_ids, ref_ids)
    np.testing.assert_array_equal(actual_offsets, ref_offsets)


# ---------------------------------------------------------------------------
# Rust fast-path tests using the real ICU tokenizer
# The Rust path (encode_and_offsets_rs) is active when:
#   - softmatcha_rs is importable (Rust extension built with maturin)
#   - the tokenizer exposes get_span_bounds()  (TokenizerICU does)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def icu_tok_with_rust_vocab():
    """Build a real TokenizerICU + matching _worker_rust_vocab for Rust-path tests."""
    import json, tempfile, os
    vocab_words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "hello", "world", "café", "bar", "a", "be", "in", "on", "at",
        "中文", "test", "leading", "spaces",
    ]
    d = {w: i for i, w in enumerate(vocab_words)}
    d["<unk>"] = len(vocab_words)

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "vocab.json"), "w") as f:
            json.dump(d, f)

        from softmatcha.tokenizers.icu import TokenizerICU
        tok = TokenizerICU.build(TokenizerICU.Config(name_or_path=tmpdir, lang="en"))

        try:
            from softmatcha_rs import build_rust_vocab
            rust_vocab = build_rust_vocab(list(tok.dictionary.items()), tok.unk_idx)
        except ImportError:
            pytest.skip("softmatcha_rs not built; skipping Rust-path tests")

        yield tok, rust_vocab


@pytest.mark.parametrize("line", [
    "",
    "   ",
    "hello",
    "hello world",
    "Hello World",
    "the quick brown fox jumps over the lazy dog",
    "café bar",
    "hello 中文 world",
    "  leading spaces",
    "test test test",          # repeated tokens — ICU gives correct offsets
    "The the fox",             # mixed case
])
def test_rust_path_matches_icu_python_path(line, icu_tok_with_rust_vocab):
    """The Rust fast path must agree with the Python ICU path on all inputs.

    Both paths use ICU break-iterator positions, so they should give identical
    results.  Note: for repeated tokens like 'the the the', the Rust+ICU path
    correctly gives distinct offsets (e.g. [0,4,8]), unlike the old Python
    reference which had the position-0 quirk ([0,0,0]).
    """
    tok, rust_vocab = icu_tok_with_rust_vocab

    # Python ICU path
    symbols, char_positions = tok.tokenize_raw_with_char_offsets(line)
    n = len(symbols)
    if n == 0:
        py_ids = np.empty(0, dtype=np.uint32)
        py_offs = np.empty(0, dtype=np.uint32)
    else:
        d = tok.dictionary; unk = tok.unk_idx
        py_ids = np.empty(n, dtype=np.uint32)
        py_offs = np.empty(n, dtype=np.uint32)
        if line.isascii():
            for i, (sym, cp) in enumerate(zip(symbols, char_positions)):
                py_ids[i] = d.get(sym.lower(), unk)
                py_offs[i] = cp
        else:
            line_bytes_np = np.frombuffer(line.encode("utf-8"), dtype=np.uint8)
            char_byte = np.where((line_bytes_np < 0x80) | (line_bytes_np >= 0xC0))[0]
            for i, (sym, cp) in enumerate(zip(symbols, char_positions)):
                py_ids[i] = d.get(sym.lower(), unk)
                py_offs[i] = int(char_byte[cp]) if cp < len(char_byte) else 0

    # Rust path
    from softmatcha_rs import encode_and_offsets_rs
    starts_np, ends_np = tok.get_span_bounds(line)
    if len(starts_np) == 0:
        rust_ids = np.empty(0, dtype=np.uint32)
        rust_offs = np.empty(0, dtype=np.uint32)
    else:
        rust_ids, rust_offs = encode_and_offsets_rs(line, starts_np, ends_np, rust_vocab)

    np.testing.assert_array_equal(rust_ids, py_ids)
    np.testing.assert_array_equal(rust_offs, py_offs)


