#!/usr/bin/env python3
"""
Benchmark: Python encode+spans vs Rust encode_and_spans_rs

Measures time to run encode() + get_span_start_positions() on batches of
synthetic English text lines, comparing the pure-Python implementations
against the Rust implementation.

Usage:
    uv run python benchmarks/bench_tokenize_rs.py
    uv run python benchmarks/bench_tokenize_rs.py --lines 5000 --warmup 2 --runs 5
"""
from __future__ import annotations

import argparse
import time
import random
import string
import numpy as np
from icu import BreakIterator, Locale
from softmatcha_rs import init_vocab_rs, encode_and_spans_rs, encode_and_spans_positions_rs
from softmatcha.tokenizers.base import Tokenizer


# ---------------------------------------------------------------------------
# Reference Python implementations (from base.py)
# ---------------------------------------------------------------------------

def py_encode(tokens: list[str], dictionary: dict[str, int], unk_idx: int) -> np.ndarray:
    n = len(tokens)
    out = np.empty(n, dtype=np.uint32)
    for i, tok in enumerate(tokens):
        out[i] = dictionary.get(tok, unk_idx)
    return out


def py_get_span_start_positions(line: str, tokens: list[str]) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

WORD_LIST = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "hello", "world", "python", "rust", "fast", "slow", "data", "text",
    "search", "index", "token", "encode", "span", "byte", "unicode",
    "natural", "language", "processing", "machine", "learning", "model",
    "corpus", "document", "sentence", "word", "character", "offset",
    "position", "start", "end", "count", "size", "length", "buffer",
    "memory", "file", "read", "write", "open", "close", "parse", "build",
]


def make_vocab(words: list[str]) -> tuple[dict[str, int], int]:
    d = {w: i for i, w in enumerate(words)}
    unk = len(words)
    return d, unk


def generate_lines(n: int, avg_words: int, seed: int = 42) -> list[tuple[str, list[str]]]:
    """Return list of (line, raw_tokens) pairs."""
    rng = random.Random(seed)
    result = []
    for _ in range(n):
        k = max(1, int(rng.gauss(avg_words, avg_words * 0.3)))
        words = [rng.choice(WORD_LIST) for _ in range(k)]
        line = " ".join(words)
        result.append((line, words))
    return result


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_python(lines: list[tuple[str, list[str]]], dictionary: dict[str, int], unk_idx: int) -> float:
    t0 = time.perf_counter()
    for line, tokens in lines:
        lower = [t.lower() for t in tokens]
        py_encode(lower, dictionary, unk_idx)
        py_get_span_start_positions(line, tokens)
    return time.perf_counter() - t0


def _apply_bi_strings(bi: BreakIterator, text: str) -> list[str]:
    """Reference apply_break_iterator: ICU boundaries → filtered list[str]."""
    bi.setText(text)
    parts, p0 = [], 0
    for p1 in bi:
        part = text[p0:p1].strip()
        if part:
            parts.append(part)
        p0 = p1
    return parts


def time_full_string(lines: list[tuple[str, list[str]]], bi: BreakIterator) -> float:
    """Current production path: ICU setText→list[str] → encode_and_spans_rs."""
    t0 = time.perf_counter()
    for line, _ in lines:
        tokens = _apply_bi_strings(bi, line)
        encode_and_spans_rs(line, tokens)
    return time.perf_counter() - t0


def time_full_positions(lines: list[tuple[str, list[str]]], bi: BreakIterator) -> float:
    """New fast path: ICU setText→list[int] → encode_and_spans_positions_rs."""
    t0 = time.perf_counter()
    for line, _ in lines:
        bi.setText(line)
        encode_and_spans_positions_rs(line, list(bi))
    return time.perf_counter() - t0


def total_tokens(lines: list[tuple[str, list[str]]]) -> int:
    return sum(len(toks) for _, toks in lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(n_lines: int, avg_words: int, warmup: int, runs: int, label: str,
                  bi: BreakIterator):
    lines = generate_lines(n_lines, avg_words)
    ntok = total_tokens(lines)
    print(f"\n{'─'*60}")
    print(f"  {label}: {n_lines} lines, ~{avg_words} words/line, {ntok:,} tokens total")
    print(f"{'─'*60}")

    # Warmup
    for _ in range(warmup):
        time_python(lines, vocab, unk_idx)
        time_full_string(lines, bi)
        time_full_positions(lines, bi)

    py_times, str_times, pos_times = [], [], []
    for _ in range(runs):
        py_times.append(time_python(lines, vocab, unk_idx))
        str_times.append(time_full_string(lines, bi))
        pos_times.append(time_full_positions(lines, bi))

    py_best = min(py_times)
    str_best = min(str_times)
    pos_best = min(pos_times)

    py_mtok  = ntok / py_best  / 1e6
    str_mtok = ntok / str_best / 1e6
    pos_mtok = ntok / pos_best / 1e6

    print(f"  Python (ICU+encode+spans)          best={py_best*1000:7.1f} ms   {py_mtok:5.2f} Mtok/s")
    print(f"  Rust strings  (ICU→list[str]→Rust) best={str_best*1000:7.1f} ms   {str_mtok:5.2f} Mtok/s   {py_best/str_best:.1f}× over Python")
    print(f"  Rust positions(ICU→list[int]→Rust) best={pos_best*1000:7.1f} ms   {pos_mtok:5.2f} Mtok/s   {py_best/pos_best:.1f}× over Python   {str_best/pos_best:.2f}× over strings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=2000, help="Lines per benchmark scenario")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=5, help="Timed iterations")
    args = parser.parse_args()

    vocab, unk_idx = make_vocab(WORD_LIST)

    # Initialise Rust thread-local vocab
    init_vocab_rs(
        list(vocab.keys()),
        [np.uint32(v) for v in vocab.values()],
        np.uint32(unk_idx),
    )

    print(f"\nVocab size: {len(vocab):,} | unk_idx: {unk_idx}")
    print(f"Warmup: {args.warmup}  |  Runs: {args.runs}  |  Reporting best-of-{args.runs}")

    bi = BreakIterator.createWordInstance(Locale("en"))

    for avg_words, label in [
        (10, "short lines (~10 words)"),
        (100, "medium lines (~100 words)"),
        (1000, "long lines (~1000 words)"),
    ]:
        run_benchmark(args.lines, avg_words, args.warmup, args.runs, label, bi)

    print()
