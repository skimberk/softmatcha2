"""
Correctness tests for the optimised pre-phase line counter.

Compares the new parallel block-scan implementation against the reference
readline() approach on a variety of synthetic inputs.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

# ── reference: exact copy of the old readline loop ───────────────────────────

def _old_scan(path: str) -> np.ndarray:
    """Original readline()-based line scanner (reference implementation)."""
    LINES_SIZE = 4_096
    lines_byt = np.zeros(LINES_SIZE, dtype=np.uint64)
    num_lines = 0
    with open(path, mode="rb") as f:
        while True:
            pos = f.tell()
            lines_byt[num_lines] = pos
            line = f.readline()
            if not line:
                break
            num_lines += 1
            if num_lines + 1024 > LINES_SIZE:
                LINES_SIZE *= 2
                lines_byt = np.resize(lines_byt, LINES_SIZE)
    return num_lines, lines_byt[:num_lines + 1].copy()


# ── new: the implementation now living in tokenize.py (inlined for testing) ──

from concurrent.futures import ThreadPoolExecutor

_BLOCK = 64 * 1024 * 1024


def _scan_region(fd: int, region_start: int, region_end: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    pos = region_start
    while pos < region_end:
        n = min(_BLOCK, region_end - pos)
        block = os.pread(fd, n, pos)
        if not block:
            break
        arr = np.frombuffer(block, dtype=np.uint8)
        nl_offsets = np.where(arr == ord('\n'))[0]
        if len(nl_offsets):
            parts.append((pos + nl_offsets + 1).astype(np.uint64))
        pos += len(block)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)


def _new_scan(path: str, num_workers: int = 4):
    """New parallel block scanner (mirrors the tokenize.py implementation)."""
    file_size = os.path.getsize(path)
    last_newline = True
    if file_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            last_newline = (f.read(1) == b'\n')

    chunk = max(_BLOCK, (file_size + num_workers - 1) // num_workers)
    regions = []
    for i in range(num_workers):
        s = i * chunk
        if s >= file_size:
            break
        regions.append((s, min(s + chunk, file_size)))

    fd = os.open(path, os.O_RDONLY)
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            parts = list(pool.map(lambda r: _scan_region(fd, r[0], r[1]), regions))
    finally:
        os.close(fd)

    all_starts = np.concatenate(parts) if any(len(p) for p in parts) else np.empty(0, dtype=np.uint64)
    num_lines = len(all_starts)

    lines_byt = np.zeros(num_lines + 2, dtype=np.uint64)
    lines_byt[0] = 0
    if num_lines:
        lines_byt[1 : num_lines + 1] = all_starts

    if file_size > 0 and not last_newline:
        num_lines += 1
        # Store the EOF position at lines_byt[num_lines], matching old behaviour.
        if num_lines < len(lines_byt):
            lines_byt[num_lines] = file_size

    return num_lines, lines_byt[:num_lines + 1].copy()


# ── helpers ───────────────────────────────────────────────────────────────────

def _write(path: str, content: bytes) -> None:
    with open(path, "wb") as f:
        f.write(content)


def _check(content: bytes, num_workers: int = 4) -> None:
    """Assert old and new produce identical (num_lines, lines_byt) for content."""
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        _write(tf.name, content)
        try:
            old_n, old_byt = _old_scan(tf.name)
            new_n, new_byt = _new_scan(tf.name, num_workers=num_workers)
            assert old_n == new_n, f"line count mismatch: old={old_n} new={new_n}"
            # old stores an extra EOF entry; new may or may not — compare the shared prefix
            shared = min(len(old_byt), len(new_byt))
            assert np.array_equal(old_byt[:shared], new_byt[:shared]), (
                f"lines_byt mismatch:\n  old={old_byt[:shared]}\n  new={new_byt[:shared]}"
            )
        finally:
            os.unlink(tf.name)


# ── test cases ────────────────────────────────────────────────────────────────

class TestLineCounter:
    def test_empty_file(self):
        _check(b"")

    def test_single_line_with_newline(self):
        _check(b"hello world\n")

    def test_single_line_no_newline(self):
        _check(b"hello world")

    def test_two_lines(self):
        _check(b"line one\nline two\n")

    def test_two_lines_no_trailing_newline(self):
        _check(b"line one\nline two")

    def test_blank_lines(self):
        _check(b"a\n\nb\n\n\nc\n")

    def test_many_short_lines(self):
        content = b"\n".join(f"line {i}".encode() for i in range(10_000)) + b"\n"
        _check(content)

    def test_utf8_multibyte(self):
        content = "héllo wörld\n日本語テスト\ncafé naïve\n".encode("utf-8")
        _check(content)

    def test_single_worker(self):
        content = b"\n".join(f"line {i}".encode() for i in range(1_000)) + b"\n"
        _check(content, num_workers=1)

    def test_more_workers_than_lines(self):
        """File smaller than one block — all workers see the same single region."""
        content = b"short\nfile\n"
        _check(content, num_workers=8)

    def test_line_starts_are_correct(self):
        """Verify each lines_byt entry actually points to the start of that line."""
        lines = [f"line{i} content here".encode() for i in range(100)]
        content = b"\n".join(lines) + b"\n"
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            _write(tf.name, content)
            try:
                n, byt = _new_scan(tf.name, num_workers=2)
                assert n == 100
                with open(tf.name, "rb") as f:
                    for i in range(100):
                        f.seek(byt[i])
                        data = f.read(len(lines[i]))
                        assert data == lines[i], f"line {i}: expected {lines[i]!r} got {data!r}"
            finally:
                os.unlink(tf.name)
