#!/usr/bin/env python3
"""
Benchmark: pre-phase line counting (phase 1-1 of softmatcha-index).

Generates a synthetic file, then times four approaches:
  old   – original readline() loop
  new1  – O1+O2: 64 MB block reads + numpy newline scan, single-threaded
  new2  – O1+O2+O3: same but parallel (ThreadPoolExecutor + os.pread)
  new3  – O1+O2+O3 with more threads

Verifies that all approaches produce identical lines_byt arrays.

Usage:
    uv run python benchmarks/bench_prephase.py
    uv run python benchmarks/bench_prephase.py --size-mb 1000 --workers 8
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# ── synthetic file generator ──────────────────────────────────────────────────

def make_synthetic_file(path: str, target_mb: int, seed: int = 42) -> int:
    """Write a file of ~target_mb MB with mixed short and long lines."""
    rng = random.Random(seed)
    vocab = "the quick brown fox jumps over the lazy dog python rust numpy".split()
    written = 0
    target = target_mb * 1024 * 1024
    with open(path, "wb") as f:
        while written < target:
            n_words = rng.choice([5, 20, 100, 500])   # mix of line lengths
            line = (" ".join(rng.choices(vocab, k=n_words)) + "\n").encode()
            f.write(line)
            written += len(line)
    return written


# ── approach 0: original readline() ──────────────────────────────────────────

def scan_readline(input_file: str) -> np.ndarray:
    """Exact replica of the current tokenize.py phase 1-1 (stripped of tqdm/memmap)."""
    file_size = os.path.getsize(input_file)
    # mimic the doubling-array growth
    LINES_SIZE = 4_096
    lines_byt = np.zeros(LINES_SIZE, dtype=np.uint64)
    num_lines = 0
    with open(input_file, mode="rb") as f:
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
    return lines_byt[:num_lines + 1].copy()   # include the EOF entry


# ── approach 1: O1+O2 block + numpy, single-threaded ─────────────────────────

BLOCK_SIZE = 64 * 1024 * 1024   # 64 MB


def scan_block_single(input_file: str) -> np.ndarray:
    """Block read + numpy newline scan, single-threaded."""
    file_size = os.path.getsize(input_file)
    est_lines = max(4_096, int(file_size / 512 * 1.5))   # generous estimate
    lines_byt = np.zeros(est_lines, dtype=np.uint64)
    lines_byt[0] = 0
    num_lines = 0
    byte_pos = 0
    last_newline = True   # treat start-of-file as "after newline" for edge case

    with open(input_file, "rb") as f:
        while True:
            block = f.read(BLOCK_SIZE)
            if not block:
                break
            arr = np.frombuffer(block, dtype=np.uint8)
            nl_offsets = np.where(arr == ord('\n'))[0]
            n = len(nl_offsets)
            if n:
                abs_starts = (byte_pos + nl_offsets + 1).astype(np.uint64)
                need = num_lines + 1 + n
                if need >= len(lines_byt):
                    lines_byt = np.resize(lines_byt, max(need + 4096, len(lines_byt) * 2))
                lines_byt[num_lines + 1 : num_lines + 1 + n] = abs_starts
                num_lines += n
            last_newline = block[-1] == ord('\n')
            byte_pos += len(block)

    if byte_pos > 0 and not last_newline:
        num_lines += 1   # count the final line without a trailing \n

    return lines_byt[:num_lines + 1].copy()


# ── approach 2: O1+O2+O3 parallel block scan ─────────────────────────────────

def _scan_region(fd: int, region_start: int, region_end: int) -> np.ndarray:
    """Scan [region_start, region_end) for newlines; return absolute line-start positions."""
    results: list[np.ndarray] = []
    pos = region_start
    while pos < region_end:
        n = min(BLOCK_SIZE, region_end - pos)
        block = os.pread(fd, n, pos)
        if not block:
            break
        arr = np.frombuffer(block, dtype=np.uint8)
        nl_offsets = np.where(arr == ord('\n'))[0]
        if len(nl_offsets):
            results.append((pos + nl_offsets + 1).astype(np.uint64))
        pos += len(block)
    return np.concatenate(results) if results else np.empty(0, dtype=np.uint64)


def scan_block_parallel(input_file: str, num_workers: int) -> np.ndarray:
    """Parallel block read using os.pread; regions processed by ThreadPoolExecutor."""
    file_size = os.path.getsize(input_file)
    # find the last byte to handle trailing-newline edge case
    with open(input_file, "rb") as f:
        f.seek(-1, 2)
        last_byte = f.read(1)
    last_newline = (last_byte == b'\n')

    # split into equal regions
    chunk = max(BLOCK_SIZE, (file_size + num_workers - 1) // num_workers)
    regions = []
    for i in range(num_workers):
        start = i * chunk
        if start >= file_size:
            break
        regions.append((start, min(start + chunk, file_size)))

    fd = os.open(input_file, os.O_RDONLY)
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            parts = list(pool.map(lambda r: _scan_region(fd, r[0], r[1]), regions))
    finally:
        os.close(fd)

    # concatenate in order (each part is sorted, regions are in order)
    all_starts = np.concatenate(parts) if any(len(p) for p in parts) else np.empty(0, dtype=np.uint64)
    num_lines = len(all_starts)

    # build final lines_byt: prepend 0, append nothing (EOF entry optional)
    lines_byt = np.empty(num_lines + 1, dtype=np.uint64)
    lines_byt[0] = 0
    if num_lines:
        lines_byt[1:] = all_starts

    if not last_newline and file_size > 0:
        num_lines += 1
        # lines_byt already has lines_byt[num_lines-1] = start of that last partial line

    return lines_byt[:num_lines + 1].copy()


# ── timing harness ────────────────────────────────────────────────────────────

def time_fn(fn, *args, runs: int = 3) -> tuple[float, object]:
    result = None
    best = float("inf")
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        result = fn(*args)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best, result


def fmt(seconds: float, mb: float) -> str:
    return f"{seconds*1000:7.0f} ms   {mb/seconds:6.0f} MB/s"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=int, default=500)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        tmp_path = tf.name

    try:
        print(f"Generating {args.size_mb} MB synthetic file … ", end="", flush=True)
        actual_bytes = make_synthetic_file(tmp_path, args.size_mb)
        actual_mb = actual_bytes / 1024 / 1024
        print(f"done ({actual_mb:.1f} MB)")

        print(f"\nBest of {args.runs} runs each:\n")

        t_old, r_old = time_fn(scan_readline, tmp_path, runs=args.runs)
        n_lines = len(r_old) - 1
        print(f"  old  (readline)          {fmt(t_old, actual_mb)}   {n_lines:,} lines")

        t_new1, r_new1 = time_fn(scan_block_single, tmp_path, runs=args.runs)
        print(f"  new1 (block+numpy 1T)    {fmt(t_new1, actual_mb)}   speedup {t_old/t_new1:.1f}×")

        for nw in sorted({2, 4, args.workers}):
            if nw > os.cpu_count():
                continue
            t_newp, r_newp = time_fn(scan_block_parallel, tmp_path, nw, runs=args.runs)
            print(f"  new2 (block+numpy {nw}T)    {fmt(t_newp, actual_mb)}   speedup {t_old/t_newp:.1f}×")

        # Correctness check
        print("\nCorrectness check:")
        r_new1_trimmed = r_new1[:len(r_old)]
        match1 = np.array_equal(r_old, r_new1_trimmed)
        print(f"  old == new1 (single-thread): {'✓ PASS' if match1 else '✗ FAIL'}")
        _, r_newp_last = time_fn(scan_block_parallel, tmp_path, args.workers, runs=1)
        r_newp_trimmed = r_newp_last[:len(r_old)]
        match2 = np.array_equal(r_old, r_newp_trimmed)
        print(f"  old == new2 (parallel):      {'✓ PASS' if match2 else '✗ FAIL'}")

        if not match1:
            for i, (a, b) in enumerate(zip(r_old, r_new1_trimmed)):
                if a != b:
                    print(f"    first diff at index {i}: old={a} new={b}")
                    break

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
