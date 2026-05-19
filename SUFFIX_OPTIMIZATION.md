# Suffix Array Construction — Optimization Report

## Context

The suffix array is built in `rust/src/index/` via `build_sa_rs` (called from `build_index` in Python).
The three phases print:

```
Phase 1 begins.. (~0% >> ~10%)      # ~4 s
Phase 2 begins.. (~10% >> ~90%)     # dominant cost
Phase 3 begins.. (~90% >> 100%)     # fast
```

This report focuses on **Phase 2**, which took 112 seconds on the 6 GB test corpus (1.25B tokens)
and is projected to take many hours on the final 1.5 trillion-token corpus without changes.

---

## Phase 2 Algorithm Summary

Phase 2 builds the complete suffix array in two sub-phases, repeated for `num_shard` shards:

**Phase 2a (scatter)** — for each shard, scan the entire token corpus and write
`(position, 256-bit hash)` pairs to disk, partitioned into `num_samples+1` buckets.

**Phase 2b (sort)** — for each partition bucket, read the pairs back from disk,
sort by hash key, write sorted positions to `sa.bin`, and write compressed index
entries to `index.bin` and `rough.bin`.

Key parameters (derived from `mem_size` and `num_shards`):

| Parameter | Formula | Test value (6 GB) | Final value (1.5T tokens) |
|---|---|---|---|
| `chunk_size` | `mem_size × 8333` | 416 M tokens | 416 M tokens |
| `num_loops` | `⌈N / chunk_size⌉` | 3 | 3,601 |
| `num_samples` | `N / chunk_size + 1` | 3 | 3,601 |
| `num_partitions` | `num_samples + 1` | 4 | 3,602 |
| `partition_size` | `N / num_partitions` | 312 M items | 416 M items |
| `sa_size` (bytes/pos) | `⌈log₂₅₆(max_tokens)⌉` | 6 | 6 |

---

## Intermediate File Sizes at 1.5 Trillion Tokens

| File | Content | Size | Read by search tool? |
|---|---|---|---|
| `index.bin` | 32-byte hash key per position | **48 TB** | **Yes** |
| `sa.bin` | 6-byte position per token | 9 TB | No (rebuilt into final index) |
| **Total I/O** | written + read back | **114 TB** | — |

---

## Bottleneck 1 — The 8 NVMe SSDs Are Completely Unused

The current code writes all files to a single directory with a single `File` handle per file.
There is no striping, no parallel placement. The 40 TB of NVMe capacity at **8 × 7 GB/s = 56 GB/s**
is available, but Phase 2 uses only 7 GB/s (one drive).

| Configuration | I/O rate | Time for 114 TB |
|---|---|---|
| Current (1 SSD path) | 7 GB/s | **4.5 hours** |
| 8 SSDs striped | 56 GB/s | **34 minutes** |

**Fix options:**

- **OS-level RAID-0** across all 8 devices — zero code changes, the search tool sees single
  files and needs no modification. This is the right default choice given the downstream
  `index.bin` dependency.
- **Code-level striping** — split each file across 8 paths in Rust. Gives control over
  stripe size, but requires the search tool to understand the multi-file layout.

---

## Bottleneck 2 — `chunk_size` Is ~15× Too Small for the Final Machine

With `mem_size=50000`, `chunk_size = 416M` tokens. This creates **3,602 partitions** at the
1.5T scale. With 1 TB of RAM, `mem_size` can be increased to ~900,000, giving
`chunk_size ≈ 7.5B` tokens and only **~202 partitions**.

This is a **single parameter change** to the indexer invocation:

```
--mem_size=900000   # instead of 50000
```

Effects at 1.5T tokens:

| Metric | `mem_size=50000` | `mem_size=900000` |
|---|---|---|
| `num_partitions` | 3,602 | ~202 |
| `partition_size` | 416 M items | ~7.4 B items |
| Phase 2b peak memory (recs+idrg) | 30 GB | ~535 GB |
| Phase 2a peak memory (sub\_sa+sub\_id) | ~17 GB | ~294 GB |
| Phase 2b sort count | 3,602 | ~202 |

Both fit within 1 TB RAM (Phase 2a and 2b don't overlap). The tradeoff is that each
individual sort is larger, but far fewer sorts run overall.

---

## Bottleneck 3 — Comparison Sort Does Not Scale to This Problem Size

`par_sort_unstable_by` on `([u64; 4], u64)` is O(N log N) with cache-unfriendly
random memory access. For a 30 GB working set (partition_size × 40 bytes),
the data overwhelms L3 cache and cache-miss cost dominates.

Estimated Phase 2b sort time (128 cores, optimistic):

| Algorithm | Time/partition | 202 partitions | 3,602 partitions |
|---|---|---|---|
| `par_sort_unstable_by` | ~90 s | ~5 h | >22 h |
| Parallel radix sort | ~24 s | **80 min** | ~24 h |

The sort key is a **fixed-width 256-bit integer** (four consecutive u64 values),
which is ideal for radix sort — no comparator function, purely sequential
memory access per pass. An 8-pass (32-bit digits) or 16-pass (16-bit digits)
parallel radix sort gives O(N) time with much better cache behavior at large N.

With `mem_size=900000` + radix sort: **~80 minutes** for all Phase 2b sorts.

---

## Bottleneck 4 — Phase 2a Scans the Token Corpus 3× Unnecessarily

With `num_shard=3`, Phase 2a runs once per shard. Each run classifies **all** 1.5T tokens
(step `<2-1>`), but only writes the 1/3 belonging to the current shard (step `<2-3>`).
The other 2/3 of each classify pass is discarded.

A single pass suffices: the per-chunk buffers (`sub_sa`, `sub_id`) are already sized
by `chunk_size`, not by shard — they hold data for all partitions simultaneously.
Setting `num_shard=1` eliminates the two redundant corpus scans.

**Fix:** Change `num_shards=3` to `num_shards=1` in `build_main.py`.

| `num_shard` | Token reads (Phase 2a) | Time at 56 GB/s |
|---|---|---|
| 3 (current) | 18 TB | ~5.4 min |
| 1 | 6 TB | ~1.8 min |

This saves ~3.6 minutes — secondary to the I/O and sort bottlenecks, but free.

There is also a double `compress_build` call per in-shard token within each Phase 2a
chunk (step `<2-1>` classifies all tokens, step `<2-3>` recomputes the key for in-shard
tokens). At 1.5T scale this is compute-overlapped with I/O and not the primary bottleneck.

---

## The 48 TB `index.bin` — Design Constraint

`index.bin` stores a compressed 32-byte `(hash_key, rough_offset)` entry for every
unique hash boundary in the sorted suffix array. It is:

- Written by Phase 2b during suffix array construction
- Read by the downstream search tool at query time for binary search

The format **cannot change** without modifying the search tool. However, how it is
*generated* and *stored* can change freely. The right approach is to stripe `index.bin`
across all 8 SSDs at the OS level (RAID-0), which is transparent to both the indexer
and search tool.

An alternative — skipping `index.bin` during Phase 2a and recomputing hash keys in
Phase 2b from token positions — is **not viable** at this scale. Positions in each
partition bucket are uniformly scattered across the 6 TB token array; reading them
in scattered order requires 1.5T random 48-byte reads from NVMe (~100 μs each),
which would take weeks.

---

## Recommended Implementation Order

| Priority | Change | Effort | Estimated impact (1.5T tokens) |
|---|---|---|---|
| 1 | **OS RAID-0 across 8 SSDs** | Zero code changes | I/O: 4.5 h → 34 min |
| 2 | **Increase `mem_size` to ~900,000** | 1-line parameter change | Partitions: 3,602 → 202 |
| 3 | **Set `num_shards=1`** | 1-line parameter change | Phase 2a reads: 18 TB → 6 TB |
| 4 | **Radix sort in Phase 2b** | ~100 lines Rust | Sorts: 5 h → 80 min |

The first three changes require no code and deliver most of the improvement.
The radix sort is essential if the corpus grows beyond ~5T tokens or if
the comparison sort proves to be the bottleneck after the other fixes.

---

## Projected Phase 2 Runtime at 1.5T Tokens

After all four changes (8-SSD RAID-0, `mem_size=900000`, `num_shards=1`, radix sort):

| Sub-phase | Dominant cost | Estimated time |
|---|---|---|
| Phase 2a classify + write | Token reads (6 TB) + index.bin writes (48 TB) | ~60 min |
| Phase 2b read + sort + write | index.bin reads (48 TB) + 202 radix sorts | ~115 min |
| **Total Phase 2** | | **~3 hours** |

vs. current algorithm (single SSD path, `mem_size=50000`, `num_shards=3`, comparison sort):
estimated **>15 hours** at 1.5T tokens.
