# Basic Codebase Context — softmatcha2

Orientation guide for newcomers. For non-obvious gotchas and optimization notes,
see `CLAUDE_CONTEXT.md` and `SUFFIX_OPTIMIZATION.md`.

---

## What Is This?

**SoftMatcha** is a fast fuzzy pattern matcher for large text corpora. It builds an
inverted suffix-array index over a tokenized corpus and supports approximate
(embedding-similarity-based) pattern search with sub-second latency.

---

## Build & Setup

Python 3.12 is pinned. Always use `uv`:

```bash
uv python pin 3.12
uv sync
uv run maturin develop --release --manifest-path rust/Cargo.toml
```

The last command compiles the Rust extension (`softmatcha_rs`). **You must re-run it after
any change to files under `rust/src/`.** Without it, `softmatcha_rs` either doesn't exist
or is stale, and imports will fail or give wrong results.

Run tests:
```bash
uv run pytest tests/ -v
```

---

## CLI Entry Points

All defined in `pyproject.toml` under `[project.scripts]`:

| Command | Entry point | Purpose |
|---|---|---|
| `softmatcha-index` | `softmatcha.cli.build_main:cli_main` | Build index from corpus |
| `softmatcha-search` | `softmatcha.cli.search_main:cli_main` | Fuzzy pattern search |
| `softmatcha-exact` | `softmatcha.cli.exact_main:cli_main` | Exact pattern search |
| `softmatcha-estimate` | `softmatcha.cli.estimate:cli_main` | Estimate token count |
| `softmatcha-bench` | `softmatcha.cli.bench_main:cli_main` | Benchmarking |

Typical indexer invocation:
```bash
uv run softmatcha-index \
  --index corpus2 \
  --mem_size=50000 \
  --buffer_size 5000 \
  --mem_size_ex=1000 \
  --backend=fasttext \
  --model=fasttext-en-vectors \
  corpus.txt
```

`--mem_size` (MB) controls how much RAM the indexer uses during suffix array construction.
Set it as high as safely possible — it directly controls partition count and sort size.
`--mem_size_ex` controls how much RAM the search tool uses at query time.

---

## Project Layout

```
softmatcha2/
├── src/softmatcha/
│   ├── cli/               # CLI entry points
│   │   ├── build_main.py  # softmatcha-index logic + parameter derivation
│   │   ├── search_main.py # softmatcha-search
│   │   └── exact_main.py  # softmatcha-exact
│   ├── index/
│   │   ├── tokenize.py    # Phase 1 of indexing: tokenize corpus → tokens.bin
│   │   └── build.py       # Phase 2 of indexing: build suffix array (calls Rust)
│   ├── tokenizers/
│   │   ├── base.py        # Abstract Tokenizer class
│   │   ├── icu.py         # ICU tokenizer (used by fasttext backend)
│   │   ├── fasttext.py    # FastText router (→ ICU, MeCab, or Moses by language)
│   │   ├── mecab.py       # Japanese tokenizer
│   │   ├── moses.py       # Greek/Latin/Hebrew tokenizer
│   │   ├── transformers.py# BERT/RoBERTa etc.
│   │   ├── gensim.py      # GloVe
│   │   └── llama.py       # Llama-style models
│   ├── embeddings/
│   │   ├── base.py        # Abstract Embedding class
│   │   ├── fasttext.py    # FastText embeddings
│   │   ├── gensim.py      # GloVe embeddings
│   │   ├── transformers.py# BERT/RoBERTa embeddings
│   │   └── llama.py       # Llama embeddings
│   ├── search/
│   │   └── search.py      # Search logic (uses Rust binary search)
│   ├── struct/
│   │   ├── pattern.py     # Pattern data structure
│   │   └── token_embeddings.py
│   ├── utils/
│   │   ├── io.py          # buffer_lines(), read_lines() — corpus I/O
│   │   ├── makefile.py    # make_file() — allocate binary files
│   │   ├── custom_tqdm.py # Progress bar wrapper
│   │   └── fasttext.py    # FastText model download helper
│   ├── registry.py        # Plugin registry (register/get_tokenizer, register/get_embedding)
│   ├── configs.py         # argparse configuration
│   └── stopwatch.py       # Timing utilities (with timers["key"]: ...)
├── rust/
│   ├── Cargo.toml         # Rust dependencies (pyo3, numpy, rayon, icu_segmenter, ...)
│   └── src/
│       ├── lib.rs         # PyO3 module definition — register all Python-callable functions
│       ├── tokenize.rs    # RustVocab, encode_and_offsets_rs, tokenize_and_encode_rs
│       ├── helper.rs      # compress_build(), compress(), retrieve_value(), etc.
│       ├── index/         # Suffix array construction (Phase 1, 2, 3)
│       │   ├── main.rs    # build_sa_rs() entry point
│       │   ├── phase_1.rs # Random sampling, partition boundaries
│       │   ├── phase_2.rs # Phase 2 orchestrator (shards)
│       │   ├── phase_2a.rs# Scatter: classify tokens → write to buckets
│       │   ├── phase_2b.rs# Sort: read buckets, sort, write suffix array
│       │   ├── phase_3a.rs# 2-gram and 3-gram bit tables
│       │   ├── phase_3b.rs# Token frequency counts
│       │   └── memmap.rs  # FastMmapVec<T> wrapper
│       └── search/        # Binary search over suffix array at query time
├── tests/
│   ├── conftest.py        # softmatcha_rs import (real or mock fallback)
│   ├── test_tokenize_encode_offsets.py  # tokenize_encode_offsets unit tests
│   └── test_icu4x_validation.py         # icu4x vs PyICU validation
├── CLAUDE_CONTEXT.md      # Non-obvious gotchas and optimization discoveries
├── SUFFIX_OPTIMIZATION.md # Phase 2 suffix array bottleneck analysis
└── pyproject.toml         # Dependencies, entry points, dev tools
```

---

## Data Flow: Indexing

```
corpus.txt
    │
    ▼  [tokenize.py — ProcessPoolExecutor]
    │  tokenize_encode_offsets(line) → (token_ids[], byte_offsets[])
    │  worker globals: _worker_tokenizer, _worker_rust_vocab, _worker_use_icu4x
    │
    ▼  [binary files written to index_path/]
    ├── tokens.bin       uint32[]  — token IDs for every position in corpus
    ├── offset.bin       mixed     — sparse-encoded byte offsets (for search result display)
    ├── metadata.bin     uint64[]  — num_tokens, num_lines, vocab size, etc.
    │
    ▼  [build.py → build_sa_rs() Rust]
    │
    ▼  [Rust Phase 1: random sample → partition boundaries]
    │
    ▼  [Rust Phase 2: suffix array construction]
    ├── sa.bin           bytes     — sorted token positions (SA_SIZE bytes each)
    ├── index.bin        uint64[]  — compressed (hash, rough_offset) entries for binary search
    └── rough.bin        uint64[]  — sparse index at rough_div boundaries
    │
    ▼  [Rust Phase 3: frequency + n-gram tables]
    ├── metadata.bin     (extended with freq[], pair[], trio[], norm[])
    └── [tokens.bin deleted after indexing]
```

---

## Tokenizer + Embedding Registry

The `@register("name")` decorator in `registry.py` maps backend names to classes.
`get_tokenizer("fasttext")` returns `TokenizerFasttext`; `get_embedding("fasttext")`
returns `EmbeddingFasttext`.

**FastText backend** (most common, used with `--backend=fasttext`):
- `TokenizerFasttext` routes by language from the model name:
  - `fasttext-ja-vectors` → `TokenizerMecab`
  - `fasttext-el/la/he-vectors` → `TokenizerMoses`
  - all others → `TokenizerICU`
- Models are downloaded from HuggingFace (`facebook/fasttext-{lang}-vectors`) on first use
- `vocab.json` is loaded from the downloaded model directory with `simdjson`

**Tokenizer interface** (all backends implement):
```python
class Tokenizer:
    cfg: Config
    dictionary: dict[str, int]   # word → token ID
    tokens: dict[int, str]       # token ID → word (reverse)
    def tokenize(line) -> list[str]          # lowercased tokens
    def tokenize_raw(line) -> list[str]      # case-preserving tokens
    def encode(tokens) -> NDArray[uint32]    # dict lookup → IDs
    def get_span_start_positions(line, tokens) -> NDArray[uint32]  # byte offsets
    def tokenize_raw_with_char_offsets(line) -> (list[str], list[int])  # optimized
    def get_span_bounds(line) -> (np.ndarray, np.ndarray)  # ICU only, for Rust path
```

`TokenizerICU.build(cfg)` is a classmethod that loads `vocab.json`, sets the class-level
`_tokenizer` (a `icu_tokenizer.Tokenizer` PyICU wrapper), and returns a new instance.

---

## Key Parameters in `build_main.py`

| Param | Flag | Default | Effect |
|---|---|---|---|
| `mem_size` | `--mem_size` | 500 | MB of RAM for tokenization + suffix array chunk size |
| `mem_size_ex` | `--mem_size_ex` | 100 | MB of RAM for search-time structures |
| `buffer_size` | `--buffer_size` | 2500 | Lines per worker batch during tokenization |
| `max_vocab` | — | 2^19 = 524,288 | Token IDs are capped at this value |
| `num_shards` | — | 3 (hardcoded) | Passes over corpus in suffix array Phase 2a |
| `write_thread` | — | 4 (hardcoded) | Parallel disk write threads in Rust Phase 2 |
| `rough_div` | — | derived | Granularity of the sparse rough index |
| `pair_cons`/`trio_cons` | — | derived from mem_size_ex | N-gram table sizes |

`chunk_size` (passed to Rust) = `mem_size × (1_000_000 // 120)`.
This controls partition count: more RAM → larger chunks → fewer partitions → faster sorts.

---

## Binary File Formats (index_path/)

| File | Type | Content |
|---|---|---|
| `tokens.bin` | `uint32[]` | Token ID for each corpus position; deleted after indexing |
| `offset.bin` | mixed | Sparse byte-offset encoding of token positions within lines |
| `metadata.bin` | `uint64[]` | Header: counts, sizes, vocab, pair/trio tables, norm values |
| `sa.bin` | `uint8[]` (SA_SIZE bytes/entry) | Suffix array: sorted token positions |
| `index.bin` | `uint64[]` (4 per entry) | Compressed hash+offset entries; read by search tool |
| `rough.bin` | `uint64[]` (5 per entry) | Sparse index at `rough_div` boundaries |
| `tmp1.bin`, `tmp2.bin` | `uint64[]` | Line byte positions and token offsets (deleted after tokenize) |
| `tmp3.bin` | `uint32[]` | Raw within-line byte offsets before sparse encoding (deleted) |

`metadata.bin` layout (in uint64 slots):
- `[0]` = num_tokens, `[1]` = num_lines, `[4]` = TOKEN_SIZE (pre-allocated slots),
  `[5]` = LINES_SIZE, `[6]` = MAX_VOCAB, `[7]` = DATA_BEGIN (where freq/norm/pair/trio start)
- From DATA_BEGIN: freq[], norm[] (as float32), pair[] (bit table), trio[] (bit table)
- `[8..13]` = pair_cons, trio_cons, rough_div, SA_SIZE, FILE_PAIR, FILE_TRIO

---

## Rust Functions Exposed to Python

All registered in `rust/src/lib.rs`:

| Function | Purpose |
|---|---|
| `build_sa_rs(tokens, freq, pair, trio, ...)` | Full suffix array construction (Phase 1+2+3) |
| `enumerate_candidates_rs(...)` | Search: find candidate matches in suffix array |
| `get_match_range_rs(...)` | Search: binary search for a pattern's range in SA |
| `build_rust_vocab(items, unk_idx)` | Build Rust-native HashMap for fast token encoding |
| `encode_and_offsets_rs(line, starts, ends, vocab)` | Encode tokens + compute byte offsets (Python ICU + Rust encode) |
| `tokenize_and_encode_rs(line, vocab)` | Full tokenize+encode using icu4x (fastest path) |

`RustVocab` is a `#[pyclass]` holding `HashMap<String, u32>` + `unk_idx`.

---

## `tokenize_encode_offsets` — The Hot Path

This function is called for every line in the corpus via `ProcessPoolExecutor.map()`.
It returns `(token_ids: uint32[], byte_offsets: uint32[])`.

Three execution paths, selected at `init_worker` time:

1. **icu4x Rust** (`_worker_use_icu4x=True`): `tokenize_and_encode_rs(line, _worker_rust_vocab)`
   — active when `softmatcha_rs` is built and tokenizer is `TokenizerICU`.
   Entire word segmentation + encode runs in Rust. **~2.2 μs/line.**

2. **Python ICU + Rust encode** (`_worker_rust_vocab` set, `get_span_bounds` available):
   ICU break iterator gets char positions, Rust encodes + converts to byte offsets.
   **~6 μs/line.**

3. **Python fallback**: `tokenize_raw_with_char_offsets()` + Python encode loop.
   Used for non-ICU tokenizers (Moses, MeCab, Transformers).
   **~9-16 μs/line.**

`init_worker` sets these globals per worker process; `tokenize_encode_offsets` reads them.

---

## `offset.bin` — Sparse Byte Offset Encoding

The `offset.bin` file is not a simple array — it uses a compact encoding:

- **Main region** (`byte_offset1`, uint8[]): delta from previous token's byte position,
  capped at 255. Size = num_tokens bytes.
- **Coarse region** (`byte_offset2`, uint64[]): absolute byte position at every 256-token
  boundary. Size = ceil(num_tokens/256) × 8 bytes.
- **Exception list** (`byte_offset3`, uint64[]): pairs of (token_index, true_delta) for
  cases where the delta exceeds 255 bytes. Written after the main+coarse regions.

The `fill_all` Numba function populates this encoding in the post-tokenize phase.

---

## `compress_build` / `compress` — The Hash Key Format

These functions in `rust/src/helper.rs` encode 12 consecutive token IDs (each 20 bits)
into 4 × u64 (256 bits) for suffix array sorting and search. The packing is a custom
bit-interleaving scheme — NOT a simple concatenation. The 5th u64 (in `compress_build`)
holds the position index and is stripped before sorting.

`compress` (search time) takes a pattern of ≤12 tokens and can produce either a
lower-bound or upper-bound key (for binary search range queries).

`retrieve_value(seq, pos)` reverses the encoding: extracts token at position `pos`
(0–11) from a packed [u64; 4].

---

## Adding a New Tokenizer Backend

1. Create `src/softmatcha/tokenizers/your_backend.py` subclassing `Tokenizer`
2. Implement `build(cls, cfg)`, `tokenize(line)`, `unk_idx`
3. Decorate the class: `@register("your_backend_name")`
4. Import it in `src/softmatcha/tokenizers/__init__.py`
5. Create a matching `EmbeddingYourBackend` in `src/softmatcha/embeddings/`

The `@register` decorator adds the class to a global dict; `get_tokenizer("name")` looks
it up. The string name must match the `--backend` CLI argument.

---

## Tests

```bash
uv run pytest tests/ -v                      # all tests
uv run pytest tests/test_tokenize_encode_offsets.py -v   # unit tests (mock tokenizer)
uv run pytest tests/test_icu4x_validation.py -v          # icu4x vs PyICU validation
```

`tests/conftest.py` — loads real `softmatcha_rs` or installs a MagicMock.
`src/softmatcha/conftest.py` — separate conftest for the package itself (BERT/GloVe fixtures).

The unit tests use `SimpleTestTokenizer` (whitespace-splitting mock) to avoid needing
any model files. The icu4x validation tests use a real `TokenizerICU` built from a
temp directory with a synthetic `vocab.json`.

---

## Common Development Tasks

**After editing any Rust file:**
```bash
uv run maturin develop --release --manifest-path rust/Cargo.toml
```

**Profile a specific function:**
```python
import importlib, time
tokenize_module = importlib.import_module("softmatcha.index.tokenize")
# ... set up _worker_tokenizer ... then time calls
```

**Inspect a built index:**
```python
import numpy as np
meta = np.memmap("corpus2/metadata.bin", dtype=np.uint64, mode='r')
print(f"num_tokens={meta[0]:,}  num_lines={meta[1]:,}  max_vocab={meta[6]:,}")
```

**Check which tokenizer backend is used for a language:**
```python
from softmatcha.tokenizers.fasttext import TokenizerFasttext
# Routes based on regex match on model name:
# fasttext-ja → MeCab, fasttext-el/la/he → Moses, everything else → ICU
```
