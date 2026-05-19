# Claude Context — softmatcha2 Codebase

Non-obvious discoveries, gotchas, and hard-won insights from working with this codebase.
See also `SUFFIX_OPTIMIZATION.md` for the full Phase 2 suffix array analysis.

---

## Python Import Gotcha — `softmatcha.index.tokenize`

**DO NOT** do this:
```python
import softmatcha.index.tokenize as tokenize_module
# tokenize_module is now the `tokenize` FUNCTION, not the module!
```

`softmatcha/index/__init__.py` re-exports `from .tokenize import tokenize`, which shadows the
submodule name in the package namespace. Python's `import a.b.c as x` follows attribute
lookup (`a.b.c`) not `sys.modules`, so `x` gets the function.

**Always use:**
```python
import importlib
tokenize_module = importlib.import_module("softmatcha.index.tokenize")
```

---

## `softmatcha_rs` Mock in conftest.py

`softmatcha/index/__init__.py` imports `build_index` from `build.py`, which imports
`softmatcha_rs`. This fails at import time if the Rust extension hasn't been built.

The `tests/conftest.py` works around this by trying the real import first:
```python
if "softmatcha_rs" not in sys.modules:
    try:
        import softmatcha_rs
    except ImportError:
        sys.modules["softmatcha_rs"] = unittest.mock.MagicMock()
```

**Critical:** If you always-mock (original version), Rust-path tests get a MagicMock whose
`encode_and_offsets_rs()` returns something that can't be unpacked as a 2-tuple, causing
mysterious `ValueError: not enough values to unpack`. Always try the real import first.

---

## `Tokenizer.__init__` Sets a CLASS Attribute, Not Instance

```python
def __init__(self, cfg, tokenizer, dictionary):
    type(self)._tokenizer = tokenizer  # CLASS attribute!
```

`_tokenizer` (e.g. the PyICU `icu_tokenizer.Tokenizer` object) is shared across ALL
instances of the same subclass. In `init_worker`, `tokenizer.build(cfg)` is called but its
return value is discarded — the only effect is refreshing the class-level `_tokenizer`.

This means you can't have two `TokenizerICU` instances with different underlying ICU
tokenizers in the same process.

---

## Worker Globals — Must Save/Restore All Three in Tests

`tokenize.py` has three module-level globals used by `tokenize_encode_offsets`:
```python
_worker_tokenizer = None
_worker_rust_vocab = None
_worker_use_icu4x = False
```

Tests that install a mock tokenizer via `set_tokenizer()` must reset all three, not just
`_worker_tokenizer`. Otherwise, globals set by one test file (e.g. `_worker_use_icu4x=True`
from the icu4x validation tests) leak into other test files and cause wrong code paths.

The autouse fixture and `set_tokenizer()` must both explicitly reset all three.

---

## `apply_break_iterator` vs `getRuleStatus()` — Critical Behavior Difference

`tokenize_raw()` calls `icu_tokenizer`'s `apply_break_iterator`, which includes **every
non-empty span after strip** — this means hyphens, periods, punctuation, etc. are returned
as tokens alongside words:

- `"hello-world"` → `["hello", "-", "world"]` (3 tokens)
- `"Hello world."` → `["Hello", "world", "."]` (3 tokens)

`getRuleStatus() != 0` (UBRK_WORD_LETTER=200 or UBRK_WORD_NUMBER=400) filters to **only
word-like spans** — skipping punctuation entirely:

- `"hello-world"` → `["hello", "world"]` (2 tokens)

All optimized paths (`tokenize_raw_with_char_offsets`, `get_span_bounds`,
`tokenize_and_encode_rs`) MUST use the `apply_break_iterator` logic (include all non-empty
spans) to match the original behavior. Using `getRuleStatus()` silently drops punctuation
tokens, changing the indexed sequence.

---

## ICU BreakIterator Character Positions — Key Property

ICU's word-break iterator **never mixes content and whitespace in the same span**. Every
span is either pure whitespace OR pure non-whitespace content. Therefore:

- `span.strip() == span` for every non-whitespace span
- The token starts at exactly `p0` (the span start position) within the stripped text
- `span.find(token) == 0` always for word and punctuation spans
- You can use `leading + p0` directly as the token's char position — no need to search

This makes `tokenize_raw_with_char_offsets` simpler and faster: no `span.find()` needed.

---

## ICU 78.3 Emoji Quirk

On the test/EC2 machine, **ICU 78.3** (via PyICU) has a quirk with emoji followed by a
space:

```
"hello 🎉 world" → ["hello", "🎉 ", "w", "orld"]
```

The trailing space gets grouped with 🎉 into one span, causing "world" to split into "w"
and "orld". **icu4x 2.2** (the Rust library) does not have this quirk and correctly
produces `["hello", "🎉", "world"]`.

This is an ICU data-version difference, not an algorithm difference. For the English
fasttext use case, emoji followed immediately by ASCII words is essentially nonexistent in
practice. The `test_utf8_emoji` test documents this and verifies offsets are valid rather
than requiring exact PyICU match.

---

## icu_segmenter 2.2 API (Non-Obvious)

The API changed between expected and actual in icu_segmenter 2.2:

```rust
// WRONG (doesn't compile):
static WORD_SEGMENTER: WordSegmenter = WordSegmenter::new_auto();

// CORRECT:
use icu_segmenter::{WordSegmenter, WordSegmenterBorrowed};
use icu_segmenter::options::WordBreakInvariantOptions;

thread_local! {
    static WORD_SEGMENTER: WordSegmenterBorrowed<'static> =
        WordSegmenter::new_auto(WordBreakInvariantOptions::default());
}
```

`new_auto()` returns `WordSegmenterBorrowed<'static>`, not `WordSegmenter`. The owned
`WordSegmenter` type is a thin wrapper; all methods live on `WordSegmenterBorrowed`.

`is_word_like()` is called on the **iterator** after each `next()`, not on the segmenter:
```rust
let mut iter = seg.segment_str(trimmed);
let mut prev = 0;
while let Some(boundary) = iter.next() {
    if iter.is_word_like() { /* word-like span */ }
    prev = boundary;
}
```

---

## PyO3 Numpy Return Type

Must use `Bound<'py, PyArray1<T>>`, not `Py<PyArray1<T>>`. The correct pattern (matching
the existing search code in `rust/src/search/main.rs`):

```rust
#[pyfunction]
pub fn my_fn<'py>(py: Python<'py>, ...) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let vec: Vec<u32> = ...;
    Ok((
        PyArray1::from_vec(py, vec),
        PyArray1::from_vec(py, other_vec),
    ))
}
```

---

## The Position-0 Quirk in `get_span_start_positions`

The original `base.py` implementation has a subtle bug:

```python
if start_position > 0:
    start_position += len(token)  # does NOT advance if token found at position 0
```

When the first token is at character position 0, `start_position` stays 0. All subsequent
searches for the SAME token string start from 0, so they also find position 0. Result:
`"the the the"` → offsets `[0, 0, 0]` instead of `[0, 4, 8]`.

The optimized paths (ICU `getRuleStatus()`/icu4x) give CORRECT offsets `[0, 4, 8]`.
The test suite documents `[0, 0, 0]` as "existing behavior" using the mock whitespace
tokenizer (which uses the base-class default `tokenize_raw_with_char_offsets`). In
production with the ICU tokenizer, correct offsets are produced.

---

## The `black_list` Bug in `fill_all` (Fixed)

In `tokenize.py`'s Numba JIT function `fill_all`, the original guard condition was:

```python
# BUGGY — allows writing 2*(N-1) into an N-element array:
if cur - prv >= 255 and black_cnt[worker_id] < len(black_list[worker_id]):
```

Each "exception" entry occupies **two** consecutive uint64 slots: `[2*k]` = position,
`[2*k+1]` = delta. The correct condition:

```python
# FIXED:
if cur - prv >= 255 and black_cnt[worker_id] < len(black_list[worker_id]) // 2:
```

Numba JIT runs without bounds checking, so the out-of-bounds writes are silent. The crash
manifests later in the Python readback loop as:
```
IndexError: index 12608 is out of bounds for axis 0 with size 12608
```
at `black_list[i][2 * j + 0]` where `j` reaches `len(black_list[i]) // 2`.

The bug is triggered when a corpus generates enough "exception" token pairs (consecutive
tokens more than 255 bytes apart in the file) to fill the pre-allocated black list.

---

## `bytes_rec` vs Absolute File Offsets in `fill_all`

`bytes_rec` (the `tmp3.bin` mmap, called `rec_i32` inside `fill_all`) stores **relative
byte offsets within each line**, NOT absolute file positions. The absolute position is:

```python
cur = lines_byt[current_line] + rec_i32[j]
```

where `lines_byt[c]` = absolute byte offset of line `c` in the input corpus file.
`rec_i32[j]` = byte offset of token `j` within its line (from `tokenize_encode_offsets`).

The black-list exception mechanism (`byte_offset3`) records token pairs where the
absolute delta `cur - prv >= 255` bytes. Most exceptions occur at line boundaries
(large jump from end of one line to start of next), not within lines.

---

## `init_worker` Ignores the Return Value of `tokenizer.build(cfg)`

```python
def init_worker(tokenizer, cfg):
    global _worker_tokenizer
    _worker_tokenizer = tokenizer
    tokenizer.build(cfg)  # return value intentionally discarded
```

`build()` is a classmethod that constructs a new instance. The discarded return value
doesn't matter — the only observable effect is that `type(tokenizer)._tokenizer` (the
class-level ICU break iterator) gets refreshed with a new `icu_tokenizer.Tokenizer`.
This ensures the per-process tokenizer state is fresh after forking.

---

## `SA_SIZE` Computation (Suffix Array Position Width)

```python
SA_SIZE = 0
for i in range(8):
    if (2 ** (i * 8)) >= max_tokens:
        SA_SIZE = i
        break
```

This is the minimum number of bytes needed to represent any position index up to
`max_tokens`. For 1.5T tokens: `2^40 = 1.09T < 1.5T`, `2^48 = 281T > 1.5T` → SA_SIZE = 6.
This affects all intermediate file sizes: `sa.bin` = N × SA_SIZE bytes.

---

## Phase 2 Key Structural Facts

- **`index.bin`** stores compressed `(hash_key, rough_offset)` entries — one entry per
  unique hash boundary in the sorted suffix array (NOT one entry per token position). The
  search tool does binary search in this file. Its format is fixed by the write logic in
  `phase_2b.rs` and cannot be changed without modifying the search tool.

- **`rough.bin`** stores a sparse index at `rough_div` boundaries (every `rough_div`
  positions in the suffix array). It holds 5 × u64 per entry: 4 for the hash key + 1 for
  the corresponding `index.bin` offset. Used to narrow binary search range.

- **`num_samples`** (partitions - 1) is derived from `num_tokens / chunk_size + 1`, not
  configurable directly. The only way to change the number of partitions is to change
  `chunk_size` via `mem_size`.

- **Phase 2a buffers** (`sub_sa`, `sub_id`) are sized by `chunk_size`, NOT by shard size.
  With `num_shard=3`, only 1/3 of each buffer gets used per shard. Setting `num_shard=1`
  uses the full buffer and eliminates 2/3 of the corpus-scan work in Phase 2a.

- **`compress_build`** packs 12 consecutive u32 tokens (each 20-bit vocabulary ID) into
  [u64; 5]: the first 4 u64s are the sort key, the 5th is the position index. The packing
  is a custom bit-interleaving scheme (not straightforward concatenation). `compress`
  (without `_build`) is the search-time variant that takes a pattern of ≤12 tokens.

---

## Testing Infrastructure Summary

| File | Purpose |
|---|---|
| `tests/conftest.py` | Installs real `softmatcha_rs` or MagicMock fallback |
| `tests/test_tokenize_encode_offsets.py` | Unit tests for `tokenize_encode_offsets` using `SimpleTestTokenizer` (whitespace splitter); also tests Rust path with real ICU tokenizer |
| `tests/test_icu4x_validation.py` | Validates icu4x Rust word segmentation against PyICU; 2000+ random lines, edge cases, byte-offset verification |

Run all tests: `uv run pytest tests/ -v`

Rebuild Rust after any change to `rust/src/`: `uv run maturin develop --release --manifest-path rust/Cargo.toml`

---

## Optimization Summary — `tokenize_encode_offsets`

Original: **15.9 μs/line** (real ICU tokenizer, 1.25B token corpus)

The function went through three optimization layers:

1. **Combine encode+offset loops, ASCII fast path, byte-search for non-ASCII**
   - Eliminates intermediate list `[sym.lower() for sym in symbols]`
   - ASCII: `char_pos == byte_pos`, skip all encoding
   - Non-ASCII: encode line once, `bytes.find()` instead of cumulative `str.encode()` slices
   - Result: ~9.1 μs/line (1.75× speedup)

2. **Capture ICU break iterator positions directly** (eliminates `str.find` re-scan)
   - `tokenize_raw_with_char_offsets()` in `TokenizerICU`: uses `p0` from break iterator loop
   - `get_span_bounds()`: returns numpy uint32 arrays of (starts, ends) for Rust
   - Eliminates `get_span_start_positions()` entirely
   - Result: ~8.8 μs/line (1.81× speedup)

3. **Full icu4x Rust tokenization** (`tokenize_and_encode_rs`)
   - `WORD_SEGMENTER` stored in `thread_local!`, one per worker
   - Entire classify+encode+offset loop in Rust, zero Python/C++ per-boundary calls
   - Byte positions come directly from icu4x's UTF-8 byte indices (no char→byte conversion)
   - Result: **2.17 μs/line (7.37× speedup)**

The icu4x path activates when `_worker_use_icu4x=True`, set in `init_worker` when
`softmatcha_rs` is available and the tokenizer has `get_span_bounds()` (i.e., is ICU-based).
