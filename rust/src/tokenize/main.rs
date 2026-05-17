use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::PyArray1;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

// ---------------------------------------------------------------------------
// Fast inline FxHash64 — deterministic, no external dependency.
// Same algorithm used by rustc and Firefox; produces well-distributed 64-bit
// values for the short strings (< 30 chars) typical in NLP vocabularies.
// ---------------------------------------------------------------------------
#[inline]
fn fxhash64(bytes: &[u8]) -> u64 {
    const K: u64 = 0x517cc1b727220a95;
    let mut hash: u64 = 0;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().unwrap());
        hash = hash.rotate_left(5) ^ word;
        hash = hash.wrapping_mul(K);
    }
    for &b in chunks.remainder() {
        hash = hash.rotate_left(5) ^ (b as u64);
        hash = hash.wrapping_mul(K);
    }
    hash
}

// ---------------------------------------------------------------------------
// Identity hasher — vocab keys ARE fxhash64 outputs (well-distributed u64),
// so pass them straight through rather than hashing again.
// Eliminates the AHash re-hash cost (~3 ns/lookup) on top of fxhash64.
// ---------------------------------------------------------------------------
#[derive(Default, Clone)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    #[inline]
    fn write_u64(&mut self, i: u64) { self.0 = i; }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Defensive fallback; for u64 keys Rust always calls write_u64.
        if bytes.len() == 8 {
            self.0 = u64::from_ne_bytes(bytes.try_into().unwrap());
        }
    }
    #[inline]
    fn finish(&self) -> u64 { self.0 }
}

#[derive(Default, Clone)]
struct BuildIdentityHasher;

impl BuildHasher for BuildIdentityHasher {
    type Hasher = IdentityHasher;
    #[inline]
    fn build_hasher(&self) -> IdentityHasher { IdentityHasher(0) }
}

// ---------------------------------------------------------------------------
// Thread-local state
// Keys: fxhash64(lowercase_word_bytes) — avoids storing heap-allocated Strings
// and eliminates the string memcmp (→ __memcmp_evex_movbe) during lookup.
// ---------------------------------------------------------------------------
thread_local! {
    static VOCAB: RefCell<HashMap<u64, u32, BuildIdentityHasher>> =
        RefCell::new(HashMap::with_hasher(BuildIdentityHasher));
    static UNK_IDX: Cell<u32> = const { Cell::new(0) };
}

/// Initialize the thread-local vocabulary. Call once per worker process after
/// the tokenizer is built.
///
/// Keys are hashed to u64 via fxhash64(lowercase_bytes) so that lookup only
/// needs a u64 comparison instead of a full string memcmp.
#[pyfunction]
pub fn init_vocab_rs(keys: Vec<String>, values: Vec<u32>, unk_idx: u32) -> PyResult<()> {
    UNK_IDX.with(|u| u.set(unk_idx));
    VOCAB.with(|v| {
        let mut map = v.borrow_mut();
        map.clear();
        map.reserve(keys.len());
        for (k, id) in keys.into_iter().zip(values) {
            // Lowercase defensively (vocab should already be lowercase in practice).
            let hash = if k.bytes().any(|b| b.is_ascii_uppercase()) {
                fxhash64(k.to_lowercase().as_bytes())
            } else {
                fxhash64(k.as_bytes())
            };
            map.insert(hash, id);
        }
    });
    Ok(())
}

/// Search for `needle` in `haystack`, returning the byte offset of the first match.
#[inline]
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Combined encode + span calculation from ICU BreakIterator char-end positions.
///
/// This is the high-performance replacement for encode_and_spans_rs. Instead of receiving
/// a list[str] of token strings (which requires N Python string allocations per line), it
/// receives the raw char-end positions from pyicu's BreakIterator (just N Python ints).
///
/// char_ends: the boundary positions produced by `list(break_iterator)` in Python — each
///            value is the char index AFTER a span, in strictly ascending order.
/// line: the original text that was passed to break_iterator.setText().
///
/// Rust walks char_indices() once (O(L)) to convert char positions to byte positions,
/// then filters whitespace-only spans, trims, hashes the token bytes, and does vocab
/// lookup — all in one pass without any intermediate Python string objects.
///
/// Returns (token_ids, span_start_bytes) as numpy u32 arrays.
/// Requires init_vocab_rs to have been called in this process first.
#[pyfunction]
pub fn encode_and_spans_positions_rs<'py>(
    py: Python<'py>,
    line: String,
    char_ends: Vec<u32>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let n_ends = char_ends.len();
    if n_ends == 0 {
        return Ok((
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
        ));
    }

    let line_bytes = line.as_bytes();

    // Step 1: Walk char_indices() once to map each char-end position → its byte offset.
    // char_ends is strictly ascending (BreakIterator guarantees this).
    let mut byte_ends: Vec<usize> = Vec::with_capacity(n_ends);
    let mut end_idx = 0usize;
    let mut char_count = 0usize;

    for (byte_off, _) in line.char_indices() {
        while end_idx < n_ends && char_ends[end_idx] as usize == char_count {
            byte_ends.push(byte_off);
            end_idx += 1;
        }
        if end_idx >= n_ends {
            break;
        }
        char_count += 1;
    }
    // Any remaining char_ends at or past EOF → use the total byte length.
    while end_idx < n_ends {
        byte_ends.push(line_bytes.len());
        end_idx += 1;
    }

    // Step 2: Process each span — trim whitespace, hash bytes, vocab lookup.
    let mut token_ids: Vec<u32> = Vec::with_capacity(n_ends / 2 + 1);
    let mut span_starts: Vec<u32> = Vec::with_capacity(n_ends / 2 + 1);
    let mut prev_byte = 0usize;

    VOCAB.with(|v| {
        let vocab = v.borrow();
        let unk = UNK_IDX.with(|u| u.get());

        for byte_end in &byte_ends {
            let byte_end = *byte_end;
            let span = &line_bytes[prev_byte..byte_end];
            let span_start = prev_byte;
            prev_byte = byte_end;

            // Trim ASCII whitespace — mirrors Python's str.strip().
            let lead = span.iter().take_while(|&&b| b.is_ascii_whitespace()).count();
            let tail  = span.iter().rev().take_while(|&&b| b.is_ascii_whitespace()).count();
            if lead + tail >= span.len() {
                continue; // whitespace-only span: skip
            }
            let token_bytes = &span[lead..span.len() - tail];
            let token_byte_start = (span_start + lead) as u32;

            // Validate UTF-8 (ICU input is always valid, but be defensive).
            let token_str = match std::str::from_utf8(token_bytes) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Hash the token bytes for lookup. Only allocate for uppercase tokens.
            let hash = if token_str.bytes().any(|b| b.is_ascii_uppercase()) {
                fxhash64(token_str.to_lowercase().as_bytes())
            } else {
                fxhash64(token_bytes)
            };

            span_starts.push(token_byte_start);
            token_ids.push(vocab.get(&hash).copied().unwrap_or(unk));
        }
    });

    Ok((
        PyArray1::from_vec(py, token_ids),
        PyArray1::from_vec(py, span_starts),
    ))
}

/// Combined encode + get_span_start_positions for a single line.
///
/// Returns (token_ids, span_start_bytes) as numpy u32 arrays.
/// Equivalent to calling tokenizer.encode([t.lower() for t in raw_tokens]) and
/// tokenizer.get_span_start_positions(line, raw_tokens) from Python, but in one Rust pass.
///
/// `raw_tokens` is borrowed directly from Python's list — no String copies are made.
/// This eliminates the N malloc+memcpy on entry and N free on exit that were the dominant
/// hotspot (drop_in_place<Vec<String>>) in the py-spy profile.
///
/// Requires init_vocab_rs to have been called in this process first.
#[pyfunction]
pub fn encode_and_spans_rs<'py>(
    py: Python<'py>,
    line: &str,                       // borrow Python's str buffer, no copy
    raw_tokens: &Bound<'py, PyList>, // borrow Python's list, no String copies
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let n = raw_tokens.len();
    let mut token_ids = vec![0u32; n];
    let mut span_starts = vec![0u32; n];

    let line_bytes = line.as_bytes();
    let mut byte_pos = 0usize;

    // Pre-fetch item handles (just pointer-sized Bound refs, no String data copied).
    let items: Vec<Bound<'py, PyAny>> = (0..n)
        .map(|i| raw_tokens.get_item(i))
        .collect::<PyResult<_>>()?;

    VOCAB.with(|v| -> PyResult<()> {
        let vocab = v.borrow();
        let unk = UNK_IDX.with(|u| u.get());

        for (i, item) in items.iter().enumerate() {
            // Borrow the UTF-8 bytes directly from Python's PyUnicode internal buffer.
            // No allocation: raw_token is a &str pointing into Python-managed memory.
            let raw_token: &str = item.extract::<&str>()?;
            let token_bytes = raw_token.as_bytes();

            // Find byte offset of this token in the line from the current scan position.
            if let Some(offset) = find_subslice(&line_bytes[byte_pos..], token_bytes) {
                span_starts[i] = (byte_pos + offset) as u32;
                byte_pos += offset + token_bytes.len().max(1);
            } else {
                span_starts[i] = byte_pos as u32;
            }

            // Hash token bytes (lowercasing only if uppercase present) and look up.
            // Only allocates a String for the rare uppercase-token case.
            let hash = if token_bytes.iter().any(|b| b.is_ascii_uppercase()) {
                fxhash64(raw_token.to_lowercase().as_bytes())
            } else {
                fxhash64(token_bytes)
            };
            token_ids[i] = vocab.get(&hash).copied().unwrap_or(unk);
        }
        Ok(())
    })?;

    Ok((
        PyArray1::from_vec(py, token_ids),
        PyArray1::from_vec(py, span_starts),
    ))
}
