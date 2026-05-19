use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use std::collections::HashMap;

/// Rust-native vocabulary table, built once per worker and reused for every
/// line.  Storing it in Rust avoids Python dict-lookup overhead per token.
#[pyclass]
pub struct RustVocab {
    map: HashMap<String, u32>,
    unk_idx: u32,
}

#[pymethods]
impl RustVocab {
    #[new]
    pub fn new(items: Vec<(String, u32)>, unk_idx: u32) -> Self {
        let mut map = HashMap::with_capacity(items.len() * 2);
        for (k, v) in items {
            map.insert(k, v);
        }
        RustVocab { map, unk_idx }
    }
}

/// Build a RustVocab from a Python list of (word, id) pairs.
#[pyfunction]
pub fn build_rust_vocab(items: Vec<(String, u32)>, unk_idx: u32) -> RustVocab {
    RustVocab::new(items, unk_idx)
}

/// Encode tokens and compute byte offsets from ICU span char-position boundaries.
///
/// `line`        – the original text line (not stripped).
/// `span_starts` – uint32 numpy array of char positions where each token starts.
/// `span_ends`   – uint32 numpy array of char positions where each token ends (exclusive).
/// `vocab`       – pre-built RustVocab.
///
/// Returns `(token_ids, byte_offsets)` as uint32 numpy arrays.
///
/// ASCII fast path: char position == byte position, no scan needed.
/// Non-ASCII path: single left-to-right character scan to resolve all char→byte
/// positions in one pass, then per-token extraction and lookup.
#[pyfunction]
pub fn encode_and_offsets_rs<'py>(
    py: Python<'py>,
    line: &str,
    span_starts: PyReadonlyArray1<'py, u32>,
    span_ends: PyReadonlyArray1<'py, u32>,
    vocab: &RustVocab,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let starts = span_starts.as_slice()?;
    let ends   = span_ends.as_slice()?;
    let n = starts.len();

    let mut token_ids    = vec![vocab.unk_idx; n];
    let mut byte_offsets = vec![0u32; n];

    if line.is_ascii() {
        // ----------------------------------------------------------------
        // ASCII fast path: char pos == byte pos, no scan required.
        // ----------------------------------------------------------------
        for i in 0..n {
            let s = starts[i] as usize;
            let e = ends[i] as usize;
            let token = &line[s..e];
            byte_offsets[i] = starts[i];

            // Avoid heap allocation when token is already lowercase (common).
            let id = if token.bytes().any(|b| b.is_ascii_uppercase()) {
                let lower = token.to_ascii_lowercase();
                vocab.map.get(&lower)
            } else {
                vocab.map.get(token)
            };
            token_ids[i] = *id.unwrap_or(&vocab.unk_idx);
        }
    } else {
        // ----------------------------------------------------------------
        // Non-ASCII path: single linear scan to compute byte positions.
        // ----------------------------------------------------------------
        // starts and ends are both sorted ascending (spans are non-overlapping,
        // left-to-right).  We walk the line character by character and fill in
        // byte_starts / byte_ends using two pointers.
        let mut byte_starts = vec![0u32; n];
        let mut byte_ends   = vec![0u32; n];

        let mut start_ptr = 0usize;
        let mut end_ptr   = 0usize;
        let mut char_i    = 0u32;

        for (byte_pos, _) in line.char_indices() {
            let b = byte_pos as u32;
            while start_ptr < n && starts[start_ptr] == char_i {
                byte_starts[start_ptr] = b;
                start_ptr += 1;
            }
            while end_ptr < n && ends[end_ptr] == char_i {
                byte_ends[end_ptr] = b;
                end_ptr += 1;
            }
            char_i += 1;
        }
        // Positions that fall at the very end of the string.
        let end_b = line.len() as u32;
        while start_ptr < n { byte_starts[start_ptr] = end_b; start_ptr += 1; }
        while end_ptr   < n { byte_ends[end_ptr]   = end_b; end_ptr   += 1; }

        for i in 0..n {
            let s = byte_starts[i] as usize;
            let e = byte_ends[i] as usize;
            let token = &line[s..e];
            byte_offsets[i] = byte_starts[i];

            let id = if token.chars().any(|c| c.is_uppercase()) {
                let lower = token.to_lowercase();
                vocab.map.get(&lower)
            } else {
                vocab.map.get(token)
            };
            token_ids[i] = *id.unwrap_or(&vocab.unk_idx);
        }
    }

    Ok((
        PyArray1::from_vec(py, token_ids),
        PyArray1::from_vec(py, byte_offsets),
    ))
}
