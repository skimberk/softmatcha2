use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use std::collections::HashMap;
use icu_segmenter::{WordSegmenter, WordSegmenterBorrowed};
use icu_segmenter::options::WordBreakInvariantOptions;

// One segmenter per thread (each worker process uses one thread for tokenisation).
// WordSegmenterBorrowed<'static> references compiled-in Unicode data and is cheap to keep.
thread_local! {
    static WORD_SEGMENTER: WordSegmenterBorrowed<'static> =
        WordSegmenter::new_auto(WordBreakInvariantOptions::default());
}

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

/// Tokenise a single line using icu4x word segmentation and encode with vocab.
///
/// Replaces the Python ICU path entirely: strips the line, runs the icu4x
/// WordSegmenter (same Unicode word-break rules as PyICU), looks up lowercase
/// token IDs in the Rust vocab table, and returns byte offsets directly from
/// the UTF-8 byte positions produced by the Rust segmenter.
///
/// No Python↔C++ per-boundary calls; the entire segmentation loop runs in Rust.
#[pyfunction]
pub fn tokenize_and_encode_rs<'py>(
    py: Python<'py>,
    line: &str,
    vocab: &RustVocab,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let trimmed = line.trim();
    let leading_bytes = line.len() - line.trim_start().len();

    let (token_ids, byte_offsets) = WORD_SEGMENTER.with(|seg| {
        let mut token_ids: Vec<u32> = Vec::new();
        let mut byte_offsets: Vec<u32> = Vec::new();

        if trimmed.is_empty() {
            return (token_ids, byte_offsets);
        }

        let mut iter = seg.segment_str(trimmed);
        let mut prev = 0usize;

        while let Some(boundary) = iter.next() {
            let span = &trimmed[prev..boundary];
            let token = span.trim();
            if !token.is_empty() {
                let leading_in_span = span.len() - span.trim_start().len();
                byte_offsets.push((leading_bytes + prev + leading_in_span) as u32);

                let id = if token.is_ascii() {
                    if token.bytes().any(|b| b.is_ascii_uppercase()) {
                        let lower = token.to_ascii_lowercase();
                        vocab.map.get(&lower)
                    } else {
                        vocab.map.get(token)
                    }
                } else {
                    if token.chars().any(|c| c.is_uppercase()) {
                        let lower = token.to_lowercase();
                        vocab.map.get(&lower)
                    } else {
                        vocab.map.get(token)
                    }
                };
                token_ids.push(*id.unwrap_or(&vocab.unk_idx));
            }
            prev = boundary;
        }

        (token_ids, byte_offsets)
    });

    Ok((
        PyArray1::from_vec(py, token_ids),
        PyArray1::from_vec(py, byte_offsets),
    ))
}

/// Batch version of tokenize_and_encode_rs.
///
/// Processes a list of lines in a single Rust call, returning three arrays:
///   - cat_token_ids:   all token IDs concatenated across lines
///   - cat_byte_offsets: all byte offsets concatenated across lines
///   - lengths:         number of tokens produced per input line
///
/// Using a batch call instead of one call per line avoids:
///   - the Python→Rust boundary crossing overhead per line
///   - per-line numpy array allocation (2 arrays per line → 3 total)
///   - the thread-local WORD_SEGMENTER lookup overhead per line
///
/// The WORD_SEGMENTER TLS variable is accessed exactly once for the entire
/// batch; all lines are processed inside a single `with()` closure.
#[pyfunction]
pub fn tokenize_batch_rs<'py>(
    py: Python<'py>,
    lines: Vec<String>,
    vocab: &RustVocab,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
)> {
    let n = lines.len();
    let (all_token_ids, all_byte_offsets, lengths) = WORD_SEGMENTER.with(|seg| {
        let mut token_ids: Vec<u32> = Vec::with_capacity(n * 8);
        let mut byte_offsets: Vec<u32> = Vec::with_capacity(n * 8);
        let mut lengths: Vec<u32> = Vec::with_capacity(n);

        for line in &lines {
            let trimmed = line.trim();
            let leading_bytes = line.len() - line.trim_start().len();
            let start_count = token_ids.len();

            if !trimmed.is_empty() {
                let mut iter = seg.segment_str(trimmed);
                let mut prev = 0usize;

                while let Some(boundary) = iter.next() {
                    let span = &trimmed[prev..boundary];
                    let token = span.trim();
                    if !token.is_empty() {
                        let leading_in_span = span.len() - span.trim_start().len();
                        byte_offsets.push((leading_bytes + prev + leading_in_span) as u32);

                        let id = if token.is_ascii() {
                            if token.bytes().any(|b| b.is_ascii_uppercase()) {
                                let lower = token.to_ascii_lowercase();
                                vocab.map.get(&lower)
                            } else {
                                vocab.map.get(token)
                            }
                        } else {
                            if token.chars().any(|c| c.is_uppercase()) {
                                let lower = token.to_lowercase();
                                vocab.map.get(&lower)
                            } else {
                                vocab.map.get(token)
                            }
                        };
                        token_ids.push(*id.unwrap_or(&vocab.unk_idx));
                    }
                    prev = boundary;
                }
            }

            lengths.push((token_ids.len() - start_count) as u32);
        }

        (token_ids, byte_offsets, lengths)
    });

    Ok((
        PyArray1::from_vec(py, all_token_ids),
        PyArray1::from_vec(py, all_byte_offsets),
        PyArray1::from_vec(py, lengths),
    ))
}
