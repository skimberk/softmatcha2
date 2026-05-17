use pyo3::prelude::*;
use numpy::PyArray1;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

thread_local! {
    static VOCAB: RefCell<HashMap<String, u32>> = RefCell::new(HashMap::new());
    static UNK_IDX: Cell<u32> = const { Cell::new(0) };
}

/// Initialize the thread-local vocabulary. Call once per worker process after the tokenizer is built.
#[pyfunction]
pub fn init_vocab_rs(keys: Vec<String>, values: Vec<u32>, unk_idx: u32) -> PyResult<()> {
    UNK_IDX.with(|u| u.set(unk_idx));
    VOCAB.with(|v| {
        let mut map = v.borrow_mut();
        map.clear();
        map.reserve(keys.len());
        for (k, id) in keys.into_iter().zip(values) {
            map.insert(k, id);
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

/// Combined encode + get_span_start_positions for a single line.
///
/// Returns (token_ids, span_start_bytes) as numpy u32 arrays.
/// Equivalent to calling tokenizer.encode([t.lower() for t in raw_tokens]) and
/// tokenizer.get_span_start_positions(line, raw_tokens) from Python, but in one Rust pass.
///
/// Requires init_vocab_rs to have been called in this process first.
#[pyfunction]
pub fn encode_and_spans_rs<'py>(
    py: Python<'py>,
    line: String,
    raw_tokens: Vec<String>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let n = raw_tokens.len();
    let mut token_ids = vec![0u32; n];
    let mut span_starts = vec![0u32; n];

    let line_bytes = line.as_bytes();
    let mut byte_pos = 0usize;

    VOCAB.with(|v| {
        let vocab = v.borrow();
        let unk = UNK_IDX.with(|u| u.get());

        for (i, raw_token) in raw_tokens.iter().enumerate() {
            let token_bytes = raw_token.as_bytes();

            // Find byte offset of this token in the line from the current scan position.
            // Searching bytes directly gives the UTF-8 byte offset without a char→byte round-trip.
            if let Some(offset) = find_subslice(&line_bytes[byte_pos..], token_bytes) {
                span_starts[i] = (byte_pos + offset) as u32;
                byte_pos += offset + token_bytes.len().max(1);
            } else {
                // Token not found — use current position as a safe fallback.
                span_starts[i] = byte_pos as u32;
            }

            // Lowercase for vocabulary lookup. Skip the allocation when already lowercase (common).
            let is_upper = raw_token.bytes().any(|b| b.is_ascii_uppercase());
            let lower: String = if is_upper {
                raw_token.to_lowercase()
            } else {
                raw_token.clone()
            };
            token_ids[i] = vocab.get(lower.as_str()).copied().unwrap_or(unk);
        }
    });

    Ok((
        PyArray1::from_vec(py, token_ids),
        PyArray1::from_vec(py, span_starts),
    ))
}
