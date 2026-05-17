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
/// then filters whitespace-only spans, trims, lowercases, and does vocab lookup —
/// all in one pass without any intermediate Python string objects.
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
    // The byte END of the span that ends at char C is the byte START of char C.
    let mut byte_ends: Vec<usize> = Vec::with_capacity(n_ends);
    let mut end_idx = 0usize;
    let mut char_count = 0usize;

    for (byte_off, _) in line.char_indices() {
        // Flush all char_ends that equal the current char index.
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

    // Step 2: Process each span [prev_byte..byte_end), trimming whitespace and encoding.
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

            // Validate UTF-8 and get a &str view (ICU input is always valid UTF-8).
            let token_str = match std::str::from_utf8(token_bytes) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Lowercase for vocab lookup. Avoid allocation when already lowercase.
            let lower_owned: Option<String> = if token_str.bytes().any(|b| b.is_ascii_uppercase()) {
                Some(token_str.to_lowercase())
            } else {
                None
            };
            let lookup: &str = lower_owned.as_deref().unwrap_or(token_str);

            span_starts.push(token_byte_start);
            token_ids.push(vocab.get(lookup).copied().unwrap_or(unk));
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
