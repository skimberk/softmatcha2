use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod index;
mod search;
mod helper;
mod tokenize;
use std::sync::Once;
use crate::index::main::build_sa_rs;
use crate::search::main::enumerate_candidates_rs;
use crate::search::main::get_match_range_rs;
use crate::tokenize::{RustVocab, build_rust_vocab, encode_and_offsets_rs};
static INIT_LOGGER: Once = Once::new();


#[pymodule]
fn softmatcha_rs(m: &Bound<PyModule>) -> PyResult<()> {
	INIT_LOGGER.call_once(|| {
		env_logger::Builder::from_default_env()
			.format(|buf, record| {
				use std::io::Write;
				writeln!(
					buf,
					"| {} | {:4} | {}",
					chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
					record.level(),
					record.args()
				)
			})
			.filter_level(log::LevelFilter::Info)
			.init();
	});
	m.add_function(wrap_pyfunction!(enumerate_candidates_rs, m)?)?;
	m.add_function(wrap_pyfunction!(build_sa_rs, m)?)?;
	m.add_function(wrap_pyfunction!(get_match_range_rs, m)?)?;
	m.add_function(wrap_pyfunction!(build_rust_vocab, m)?)?;
	m.add_function(wrap_pyfunction!(encode_and_offsets_rs, m)?)?;
	m.add_class::<RustVocab>()?;
	Ok(())
}
