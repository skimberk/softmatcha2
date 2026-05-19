"""Pytest configuration for the tests/ directory.

softmatcha_rs is a Rust extension used only by build.py (index building).
tokenize.py does not call it, but importing softmatcha.index triggers
softmatcha/index/__init__.py which re-exports build_index — causing an
import-time failure if the Rust extension has not been compiled.

Try to import the real extension first; fall back to a MagicMock only if it
is not available (e.g. maturin develop has not been run yet).  Using the real
extension when available ensures that Rust-path tests in test_tokenize_encode_offsets
get the actual implementation rather than a mock.
"""
import sys
import unittest.mock

if "softmatcha_rs" not in sys.modules:
    try:
        import softmatcha_rs  # noqa: F401 — real extension, keep in sys.modules
    except ImportError:
        sys.modules["softmatcha_rs"] = unittest.mock.MagicMock()
