"""Pytest configuration for the tests/ directory.

softmatcha_rs is a Rust extension used only by build.py (index building).
tokenize.py does not call it, but importing softmatcha.index triggers
softmatcha/index/__init__.py which re-exports build_index — causing an
import-time failure if the Rust extension has not been compiled.

Mocking the module here lets tokenize.py be imported and tested without
requiring `maturin develop` to have been run.
"""
import sys
import unittest.mock

if "softmatcha_rs" not in sys.modules:
    sys.modules["softmatcha_rs"] = unittest.mock.MagicMock()
