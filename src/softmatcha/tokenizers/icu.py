from __future__ import annotations

import os.path
from dataclasses import dataclass

import numpy as np
import simdjson

from .base import Tokenizer


class TokenizerICU(Tokenizer):
	@dataclass
	class Config(Tokenizer.Config):
		"""Configuration for tokenizer.

		name_or_path (str): Model name or path.
		lang (str): Language code.
		"""

		lang: str = "en"

	@property
	def unk_idx(self) -> int:
		"""Return the unknown index."""
		return self.dictionary[self.UNK_TOKEN]

	@classmethod
	def build(cls, cfg: TokenizerICU.Config) -> TokenizerICU:
		"""Build an tokenizer class.

		Args:
			cfg (TokenizerICU.Config): Tokenizer configuration.

		Returns:
			TokenizerICU: This class.
		"""
		parser = simdjson.Parser()
		dictionary = parser.load(os.path.join(cfg.name_or_path, "vocab.json"), True)
		dictionary[cls.UNK_TOKEN] = max(dictionary.values()) + 1

		import icu_tokenizer

		return cls(cfg, icu_tokenizer.Tokenizer(lang=cfg.lang), dictionary)

	def tokenize(self, line: str) -> list[str]:
		"""Tokenize the input line.

		Args:
			line (str): An input line.

		Returns:
			list[str]: The tokenized line.
		"""
		line = line.strip()
		line = line.lower()
		return self._tokenizer.tokenize(line)

	def tokenize_raw(self, line: str) -> list[str]:
		line = line.strip()
		return self._tokenizer.tokenize(line)

	def get_span_bounds(self, line: str) -> tuple[np.ndarray, np.ndarray]:
		"""Return (span_starts, span_ends) as uint32 numpy arrays (char positions).

		Intended for the Rust fast path in tokenize_encode_offsets: the caller
		passes these zero-copy to encode_and_offsets_rs, which extracts tokens
		from the line itself and avoids per-token Python string marshaling.

		Falls back to deriving span bounds from tokenize_raw_with_char_offsets
		if protected_patterns are active (rare; not the default configuration).
		"""
		if self._tokenizer.protected_patterns:
			tokens, starts = self.tokenize_raw_with_char_offsets(line)
			starts_np = np.array(starts, dtype=np.uint32)
			ends_np = np.array(
				[s + len(t) for s, t in zip(starts, tokens)], dtype=np.uint32
			)
			return starts_np, ends_np

		stripped = line.strip()
		leading = len(line) - len(line.lstrip())

		bi = self._tokenizer.break_iterator
		bi.setText(stripped)
		word_starts: list[int] = []
		word_ends: list[int] = []
		p0 = 0
		for p1 in bi:
			span = stripped[p0:p1]
			if span.strip():  # non-whitespace span → include (matches apply_break_iterator)
				word_starts.append(leading + p0)
				word_ends.append(leading + p1)
			p0 = p1

		return (
			np.array(word_starts, dtype=np.uint32),
			np.array(word_ends, dtype=np.uint32),
		)

	def tokenize_raw_with_char_offsets(self, line: str) -> tuple[list[str], list[int]]:
		"""Capture token char-positions directly from the ICU break iterator.

		Matches the behavior of apply_break_iterator exactly: includes every
		span whose content is non-empty after stripping whitespace.  This means
		punctuation marks (hyphens, periods, etc.) ARE included, exactly as
		tokenize_raw() does.  Only pure-whitespace spans are dropped.

		Falls back to the base-class implementation if protected_patterns are
		active (email/URL protection alters the text before the break iterator
		runs, invalidating the raw char positions).
		"""
		if self._tokenizer.protected_patterns:
			return super().tokenize_raw_with_char_offsets(line)

		stripped = line.strip()
		leading = len(line) - len(line.lstrip())

		bi = self._tokenizer.break_iterator
		bi.setText(stripped)
		tokens: list[str] = []
		char_positions: list[int] = []
		p0 = 0
		for p1 in bi:
			span = stripped[p0:p1]
			token = span.strip()
			if token:
				# ICU word-break spans never mix content with surrounding whitespace,
				# so the token always starts at the beginning of the span.
				char_positions.append(leading + p0)
				tokens.append(token)
			p0 = p1
		return tokens, char_positions