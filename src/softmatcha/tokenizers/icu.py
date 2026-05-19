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
		get_status = bi.getRuleStatus
		word_starts: list[int] = []
		word_ends: list[int] = []
		p0 = 0
		for p1 in bi:
			if get_status():
				word_starts.append(leading + p0)
				word_ends.append(leading + p1)
			p0 = p1

		return (
			np.array(word_starts, dtype=np.uint32),
			np.array(word_ends, dtype=np.uint32),
		)

	def tokenize_raw_with_char_offsets(self, line: str) -> tuple[list[str], list[int]]:
		"""Capture token char-positions directly from the ICU break iterator.

		The ICU BreakIterator already tracks span boundaries (p0, p1) during
		tokenization.  The default tokenize_raw() discards those positions and
		returns only the token strings, forcing tokenize_encode_offsets to
		re-scan the line with str.find to recover them.  This override keeps p0
		for every word span, so no second scan is needed.

		getRuleStatus() returns 0 for non-word spans (spaces, punctuation) and
		non-zero (UBRK_WORD_LETTER=200, UBRK_WORD_NUMBER=400) for word spans.
		This lets us skip the per-span strip()+find() calls that apply_break_iterator
		uses, while still correctly filtering out non-word spans.

		Falls back to the base-class implementation if protected_patterns are
		active (email/URL protection alters the text before the break iterator
		runs, invalidating the raw char positions).
		"""
		if self._tokenizer.protected_patterns:
			return super().tokenize_raw_with_char_offsets(line)

		stripped = line.strip()
		# char offset of 'stripped' within the original 'line'
		leading = len(line) - len(line.lstrip())

		bi = self._tokenizer.break_iterator
		bi.setText(stripped)
		get_status = bi.getRuleStatus  # cache method lookup
		tokens: list[str] = []
		char_positions: list[int] = []
		p0 = 0
		for p1 in bi:
			if get_status():  # non-zero = word span; skip spaces/punctuation
				char_positions.append(leading + p0)
				tokens.append(stripped[p0:p1])
			p0 = p1
		return tokens, char_positions