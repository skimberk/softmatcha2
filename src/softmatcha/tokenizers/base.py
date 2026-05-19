from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any
from softmatcha.typing import NDArrayU32, NDArrayU64
import numpy as np


class Tokenizer(abc.ABC):
	"""Tokenizer class."""

	_tokenizer: Any

	@dataclass
	class Config:
		"""Configuration for tokenizer.

		name_or_path (str): Model name or path.
		"""

		name_or_path: str = field(default="", metadata={"cmd": False})

	def __init__(self, cfg: Config, tokenizer: Any, dictionary: dict[str, int]) -> None:
		self.cfg = cfg
		type(self)._tokenizer = tokenizer
		self.dictionary = dictionary
		self.tokens: dict[int, str] = {idx: token for token, idx in dictionary.items()}

	def __len__(self) -> int:
		return max(self.tokens) + 1

	UNK_TOKEN = "<unk>"

	@property
	@abc.abstractmethod
	def unk_idx(self) -> int:
		"""Return the unknown index."""

	@classmethod
	@abc.abstractmethod
	def build(cls, cfg: Config) -> Tokenizer:
		"""Build an tokenizer class.

		Args:
			cfg (Tokenizer.Config): Tokenizer configuration.

		Returns:
			Tokenizer: This class.
		"""

	@abc.abstractmethod
	def tokenize(self, line: str) -> list[str]:
		"""Tokenize the input line.

		Args:
			line (str): An input line.

		Returns:
			list[str]: The tokenized line.
		"""

	def tokenize_raw_with_char_offsets(self, line: str) -> tuple[list[str], list[int]]:
		"""Tokenize and return (tokens, char_positions_in_original_line).

		Default implementation: calls tokenize_raw then linearly scans for each
		token's position.  Subclasses that track positions during tokenization
		(e.g. TokenizerICU) can override this to avoid the second scan.
		"""
		tokens = self.tokenize_raw(line)
		char_positions: list[int] = []
		char_pos = 0
		for tok in tokens:
			found = line.find(tok, char_pos)
			char_positions.append(found if found >= 0 else 0)
			if found > 0:
				char_pos = found + len(tok)
		return tokens, char_positions

	def get_span_start_positions(self, line: str, tokens: list[str]) -> NDArrayU32:
		"""Get the start positions of spans.

		Args:
			line (str): An input line.
			tokens (list[str]): The tokenized line.

		Returns:
			list[int]: The start positions of token spans.
		"""
		span_starts = np.empty(len(tokens), dtype=np.uint32)
		start_position = 0
		prev_pos = 0
		cumsum = 0
		for i, token in enumerate(tokens):
			start_position = line.find(token, start_position)
			cumsum += len(line[prev_pos:start_position].encode("utf-8"))
			span_starts[i] = max(0, cumsum)
			prev_pos = start_position
			if start_position > 0:
				start_position += len(token)
		return span_starts
	
	def encode(self, tokens: list[str]) -> NDArrayU32:
		n = len(tokens)
		out = np.empty(n, dtype=np.uint32)
		d = self.dictionary
		unk = self.unk_idx
		for i, tok in enumerate(tokens):
			out[i] = d.get(tok, unk)
		return out

	def decode(self, indices: list[int]) -> list[str]:
		"""Decode token IDs into tokens.

		Args:
			indices (list[int]): Input token IDs.

		Returns:
			list[str]: The token sequence.
		"""
		return [self.tokens.get(idx, self.UNK_TOKEN) for idx in indices]

	def __call__(self, line: str) -> NDArrayU32:
		"""Tokenize and encode a line.

		Args:
			line (str): An input line.

		Returns:
			list[int]: The token ID sequence.
		"""
		return self.encode(self.tokenize(line))
