from __future__ import annotations
import os
import gc
import random
import logging
import numba as nb
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from softmatcha import stopwatch
from softmatcha.tokenizers import Tokenizer
from softmatcha.utils.io import buffer_lines
from softmatcha.utils.makefile import make_file
from softmatcha.utils.custom_tqdm import CustomTqdm
logger = logging.getLogger(__name__)
_worker_tokenizer = None

try:
	from softmatcha_rs import init_vocab_rs as _init_vocab_rs, encode_and_spans_rs as _encode_and_spans_rs
	_HAS_RUST_TOKENIZE = True
except ImportError:
	_HAS_RUST_TOKENIZE = False

try:
	from softmatcha_rs import encode_and_spans_positions_rs as _encode_and_spans_positions_rs
	_HAS_RUST_POSITIONS = True
except ImportError:
	_HAS_RUST_POSITIONS = False

# Set to True in init_worker when the positions fast path is safe to use.
# Requires: Rust available + ICU tokenizer with no URL/hyphen protection patterns.
_USE_POSITIONS_PATH = False

# posix_fadvise hints — standard on Linux, graceful no-op everywhere else.
# FADV_SEQUENTIAL: tell the kernel to prefetch ahead aggressively.
# FADV_DONTNEED:   tell the kernel we are done with a region; evict it from
#                  page cache immediately so RAM is free for the next read.
#                  Without this, reading a 600 GB file on a 128 GB machine
#                  causes the kernel to thrash its own cache (visible as high
#                  %wa and %sy in top).
try:
	import ctypes as _ctypes
	import ctypes.util as _ctu
	_libc = _ctypes.CDLL(_ctu.find_library("c"), use_errno=True)
	_FADV_SEQUENTIAL = 2
	_FADV_DONTNEED   = 4
	# Probe that the symbol actually exists (it does not on macOS).
	_posix_fadvise = _libc.posix_fadvise
	def _fadvise(fd: int, offset: int, length: int, advice: int) -> None:
		_posix_fadvise(fd,
		               _ctypes.c_long(offset),
		               _ctypes.c_long(length),
		               _ctypes.c_int(advice))
except Exception:
	def _fadvise(fd: int, offset: int, length: int, advice: int) -> None:
		pass  # macOS / Windows: no-op


# =====================================================================================================================
# Preparation
# =====================================================================================================================
def init_worker(tokenizer: Tokenizer, cfg):
	global _worker_tokenizer, _USE_POSITIONS_PATH
	_worker_tokenizer = tokenizer
	tokenizer.build(cfg)
	if _HAS_RUST_TOKENIZE:
		keys = list(tokenizer.dictionary.keys())
		values = [int(v) for v in tokenizer.dictionary.values()]
		_init_vocab_rs(keys, values, int(tokenizer.unk_idx))
		# Enable the positions fast path when the ICU tokenizer has no URL/hyphen
		# protection patterns — the common case. With protection patterns active,
		# tokenize() rewrites some tokens before BreakIterator sees them, so we
		# cannot bypass it and must fall back to the string-token path.
		_USE_POSITIONS_PATH = (
			_HAS_RUST_POSITIONS
			and hasattr(tokenizer, '_tokenizer')
			and hasattr(tokenizer._tokenizer, 'break_iterator')
			and not getattr(tokenizer._tokenizer, 'protected_patterns', True)
		)

def tokenize_count(line: str):
	global _worker_tokenizer
	symbols = _worker_tokenizer.tokenize(line)
	return len(symbols)

def tokenize_encode_offsets(line: str):
	global _worker_tokenizer
	if _USE_POSITIONS_PATH:
		# Fast path: collect ICU char-boundary positions (N Python ints) and pass
		# them directly to Rust — no intermediate Python string objects created.
		bi = _worker_tokenizer._tokenizer.break_iterator
		bi.setText(line)
		token_ids, offsets = _encode_and_spans_positions_rs(line, list(bi))
	elif _HAS_RUST_TOKENIZE:
		# Rust encode+spans with string tokens (URL/hyphen protection active).
		symbols = _worker_tokenizer.tokenize_raw(line)
		token_ids, offsets = _encode_and_spans_rs(line, symbols)
	else:
		# Full Python fallback.
		symbols = _worker_tokenizer.tokenize_raw(line)
		token_ids = _worker_tokenizer.encode([sym.lower() for sym in symbols])
		offsets = _worker_tokenizer.get_span_start_positions(line, symbols)
	return token_ids, offsets

def get_custom_tqdm(num):
	return CustomTqdm(
		total=num,
		bar_format="{bar:64} {n_fmt}/{total_fmt} ETA {remaining}",
		ascii="░█",
		dynamic_ncols=True
	)

def read_random_chunk_safe(file_path, start_pos, chunk_size):
	with open(file_path, "rb") as f:
		f.seek(start_pos)
		if start_pos > 0:
			while True:
				byte = f.read(1)
				if not byte:
					return "", 0
				if not (0x80 <= byte[0] <= 0xBF):
					break
		buffer = f.read(chunk_size)
		while True:
			byte = f.read(1)
			if not byte:
				break
			if 0x80 <= byte[0] <= 0xBF:
				buffer += byte
			else:
				f.seek(-1, 1)
				break
		actual_byte_count = len(buffer)
		text = buffer.decode("utf-8", errors="replace")
		return text, actual_byte_count


def _read_chunk_with_fd(fd: int, start_pos: int, chunk_size: int, file_size: int):
	"""Like read_random_chunk_safe but uses an already-open fd via os.pread (O4)."""
	if start_pos >= file_size:
		return "", 0
	# Skip UTF-8 continuation bytes at the start position.
	offset = 0
	if start_pos > 0:
		for offset in range(8):
			raw = os.pread(fd, 1, start_pos + offset)
			if not raw:
				return "", 0
			if not (0x80 <= raw[0] <= 0xBF):
				break
	read_start = start_pos + offset
	raw_buf = bytearray(os.pread(fd, chunk_size, read_start))
	# Consume trailing UTF-8 continuation bytes to avoid splitting a codepoint.
	trail_pos = read_start + len(raw_buf)
	while trail_pos < file_size:
		b = os.pread(fd, 1, trail_pos)
		if not b or not (0x80 <= b[0] <= 0xBF):
			break
		raw_buf += b
		trail_pos += 1
	actual_byte_count = len(raw_buf)
	text = bytes(raw_buf).decode("utf-8", errors="replace")
	return text, actual_byte_count

def return_number_of_tokens(lines, num_workers, tokenizer):
	with concurrent.futures.ProcessPoolExecutor(
		max_workers=num_workers,
		initializer=init_worker,
		initargs=(tokenizer, tokenizer.cfg)
	) as executor:
		return sum(list(executor.map(tokenize_count, lines, chunksize=(len(lines) + num_workers - 1) // num_workers)))



# =================================================================================================================
# Main Tokenize Function
# =================================================================================================================
def tokenize(
	index_path: str,
	input_file: str,
	tokenizer: Tokenizer,
	num_workers: int,
	buffer_size: int,
	max_vocab: int,
) -> None:
	
	MAX_VOCAB = max_vocab
	chunk = 1_000_000
	num_chunk = (os.path.getsize(input_file) + chunk - 1) // chunk
	num_retries = 0

	while True:
		# =============================================================================================================
		# 1. count lines & estimate #tokens
		# =============================================================================================================
		# 1-0. make temporary files
		# Pre-allocate based on file size to avoid repeated doubling resizes.
		# Assume ~512 bytes/line on average with a 1.5× safety margin; fall back
		# to doubling if the estimate is too small (same logic as before, just rare).
		_file_size_for_est = os.path.getsize(input_file)
		LINES_SIZE = max(4_096, int(_file_size_for_est / 512 * 1.5) + 4096)
		index_path_ = Path(index_path)
		index_path_.mkdir(parents=True, exist_ok=True)
		if True:
			bin_path1 = index_path_ / "tmp1.bin"
			make_file(bin_path1, LINES_SIZE * 8)
			lines_tkn = np.memmap(bin_path1, dtype=np.uint64, mode='w+', shape=(LINES_SIZE,))
			bin_path2 = index_path_ / "tmp2.bin"
			make_file(bin_path2, LINES_SIZE * 8)
			lines_byt = np.memmap(bin_path2, dtype=np.uint64, mode='w+', shape=(LINES_SIZE,))

		# 1-1. count number of lines (parallel block scan)
		# Each worker thread reads a non-overlapping region of the file using
		# os.pread() (no seek, thread-safe) and returns the byte positions of the
		# start of every line in that region. np.where releases the GIL so threads
		# run truly in parallel on both I/O and CPU.
		_BLOCK = 256 * 1024 * 1024  # 256 MB — fewer pread() syscalls than 64 MB

		bar1 = get_custom_tqdm(num_chunk)

		def _scan_region(fd: int, region_start: int, region_end: int) -> np.ndarray:
			"""Return absolute byte positions of line starts inside [region_start, region_end)."""
			# Hint: read this region sequentially → aggressive kernel prefetch.
			_fadvise(fd, region_start, region_end - region_start, _FADV_SEQUENTIAL)
			parts: list[np.ndarray] = []
			pos = region_start
			while pos < region_end:
				n = min(_BLOCK, region_end - pos)
				block = os.pread(fd, n, pos)
				if not block:
					break
				arr = np.frombuffer(block, dtype=np.uint8)
				nl_offsets = np.where(arr == ord('\n'))[0]
				if len(nl_offsets):
					parts.append((pos + nl_offsets + 1).astype(np.uint64))
				# Evict this block from the page cache — we will never read it again.
				# Prevents cache thrashing when file_size >> RAM.
				_fadvise(fd, pos, len(block), _FADV_DONTNEED)
				# Update the progress bar as each block is processed (thread-safe).
				bar1.update(len(block) // chunk)
				pos += len(block)
			return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)
		num_lines = 0

		# Determine last byte to handle files not ending with '\n'.
		_last_byte_is_newline = True
		if _file_size_for_est > 0:
			with open(input_file, "rb") as _f:
				_f.seek(-1, 2)
				_last_byte_is_newline = (_f.read(1) == b'\n')

		_chunk_size = max(_BLOCK, (_file_size_for_est + num_workers - 1) // num_workers)
		_regions = []
		for _i in range(num_workers):
			_s = _i * _chunk_size
			if _s >= _file_size_for_est:
				break
			_regions.append((_s, min(_s + _chunk_size, _file_size_for_est)))

		_fd = os.open(input_file, os.O_RDONLY)
		try:
			with ThreadPoolExecutor(max_workers=num_workers) as _pool:
				_parts = list(_pool.map(lambda r: _scan_region(_fd, r[0], r[1]), _regions))
		finally:
			os.close(_fd)

		# Merge: regions are processed in order, so concatenation is already sorted.
		_all_starts = np.concatenate(_parts) if any(len(p) for p in _parts) else np.empty(0, dtype=np.uint64)
		num_lines = len(_all_starts)

		# Grow lines_byt / lines_tkn if the pre-allocation was too small (rare).
		if num_lines + 2 > LINES_SIZE:
			lines_tkn.flush(); lines_byt.flush(); gc.collect()
			del lines_byt; del lines_tkn
			LINES_SIZE = num_lines + 4096
			with open(bin_path1, "r+b") as _fr: _fr.truncate(LINES_SIZE * 8)
			with open(bin_path2, "r+b") as _fr: _fr.truncate(LINES_SIZE * 8)
			lines_tkn = np.memmap(bin_path1, dtype=np.uint64, mode='r+', shape=(LINES_SIZE,))
			lines_byt = np.memmap(bin_path2, dtype=np.uint64, mode='r+', shape=(LINES_SIZE,))

		# Populate lines_byt: line 0 starts at byte 0, rest from scan.
		lines_byt[0] = 0
		if num_lines:
			lines_byt[1 : num_lines + 1] = _all_starts

		# If the file does not end with '\n', there is one more (partial) line.
		if _file_size_for_est > 0 and not _last_byte_is_newline:
			num_lines += 1
			# Store the EOF position at lines_byt[num_lines], matching the original
			# behaviour where the loop's final iteration sets this entry.
			if num_lines < LINES_SIZE:
				lines_byt[num_lines] = _file_size_for_est

		bar1.update(bar1.total - bar1.n)
		bar1.close()

		# 1-2. estimate number of tokens (small)
		logger.info(f"Estimating number of tokens... (<3 min)")
		total_bytes = os.path.getsize(input_file)
		sub_bytes = 0
		sub_token = 0
		if os.path.getsize(input_file) < 400_000_000:
			text, sub_bytes = read_random_chunk_safe(input_file, 0, os.path.getsize(input_file))
			lines = text.splitlines()
			sub_token = return_number_of_tokens(lines, num_workers, tokenizer)
			est_tokens = sub_token + 1024
		
		# 1-3. estimate number of tokens (large)
		# O4: open the file once and use os.pread for all 40 000 random reads,
		# eliminating 40 000 open()/close() syscall pairs.
		else:
			sub_chunk_est = 10_000
			lines = []
			_est_fd = os.open(input_file, os.O_RDONLY)
			try:
				for i in range(40_000):
					pos = random.randint(0, total_bytes - sub_chunk_est)
					v, _nb = _read_chunk_with_fd(_est_fd, pos, sub_chunk_est, total_bytes)
					sub_bytes += _nb
					for u in v.splitlines():
						lines.append(u)
			finally:
				os.close(_est_fd)
			sub_token = return_number_of_tokens(lines, num_workers, tokenizer)
			safe_ratio = 1.3 + 0.05 * ((2 ** num_retries) - 1)
			est_tokens = int(safe_ratio * total_bytes * sub_token / sub_bytes)


		# =============================================================================================================
		# 2. make a file
		# =============================================================================================================
		# 2-1. decide filesize
		TOKEN_SIZE = (est_tokens // 1024 + 2) * 1024
		
		# 2-2. make binary files
		if True:
			bin_path = index_path_ / "tokens.bin"
			make_file(bin_path, TOKEN_SIZE * 4)
			tokens = np.memmap(bin_path, dtype=np.uint32, mode='w+', shape=(TOKEN_SIZE,))
		if True:
			bin_path = index_path_ / "offset.bin"
			make_file(bin_path, TOKEN_SIZE + TOKEN_SIZE // 32)
			offsets = np.memmap(bin_path, dtype=np.uint8, mode='w+', shape=(TOKEN_SIZE + TOKEN_SIZE // 32,))
			byte_offset1  = offsets[0 : TOKEN_SIZE]
			byte_offset2  = offsets[TOKEN_SIZE : ].view(np.uint64)
		if True:
			bin_path = index_path_ / "metadata.bin"
			make_file(bin_path, 2048 * 8)
			initial = np.memmap(bin_path, dtype=np.uint64, mode='w+', shape=(2048,))
		
		# 2-3. make a temporary file
		if True:
			bin_path = index_path_ / "tmp3.bin"
			bytes_rec = np.memmap(bin_path, dtype=np.uint32, mode='w+', shape=(TOKEN_SIZE,))
		
		# 2-4. output number of lines & tokens
		logger.info(f"Tokenize pre-phase finished")
		logger.info(f"#Lines     : {num_lines:,}")
		logger.info(f"#Tokens    : {est_tokens:,} est.")

		# 2-5. estimate the maximum vocabulary
		max_vocab_in_list = 0
		for i in tokenizer.tokens:
			max_vocab_in_list = max(max_vocab_in_list, i)
		if len(tokenizer.tokens) <= MAX_VOCAB:
			logger.info(f"#Vocabulary: {max_vocab_in_list + 1:,}")
		else:
			logger.info(f"#Vocabulary: {max_vocab_in_list + 1:,} \x1b[31m(Capped to {MAX_VOCAB:,})\x1b[0m")


		# =============================================================================================================
		# 3. tokenize all
		# =============================================================================================================
		logger.info(f"Tokenize begins...")
		ctokens = 0
		clines = 0
		ct = 0
		failed = False
		# Use a smaller chunksize than buffer_size so results trickle back from
		# workers throughout the batch rather than all arriving at once at the end.
		# chunksize=100 means each worker returns results every ~100 lines, giving
		# frequent bar updates without meaningful IPC overhead.
		_map_chunksize = max(1, buffer_size // 25)
		bar2 = get_custom_tqdm(num_lines)
		with concurrent.futures.ProcessPoolExecutor(
			max_workers=num_workers,
			initializer=init_worker,
			initargs=(tokenizer, tokenizer.cfg)
		) as executor:
			with stopwatch.timers["tokenize"]:
				for buffer in buffer_lines(input_file, buffer_size*num_workers, num_chunk, chunk, show_bar=False):
					init_ctokens = ctokens
					token_lengths = []

					# Iterate over map results one at a time so the bar updates as
					# each chunk of lines finishes, not only after the entire buffer.
					for token_seq, offset_seq in executor.map(
						tokenize_encode_offsets, buffer, chunksize=_map_chunksize
					):
						length = token_seq.shape[0]
						if ctokens + length > TOKEN_SIZE:
							failed = True
							break
						tokens[ctokens : ctokens+length] = token_seq
						bytes_rec[ct : ct+length] = offset_seq
						ctokens += length
						ct += length
						token_lengths.append(length)
						bar2.update(1)

					if failed:
						break

					# copy the line-start token indices
					cum = init_ctokens
					for k, ln in enumerate(token_lengths):
						lines_tkn[clines + k] = cum
						cum += ln
					clines += len(token_lengths)
		bar2.close()
		if failed == True:
			logger.info(f"Failed to estimate the number of tokens. repeating...")
			num_retries += 1
			continue
		num_tokens = ctokens
		lines_tkn[clines] = num_tokens
		logger.info(f"#Tokens    : {ctokens:,}")
		break


	# =================================================================================================================
	# 4. get other informations
	# =================================================================================================================
	# 4-1. functions to fill remaining tokens
	@nb.njit(cache=True, parallel=True)
	def fill_max(token, fst, lst, MAX):
		chunk = (lst - fst + num_workers - 1) // num_workers
		for worker_id in nb.prange(num_workers):
			stt = min(lst, fst + chunk * (worker_id + 0))
			end = min(lst, fst + chunk * (worker_id + 1))
			for i in range(stt, end):
				token[i] = MAX
	
	# 4-2. function to cap tokens
	@nb.njit(cache=True, parallel=True)
	def cap_tokens(token, num_tokens, MAX):
		chunk = (num_tokens + num_workers - 1) // num_workers
		for worker_id in nb.prange(num_workers):
			stt = min(num_tokens, chunk * (worker_id + 0))
			end = min(num_tokens, chunk * (worker_id + 1))
			for i in range(stt, end):
				token[i] = min(token[i], MAX)
	
	# 4-2. function to fill byte offsets
	@nb.njit(cache=True, parallel=True)
	def fill_all(num_tokens, rec_i32, byte_offset1, byte_offset2, lines_tkn, lines_byt, black_cnt, black_list):
		chunk = (num_tokens + num_workers - 1) // num_workers
		chunk = ((chunk + 255) // 256) * 256
		for worker_id in nb.prange(num_workers):
			stt = min(num_tokens, chunk * (worker_id + 0))
			end = min(num_tokens, chunk * (worker_id + 1))
			ok = 0
			ng = num_lines
			while ng - ok > 1:
				mid = (ok + ng) // 2
				if lines_tkn[mid] <= stt:
					ok = mid
				else:
					ng = mid
			current_line = ok
			while current_line < num_lines and lines_tkn[current_line + 1] < stt:
				current_line += 1

			# Process by 256 bytes
			for i in range(stt, end, 256):
				byte_offset2[i >> 8] = lines_byt[current_line] + rec_i32[i]
				prv = 0
				for j in range(i, min(num_tokens, i + 256)):
					cur = lines_byt[current_line] + rec_i32[j]
					if j != i:
						byte_offset1[j] = min(255, cur - prv)
						if cur - prv >= 255 and black_cnt[worker_id] < len(black_list[worker_id]):
							black_list[worker_id][2 * black_cnt[worker_id] + 0] = j - 1
							black_list[worker_id][2 * black_cnt[worker_id] + 1] = cur - prv
							black_cnt[worker_id] += 1
						prv = cur
					else:
						prv = cur
					while current_line < num_lines and lines_tkn[current_line + 1] <= j + 1:
						current_line += 1
	
	# 4-3. registration
	with stopwatch.timers["register"]:
		black_cnt  = np.zeros((num_workers), dtype=np.uint64)
		black_list = np.zeros((num_workers, (num_tokens // (num_workers * 10_000)) + 1_000), dtype=np.uint64)
		logger.info(f"Tokenize final processing begins..")
		logger.info(f"<this may take 5-10% of tokenize time>")
		fill_max(tokens, num_tokens, TOKEN_SIZE, MAX_VOCAB - 1)
		cap_tokens(tokens, num_tokens, MAX_VOCAB - 1)
		logger.info(f"Tokenize final processing 1/2 finished")
		fill_all(num_tokens, bytes_rec, byte_offset1, byte_offset2, lines_tkn, lines_byt, black_cnt, black_list)
		logger.info(f"Tokenize final processing 2/2 finished")
		logger.info(f"=====================================================")

		# black list (word length >= 256)
		black_sum = 0
		for i in range(num_workers):
			black_sum += black_cnt[i]
		byte_offset1.flush()
		byte_offset2.flush()
		offsets.flush()
		offsets._mmap.close()
		del offsets
		del byte_offset1
		del byte_offset2
		bin_path = index_path_ / "offset.bin"
		with open(bin_path, "ab") as f:
			f.truncate((TOKEN_SIZE + TOKEN_SIZE // 32) + black_sum * 16)
		offsets = np.memmap(bin_path, dtype=np.uint8, mode='r+', shape=(TOKEN_SIZE + TOKEN_SIZE // 32 + black_sum * 16,))
		byte_offset3  = offsets[TOKEN_SIZE + TOKEN_SIZE // 32 : ].view(np.uint64)
		cnts = 0
		for i in range(num_workers):
			for j in range(black_cnt[i]):
				byte_offset3[0 * black_sum + cnts] = black_list[i][2 * j + 0]
				byte_offset3[1 * black_sum + cnts] = black_list[i][2 * j + 1]
				cnts += 1
		logger.info(f"Exceptions = {cnts:,}")
	

	# =================================================================================================================
	# 5. final
	# =================================================================================================================
	# 5-1. record final data
	initial[ 0] = num_tokens
	initial[ 1] = num_lines
	initial[ 4] = TOKEN_SIZE
	initial[ 5] = LINES_SIZE
	initial[ 6] = MAX_VOCAB
	initial[ 7] = os.path.getsize(index_path_ / "metadata.bin")
	for i in range(len(input_file)):
		initial[512+i] = ord(input_file[i])
	initial.flush()
	offsets.flush()
	tokens.flush()
	del initial
	del offsets
	del tokens

	# 5-2. delete temporary file
	if os.path.exists(index_path_ / "tmp1.bin"):
		os.remove(index_path_ / "tmp1.bin")
	if os.path.exists(index_path_ / "tmp2.bin"):
		os.remove(index_path_ / "tmp2.bin")
	if os.path.exists(index_path_ / "tmp3.bin"):
		os.remove(index_path_ / "tmp3.bin")

	# 5-3. return number of tokens
	return num_tokens