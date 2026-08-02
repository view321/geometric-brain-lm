"""Memory-mapped token loader for the packed bins written by `prep_data.py`.

Imports nothing but torch, numpy, tokenizers and the stdlib, so the GPU box
never needs `datasets` or `transformers`. The bins are read via np.memmap, so
a multi-GB corpus costs no RAM. Point it at the directory holding train.bin,
val.bin, meta.json and tok8k.json.

    python -c "from data import TokenStream; s=TokenStream('./data'); print(s.meta)"
    python -c "from data import TokenStream; s=TokenStream('./data','cuda'); print(s.get_batch('train',8,512)[0].shape)"
    python -c "from data import load_tokenizer; print(load_tokenizer('./data').decode([1,2,3]))"
"""
import json
import os

import numpy as np
import torch
from tokenizers import Tokenizer

SPLITS = ("train", "val")
META_FILE = "meta.json"
DEFAULT_TOKENIZER_FILE = "tok8k.json"


class TokenStream:
    """Random and sequential views over the packed uint16 token bins.

    Attributes:
        vocab_size, eos_id, meta, val_utf8_bytes, train_utf8_bytes, device
    """

    #: Cap on the training tokens `get_batch` will sample from; 0 = the whole
    #: split. Set this to force repeated epochs over a small subset. At the full
    #: corpus a short run never revisits a token, so there is no overfitting
    #: pressure at all and an inductive bias has nothing to show -- which makes
    #: the unconstrained model look equal when it is merely unconstrained on
    #: data it will never see twice.
    train_limit: int = 0

    def __init__(self, data_dir: str, device: str = "cpu", seed: int = 1234) -> None:
        self.data_dir = os.path.abspath(data_dir)
        meta_path = os.path.join(self.data_dir, META_FILE)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{meta_path} not found -- run prep_data.py --data-dir {data_dir} first")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.vocab_size = int(self.meta["vocab_size"])
        self.eos_id = int(self.meta["eos_id"])
        self.val_utf8_bytes = int(self.meta["val_utf8_bytes"])
        self.train_utf8_bytes = int(self.meta["train_utf8_bytes"])

        self.device = torch.device(device)
        self.pin = self.device.type == "cuda"
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        self._data = {}
        for split in SPLITS:
            path = os.path.join(self.data_dir, split + ".bin")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found -- run prep_data.py --data-dir {data_dir} first")
            # read-only memmap: the OS pages it in, we never hold the array.
            self._data[split] = np.memmap(path, dtype=np.uint16, mode="r")

    # ------------------------------------------------------------------
    def _split(self, split: str) -> np.memmap:
        try:
            return self._data[split]
        except KeyError:
            raise KeyError(f"unknown split {split!r}, expected one of {SPLITS}") from None

    def n_tokens(self, split: str) -> int:
        """Number of tokens in `split`."""
        return int(self._split(split).size)

    def _check(self, split: str, seq_len: int) -> np.memmap:
        arr = self._split(split)
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if seq_len + 1 > arr.size:
            raise ValueError(
                f"seq_len+1 ({seq_len + 1}) exceeds the {arr.size} tokens in split "
                f"{split!r} ({os.path.join(self.data_dir, split + '.bin')}); "
                f"use a shorter seq_len or prepare more data")
        return arr

    def _to_device(self, x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(x)
        if self.pin:
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    # ------------------------------------------------------------------
    def get_batch(self, split: str, batch_size: int, seq_len: int):
        """Random-offset batch. Returns (x, y) int64 [B, T] on self.device,
        y = x shifted by 1."""
        arr = self._check(split, seq_len)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        # valid starts: i + seq_len + 1 <= n  =>  i in [0, n - seq_len - 1]
        usable = arr.size
        cap = self.train_limit if split == "train" else 0
        if cap:
            usable = min(usable, cap)
            if usable <= seq_len:
                raise ValueError(
                    f"train_limit={cap} is too small for seq_len={seq_len}")
        starts = self._rng.integers(0, usable - seq_len, size=batch_size)
        x = np.empty((batch_size, seq_len), dtype=np.int64)
        y = np.empty((batch_size, seq_len), dtype=np.int64)
        for row, i in enumerate(starts):
            i = int(i)
            window = np.asarray(arr[i:i + seq_len + 1], dtype=np.int64)
            x[row] = window[:-1]
            y[row] = window[1:]
        return self._to_device(x), self._to_device(y)

    def sequential_batches(self, split: str, batch_size: int, seq_len: int):
        """Deterministic non-overlapping generator over the whole split, for eval.
        Yields (x, y) int64 [B, T]. Drops a final partial batch."""
        arr = self._check(split, seq_len)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        n_windows = (arr.size - 1) // seq_len
        n_batches = n_windows // batch_size
        for b in range(n_batches):
            x = np.empty((batch_size, seq_len), dtype=np.int64)
            y = np.empty((batch_size, seq_len), dtype=np.int64)
            for row in range(batch_size):
                i = (b * batch_size + row) * seq_len
                window = np.asarray(arr[i:i + seq_len + 1], dtype=np.int64)
                x[row] = window[:-1]
                y[row] = window[1:]
            yield self._to_device(x), self._to_device(y)


def load_tokenizer(data_dir: str) -> "Tokenizer":
    """Load the byte-level BPE saved next to the bins."""
    data_dir = os.path.abspath(data_dir)
    name = DEFAULT_TOKENIZER_FILE
    meta_path = os.path.join(data_dir, META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            name = os.path.basename(json.load(f).get("tokenizer_file") or name)
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run prep_data.py --data-dir {data_dir} first")
    return Tokenizer.from_file(path)
