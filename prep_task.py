"""Synthetic algorithmic tasks, written in the same format as prep_data.py.

Language modelling on TinyStories cannot demonstrate test-time compute scaling,
because next-token prediction there does not require any computation to scale.
Four propagation rounds saturate it, and every measurement so far agrees:
perplexity is minimised at the trained depth and rises on both sides. A model
cannot show that thinking longer helps on a task where thinking does not help.

These tasks do require it, and their difficulty is a dial:

  parity   1 0 1 1 = O      accumulate over the whole sequence; no shortcut
  copy     a c b | a c b    retrieve verbatim from context; pure memory
  sort     3 1 2 > 1 2 3    compare and reorder; needs intermediate results

The claim to test is that a model trained on short instances keeps improving on
longer ones as inference rounds increase. That is compute scaling with problem
difficulty, and unlike perplexity on a story corpus it cannot be faked by
better local statistics.

    python prep_task.py --task parity --min-len 4 --max-len 16
    python prep_task.py --task copy --data-dir ./data_copy
    python train.py --preset think --data-dir ./data_parity --steps 5000
    python evaluate.py --ckpt runs/.../best.pt --scaling
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

EOS = "<|endoftext|>"
DIGITS = [str(i) for i in range(10)]
LETTERS = [chr(ord("a") + i) for i in range(20)]
MARKS = ["=", "|", ">", "E", "O"]
VOCAB = [EOS] + DIGITS + LETTERS + MARKS


def build_tokenizer() -> Tokenizer:
    """Whitespace WordLevel over a fixed symbol set -- exact, no merges."""
    tok = Tokenizer(models.WordLevel(
        vocab={s: i for i, s in enumerate(VOCAB)}, unk_token=EOS))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def gen_parity(rng: random.Random, length: int) -> str:
    bits = [rng.choice("01") for _ in range(length)]
    answer = "O" if sum(int(b) for b in bits) % 2 else "E"
    return " ".join(bits) + " = " + answer


def gen_copy(rng: random.Random, length: int) -> str:
    seq = [rng.choice(LETTERS) for _ in range(length)]
    return " ".join(seq) + " | " + " ".join(seq)


def gen_sort(rng: random.Random, length: int) -> str:
    nums = [rng.randrange(10) for _ in range(length)]
    return " ".join(map(str, nums)) + " > " + " ".join(map(str, sorted(nums)))


TASKS = {"parity": gen_parity, "copy": gen_copy, "sort": gen_sort}


def build_split(task: str, n: int, lo: int, hi: int, seed: int):
    rng = random.Random(seed)
    gen = TASKS[task]
    return [gen(rng, rng.randint(lo, hi)) for _ in range(n)]


def pack(examples, tok: Tokenizer, path: str) -> tuple[int, int]:
    ids: list[int] = []
    raw = 0
    for text in examples:
        ids.extend(tok.encode(text).ids)
        ids.append(0)                      # EOS between examples
        raw += len(text.encode("utf-8"))
    arr = np.asarray(ids, dtype=np.uint16)
    arr.tofile(path)
    return arr.size, raw


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="parity", choices=sorted(TASKS))
    ap.add_argument("--data-dir", default=None, help="default ./data_<task>")
    ap.add_argument("--train", type=int, default=400000, help="training examples")
    ap.add_argument("--val", type=int, default=8000)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--val-max-len", type=int, default=0,
                    help="longer instances for the val split, to test whether "
                         "extra inference rounds buy length extrapolation; "
                         "0 = same range as training")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    data_dir = args.data_dir or f"./data_{args.task}"
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    tok = build_tokenizer()
    tok.save(os.path.join(data_dir, "tok8k.json"))

    val_hi = args.val_max_len or args.max_len
    train = build_split(args.task, args.train, args.min_len, args.max_len, args.seed)
    val = build_split(args.task, args.val, args.min_len, val_hi, args.seed + 1)

    n_tr, b_tr = pack(train, tok, os.path.join(data_dir, "train.bin"))
    n_va, b_va = pack(val, tok, os.path.join(data_dir, "val.bin"))

    meta = {
        "vocab_size": len(VOCAB), "train_tokens": n_tr, "val_tokens": n_va,
        "val_utf8_bytes": b_va, "train_utf8_bytes": b_tr, "eos_id": 0,
        "source": f"synthetic:{args.task}", "tokenizer_file": "tok8k.json",
        "task": args.task, "train_len": [args.min_len, args.max_len],
        "val_len": [args.min_len, val_hi],
    }
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"task={args.task}  vocab={len(VOCAB)}")
    print(f"  train {args.train:,} examples, lengths {args.min_len}-{args.max_len}"
          f"  -> {n_tr:,} tokens")
    print(f"  val   {args.val:,} examples, lengths {args.min_len}-{val_hi}"
          f"  -> {n_va:,} tokens")
    print(f"  example: {train[0]!r}")
    print(f"wrote {data_dir}")
    if val_hi > args.max_len:
        print(f"\nval reaches length {val_hi} against a training max of "
              f"{args.max_len}, so held-out loss includes length extrapolation. "
              f"That is where extra inference rounds should pay off if they ever do.")


if __name__ == "__main__":
    main()
