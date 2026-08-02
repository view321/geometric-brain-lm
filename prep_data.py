"""Fetch TinyStories, train an 8k byte-level BPE, and pack it into uint16 bins.

Run this once on a machine with network access; `data.py` then only needs
torch + numpy + tokenizers and reads the bins. Copy the whole `brain/` folder
to the GPU box and you can train offline. Falls back to a raw HTTPS download
if `datasets` is missing, and to a small synthetic corpus if there is no
network at all (or if you pass --smoke).

    python prep_data.py                                  # everything, into ./data
    python prep_data.py --vocab-size 8192 --val-stories 20000
    python prep_data.py --max-train-stories 200000       # quick subset run
    python prep_data.py --smoke --data-dir ./data_smoke --vocab-size 512
"""
import argparse
import io
import json
import os
import random
import sys
import time
import urllib.request

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

EOS = "<|endoftext|>"
EOS_ID = 0
TOKENIZER_FILE = "tok8k.json"

HF_REPO = "roneneldan/TinyStories"
HF_BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
RAW_FILES = {"train": "TinyStories-train.txt", "val": "TinyStories-valid.txt"}

ENCODE_BATCH = 2000          # stories handed to the rust tokenizer at once
FLUSH_TOKENS = 8 << 20       # ~16 MB of uint16 held before hitting disk
SYNTHETIC_BYTES = 2_000_000  # target size of the offline fallback corpus
SYNTHETIC_SEED = 20240517


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------
def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024.0


class Corpus:
    """A named pair of story iterators. `stories(split)` is re-iterable."""

    source = "?"

    def stories(self, split):
        raise NotImplementedError


class DatasetsCorpus(Corpus):
    """Path (a): the `datasets` library is installed."""

    source = "hf-datasets:" + HF_REPO

    def __init__(self, ds):
        self.ds = ds

    def stories(self, split):
        key = "train" if split == "train" else "validation"
        for row in self.ds[key]:
            text = (row["text"] or "").strip()
            if text:
                yield text


class RawTextCorpus(Corpus):
    """Path (b): the raw .txt dumps, streamed off disk and split on EOS."""

    source = "https:" + HF_BASE

    def __init__(self, paths):
        self.paths = paths

    def stories(self, split):
        with io.open(self.paths[split], "r", encoding="utf-8", errors="replace") as f:
            tail = ""
            while True:
                chunk = f.read(1 << 22)
                if not chunk:
                    break
                parts = (tail + chunk).split(EOS)
                tail = parts.pop()
                for part in parts:
                    part = part.strip()
                    if part:
                        yield part
            tail = tail.strip()
            if tail:
                yield tail


class SyntheticCorpus(Corpus):
    """Path (c): fixed-seed templated stories, so the pipeline runs offline."""

    source = "synthetic"

    def __init__(self, train, val):
        self._train, self._val = train, val

    def stories(self, split):
        return iter(self._train if split == "train" else self._val)


NAMES = ["Tom", "Lily", "Ben", "Mia", "Sam", "Anna", "Max", "Zoe", "Leo", "Ruby",
         "Jack", "Nora", "Finn", "Ivy", "Otto", "Elsa", "Kai", "Rose", "Hugo", "Poppy"]
ANIMALS = ["cat", "dog", "bird", "frog", "bear", "fox", "duck", "mouse", "rabbit", "turtle"]
THINGS = ["ball", "kite", "box", "hat", "book", "cup", "star", "boat", "drum", "flower",
          "cookie", "lamp", "key", "shell", "bell"]
PLACES = ["park", "garden", "beach", "forest", "kitchen", "barn", "hill", "pond", "market", "attic"]
ADJS = ["big", "small", "red", "blue", "shiny", "old", "soft", "funny", "warm", "tiny",
        "happy", "quiet", "brave", "silly", "bright"]
FEELINGS = ["happy", "sad", "scared", "proud", "excited", "surprised", "sleepy", "glad"]
VERBS = ["found", "lost", "hid", "shared", "washed", "painted", "carried", "dropped", "fixed"]

TEMPLATES = [
    "Once upon a time, there was a {adj} {animal} named {name}. {name} lived near the {place}.\n"
    "One day {name} {verb} a {adj2} {thing}. It made {name} feel very {feel}.\n"
    "\"Look at my {thing}!\" said {name}. {name2} came to the {place} and smiled.\n"
    "They played with the {thing} until the sun went down. Then they went home and slept.",

    "{name} was a {adj} kid who loved the {place}. Every morning {name} took a {thing} along.\n"
    "One morning a {animal} came out and looked at the {thing}. {name} was {feel}.\n"
    "\"Do you want to play?\" asked {name}. The {animal} said yes and they {verb} the {thing} together.\n"
    "From that day on, {name} and the {animal} were the best of friends.",

    "There was a {adj} {thing} in the {place}. Nobody knew who it belonged to.\n"
    "{name} and {name2} looked at the {adj2} {thing} for a long time.\n"
    "\"Maybe the {animal} left it here,\" said {name2}. They were a little {feel}.\n"
    "So they {verb} the {thing} and gave it back. The {animal} was very {feel2} and said thank you.",

    "One cold day, {name} walked to the {place} with a {adj} {thing}.\n"
    "A {adj2} {animal} was sitting there all alone and looked {feel}.\n"
    "{name} sat down and shared the {thing}. The {animal} started to smile.\n"
    "\"Sharing is nice,\" said {name}. They stayed at the {place} and told each other stories.",
]


def make_synthetic(target_bytes=SYNTHETIC_BYTES, seed=SYNTHETIC_SEED):
    rng = random.Random(seed)
    stories, total = [], 0
    while total < target_bytes:
        name, name2 = rng.sample(NAMES, 2)
        feel, feel2 = rng.sample(FEELINGS, 2)
        adj, adj2 = rng.sample(ADJS, 2)
        text = rng.choice(TEMPLATES).format(
            name=name, name2=name2, animal=rng.choice(ANIMALS), thing=rng.choice(THINGS),
            place=rng.choice(PLACES), adj=adj, adj2=adj2, feel=feel, feel2=feel2,
            verb=rng.choice(VERBS),
        )
        stories.append(text)
        total += len(text.encode("utf-8"))
    return stories


def _download(url, dest):
    """Stream `url` to `dest` with a one-line progress indicator."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  cached  {os.path.basename(dest)}  ({_human(os.path.getsize(dest))})", flush=True)
        return
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "prep_data/1.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = f"{100.0 * done / total:5.1f}%" if total else "  ?  "
            rate = done / max(time.time() - t0, 1e-6) / (1 << 20)
            print(f"\r  {os.path.basename(dest)}  {pct}  {_human(done)}  {rate:5.1f} MB/s",
                  end="", flush=True)
    print(flush=True)
    os.replace(tmp, dest)


def acquire(data_dir, smoke):
    """Try datasets -> raw HTTPS -> synthetic. Returns a Corpus."""
    if smoke:
        print("[source] --smoke: generating synthetic corpus (no network)", flush=True)
        stories = make_synthetic()
        n_val = max(1, min(len(stories) // 10, 2000))
        return SyntheticCorpus(stories[n_val:], stories[:n_val])

    try:
        from datasets import load_dataset
    except ImportError:
        print("[source] `datasets` not importable -> trying raw HTTPS download", flush=True)
    else:
        try:
            print(f"[source] (a) datasets.load_dataset({HF_REPO!r}) ...", flush=True)
            ds = load_dataset(HF_REPO)
            print("[source] OK: using the `datasets` library", flush=True)
            return DatasetsCorpus(ds)
        except Exception as e:
            print(f"[source] datasets failed ({type(e).__name__}: {e}) -> trying raw HTTPS",
                  flush=True)

    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    try:
        print(f"[source] (b) HTTPS download from {HF_BASE}", flush=True)
        paths = {}
        for split, name in RAW_FILES.items():
            paths[split] = os.path.join(raw_dir, name)
            _download(HF_BASE + name, paths[split])
        print("[source] OK: using the raw TinyStories .txt files", flush=True)
        return RawTextCorpus(paths)
    except Exception as e:
        print(f"[source] HTTPS download failed ({type(e).__name__}: {e})", flush=True)

    print("[source] (c) both network paths failed -> synthetic corpus", flush=True)
    stories = make_synthetic()
    n_val = max(1, min(len(stories) // 10, 2000))
    return SyntheticCorpus(stories[n_val:], stories[:n_val])


# --------------------------------------------------------------------------
# tokenizer + packing
# --------------------------------------------------------------------------
def _chunks(it, n):
    buf = []
    for item in it:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _limited(it, limit):
    if not limit:
        return it

    def gen():
        for i, item in enumerate(it):
            if i >= limit:
                return
            yield item

    return gen()


def train_tokenizer(corpus, vocab_size, limit, out_path):
    """Byte-level BPE: ByteLevel pre-tokenizer + decoder round-trips any UTF-8."""
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],                              # -> id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes: lossless
        show_progress=True,
    )
    n = [0]

    def feed():
        for s in _limited(corpus.stories("train"), limit):
            n[0] += 1
            yield s

    where = f"up to {limit:,}" if limit else "all"
    print(f"\n[bpe] training vocab_size={vocab_size} on {where} train stories ...", flush=True)
    t0 = time.time()
    tok.train_from_iterator(feed(), trainer=trainer)
    got = tok.get_vocab_size()
    assert tok.token_to_id(EOS) == EOS_ID, f"{EOS} landed at id {tok.token_to_id(EOS)}, want {EOS_ID}"
    tok.save(out_path)
    print(f"[bpe] {got:,} tokens from {n[0]:,} stories in {time.time() - t0:.1f}s -> {out_path}",
          flush=True)
    return tok


def _encode_batch(tok, batch):
    fast = getattr(tok, "encode_batch_fast", None)
    encs = fast(batch) if fast is not None else tok.encode_batch(batch)
    return [e.ids for e in encs]


def pack(tok, stories, out_path, label):
    """Tokenize, append EOS after each story, stream uint16 to `out_path`."""
    n_tokens = n_bytes = n_stories = 0
    buf = []
    t0 = time.time()
    with open(out_path, "wb") as f:
        def flush():
            nonlocal n_tokens
            if buf:
                arr = np.asarray(buf, dtype=np.uint16)
                arr.tofile(f)
                n_tokens += arr.size
                buf.clear()

        for batch in _chunks(stories, ENCODE_BATCH):
            n_bytes += sum(len(s.encode("utf-8")) for s in batch)
            for ids in _encode_batch(tok, batch):
                buf.extend(ids)
                buf.append(EOS_ID)
            n_stories += len(batch)
            if len(buf) >= FLUSH_TOKENS:
                flush()
                print(f"\r  [{label}] {n_stories:,} stories  {n_tokens:,} tokens "
                      f"({time.time() - t0:.0f}s)", end="", flush=True)
        flush()
    print(f"\r  [{label}] {n_stories:,} stories  {n_tokens:,} tokens  "
          f"{_human(n_bytes)} raw  ({time.time() - t0:.0f}s) -> {out_path}", flush=True)
    return n_tokens, n_bytes, n_stories


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--val-stories", type=int, default=20000)
    ap.add_argument("--max-train-stories", type=int, default=0, help="0 = all")
    ap.add_argument("--bpe-train-stories", type=int, default=200000,
                    help="stories sampled to fit the BPE (0 = all)")
    ap.add_argument("--smoke", action="store_true",
                    help="skip the network and use the synthetic corpus")
    args = ap.parse_args()

    assert args.vocab_size <= 65536, f"vocab {args.vocab_size} does not fit in uint16"
    os.makedirs(args.data_dir, exist_ok=True)
    data_dir = os.path.abspath(args.data_dir)

    corpus = acquire(data_dir, args.smoke)

    tok_path = os.path.join(data_dir, TOKENIZER_FILE)
    bpe_limit = args.bpe_train_stories
    if args.max_train_stories:
        bpe_limit = min(bpe_limit or args.max_train_stories, args.max_train_stories)
    tok = train_tokenizer(corpus, args.vocab_size, bpe_limit, tok_path)
    vocab_size = tok.get_vocab_size()
    assert vocab_size <= 65536, f"vocab {vocab_size} does not fit in uint16"

    print("\n[pack] writing uint16 bins ...", flush=True)
    val_path = os.path.join(data_dir, "val.bin")
    train_path = os.path.join(data_dir, "train.bin")
    val_tokens, val_bytes, val_stories = pack(
        tok, _limited(corpus.stories("val"), args.val_stories), val_path, "val")
    train_tokens, train_bytes, train_stories = pack(
        tok, _limited(corpus.stories("train"), args.max_train_stories), train_path, "train")

    meta = {
        "vocab_size": vocab_size,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "val_utf8_bytes": val_bytes,
        "train_utf8_bytes": train_bytes,
        "eos_id": EOS_ID,
        "source": corpus.source,
        "tokenizer_file": TOKENIZER_FILE,
    }
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"\n{'':7s} {'stories':>12s} {'tokens':>14s} {'utf8 bytes':>14s} {'bytes/token':>12s}")
    print("-" * 62)
    for name, st, tk, by in (("train", train_stories, train_tokens, train_bytes),
                             ("val", val_stories, val_tokens, val_bytes)):
        ratio = by / tk if tk else 0.0
        print(f"{name:7s} {st:12,d} {tk:14,d} {by:14,d} {ratio:12.3f}")
    tot_t, tot_b = train_tokens + val_tokens, train_bytes + val_bytes
    print("-" * 62)
    print(f"{'total':7s} {train_stories + val_stories:12,d} {tot_t:14,d} {tot_b:14,d} "
          f"{(tot_b / tot_t if tot_t else 0.0):12.3f}")
    print(f"\nvocab_size={vocab_size}  eos_id={EOS_ID}  source={corpus.source}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
