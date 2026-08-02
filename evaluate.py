"""Held-out scoring, sampling, and geometry diagnostics.

Two loss numbers are reported and they answer different questions:

  * token perplexity is the primary metric *within* this study, because every
    model here shares the same 8k tokenizer, so it is directly comparable;
  * bits-per-byte normalises the tokenizer away, and is the only number that
    can be put next to a model with a different vocabulary (GPT-2, Pythia, ...).

    python evaluate.py --ckpt runs/brain-d16/best.pt --sample 4
    python evaluate.py --ckpt runs/brain-d16/best.pt --full --diagnostics
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


def _detach(state):
    if state is None:
        return None
    if isinstance(state, (tuple, list)):
        return type(state)(_detach(s) for s in state)
    return state.detach()


@torch.no_grad()
def evaluate(model, stream, *, batch_size: int, seq_len: int, tbptt: int,
             max_batches: int = 0, tokenizer=None, split: str = "val",
             autocast_dtype=None, range_start: int = 0,
             range_stop: int | None = None) -> dict:
    """Teacher-forced held-out loss.

    Batches are consumed sequentially and the recurrent state is carried within
    a batch but reset between batches, matching how training sees the data.
    """
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    k_sum, k_count, fired_total = 0.0, 0, None

    ctx = (torch.autocast(device.type, dtype=autocast_dtype)
           if autocast_dtype is not None else torch.enable_grad())
    use_ctx = autocast_dtype is not None

    batches = stream.sequential_batches(split, batch_size, seq_len,
                                        start=range_start, stop=range_stop)
    for i, (x, y) in enumerate(batches):
        if max_batches and i >= max_batches:
            break
        state = model.init_state(x.shape[0], device)
        for start in range(0, x.shape[1], tbptt):
            xc = x[:, start:start + tbptt]
            yc = y[:, start:start + tbptt]
            if use_ctx:
                with ctx:
                    logits, state, stats = model(xc, state, collect_stats=True)
            else:
                logits, state, stats = model(xc, state, collect_stats=True)
            state = _detach(state)
            nll = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                yc.reshape(-1), reduction="sum",
            )
            total_nll += nll.item()
            total_tokens += yc.numel()

            if stats.get("k_count"):
                k_sum += float(stats["k_sum"]) / stats["k_count"]
                k_count += 1
            if "fired" in stats:
                f = stats["fired"]
                fired_total = f.clone() if fired_total is None else fired_total + f

        if tokenizer is not None:
            rows = y.detach().cpu().tolist()
            for text in tokenizer.decode_batch(rows, skip_special_tokens=True):
                total_bytes += len(text.encode("utf-8"))

    if was_training:
        model.train()

    if total_tokens == 0:
        raise RuntimeError("evaluation consumed zero tokens; is the split empty?")

    mean_nll = total_nll / total_tokens
    out = {
        "loss": mean_nll,
        "ppl": math.exp(min(mean_nll, 60.0)),
        "tokens": total_tokens,
    }
    if total_bytes:
        out["bpb"] = total_nll / LN2 / total_bytes
        out["bytes"] = total_bytes
        out["bytes_per_token"] = total_bytes / total_tokens
    if k_count:
        out["mean_k"] = k_sum / k_count
    if fired_total is not None:
        n = fired_total.numel()
        share = fired_total / fired_total.sum().clamp_min(1.0)
        nz = share[share > 0]
        out["dead_frac"] = float((fired_total == 0).float().mean())
        # Perplexity of the usage distribution, as a fraction of the neuron
        # count: 1.0 means every neuron is used equally, near 0 means a handful
        # carry everything.
        entropy = float(-(nz * nz.log()).sum())
        out["usage_uniformity"] = math.exp(entropy) / n
    return out


@torch.no_grad()
def generate(model, tokenizer, prompt: str = "", *, max_new: int = 128,
             temperature: float = 0.8, top_k: int = 50, device=None,
             seed: int | None = None) -> str:
    """Sample a continuation.

    TinyStories exists precisely so that a model this small produces readable
    text, which means samples are a real diagnostic here and not decoration --
    a model that is quietly broken reads broken long before perplexity is
    conclusive.
    """
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    ids = tokenizer.encode(prompt).ids if prompt else []
    if not ids:
        ids = [getattr(model, "eos_id", 0)]

    x = torch.tensor([ids], dtype=torch.long, device=device)
    stateful = model.init_state(1, device) is not None

    state = model.init_state(1, device) if stateful else None
    if stateful:
        logits, state, _ = model(x, state, collect_stats=False)
        logits = logits[:, -1]
    else:
        logits, _, _ = model(x, None, collect_stats=False)
        logits = logits[:, -1]

    out = list(ids)
    for _ in range(max_new):
        logits = logits.float() / max(temperature, 1e-4)
        if top_k:
            k = min(top_k, logits.shape[-1])
            cut = logits.topk(k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < cut, float("-inf"))
        probs = F.softmax(logits, dim=-1).cpu()
        nxt = int(torch.multinomial(probs, num_samples=1, generator=gen))
        out.append(nxt)
        step = torch.tensor([[nxt]], dtype=torch.long, device=device)
        if stateful:
            logits, state, _ = model(step, state, collect_stats=False)
            logits = logits[:, -1]
        else:
            window = torch.tensor([out[-model.cfg.max_seq_len:]],
                                  dtype=torch.long, device=device)
            logits, _, _ = model(window, None, collect_stats=False)
            logits = logits[:, -1]

    if was_training:
        model.train()
    return tokenizer.decode(out, skip_special_tokens=True)


@torch.no_grad()
def compute_scaling(model, stream, *, batch_size: int, seq_len: int, tbptt: int,
                    rounds=(1, 2, 3, 4, 6, 8, 12, 16, 24), batches: int = 40,
                    autocast_dtype=None) -> dict:
    """Does spending more latent iterations at inference improve the answer?

    This is the falsifiable core of "reasoning by changing state". Propagation
    rounds carry no parameters, so the same checkpoint can be run at any depth.
    If perplexity keeps falling as rounds increase -- especially past the depth
    it was trained at -- then the state update is a genuine refinement operator
    and compute is decoupled from tokens emitted. If it degrades immediately,
    the model learned one fixed transformation and there is no thinking to do.

    Also reports how far the state moves per round. A settling trajectory (the
    step size shrinking toward zero) is attractor dynamics; a flat one means the
    state is churning rather than converging on anything.
    """
    if not hasattr(model, "cfg") or not hasattr(model, "axon"):
        return {}
    original = model.cfg.rounds
    out: dict = {}
    try:
        for r in rounds:
            model.cfg.rounds = r
            m = evaluate(model, stream, batch_size=batch_size, seq_len=seq_len,
                         tbptt=tbptt, max_batches=batches,
                         autocast_dtype=autocast_dtype)
            out[f"ppl_rounds{r}"] = m["ppl"]
    finally:
        model.cfg.rounds = original

    scores = {r: out[f"ppl_rounds{r}"] for r in rounds}
    best = min(scores, key=scores.get)
    out["best_rounds"] = best
    out["best_ppl"] = scores[best]
    out["trained_rounds"] = original
    # Did extra compute beyond the training depth still help? That is the claim.
    deeper = [r for r in rounds if r > original]
    if deeper:
        out["gain_beyond_trained"] = (
            scores[original] - min(scores[r] for r in deeper)
            if original in scores else float("nan")
        )

    # Settling trajectory: relative movement of the state per round.
    device = next(model.parameters()).device
    x, _ = stream.get_batch("val", min(batch_size, 8), seq_len)
    w, signs, dec = model.edge_weights(), model.signs(), model.decays()
    state = model.init_state(x.shape[0], device)
    for t in range(min(16, seq_len)):          # warm the state on real context
        state = model.step(x[:, t], state, w, signs, None, dec, model.cfg.rounds)
    deltas = []
    prev = state
    for _ in range(max(rounds)):
        nxt, _, _ = model._propagate(prev, w, signs, dec)
        deltas.append(float((nxt - prev).norm() / prev.norm().clamp_min(1e-9)))
        prev = nxt
    out["settle_deltas"] = [round(d, 4) for d in deltas]
    out["settles"] = deltas[-1] < 0.5 * deltas[0]
    return out


@torch.no_grad()
def generalization_gap(model, stream, *, train_tokens: int, batch_size: int,
                       seq_len: int, tbptt: int, batches: int = 100,
                       tokenizer=None, autocast_dtype=None) -> dict:
    """Score the exact slice the model trained on against the slice it never saw.

    Both slices live inside train.bin and come from one corpus built in one
    pass, so nothing distinguishes them but exposure -- no train/valid file
    difference, no distribution shift, no tokenizer discrepancy. The difference
    between them is memorisation, and the held-out number is generalisation.

    This is the measurement that separates two parameterizations of equal
    expressivity. A model with more free parameters can always fit the seen
    slice harder; whether that costs it on the unseen slice is the only thing
    that says which prior is better.
    """
    if train_tokens <= 0:
        raise ValueError(
            "generalization_gap needs a split point. This checkpoint was "
            "trained on the whole corpus, so nothing was withheld from it. "
            "Pass --gap-tokens N to split anyway -- useful as a control, since "
            "a single-pass model should show a gap near zero.")
    total = stream.n_tokens("train")
    if train_tokens >= total:
        raise ValueError(
            f"training cap {train_tokens:,} covers the whole {total:,}-token "
            f"split; nothing was held out")

    common = dict(batch_size=batch_size, seq_len=seq_len, tbptt=tbptt,
                  max_batches=batches, tokenizer=tokenizer, split="train",
                  autocast_dtype=autocast_dtype)
    seen = evaluate(model, stream, range_start=0, range_stop=train_tokens, **common)
    # Skip a seq_len of slack so no evaluation window straddles the boundary.
    unseen = evaluate(model, stream, range_start=train_tokens + seq_len,
                      range_stop=None, **common)

    out = {
        "seen_loss": seen["loss"], "seen_ppl": seen["ppl"],
        "unseen_loss": unseen["loss"], "unseen_ppl": unseen["ppl"],
        "gap_nats": unseen["loss"] - seen["loss"],
        "gap_ppl_ratio": unseen["ppl"] / max(seen["ppl"], 1e-9),
        "train_tokens": train_tokens,
        "eval_tokens_each": seen["tokens"],
    }
    if "bpb" in seen and "bpb" in unseen:
        out["seen_bpb"] = seen["bpb"]
        out["unseen_bpb"] = unseen["bpb"]
    return out


@torch.no_grad()
def induction_probe(model, *, gaps=(2, 4, 8, 16, 32, 64), trials: int = 64,
                    prefix: int = 8, seed: int = 0) -> dict:
    """Can the model complete `A B ... A -> B` from its own context?

    The canonical in-context learning test, and the sharpest one available at
    this scale. Two sequences are scored: one where the pair `A B` appeared
    earlier, and a control where it did not. The gap in log-probability of `B`
    at the final position is how much the model learned from the context rather
    than from training statistics.

    A score near zero means no in-context learning at all -- the model predicts
    only from what it memorised, never from what it just read. That is
    compatible with excellent perplexity, which is exactly why it needs its own
    measurement. Reported in nats; anything under ~0.1 is noise.

    Works for the baselines too, so the geometric model can be compared against
    a GRU and a transformer on the one axis transformers are known to win.
    """
    device = next(model.parameters()).device
    vocab = model.cfg.vocab_size
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape):
        return torch.randint(0, vocab, shape, generator=gen)

    out = {}
    for gap in gaps:
        # A B <gap fillers> A   -- and a control with an unrelated pair in place
        # of A B, so the only difference is whether the context contains the
        # answer.
        a, b = rnd(trials), rnd(trials)
        x, y = rnd(trials), rnd(trials)
        head, filler = rnd(trials, prefix), rnd(trials, gap)

        treat = torch.cat([head, a[:, None], b[:, None], filler, a[:, None]], dim=1)
        ctrl = torch.cat([head, x[:, None], y[:, None], filler, a[:, None]], dim=1)

        scores = []
        for seq in (treat, ctrl):
            logits, _, _ = model(seq.to(device), None, collect_stats=False)
            logp = torch.log_softmax(logits[:, -1].float(), dim=-1)
            scores.append(logp.gather(1, b[:, None].to(device)).squeeze(1))
        out[f"induction_gap{gap}"] = float((scores[0] - scores[1]).mean())

    vals = [out[f"induction_gap{g}"] for g in gaps]
    out["induction_best"] = max(vals)
    out["induction_best_gap"] = gaps[int(torch.tensor(vals).argmax())]
    return out


@torch.no_grad()
def memory_horizon(model, *, tokens: int = 128, trials: int = 16,
                   seed: int = 0, stream=None) -> dict:
    """How long does one token's footprint survive in the state?

    Inject a probe token, record which neurons it leaves active, then feed
    unrelated tokens and track what fraction of that set is still active. The
    half-life is the effective context window of the recurrence, which is the
    number that actually bounds how far back the model can refer.

    Also reports how much of the *current* token survives into a populated
    state. If that is near zero the model is not integrating its input at all,
    it is coasting on whatever entered first.

    Run this on a trained checkpoint. Measured on random initial positions it
    describes the initialization, not the model: a trained point cloud has
    attractor structure that a random one does not, and that is exactly what
    would change these numbers.
    """
    if not hasattr(model, "axon"):
        return {}
    device = next(model.parameters()).device
    cfg = model.cfg
    gen = torch.Generator(device="cpu").manual_seed(seed)
    vocab = cfg.vocab_size

    def rand_tok(n=1):
        return torch.randint(0, vocab, (n,), generator=gen).to(device)

    w, signs, dec = model.edge_weights(), model.signs(), model.decays()
    band = cfg.n_neurons // cfg.n_bands
    curve = torch.zeros(tokens)
    per_band = torch.zeros(cfg.n_bands, tokens)
    entry: list[float] = []
    band_entry = torch.zeros(cfg.n_bands)
    band_entry_n = torch.zeros(cfg.n_bands)

    for tr in range(trials):
        state = model.init_state(1, device)
        # Warm the state so the probe lands in a populated system, not an empty one.
        for _ in range(16):
            state = model.step(rand_tok(), state, w, signs, None, dec)

        probe = rand_tok()
        state = model.step(probe, state, w, signs, None, dec)
        tracked = model._kwta(state)[2][0]
        tset = set(tracked.tolist())
        bands = [{i for i in tset if i // band == b} for b in range(cfg.n_bands)]

        for t in range(tokens):
            tok = rand_tok()
            # What fraction of THIS token's seeds survives the k-WTA?
            p = model.tok_pos(tok)
            sq = (p.pow(2).sum(-1, keepdim=True) - 2.0 * (p @ model.dend.t())
                  + model.dend.pow(2).sum(-1).unsqueeze(0)).clamp_min(0.0)
            want = set((-sq).topk(cfg.seed_n, dim=-1).indices[0].tolist())

            state = model.step(tok, state, w, signs, None, dec)
            live = set(model._kwta(state)[2][0].tolist())
            entry.append(len(want & live) / max(len(want), 1))
            for b in range(cfg.n_bands):
                wb = {i for i in want if i // band == b}
                if wb:
                    band_entry[b] += len(wb & live) / len(wb)
                    band_entry_n[b] += 1
            curve[t] += len(tset & live) / max(len(tset), 1) / trials
            for b in range(cfg.n_bands):
                if bands[b]:
                    per_band[b, t] += len(bands[b] & live) / len(bands[b]) / trials

    def half_life(c):
        below = (c < 0.5).nonzero()
        return int(below[0]) + 1 if below.numel() else f">{len(c)}"

    out = {
        "half_life": half_life(curve),
        "retention_t1": float(curve[0]),
        "retention_t8": float(curve[min(7, tokens - 1)]),
        "retention_t32": float(curve[min(31, tokens - 1)]),
        "current_token_entry": sum(entry) / len(entry),
        "trials": trials,
    }
    if cfg.n_bands > 1:
        tau = -1.0 / torch.log(dec.clamp(1e-6, 1 - 1e-6))
        out["band_half_life"] = [half_life(per_band[b]) for b in range(cfg.n_bands)]
        out["band_tau_median"] = [
            float(tau[b * band:(b + 1) * band].median()) for b in range(cfg.n_bands)
        ]
        # Read this instead of the pooled figure on a banded model. Slow bands
        # decline the current token by design, so pooling them with the fast
        # bands makes a healthy model look like it has stopped listening. What
        # matters is that the fastest band is high.
        out["band_entry"] = [
            float(band_entry[b] / band_entry_n[b].clamp_min(1))
            for b in range(cfg.n_bands)
        ]
    return out


@torch.no_grad()
def geometry_diagnostics(model, stream=None, *, batch_size: int = 8,
                         seq_len: int = 128) -> dict:
    """What the learned space actually looks like.

    These are the numbers that decide whether the architecture is interesting
    even when perplexity is not: does the geometry carry structure, and does
    adaptive fan-out track how hard the token was?
    """
    from knn import graph_stats, edge_length_stats

    if not hasattr(model, "axon"):
        return {}

    out = {
        **graph_stats(model.knn_idx, model.cfg.n_neurons),
        **edge_length_stats(model.axon, model.dend, model.knn_idx),
        "sigma": float(model.log_sigma.exp()),
        "gain": float(model.gain),
        # Multiplier on the per-neuron write rate. Near zero means the model has
        # stopped listening to its input.
        "input_gate": float(model.gate_logit.exp()),
    }
    dec = model.decays()
    tau = -1.0 / torch.log(dec.clamp(1e-6, 1 - 1e-6))
    out["tau_median"] = float(tau.median())
    out["tau_p95"] = float(tau.quantile(0.95))
    if model.cfg.dale:
        s = model.signs()
        out["excitatory_frac"] = float((s > 0).float().mean())
        out["sign_saturation"] = float(s.abs().mean())

    # Spread of the token->space map. If every token lands in the same place,
    # the input map has collapsed and nothing downstream can recover.
    tp = model.tok_pos.weight
    out["tok_pos_std"] = float(tp.std())
    out["tok_pos_spread"] = float(
        torch.cdist(tp[:512].float(), tp[:512].float()).mean()
    )

    if stream is not None:
        # Does a neuron fire harder on tokens the model finds surprising?
        # A positive correlation is the adaptive-compute claim.
        device = next(model.parameters()).device
        x, y = stream.get_batch("val", batch_size, seq_len)
        state = model.init_state(batch_size, device)
        surprisal, ks = [], []
        for t in range(seq_len):
            stats: dict = {}
            w = model.edge_weights()
            state = model.step(x[:, t], state, w, model.signs(), stats)
            logits = model.readout(state)
            nll = F.cross_entropy(logits.float(), y[:, t], reduction="none")
            surprisal.append(nll)
            ks.append(stats["k_sum"] / stats["k_count"])
        s = torch.stack(surprisal, 1).mean(0)
        k = torch.stack([kk.reshape(()) for kk in ks])
        if s.numel() > 2 and float(s.std()) > 0 and float(k.std()) > 0:
            sc = (s - s.mean()) / s.std()
            kc = (k - k.mean()) / k.std()
            out["k_surprisal_corr"] = float((sc * kc).mean())
    return out


# --------------------------------------------------------------------------

def main() -> None:
    from data import TokenStream, load_tokenizer
    from checkpoint import load_model

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", type=int, default=50, help="0 = whole split")
    ap.add_argument("--full", action="store_true", help="score the entire val split")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--sample", type=int, default=0, help="how many samples to print")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--rounds", type=int, default=None,
                    help="override propagation rounds at evaluation time. "
                         "--rounds 0 disables spreading entirely, so the gap "
                         "against the unmodified score is what the connectome "
                         "is actually contributing. Rounds carry no parameters, "
                         "so the checkpoint still loads unchanged.")
    ap.add_argument("--memory", action="store_true",
                    help="measure the effective context window of the recurrence")
    ap.add_argument("--memory-trials", type=int, default=16)
    ap.add_argument("--induction", action="store_true",
                    help="in-context learning: can the model complete "
                         "A B ... A -> B from its own context?")
    ap.add_argument("--gap", action="store_true",
                    help="score the slice the model trained on against the "
                         "slice it never saw, both inside train.bin. Needs a "
                         "checkpoint trained with --train-tokens.")
    ap.add_argument("--gap-batches", type=int, default=100,
                    help="batches per side; the same count is used for both")
    ap.add_argument("--scaling", action="store_true",
                    help="perplexity as a function of latent iterations at "
                         "inference, plus the state settling trajectory")
    ap.add_argument("--gap-tokens", type=int, default=0,
                    help="split point to use instead of the checkpoint's cap. "
                         "Run this on a full-corpus checkpoint as a control: it "
                         "never repeated a token, so its gap should be ~0, which "
                         "shows the gap measures memorisation and not some "
                         "difference between the two slices themselves.")
    ap.add_argument("--json", default=None, help="write metrics here")
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt, device=args.device)
    if args.rounds is not None:
        if not hasattr(model, "cfg"):
            raise SystemExit("--rounds only applies to the brain model")
        print(f"[ablate] propagation rounds {model.cfg.rounds} -> {args.rounds}")
        model.cfg.rounds = args.rounds
    data_dir = args.data_dir or cfg.data_dir
    stream = TokenStream(data_dir, device=args.device)
    tok = load_tokenizer(data_dir)

    bs = args.batch_size or cfg.train.batch_size
    sl = args.seq_len or cfg.train.seq_len
    metrics = evaluate(
        model, stream, batch_size=bs, seq_len=sl,
        tbptt=cfg.train.tbptt_chunk, max_batches=0 if args.full else args.batches,
        tokenizer=tok,
    )
    if args.diagnostics:
        metrics.update(geometry_diagnostics(model, stream))
    if args.memory:
        metrics.update(memory_horizon(model, trials=args.memory_trials))
    if args.induction:
        metrics.update(induction_probe(model))
    if args.scaling:
        metrics.update(compute_scaling(
            model, stream, batch_size=bs, seq_len=sl,
            tbptt=cfg.train.tbptt_chunk, batches=max(args.batches, 20)))
    if args.gap:
        # The cap is recorded in the checkpoint, so the two slices are the ones
        # this model actually saw and did not see.
        metrics.update(generalization_gap(
            model, stream, train_tokens=args.gap_tokens or cfg.train.train_tokens,
            batch_size=bs, seq_len=sl, tbptt=cfg.train.tbptt_chunk,
            batches=args.gap_batches, tokenizer=tok,
        ))

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"checkpoint": args.ckpt, "metrics": metrics}, fh, indent=2)

    for i in range(args.sample):
        print(f"\n--- sample {i + 1} " + "-" * 50)
        print(generate(model, tok, args.prompt, temperature=args.temperature,
                       device=args.device, seed=1000 + i))


if __name__ == "__main__":
    main()
