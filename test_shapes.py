"""Invariants that are cheap to check and expensive to get wrong.

Run before shipping to the GPU box; the whole file is a few seconds on CPU.

    python test_shapes.py

These are not accuracy tests. They catch the class of bug that produces a run
which trains for two days and reports a number that means nothing: a dead
gradient path, a state that silently drifts in bf16, a k-WTA that is not
actually sparse, a reseed that moves the wrong neurons.
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

from config import BrainConfig
from knn import build_knn
from model import BrainLM, param_count

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def tiny(**kw) -> BrainConfig:
    base = dict(vocab_size=64, n_neurons=256, d_space=6, seed_n=8, top_n=32,
                k_min=2, k_max=8, rounds=2, readout_rank=32, knn_chunk=64)
    base.update(kw)
    return BrainConfig(**base)


# --------------------------------------------------------------------------

def test_param_count() -> None:
    print("\nanalytic parameter count matches the real model")
    for kw in (
        {}, {"readout": "geometric"}, {"free_weights": True},
        {"d_space": 32, "readout_rank": 128}, {"n_neurons": 512, "vocab_size": 128},
    ):
        cfg = tiny(**kw)
        model = BrainLM(cfg)
        real = sum(p.numel() for p in model.parameters())
        check(f"param_count {kw or 'default'}", param_count(cfg) == real,
              f"analytic {param_count(cfg)} vs real {real}")


def test_knn() -> None:
    print("\nkNN graph is well formed")
    n, d, k = 512, 8, 16
    axon, dend = torch.randn(n, d), torch.randn(n, d)
    idx = build_knn(axon, dend, k, chunk=64, exclude_self=True)
    check("shape", tuple(idx.shape) == (n, k))
    check("indices in range", bool((idx >= 0).all() and (idx < n).all()))
    rows = torch.arange(n).unsqueeze(1)
    check("no self loops", bool((idx.long() != rows).all()))

    # Brute force one row against the chunked result.
    dist = (axon[0] - dend).pow(2).sum(-1)
    dist[0] = float("inf")
    check("matches brute force", bool((idx[0].long().sort().values
                                       == dist.topk(k, largest=False).indices.sort().values).all()))

    # Neighbours must be sorted nearest first: adaptive K takes a prefix, so an
    # unsorted graph would silently hand weak neurons the strongest edges.
    lengths = (axon[0] - dend[idx[0].long()]).pow(2).sum(-1)
    check("sorted nearest first", bool((lengths[1:] >= lengths[:-1] - 1e-6).all()))


def test_gradients() -> None:
    print("\ngradients reach every trainable parameter")
    for kw in ({}, {"readout": "geometric"}, {"dale": False}, {"homeostasis": False}):
        cfg = tiny(**kw)
        model = BrainLM(cfg)
        x = torch.randint(0, cfg.vocab_size, (3, 6))
        logits, _, stats = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size),
                               x[:, 1:].reshape(-1))
        (loss + 0.01 * model.balance_loss(stats)).backward()
        dead = [n for n, p in model.named_parameters()
                if p.requires_grad and (p.grad is None or not p.grad.abs().sum() > 0)]
        check(f"all params get gradient {kw or 'default'}", not dead, f"dead: {dead}")


def test_sparsity_and_k() -> None:
    print("\nk-WTA is actually sparse and adaptive K stays in bounds")
    cfg = tiny()
    model = BrainLM(cfg)
    state = model.init_state(2, torch.device("cpu"))
    w, s = model.edge_weights(), model.signs()
    stats: dict = {}
    for t in range(6):
        state = model.step(torch.randint(0, cfg.vocab_size, (2,)), state, w, s, stats)

    sparse, val, idx = model._kwta(state)
    check("k-WTA keeps exactly top_n", int((sparse != 0).sum(-1).max()) <= cfg.top_n)
    check("selected values are the largest",
          bool(torch.allclose(val.sort(descending=True).values, val)))

    mean_k = float(stats["k_sum"]) / stats["k_count"]
    check("mean K within [k_min, k_max]", cfg.k_min <= mean_k <= cfg.k_max,
          f"mean_k={mean_k:.2f}")
    check("state finite", bool(torch.isfinite(state).all()))
    check("state non-negative under relu", bool((state >= 0).all()))


def test_long_horizon_stability() -> None:
    print("\nstate stays bounded over a long rollout")
    cfg = tiny(rounds=3)
    model = BrainLM(cfg)
    state = model.init_state(2, torch.device("cpu"))
    w, s = model.edge_weights(), model.signs()
    norms = []
    with torch.no_grad():
        for t in range(256):
            state = model.step(torch.randint(0, cfg.vocab_size, (2,)), state, w, s)
            norms.append(float(state.norm()))
    check("finite after 256 tokens", all(math.isfinite(v) for v in norms))
    check("no runaway growth", max(norms) < 20 * (norms[8] + 1e-6),
          f"max={max(norms):.2f} early={norms[8]:.2f}")
    check("no total collapse", min(norms[8:]) > 1e-4, f"min={min(norms[8:]):.2e}")


def test_autocast_state_dtype() -> None:
    print("\nbf16 autocast does not leak into the recurrent state")
    if not hasattr(torch, "autocast"):
        print("  skip  autocast unavailable")
        return
    cfg = tiny()
    model = BrainLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 4))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, state, _ = model(x)
    check("state is fp32", state.dtype == torch.float32, f"got {state.dtype}")
    check("logits finite", bool(torch.isfinite(logits).all()))


def test_reseed() -> None:
    print("\nreseeding relocates the least-used neurons and clears their momentum")
    cfg = tiny()
    model = BrainLM(cfg)
    opt = torch.optim.AdamW(model.param_groups(1e-3, 1e-3, 0.01))

    x = torch.randint(0, cfg.vocab_size, (2, 4))
    logits, _, stats = model(x)
    F.cross_entropy(logits.reshape(-1, cfg.vocab_size),
                    x.reshape(-1)).backward()
    opt.step()

    model.usage.zero_()
    model.usage[10:] = 1.0                       # neurons 0..9 are the dead ones
    before = model.axon.detach().clone()
    moved = model.reseed_dead(frac=10 / cfg.n_neurons, noise=0.1, optimizer=opt)

    delta = (model.axon.detach() - before).abs().sum(-1)
    check("reseeded the requested count", moved == 10, f"moved={moved}")
    check("only dead neurons moved", bool((delta[10:] == 0).all()))
    check("dead neurons actually moved", bool((delta[:10] > 0).all()))
    check("graph marked dirty", int(model.knn_dirty) == 1)

    st = opt.state.get(model.axon, {})
    if "exp_avg" in st:
        check("momentum cleared for moved rows",
              bool((st["exp_avg"][:10] == 0).all()))


def test_frozen_control() -> None:
    print("\nthe frozen control really is frozen")
    cfg = tiny(freeze_positions=True, freeze_input=True)
    model = BrainLM(cfg)
    check("axon frozen", not model.axon.requires_grad)
    check("dendrite frozen", not model.dend.requires_grad)
    check("signs frozen", not model.sign_logit.requires_grad)
    check("input map frozen", not model.tok_pos.weight.requires_grad)

    before = model.axon.detach().clone()
    opt = torch.optim.AdamW(model.param_groups(1e-2, 1e-2, 0.01))
    x = torch.randint(0, cfg.vocab_size, (2, 4))
    logits, _, _ = model(x)
    F.cross_entropy(logits.reshape(-1, cfg.vocab_size), x.reshape(-1)).backward()
    opt.step()
    check("positions unchanged after a step",
          bool(torch.equal(before, model.axon.detach())))

    readout = model.r_out.weight
    check("readout still trains", readout.grad is not None and bool(readout.grad.any()))


def test_no_weight_decay_on_geometry() -> None:
    print("\ncoordinates are excluded from weight decay")
    model = BrainLM(tiny())
    groups = model.param_groups(1e-3, 5e-4, 0.1)
    geo = next(g for g in groups if g["name"] == "geometry")
    check("geometry group has zero decay", geo["weight_decay"] == 0.0)
    check("geometry group uses lr_positions", geo["lr"] == 5e-4)
    ids = {id(p) for p in geo["params"]}
    check("axon in geometry group", id(model.axon) in ids)
    check("dendrite in geometry group", id(model.dend) in ids)
    check("token map in geometry group", id(model.tok_pos.weight) in ids)
    decay = next(g for g in groups if g["name"] == "weights")
    check("readout is decayed", any(id(model.r_in) == id(p) for p in decay["params"]))


def main() -> int:
    torch.manual_seed(0)
    for fn in (test_param_count, test_knn, test_gradients, test_sparsity_and_k,
               test_long_horizon_stability, test_autocast_state_dtype,
               test_reseed, test_frozen_control, test_no_weight_decay_on_geometry):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
