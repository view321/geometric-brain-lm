"""Reference models for the ablation ladder.

The point of the experiment is comparative, so the baselines matter as much as
the model under test. Two are here; the other two rungs (free edge weights,
frozen positions) are `BrainLM` itself under different config flags, because an
ablation that shares no code with the thing it ablates proves nothing.

`GRULM` is the fair fight: the brain model is an RNN, so an RNN at the same
parameter count is the number to beat. `TransformerLM` is the ceiling, present
only for context -- losing to it is expected and uninformative.

Both expose the same interface as `BrainLM.forward` -> (logits, state, stats)
so the trainer does not have to branch.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BaselineConfig


class GRULM(nn.Module):
    """Stacked GRU with tied embeddings. Carries state for TBPTT."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.rnn = nn.GRU(
            cfg.d_model, cfg.d_model, num_layers=cfg.n_layers,
            batch_first=True, dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.emb.weight
        self.apply(_init_weights)

    def init_state(self, batch: int, device, dtype=torch.float32):
        return torch.zeros(self.cfg.n_layers, batch, self.cfg.d_model,
                           device=device, dtype=dtype)

    def forward(self, x, state=None, collect_stats: bool = True):
        h = self.drop(self.emb(x))
        if state is None:
            state = self.init_state(x.shape[0], x.device, h.dtype)
        out, state = self.rnn(h, state.contiguous())
        return self.head(self.norm(out)), state, {}

    def param_groups(self, lr, lr_positions, weight_decay):
        decay = [p for p in self.parameters() if p.requires_grad and p.ndim >= 2]
        plain = [p for p in self.parameters() if p.requires_grad and p.ndim < 2]
        return [
            {"params": decay, "lr": lr, "weight_decay": weight_decay, "name": "weights"},
            {"params": plain, "lr": lr, "weight_decay": 0.0, "name": "scalars"},
        ]

    def n_params(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "readout": 0 if self.cfg.tie_embeddings else self.head.weight.numel(),
                "geometry": 0, "input": self.emb.weight.numel()}


class _Block(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.attn(self.ln1(x)).split(c, dim=2)
        shape = (b, t, self.n_heads, c // self.n_heads)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, c)))
        return x + self.drop(self.mlp(self.ln2(x)))


class TransformerLM(nn.Module):
    """Pre-norm decoder-only transformer. The ceiling, not a fair comparison."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.emb.weight
        self.apply(_init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def init_state(self, batch: int, device, dtype=torch.float32):
        return None

    def forward(self, x, state=None, collect_stats: bool = True):
        b, t = x.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence {t} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(t, device=x.device)
        h = self.drop(self.emb(x) + self.pos(pos).unsqueeze(0))
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h)), None, {}

    def param_groups(self, lr, lr_positions, weight_decay):
        decay = [p for p in self.parameters() if p.requires_grad and p.ndim >= 2]
        plain = [p for p in self.parameters() if p.requires_grad and p.ndim < 2]
        return [
            {"params": decay, "lr": lr, "weight_decay": weight_decay, "name": "weights"},
            {"params": plain, "lr": lr, "weight_decay": 0.0, "name": "scalars"},
        ]

    def n_params(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "readout": 0 if self.cfg.tie_embeddings else self.head.weight.numel(),
                "geometry": 0, "input": self.emb.weight.numel()}


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


# --------------------------------------------------------------------------
# parameter matching
# --------------------------------------------------------------------------

def count_params(cfg: BaselineConfig) -> int:
    klass = GRULM if cfg.kind == "gru" else TransformerLM
    with torch.device("meta"):
        return sum(p.numel() for p in klass(cfg).parameters())


def match_params(cfg: BaselineConfig, target: int, lo: int = 32, hi: int = 2048) -> BaselineConfig:
    """Pick the `d_model` whose parameter count lands closest to `target`.

    Comparing a 12M-parameter brain model against a 40M-parameter GRU would
    tell you nothing, and eyeballing d_model to match is how that happens. The
    search is over multiples of n_heads so the transformer stays valid.
    """
    import copy

    step = max(cfg.n_heads, 8)
    best, best_gap = None, None
    lo = max(lo - lo % step, step)
    for d in range(lo, hi + 1, step):
        trial = copy.deepcopy(cfg)
        trial.d_model = d
        try:
            gap = abs(count_params(trial) - target)
        except Exception:
            continue
        if best_gap is None or gap < best_gap:
            best, best_gap = trial, gap
    if best is None:
        raise RuntimeError(f"no valid d_model in [{lo}, {hi}] for target {target}")
    return best


def build_baseline(cfg: BaselineConfig) -> nn.Module:
    if cfg.kind == "gru":
        return GRULM(cfg)
    if cfg.kind == "transformer":
        return TransformerLM(cfg)
    raise ValueError(f"unknown baseline kind {cfg.kind!r}")
