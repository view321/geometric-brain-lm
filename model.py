"""The geometric brain language model.

An RNN whose hidden state is a sparse k-winners-take-all activation vector over
N units, and whose recurrent weight matrix is never stored -- it is generated on
demand as a kernel function of pairwise distances between N learned points.

Forward pass for one token:

    token -> point in R^d          (tok_pos, a V x d table: the fully general
                                    map from a token id into brain space)
    point -> seed_n nearest dendrites, activated by kernel strength
    state  = decay * state + seed
    repeat `rounds` times:
        k-WTA: keep the top_n most active neurons, zero the rest
        each survivor fires along its K_i nearest outbound edges, where
          K_i scales with its own activation (adaptive compute)
        signals scatter-add into the next state, signed by Dale's law
        activation, then RMS homeostasis
    state -> logits                (sparse low-rank readout)

Shapes are static throughout -- variable K is a mask over a fixed K_max, not a
ragged gather -- so the whole thing is torch.compile-able and batches cleanly.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import BrainConfig
from knn import build_knn

EPS = 1e-6


def param_count(cfg: BrainConfig) -> int:
    """Parameter count without constructing the model.

    Instantiating just to count would run a full kNN build, which is a real
    N^2 scan; the sweep needs this number before it has a GPU warmed up, only
    to size the baselines against it.

    `test_shapes.py` asserts this against an actual model so the two cannot
    drift apart.
    """
    n, d, v = cfg.n_neurons, cfg.d_space, cfg.vocab_size
    total = 2 * n * d          # axon + dendrite coordinates
    total += v * d             # token -> brain space
    total += n                 # Dale sign logits
    total += n                 # per-neuron decay logits
    total += 3                 # log_sigma, gain, input gate
    if cfg.free_weights:
        total += n * cfg.k_max
    if cfg.readout == "lowrank":
        r = cfg.readout_rank
        total += n * r + 2 * r + r * v + v
    elif cfg.readout == "geometric":
        total += v * d + v + 1
    else:
        total += n * v
    return total


def solve_readout_rank(cfg: BrainConfig, target: int) -> int:
    """Readout rank that lands the total parameter count on `target`.

    Sweeping d changes the parameter count (2*N*d of coordinates), so a raw
    d-sweep confounds "more dimensions" with "more parameters" -- at N=32768 the
    d=64 model carries ~43% more parameters than the d=2 one. Trading the
    difference against the readout rank holds the budget fixed so the sweep
    measures the dimension and nothing else.

    Only valid for the lowrank readout; returns the current rank otherwise.
    """
    if cfg.readout != "lowrank":
        return cfg.readout_rank
    n, d, v = cfg.n_neurons, cfg.d_space, cfg.vocab_size
    fixed = 2 * n * d + v * d + 2 * n + 3 + v
    if cfg.free_weights:
        fixed += n * cfg.k_max
    per_rank = n + 2 + v
    rank = (target - fixed) // per_rank
    if rank < 8:
        raise ValueError(
            f"target {target:,} is too small for N={n}, d={d}: the coordinates "
            f"alone need {fixed:,} parameters"
        )
    return int(rank)


class BrainLM(nn.Module):
    def __init__(self, cfg: BrainConfig):
        super().__init__()
        self.cfg = cfg
        n, d = cfg.n_neurons, cfg.d_space

        # Scale the cloud so the mean squared pairwise distance is ~1 for any d,
        # which keeps sigma_init meaningful as d is swept.
        std = 1.0 / math.sqrt(2.0 * d)
        self.axon = nn.Parameter(torch.randn(n, d) * std)
        self.dend = nn.Parameter(torch.randn(n, d) * std)

        # A token id maps to a point in brain space. A V x d table is already
        # the most general such map, so there is nothing to gain from an MLP in
        # front of it.
        self.tok_pos = nn.Embedding(cfg.vocab_size, d)
        nn.init.normal_(self.tok_pos.weight, std=std)

        # Parameters a disabled feature would leave without a gradient path are
        # frozen rather than left to sit at their initialisation looking trained.
        self.sign_logit = nn.Parameter(torch.randn(n) * 0.5, requires_grad=cfg.dale)
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(cfg.sigma_init)), requires_grad=cfg.learn_sigma
        )
        self.gain = nn.Parameter(torch.tensor(1.0), requires_grad=cfg.homeostasis)

        self.gate_logit = nn.Parameter(
            torch.tensor(math.log(max(cfg.input_gate, 1e-6))),
            requires_grad=cfg.learn_input_gate,
        )

        # Time constants laid out log-uniformly and, crucially, in band order:
        # band b owns the contiguous slice [b*size, (b+1)*size), slowest first.
        # The banded k-WTA relies on that contiguity to slice without gathering.
        #
        # Each neuron's tau is confined to its own band's range rather than being
        # freely learnable. The band partition is by index, so an unconstrained
        # tau could drift out of the band whose activation budget it was
        # allocated, and the banding would then guarantee no timescale diversity
        # at all -- which is the only thing it exists to do. Confining them also
        # blocks the collapse this architecture is otherwise drawn to, where the
        # optimizer shortens every horizon because next-token loss rewards
        # immediate input sensitivity over memory.
        band = n // cfg.n_bands
        edges = torch.linspace(math.log(cfg.decay_tau_max),
                               math.log(cfg.decay_tau_min), cfg.n_bands + 1)
        self.register_buffer("log_tau_lo", edges[1:].repeat_interleave(band))
        self.register_buffer("log_tau_span",
                             (edges[:-1] - edges[1:]).repeat_interleave(band))
        # Spread within each band, so a band is a continuum rather than one tau.
        frac = torch.linspace(0.98, 0.02, band).repeat(cfg.n_bands)
        self.decay_logit = nn.Parameter(
            torch.log(frac / (1 - frac)), requires_grad=cfg.learn_decay
        )

        if cfg.free_weights:
            # Ablation: strengths are free parameters on a frozen topology.
            self.free_w = nn.Parameter(torch.rand(n, cfg.k_max) * 0.5 + 0.5)

        self._build_readout()

        self.register_buffer("knn_idx", torch.zeros(n, cfg.k_max, dtype=torch.int32))
        self.register_buffer("usage", torch.zeros(n))
        self.register_buffer("knn_dirty", torch.tensor(1, dtype=torch.uint8))

        if cfg.freeze_positions:
            self.axon.requires_grad_(False)
            self.dend.requires_grad_(False)
            self.sign_logit.requires_grad_(False)
        if cfg.freeze_input:
            self.tok_pos.weight.requires_grad_(False)

        # Under the free-weight ablation the axon coordinates have no gradient
        # path: strengths no longer come from distance, and the topology they
        # would otherwise steer is frozen. Only the geometric readout still
        # reads them. Freezing makes that explicit instead of leaving a
        # parameter that trains to nothing and inflates the reported count.
        if cfg.free_weights and cfg.readout != "geometric":
            self.axon.requires_grad_(False)

        self.rebuild_knn()
        self.calibrate_sigma()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _build_readout(self) -> None:
        cfg = self.cfg
        if cfg.readout == "lowrank":
            # Read only from the active set: a weighted sum of the rows of an
            # N x r table, then r -> V. Dominates the parameter count, which is
            # why every ablation shares it verbatim.
            self.r_in = nn.Parameter(torch.randn(cfg.n_neurons, cfg.readout_rank)
                                     / math.sqrt(cfg.readout_rank))
            self.r_norm = nn.LayerNorm(cfg.readout_rank)
            self.r_out = nn.Linear(cfg.readout_rank, cfg.vocab_size, bias=True)
        elif cfg.readout == "geometric":
            # The state is summarised as its activation-weighted centroid in
            # brain space, and every vocab item is a point; the logit is kernel
            # similarity. Almost free in parameters, and keeps the whole model
            # inside the geometry -- but a d-dimensional bottleneck is severe.
            self.out_pos = nn.Parameter(torch.randn(cfg.vocab_size, cfg.d_space)
                                        / math.sqrt(2.0 * cfg.d_space))
            self.out_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
            self.out_scale = nn.Parameter(torch.tensor(1.0))
        else:  # dense
            self.r_out_dense = nn.Parameter(
                torch.randn(cfg.n_neurons, cfg.vocab_size) / math.sqrt(cfg.n_neurons)
            )

    @torch.no_grad()
    def calibrate_sigma(self) -> float:
        """Set sigma from the realised edge lengths so the kernel starts in a
        useful range instead of saturated at 0 or 1.

        Without this the first run of a new `d` is usually dead on arrival: the
        typical nearest-neighbour distance shrinks with d, so a fixed sigma
        either makes every edge weight 1.0 (no selectivity) or 0.0 (no signal).
        Always runs, including when sigma is frozen -- a frozen sigma still has
        to start somewhere sane.
        """
        n = self.cfg.n_neurons
        rows = torch.randperm(n, device=self.axon.device)[: min(2048, n)]
        nbr = self.knn_idx[rows].long()
        dist = (self.axon[rows].unsqueeze(1) - self.dend[nbr]).pow(2).sum(-1).sqrt()
        median = dist.median().clamp_min(1e-4)
        # g(median) = exp(-median^2 / 2 sigma^2) = 0.5  =>  sigma = median / 1.177
        self.log_sigma.fill_(float(torch.log(median / 1.1774)))
        return float(self.log_sigma.exp())

    @torch.no_grad()
    def rebuild_knn(self) -> None:
        self.knn_idx.copy_(
            build_knn(
                self.axon.detach(), self.dend.detach(), self.cfg.k_max,
                chunk=self.cfg.knn_chunk, exclude_self=self.cfg.exclude_self,
            )
        )
        self.knn_dirty.fill_(0)

    # ------------------------------------------------------------------
    # kernel and weights
    # ------------------------------------------------------------------

    def kernel(self, sq_dist: torch.Tensor) -> torch.Tensor:
        """Connection strength from squared distance. Decreasing, bounded, and
        smooth so that gradients reach the coordinates."""
        sigma2 = (2.0 * self.log_sigma).exp().clamp_min(EPS)
        if self.cfg.kernel == "gaussian":
            return torch.exp(-sq_dist / (2.0 * sigma2))
        return 1.0 / (1.0 + sq_dist / sigma2)      # cauchy: heavier tail

    def signs(self) -> torch.Tensor:
        """Dale's law: neuron i is excitatory or inhibitory for all its targets."""
        if not self.cfg.dale:
            return torch.ones_like(self.sign_logit)
        soft = torch.tanh(self.sign_logit / max(self.cfg.dale_tau, EPS))
        if self.cfg.dale_hard:
            return soft + (torch.sign(soft) - soft).detach()   # straight-through
        return soft

    def _edge_weights_impl(self, axon: torch.Tensor, dend: torch.Tensor) -> torch.Tensor:
        nbr = self.knn_idx.long()                              # [N, K]
        diff = axon.unsqueeze(1) - dend[nbr]                   # [N, K, d]
        return self.kernel(diff.pow(2).sum(-1))                # [N, K]

    def edge_weights(self) -> torch.Tensor:
        """[N, K_max] live connection strengths.

        Computed once per forward and reused by every token and every
        propagation round: the coordinates only move on an optimizer step, so
        recomputing per token would multiply the saved-activation cost by
        T * rounds for identical values.
        """
        if self.cfg.free_weights:
            return self.free_w
        if self.cfg.grad_checkpoint and self.training:
            return checkpoint(self._edge_weights_impl, self.axon, self.dend,
                              use_reentrant=False)
        return self._edge_weights_impl(self.axon, self.dend)

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------

    def init_state(self, batch: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.n_neurons, device=device, dtype=dtype)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation == "relu":
            return F.relu(x)
        if self.cfg.activation == "tanh":
            return torch.tanh(x)
        return x

    def _homeostasis(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize the carried activation, and pin it to fp32.

        Homeostasis is what keeps a recurrent sparse net from either dying out
        or running away; the k-WTA caps how many neurons fire but not how hard.

        The fp32 cast matters independently: this is the last op of every round,
        so it is where the state re-enters the recurrence. Under bf16 autocast a
        state accumulated across 256 tokens x 3 rounds drifts badly, while the
        ops that actually want bf16 -- the seed distance matmul and the readout
        projection -- are upstream and keep it.
        """
        state = state.float()
        if not self.cfg.homeostasis:
            return state
        rms = state.pow(2).mean(-1, keepdim=True).add(EPS).sqrt()
        return state / rms * self.gain.float()

    def decays(self) -> torch.Tensor:
        """[N] per-neuron retention. Constant unless `learn_decay` is set."""
        if not self.cfg.learn_decay:
            return torch.full_like(self.decay_logit, self.cfg.decay)
        tau = (self.log_tau_lo
               + torch.sigmoid(self.decay_logit) * self.log_tau_span).exp()
        return torch.exp(-1.0 / tau).clamp(1e-4, 1.0 - 1e-4)

    def _kwta(self, state: torch.Tensor):
        """Hard k-winners-take-all. Returns (sparse state, values, indices).

        With `n_bands > 1` the budget is split evenly across contiguous bands
        rather than competed for globally. Slow neurons retain magnitude by
        construction, so a global top-k hands them every slot permanently and
        the fast neurons -- the ones that track the current clause -- starve.
        """
        cfg = self.cfg
        if cfg.n_bands <= 1:
            val, idx = state.topk(cfg.top_n, dim=-1)
        else:
            b = state.shape[0]
            band = cfg.n_neurons // cfg.n_bands
            per = cfg.top_n // cfg.n_bands
            v, i = state.view(b, cfg.n_bands, band).topk(per, dim=-1)
            offset = torch.arange(cfg.n_bands, device=state.device).view(1, -1, 1) * band
            val = v.reshape(b, -1)
            idx = (i + offset).reshape(b, -1)
        sparse = torch.zeros_like(state).scatter(1, idx, val)
        return sparse, val, idx

    def _propagate(self, state: torch.Tensor, w_all: torch.Tensor,
                   signs: torch.Tensor, decay: torch.Tensor):
        cfg = self.cfg
        b = state.shape[0]
        state, val, idx = self._kwta(state)

        # Adaptive compute: a neuron's fan-out scales with how strongly it fired,
        # normalised against the strongest neuron in the same batch row so the
        # rule is invariant to the overall activation scale. Taken as an explicit
        # max rather than val[:, 0] because banded selection is sorted within a
        # band but not across bands.
        peak = val.max(dim=-1, keepdim=True).values.clamp_min(EPS)
        frac = (val / peak).clamp(0.0, 1.0)
        k_i = cfg.k_min + torch.floor(frac * (cfg.k_max - cfg.k_min + 1)).clamp(
            max=cfg.k_max - cfg.k_min
        )                                                        # [B, P]
        ramp = torch.arange(cfg.k_max, device=state.device).view(1, 1, -1)
        mask = (ramp < k_i.unsqueeze(-1)).to(state.dtype)        # [B, P, K]

        nbr = self.knn_idx[idx].long()                           # [B, P, K]
        w = w_all[idx]                                           # [B, P, K]
        sgn = signs[idx].unsqueeze(-1)                           # [B, P, 1]
        signal = val.unsqueeze(-1) * w * sgn * mask              # [B, P, K]

        # Divide by the fan-out actually used, so a neuron that fires along 32
        # edges does not simply inject 8x the current of one that fires along 4.
        signal = signal / mask.sum(-1, keepdim=True).clamp_min(1.0).sqrt()

        arrived = torch.zeros_like(state).scatter_add(
            1, nbr.reshape(b, -1), signal.reshape(b, -1)
        )
        out = self._homeostasis(self._activate(decay.unsqueeze(0) * state + arrived))
        return out, idx, k_i

    def step(self, tok: torch.Tensor, state: torch.Tensor, w_all: torch.Tensor,
             signs: torch.Tensor, stats: dict | None = None,
             decay: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self.cfg
        if decay is None:
            decay = self.decays()
        p = self.tok_pos(tok)                                    # [B, d]

        # Squared distance from the injected point to every dendrite.
        sq = (p.pow(2).sum(-1, keepdim=True)
              - 2.0 * (p @ self.dend.t())
              + self.dend.pow(2).sum(-1).unsqueeze(0)).clamp_min(0.0)   # [B, N]

        sv, si = (-sq).topk(cfg.seed_n, dim=-1)
        seed = torch.zeros_like(state).scatter(1, si, self.kernel(-sv))

        # Match the seed to the state's scale before mixing. Raw kernel values
        # live in (0, 1] while a homeostatically normalised state carries
        # sqrt(N / top_n) per active neuron -- about 12x larger at the reference
        # config. Adding them directly makes the k-WTA discard ~93% of every
        # token's seeds, so the state coasts on whatever entered first instead of
        # integrating the sequence.
        if cfg.legacy_injection:
            # The pre-fix path, kept bit-for-bit so checkpoints trained before
            # the fix can still be measured as the models they actually are.
            # Everything else in this class is already backward compatible: with
            # n_bands=1 the banded k-WTA is a plain top-k, and with
            # learn_decay=False `decays()` is a constant vector equal to
            # cfg.decay.
            state = self._homeostasis(decay.unsqueeze(0) * state + seed)
        else:
            seed = seed / seed.pow(2).mean(-1, keepdim=True).add(EPS).sqrt()
            # Write rate is coupled to each neuron's own decay, which makes this
            # an exponential moving average per neuron: x <- d*x + (1-d)*u. A
            # single global gate cannot work here -- setting it high enough for
            # fast neurons to track the current token makes every token overwrite
            # the slow ones too, which is how retention collapsed from 14 tokens
            # to 3 when the injection scale alone was fixed. Coupling makes a
            # tau=250 neuron accept 0.4% per token and hold, while a tau=1.5
            # neuron accepts half and tracks. Steady-state magnitudes still match
            # across bands, since an EMA converges to the mean of its input.
            write = self.gate_logit.exp() * (1.0 - decay)         # [N]
            state = self._homeostasis(
                decay.unsqueeze(0) * state + write.unsqueeze(0) * seed
            )

        if stats is not None:
            # Switch-style load balance, measured where routing actually happens.
            # Uses the kernel's own temperature so the proxy matches selection.
            sigma2 = (2.0 * self.log_sigma).exp().clamp_min(EPS)
            probs = torch.softmax(-sq / (2.0 * sigma2), dim=-1)   # [B, N]
            stats["p_sum"] = stats.get("p_sum", 0.0) + probs.mean(0)
            hits = torch.zeros_like(state).scatter(
                1, si, torch.ones_like(sv)
            ).mean(0)
            stats["f_sum"] = stats.get("f_sum", 0.0) + hits
            stats["n_tok"] = stats.get("n_tok", 0) + 1

        for _ in range(cfg.rounds):
            state, idx, k_i = self._propagate(state, w_all, signs, decay)
            if stats is not None:
                stats["k_sum"] = stats.get("k_sum", 0.0) + k_i.mean()
                stats["k_count"] = stats.get("k_count", 0) + 1
                with torch.no_grad():
                    fired = torch.zeros(cfg.n_neurons, device=state.device)
                    fired.scatter_add_(
                        0, idx.reshape(-1),
                        torch.ones(idx.numel(), device=state.device),
                    )
                    stats["fired"] = stats.get("fired", 0.0) + fired
        return state

    # ------------------------------------------------------------------
    # readout
    # ------------------------------------------------------------------

    def readout(self, state: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        # Routed through _kwta so the readout sees the same active set the
        # dynamics do; a plain global top-k would ignore the band budget.
        if cfg.readout == "lowrank":
            _, val, idx = self._kwta(state)
            h = (self.r_in[idx] * val.unsqueeze(-1)).sum(1)      # [B, r]
            return self.r_out(self.r_norm(h))
        if cfg.readout == "geometric":
            _, val, idx = self._kwta(state)
            w = val / val.sum(-1, keepdim=True).clamp_min(EPS)
            centroid = (self.axon[idx] * w.unsqueeze(-1)).sum(1)  # [B, d]
            sq = torch.cdist(centroid, self.out_pos).pow(2)       # [B, V]
            return -sq * self.out_scale.abs() + self.out_bias
        return state @ self.r_out_dense

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None,
                collect_stats: bool = True):
        """x: [B, T] token ids. Returns (logits [B, T, V], final state, stats)."""
        b, t = x.shape
        if state is None:
            state = self.init_state(b, x.device, dtype=self.axon.dtype)

        w_all = self.edge_weights()
        signs = self.signs()
        decay = self.decays()
        stats: dict | None = {} if collect_stats else None

        logits = []
        for i in range(t):
            state = self.step(x[:, i], state, w_all, signs, stats, decay)
            logits.append(self.readout(state))
        return torch.stack(logits, dim=1), state, (stats or {})

    # ------------------------------------------------------------------
    # auxiliary loss and neuron rescue
    # ------------------------------------------------------------------

    def balance_loss(self, stats: dict) -> torch.Tensor:
        """Switch-Transformer load balancing.

        N * sum_i f_i * P_i, minimised when both the hard selection frequency
        and the soft routing probability are uniform. Without this the k-WTA
        collapses onto a few thousand neurons and the rest never receive a
        gradient, because a neuron that is never selected never moves and so is
        never selected -- the VQ-VAE codebook failure in different clothing.
        """
        if not stats or "p_sum" not in stats:
            return torch.zeros((), device=self.axon.device)
        n_tok = max(stats["n_tok"], 1)
        p = stats["p_sum"] / n_tok
        f = stats["f_sum"] / n_tok
        return self.cfg.n_neurons * (p * f).sum()

    @torch.no_grad()
    def update_usage(self, stats: dict, ema: float = 0.99) -> None:
        if "fired" not in stats:
            return
        counts = stats["fired"]
        self.usage.mul_(ema).add_(counts / counts.sum().clamp_min(1.0), alpha=1 - ema)

    @torch.no_grad()
    def reseed_dead(self, frac: float, noise: float, optimizer=None) -> int:
        """Teleport the least-used neurons next to heavily-used ones.

        The load-balancing loss alone does not rescue a neuron that is already
        at zero: it gets no gradient at all, so it cannot move toward the
        action. Relocating it is the only way back. Optimizer moments for the
        moved rows are cleared as well -- stale Adam momentum would otherwise
        drag them straight back to where they were useless.
        """
        if frac <= 0:
            return 0
        n = self.cfg.n_neurons
        k = max(1, int(n * frac))
        dead = torch.topk(self.usage, k, largest=False).indices
        live = torch.topk(self.usage, max(k, 1), largest=True).indices
        donors = live[torch.randint(0, live.numel(), (k,), device=self.usage.device)]

        scale = noise * self.axon.std().clamp_min(EPS)
        self.axon[dead] = self.axon[donors] + torch.randn_like(self.axon[dead]) * scale
        self.dend[dead] = self.dend[donors] + torch.randn_like(self.dend[dead]) * scale
        self.sign_logit[dead] = self.sign_logit[donors].clone()
        self.usage[dead] = self.usage.mean()

        if optimizer is not None:
            for param in (self.axon, self.dend):
                st = optimizer.state.get(param)
                if st:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in st:
                            st[key][dead] = 0.0
        self.knn_dirty.fill_(1)
        return k

    # ------------------------------------------------------------------
    # optimizer wiring
    # ------------------------------------------------------------------

    def param_groups(self, lr: float, lr_positions: float, weight_decay: float):
        """Coordinates get their own learning rate and never any weight decay.

        Decaying a coordinate pulls it toward the origin, which collapses the
        point cloud toward a single location and destroys the geometry the
        model is built on. This is the single easiest way to silently ruin a
        run, so it is enforced here rather than left to the caller.
        """
        geometry = {"axon", "dend", "tok_pos.weight"}
        if self.cfg.readout == "geometric":
            geometry.add("out_pos")
        # free_w is the ablation's stand-in for g(distance), which is undecayed;
        # decaying it would hand the geometric model an unearned advantage.
        scalars = {"sign_logit", "log_sigma", "gain", "out_scale", "out_bias", "free_w"}

        geo, plain, decay = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name in geometry:
                geo.append(param)
            elif name in scalars or param.ndim <= 1:
                plain.append(param)
            else:
                decay.append(param)
        return [
            {"params": geo, "lr": lr_positions, "weight_decay": 0.0, "name": "geometry"},
            {"params": plain, "lr": lr, "weight_decay": 0.0, "name": "scalars"},
            {"params": decay, "lr": lr, "weight_decay": weight_decay, "name": "weights"},
        ]

    def n_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        readout = sum(
            p.numel() for name, p in self.named_parameters()
            if name.startswith(("r_in", "r_out", "out_pos", "out_bias"))
        )
        return {
            "total": total,
            "readout": readout,
            "geometry": self.axon.numel() + self.dend.numel(),
            "input": self.tok_pos.weight.numel(),
        }
