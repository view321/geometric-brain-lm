"""Configuration for the geometric brain language model and its baselines.

Every experiment is one `RunConfig`. Presets cover the CPU smoke test, the
single-GPU reference run, and each rung of the ablation ladder.

    python -c "from config import PRESETS; print(list(PRESETS))"
    python train.py --preset ref --d-space 16
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field, asdict


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

@dataclass
class BrainConfig:
    """The kernel-parameterized sparse RNN.

    The recurrent weight matrix is never stored. Neuron `i` has an axon point
    `A[i]` and a dendrite point `D[i]` in R^d; the strength of the connection
    i -> j is `g(||A[i] - D[j]||)` for a decreasing kernel `g`. Two coordinates
    per neuron rather than one is what makes the connection directed: with a
    single point the kernel is symmetric and `w_ij == w_ji`, which cannot
    express directed information flow.
    """

    vocab_size: int = 8192
    n_neurons: int = 32768

    # The headline hyperparameter. N*d coordinates generate an N*N weight
    # matrix, and Euclidean distance matrices are heavily rank-constrained, so
    # d controls how much of the weight space is reachable at all. Sweep it.
    d_space: int = 16

    # sparsity / dynamics
    seed_n: int = 128      # neurons a token activates directly
    top_n: int = 512       # k-winners-take-all width of the carried state
    k_min: int = 4         # outbound edges used by a barely-active neuron
    k_max: int = 32        # outbound edges used by the most active neuron
    rounds: int = 3        # propagation rounds per token

    # Randomized latent depth. When rounds_max > 0, each training forward draws
    # its round count from [rounds_min, rounds_max] instead of using a fixed
    # `rounds`. The model then cannot rely on a particular iteration count and
    # has to learn a state update that keeps improving as it is applied, which
    # is what makes spending more iterations at inference pay off. `rounds`
    # remains the evaluation default.
    #
    # This is the one change that separates "compute is tied to tokens emitted"
    # from "compute can be spent without emitting anything".
    rounds_min: int = 0
    rounds_max: int = 0    # 0 = fixed depth, the original behaviour

    # Re-inject the token at every propagation round instead of only once.
    #
    # Without this the iteration is `z <- f(z)`, whose fixed points do not
    # depend on the input at all: iterating longer washes the token out and
    # deeper is monotonically worse. With it the iteration is `z <- f(z, x)`,
    # the deep-equilibrium form, whose fixed point is the state consistent with
    # this input -- so extra rounds refine an answer instead of forgetting the
    # question. This is the difference between iterating and thinking.
    inject_every_round: bool = False

    # Geometric falloff applied to the re-injection across rounds: round r gets
    # `inject_decay ** r` of it. 1.0 injects equally forever.
    #
    # Injecting at full strength every round makes the fixed point a pure
    # function of the current token -- the accumulated injection eventually
    # swamps everything the state was carrying, so deep iteration converges on
    # something that has forgotten the context. That is the measured failure:
    # the state settles cleanly, but the attractor scores worse than the
    # transient, and perplexity is minimised a few rounds in and rises
    # thereafter. Decaying the injection keeps the question present while the
    # answer forms, then lets later rounds refine without being overwritten.
    inject_decay: float = 1.0
    decay: float = 0.85    # state leak, per round and across tokens

    # Global multiplier on the per-neuron write rate, which is otherwise
    # (1 - decay_i) -- so each neuron integrates at its own timescale. 1.0 is a
    # plain exponential moving average. The seed is first normalized to the
    # state's scale; without that the raw kernel values arrive ~12x weaker than
    # the resident activation and the k-WTA evicts 93% of every token's seeds.
    input_gate: float = 1.0
    learn_input_gate: bool = True

    # Reproduces the pre-fix injection exactly: raw un-normalized seed, no write
    # gate. Set automatically when loading a checkpoint trained before the fix,
    # so its metrics describe the model that was actually trained rather than a
    # different model wearing its weights. Do not enable for new runs.
    legacy_injection: bool = False

    # Per-neuron timescales. One global decay gives every neuron the same ~14
    # token horizon; a spread of time constants lets fast neurons carry syntax
    # while slow ones hold the protagonist. This is the mechanism behind S4 and
    # Mamba, and it costs N parameters.
    learn_decay: bool = False
    decay_tau_min: float = 1.5     # fastest neuron's time constant, in tokens
    decay_tau_max: float = 250.0   # slowest

    # k-WTA budget split evenly across timescale bands. With mixed timescales a
    # single global top-k is won permanently by the slow neurons, which retain
    # magnitude by construction, so the fast ones starve. Bands guarantee every
    # timescale a fixed share of the active set. 1 = plain global top-k.
    n_bands: int = 1

    # ------------------------------------------------------------------
    # Latent reasoning. Three mechanisms that could make extra iterations
    # constructive rather than merely tolerated.
    #
    # Measured motivation: the state settles cleanly (step size falls 0.247 ->
    # 0.016 over 24 rounds) but perplexity is minimised at the trained depth and
    # rises on both sides. The dynamics converge; they converge on something
    # uninformative. Depth cannot help until iteration N+1 is more correct than
    # N, and diffusion toward a fixed point gives no reason for that.
    # ------------------------------------------------------------------

    # Loss applied at a sampled intermediate round in addition to the last, so
    # every intermediate state has to be decodable. This is what makes depth
    # monotone: a state that is not yet an answer is penalised for it. One
    # random depth per forward rather than all of them keeps the cost at 2x
    # instead of Rx, and is unbiased in expectation.
    deep_supervision: float = 0.0

    # Adaptive Computation Time. A halting head reads each round's state and
    # emits the probability of stopping there; the output is the halt-weighted
    # mixture over rounds, and `ponder_coef` prices expected depth. Without a
    # price the model always halts late; with too high a price it always halts
    # at one. This is how the model learns to *spend* time instead of having a
    # fixed amount imposed on it.
    halting: bool = False
    ponder_coef: float = 0.01

    # Predictive coding. A reconstruction head must recover the current token
    # from the state, and the injection at each round is scaled by how badly it
    # currently fails to. A state that already explains its input stops being
    # driven; one that does not keeps being corrected. This makes iteration a
    # descent on prediction error, and -- the point -- makes the fixed point the
    # state that explains the input, rather than whatever the diffusion happens
    # to land on.
    predictive_coding: bool = False
    recon_coef: float = 0.1
    pc_floor: float = 0.1        # injection retained when fully unsurprised

    kernel: str = "gaussian"     # gaussian | cauchy
    sigma_init: float = 1.0
    learn_sigma: bool = True
    activation: str = "relu"     # relu | tanh | identity
    homeostasis: bool = True     # RMS-normalize the state after each round

    # Dale's law: each neuron is purely excitatory or purely inhibitory. Without
    # it a decreasing kernel yields all-positive weights, so activation can only
    # accumulate and never cancel.
    dale: bool = True
    dale_hard: bool = False      # straight-through sign() instead of tanh
    dale_tau: float = 1.0

    readout: str = "lowrank"     # lowrank | geometric | dense
    readout_rank: int = 256

    # ablations
    free_weights: bool = False   # edge weights are free params, not g(distance)
    freeze_positions: bool = False
    freeze_input: bool = False

    # kNN graph
    knn_refresh: int = 200       # optimizer steps between graph rebuilds
    knn_chunk: int = 1024        # rows per chunk in the brute-force search
    exclude_self: bool = True

    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.k_min > self.k_max:
            raise ValueError(f"k_min ({self.k_min}) > k_max ({self.k_max})")
        if self.top_n > self.n_neurons:
            raise ValueError(f"top_n ({self.top_n}) > n_neurons ({self.n_neurons})")
        if self.seed_n > self.n_neurons:
            raise ValueError(f"seed_n ({self.seed_n}) > n_neurons ({self.n_neurons})")
        if self.kernel not in ("gaussian", "cauchy"):
            raise ValueError(f"unknown kernel {self.kernel!r}")
        if self.readout not in ("lowrank", "geometric", "dense"):
            raise ValueError(f"unknown readout {self.readout!r}")
        if self.activation not in ("relu", "tanh", "identity"):
            raise ValueError(f"unknown activation {self.activation!r}")
        if self.n_bands < 1:
            raise ValueError(f"n_bands must be >= 1, got {self.n_bands}")
        if self.n_neurons % self.n_bands:
            raise ValueError(
                f"n_neurons ({self.n_neurons}) must divide evenly into "
                f"n_bands ({self.n_bands})"
            )
        if self.top_n % self.n_bands:
            raise ValueError(
                f"top_n ({self.top_n}) must divide evenly into "
                f"n_bands ({self.n_bands})"
            )
        if not 0.0 < self.decay_tau_min <= self.decay_tau_max:
            raise ValueError("require 0 < decay_tau_min <= decay_tau_max")


@dataclass
class BaselineConfig:
    """GRU / transformer reference models."""

    kind: str = "gru"            # gru | transformer
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 4
    n_heads: int = 6             # transformer only
    dropout: float = 0.0
    tie_embeddings: bool = True
    max_seq_len: int = 512       # transformer only


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    steps: int = 20000
    batch_size: int = 32
    seq_len: int = 256
    tbptt_chunk: int = 32        # ignored by the transformer baseline

    lr: float = 3e-3
    lr_positions: float = 3e-3   # coordinates often want a different rate
    min_lr_frac: float = 0.1
    warmup: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta2: float = 0.95

    # auxiliary losses
    balance_coef: float = 1e-2   # Switch-style load balancing on the seed step
    usage_ema: float = 0.99

    # dead-neuron rescue
    reseed_interval: int = 500   # 0 disables
    reseed_frac: float = 0.01    # fraction of least-used neurons to teleport
    reseed_noise: float = 0.1
    reseed_warmup: int = 1000    # no reseeding before this step

    # Cap on distinct training tokens; 0 = the whole corpus. Comparing two
    # parameterizations at convergence on abundant in-distribution data is the
    # worst available test of an inductive bias: a constrained model can only
    # show its advantage where the unconstrained one can overfit or has too
    # little signal. Set this to force many epochs over a small subset.
    train_tokens: int = 0

    eval_interval: int = 1000
    eval_batches: int = 50
    sample_interval: int = 2000
    log_interval: int = 20
    ckpt_interval: int = 2000

    dtype: str = "auto"          # auto | bfloat16 | float16 | float32
    compile: bool = False
    seed: int = 1234


@dataclass
class RunConfig:
    name: str = "brain"
    model: str = "brain"         # brain | gru | transformer
    data_dir: str = "./data"
    out_dir: str = "./runs"
    brain: BrainConfig = field(default_factory=BrainConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def from_dict(d: dict) -> "RunConfig":
        d = dict(d)
        brain = BrainConfig(**d.pop("brain", {}))
        baseline = BaselineConfig(**d.pop("baseline", {}))
        train = TrainConfig(**d.pop("train", {}))
        return RunConfig(brain=brain, baseline=baseline, train=train, **d)

    @staticmethod
    def load(path: str) -> "RunConfig":
        with open(path, encoding="utf-8") as fh:
            return RunConfig.from_dict(json.load(fh))


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------

def _smoke() -> RunConfig:
    """Tiny enough to run a few hundred steps on a laptop CPU."""
    cfg = RunConfig(name="smoke", data_dir="./data_smoke")
    cfg.brain = BrainConfig(
        vocab_size=512, n_neurons=1024, d_space=8, seed_n=16, top_n=64,
        k_min=2, k_max=8, rounds=2, readout_rank=64, knn_refresh=50,
        knn_chunk=256,
    )
    cfg.baseline = BaselineConfig(vocab_size=512, d_model=96, n_layers=2, n_heads=3)
    cfg.train = TrainConfig(
        steps=60, batch_size=4, seq_len=32, tbptt_chunk=16, warmup=5,
        eval_interval=30, eval_batches=3, sample_interval=30, log_interval=10,
        ckpt_interval=1000, reseed_interval=20, reseed_warmup=20,
        dtype="float32",
    )
    return cfg


def _ref() -> RunConfig:
    """Single-5090 reference run. ~6-8 h at these settings."""
    cfg = RunConfig(name="brain-d16")
    cfg.brain = BrainConfig(n_neurons=32768, d_space=16)
    cfg.train = TrainConfig(steps=20000, batch_size=32, seq_len=256)
    return cfg


def _ref2() -> RunConfig:
    """Reference run with a working leaky integrator and mixed timescales.

    `ref` writes each token into the state ~12x too weakly, so the k-WTA
    discards most of it and the effective context is one clause. This preset
    fixes the injection scale and gives the neurons a spread of time constants
    from ~1.5 to ~250 tokens instead of a single 14-token horizon.
    """
    cfg = RunConfig(name="brain2-d16")
    cfg.brain = BrainConfig(
        n_neurons=32768, d_space=16, top_n=1024, seed_n=128,
        learn_decay=True, n_bands=4, input_gate=0.35,
    )
    cfg.train = TrainConfig(steps=20000, batch_size=32, seq_len=256)
    return cfg


def _big() -> RunConfig:
    cfg = RunConfig(name="brain-big")
    cfg.brain = BrainConfig(n_neurons=65536, d_space=32, top_n=1024, readout_rank=384)
    cfg.train = TrainConfig(steps=30000, batch_size=24, seq_len=256, grad_clip=1.0)
    cfg.brain.grad_checkpoint = True
    return cfg


def _free_weights() -> RunConfig:
    """Ablation: same topology and dynamics, but edge weights are free
    parameters. Isolates whether the geometry is an inductive bias or just an
    expensive compression of a weight matrix."""
    cfg = _ref()
    cfg.name = "ablate-freeweight"
    cfg.brain.free_weights = True
    cfg.brain.knn_refresh = 0        # topology frozen: no positions to follow
    return cfg


def _frozen() -> RunConfig:
    """Control: an echo state network. Positions and signs are random and
    never move; only the readout learns. If the full model does not clearly
    beat this, nothing in the geometry is being learned."""
    cfg = _ref()
    cfg.name = "control-frozen"
    cfg.brain.freeze_positions = True
    cfg.brain.freeze_input = True
    cfg.brain.knn_refresh = 0
    cfg.train.balance_coef = 0.0
    cfg.train.reseed_interval = 0
    return cfg


def _think() -> RunConfig:
    """Randomized latent depth: the state update must survive being iterated.

    Same model as ref2, trained at a depth drawn from [1, 8] rather than fixed
    at 3, and with k_max cut to 8 since mean_k sits at ~4.3 and the other 24
    candidate edges are computed only to be masked away.

    The claim to test afterwards is `evaluate.py --scaling`: perplexity should
    keep falling as inference rounds increase, including past the training
    range. That is compute decoupled from tokens emitted, measured rather than
    asserted.
    """
    cfg = _ref2()
    cfg.name = "think-d16"
    cfg.brain.rounds = 4
    cfg.brain.rounds_min = 1
    cfg.brain.rounds_max = 8
    cfg.brain.k_max = 8
    cfg.brain.k_min = 1
    cfg.brain.inject_every_round = True
    return cfg


def _think_deep() -> RunConfig:
    """`think` plus deep supervision. Cheapest of the three, most likely to work."""
    cfg = _think()
    cfg.name = "think-deep"
    cfg.brain.deep_supervision = 0.5
    return cfg


def _think_act() -> RunConfig:
    """`think` plus a halting head and a ponder cost."""
    cfg = _think()
    cfg.name = "think-act"
    cfg.brain.halting = True
    cfg.brain.ponder_coef = 0.01
    cfg.train.tbptt_chunk = 16     # halting keeps a readout per round in the graph
    return cfg


def _think_pc() -> RunConfig:
    """`think` plus predictive-coding dynamics."""
    cfg = _think()
    cfg.name = "think-pc"
    cfg.brain.predictive_coding = True
    cfg.brain.recon_coef = 0.1
    cfg.train.tbptt_chunk = 16
    return cfg


def _think_all() -> RunConfig:
    """All three at once.

    Run the single-mechanism presets first. If this wins, it cannot say which
    mechanism won, and if it loses it cannot say which one is at fault.
    """
    cfg = _think()
    cfg.name = "think-all"
    cfg.brain.deep_supervision = 0.5
    cfg.brain.halting = True
    cfg.brain.predictive_coding = True
    cfg.train.tbptt_chunk = 8
    return cfg


def _frozen2() -> RunConfig:
    """The echo-state control for `ref2`. A control has to share the
    architecture it controls for, so this tracks ref2 rather than ref."""
    cfg = _ref2()
    cfg.name = "control-frozen"
    cfg.brain.freeze_positions = True
    cfg.brain.freeze_input = True
    cfg.brain.knn_refresh = 0
    cfg.train.balance_coef = 0.0
    cfg.train.reseed_interval = 0
    return cfg


def _free_weights2() -> RunConfig:
    """The free-weight ablation for `ref2`.

    Now the most important rung in the ladder. Propagation was measured to be
    worth ~16x perplexity, so the connectome is load-bearing -- but that leaves
    open whether the work is done by the *distance kernel* or merely by having
    a sparse topology. This isolates exactly that.
    """
    cfg = _ref2()
    cfg.name = "ablate-freeweight"
    cfg.brain.free_weights = True
    cfg.brain.knn_refresh = 0
    return cfg


def _gru() -> RunConfig:
    cfg = RunConfig(name="baseline-gru", model="gru")
    cfg.baseline = BaselineConfig(kind="gru", d_model=384, n_layers=3)
    cfg.train = TrainConfig(steps=20000, batch_size=32, seq_len=256, lr=1e-3)
    return cfg


def _transformer() -> RunConfig:
    cfg = RunConfig(name="baseline-transformer", model="transformer")
    cfg.baseline = BaselineConfig(kind="transformer", d_model=384, n_layers=6, n_heads=6)
    cfg.train = TrainConfig(steps=20000, batch_size=32, seq_len=256, lr=1e-3)
    return cfg


PRESETS = {
    "smoke": _smoke,
    "ref": _ref,
    "ref2": _ref2,
    "big": _big,
    "freeweight": _free_weights,
    "frozen": _frozen,
    "freeweight2": _free_weights2,
    "frozen2": _frozen2,
    "think": _think,
    "think-deep": _think_deep,
    "think-act": _think_act,
    "think-pc": _think_pc,
    "think-all": _think_all,
    "gru": _gru,
    "transformer": _transformer,
}


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------

_SECTIONS = {"brain": BrainConfig, "baseline": BaselineConfig, "train": TrainConfig}

# Deliberately not exposed on the CLI: `vocab_size` is declared by both model
# sections and is overwritten from the dataset's meta.json at startup, so a flag
# would be silently ignored. Everything else must have a unique name.
_SKIP_CLI = {"vocab_size"}


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Expose every leaf field as `--field-name`, plus `--preset`.

    Section fields are flattened; a name collision between sections is a
    programming error and raises at import time rather than silently binding
    to whichever section came last.
    """
    parser.add_argument("--preset", default="ref", choices=sorted(PRESETS))
    parser.add_argument("--name", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--model", default=None, choices=["brain", "gru", "transformer"])

    seen: dict[str, str] = {}
    for section, klass in _SECTIONS.items():
        for f in dataclasses.fields(klass):
            if f.name in _SKIP_CLI:
                continue
            if f.name in seen:
                raise RuntimeError(
                    f"config field {f.name!r} declared in both "
                    f"{seen[f.name]!r} and {section!r}; rename one"
                )
            seen[f.name] = section
            flag = "--" + f.name.replace("_", "-")
            if f.type is bool or f.type == "bool":
                parser.add_argument(flag, dest=f.name, default=None,
                                    type=lambda s: s.lower() in ("1", "true", "yes", "y"))
            else:
                ftype = {int: int, float: float, str: str}.get(f.type, str)
                if isinstance(f.type, str):
                    ftype = {"int": int, "float": float, "str": str}.get(f.type, str)
                parser.add_argument(flag, dest=f.name, default=None, type=ftype)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    cfg = PRESETS[args.preset]()
    for key in ("name", "data_dir", "out_dir", "model"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)

    for section, klass in _SECTIONS.items():
        target = getattr(cfg, section)
        for f in dataclasses.fields(klass):
            val = getattr(args, f.name, None)
            if val is not None:
                setattr(target, f.name, val)

    # keep vocab in sync across sections; train.py overwrites both from meta.json
    cfg.baseline.kind = cfg.model if cfg.model in ("gru", "transformer") else cfg.baseline.kind
    cfg.brain.__post_init__()
    return cfg
