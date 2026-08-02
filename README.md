# Geometric brain LM

A language model whose recurrent weight matrix is never stored. Instead, each
of N neurons owns two points in R^d — an axon point and a dendrite point — and
the strength of the connection i -> j is a decreasing kernel of the distance
between `axon[i]` and `dendrite[j]`. Tokens enter by landing at a point in that
space and activating their nearest neurons; activation spreads along the
strongest outbound edges for a few rounds; a sparse readout turns the resulting
state into a next-token distribution.

Stated in ordinary ML terms:

> an RNN whose hidden state is a sparse k-winners-take-all activation vector
> over N units, and whose recurrent weight matrix is generated on demand as a
> kernel over a learned point cloud.

**The hypothesis under test.** N neurons times d coordinates generate an N x N
weight matrix, and Euclidean distance matrices are heavily rank-constrained. So
the model provably *cannot* express an arbitrary recurrent weight matrix unless
d approaches N. The question is whether that constraint is a useful prior —
whether language's recurrent structure is low-dimensional and geometric — or
simply a cap on capacity. That is what the dimension sweep measures, and it is
the only reason to build this rather than a normal RNN.

Everything else here (top-k routing, sparse message passing, adaptive
computation, trained readout on a fixed-dynamics reservoir) is well-trodden and
known to work. The distance-kernel weight parameterization is the one novel
piece, so the experiment is designed to isolate it.

---

## Quickstart on the GPU box

Only `torch`, `numpy`, and `tokenizers` are needed to train. `datasets` is
optional and used once, during data prep. There is no FAISS dependency: the kNN
graph is a chunked brute-force scan in torch, which is under a second at this
scale and installs everywhere.

```bash
python prep_data.py --vocab-size 8192
```

That downloads TinyStories, trains an 8k byte-level BPE, and writes
`data/{train,val}.bin` plus `tok8k.json` and `meta.json`. It can also run on a
machine with no GPU and the artifacts copied over — that is the intended
workflow here. Verify the port with:

```bash
python test_shapes.py
```

45 invariant checks, a few seconds on CPU. Run this first on the GPU box; it
catches the entire class of bug that otherwise trains for two days and reports a
number that means nothing.

To exercise the whole pipeline without downloading anything — synthetic corpus,
tiny model, a minute on a laptop CPU — use the smoke path:

```bash
python prep_data.py --smoke --data-dir ./data_smoke --vocab-size 512
python train.py --preset smoke --data-dir ./data_smoke --device cpu
```

Then the ladder. Calibrate first so the schedule reflects real throughput:

```bash
python sweep.py --budget-hours 44 --iso-params --plan
```

That measures tokens/sec per rung with 30-step trial runs, derives a step count
that fits the budget, prints the schedule, and exits. Drop `--plan` to run it.

---

## The ladder

Ten runs, all trained for the **same number of steps** — equal tokens seen is
the fair comparison; equal wall-clock would quietly favour whichever
architecture is fastest.

| rung | question it answers |
|---|---|
| `control-frozen` | Coordinates frozen at random, only the readout trains — an echo state network. **Run this first.** Echo state networks are embarrassingly competitive at small scale. If the full model doesn't clearly beat it, training isn't moving the geometry and everything downstream is noise. |
| `brain-d{2,4,8,16,32,64}` | The headline sweep. Where is the knee? |
| `ablate-freeweight` | Identical topology and dynamics, but edge weights are free parameters instead of `g(distance)`. Isolates whether the geometry is an inductive bias or merely a compression scheme you are paying for. |
| `baseline-gru` | The fair fight. This model is an RNN, so a parameter-matched RNN is the number to beat. |
| `baseline-transformer` | The ceiling, for context only. Losing to it is expected and uninformative. |

The two ablations are `BrainLM` itself under different config flags, not
reimplementations — an ablation that shares no code with the thing it ablates
proves nothing.

## Reading the results

`runs/REPORT.md` is rewritten after every rung, so a sweep that dies at hour 30
still leaves a readable table. It also states the two verdicts in prose.

What counts as "deserves further study":

- **Perplexity within ~20% of the parameter-matched GRU.** Genuinely
  interesting given how constrained the weight parameterization is.
- **A knee in the dimension sweep** — d=8 nearly matching d=64 would be a real
  finding about the dimensionality of language's recurrent structure. If
  perplexity keeps falling all the way to d=64, the kernel is just an expensive
  way to write down a weight matrix.
- **Beating `ablate-freeweight`.** Then the distance kernel is doing work that
  free parameters cannot, which is the strongest possible result here.
- **`usage_uniformity` high with `ratio` well below 1.** Neurons broadly used,
  edges much shorter than the ambient scale: the point cloud has organised.
- **`k_surprisal_corr` positive.** Fan-out tracks token difficulty — the
  adaptive-compute story.

And the stopping condition: **if the best model only matches `control-frozen`,
it is a reservoir.** Reservoirs are known. Stop.

## Files

| file | what it is |
|---|---|
| `config.py` | Every hyperparameter, the presets, CLI plumbing |
| `model.py` | `BrainLM` — kernel weights, k-WTA, adaptive-K propagation, readouts, reseeding |
| `knn.py` | Chunked brute-force kNN graph, plus connectome health stats |
| `baselines.py` | GRU and transformer reference models, parameter matching |
| `train.py` | One run: TBPTT, load balancing, reseeding, checkpoint/resume |
| `evaluate.py` | Held-out loss, bits-per-byte, sampling, geometry diagnostics |
| `sweep.py` | The ladder, throughput calibration, report rendering |
| `prep_data.py`, `data.py` | Tokenizer and packed token bins |
| `test_shapes.py` | Invariants worth pinning |

## Knobs that matter

`--d-space` is the experiment; everything else is support. After that:
`--n-neurons`, `--top-n` (k-WTA width, ~1–4% of N), `--rounds` (propagation
depth per token; cost is linear in it), `--k-min/--k-max` (fan-out range), and
`--tbptt-chunk` (memory against gradient horizon).

Useful invocations:

```bash
python train.py --preset ref --d-space 16 --name brain-d16
python evaluate.py --ckpt runs/brain-d16/best.pt --full --diagnostics --sample 4
python sweep.py --only control-frozen --budget-hours 2
python sweep.py --report-only
```

Every run writes `config.json`, `log.jsonl`, `summary.json`, `best.pt`, and
`last.pt`. `train.py --resume` picks up from `last.pt`.

## Things that were fixed, and would otherwise bite

These are the failure modes the design already accounts for. They are recorded
because each one is silent — the run completes and the number is wrong.

- **The state has to be able to hear its input.** Homeostasis normalises the
  state to fixed RMS, so each active neuron carries `sqrt(N / top_n)` — about 12x
  a raw kernel value at the reference config. Adding an un-normalised seed to
  that made the k-WTA discard **93% of every token's seed neurons**, so the state
  coasted on whatever entered first instead of integrating the sequence. The
  seed is now normalised to the state's scale first. Measure it with
  `evaluate.py --memory`: `current_token_entry` near zero means the model has
  stopped listening.
- **The write rate must be coupled to the decay.** Fixing the injection scale
  with a single global gate swings the failure the other way — retention
  collapsed from 14 tokens to 3, because a gate large enough for fast neurons to
  track the current token also overwrites the slow ones. Each neuron now writes
  at `1 - decay_i`, making the state a bank of exponential moving averages at
  different timescales rather than one blur.
- **A single point per neuron makes the kernel symmetric**, so `w_ij == w_ji`
  and no directed information can flow. Hence two coordinates per neuron. (This
  makes the model structurally query/key attention with a distance kernel in
  place of a dot product.)
- **A decreasing kernel gives all-positive weights**, so activation can only
  accumulate and never cancel. Dale's law — one learned sign per neuron —
  restores inhibition for N extra parameters.
- **Weight decay on coordinates collapses the point cloud toward the origin.**
  `param_groups()` enforces zero decay on all geometry rather than leaving it
  to the caller.
- **Dead neurons are self-reinforcing.** Under hard k-WTA an unselected neuron
  gets no gradient, so it never moves, so it is never selected — VQ-VAE
  codebook collapse in different clothing. Three defences: a load-balancing
  auxiliary loss, periodic relocation of the least-used neurons next to
  heavily-used ones, and clearing the Adam moments of relocated rows so stale
  momentum doesn't drag them straight back.
- **bf16 drifts across a long recurrence.** The state is pinned to fp32 at the
  homeostasis step, which is where it re-enters the loop; the ops that actually
  want bf16 (the seed distance matmul, the readout projection) are upstream and
  keep it.
- **Edge weights recomputed per token would be identical and 768x more
  expensive.** Coordinates only move on an optimizer step, so `edge_weights()`
  runs once per forward and is reused by every token and round.
- **Sigma has to be calibrated to the realised edge lengths.** Nearest-neighbour
  distance shrinks with d, so a fixed sigma makes every weight 1.0 or 0.0
  depending on which d you picked. `calibrate_sigma()` runs at init.
- **The frozen control must share the reference architecture.** A control at a
  different d is a control for a run nobody is doing.

## Known confounds

- **The readout dominates the parameter count** (~10.5M of ~11.7M at N=32768,
  r=256). Every rung shares it verbatim so comparisons are clean, but the
  absolute parameter count is not a measure of the recurrent model's size.
- **Larger d buys parameters as well as dimensions** — 43% more at d=64 than
  d=2. `--iso-params` removes this by trading readout rank against coordinate
  count, holding every rung within ~0.2% of the reference budget. Prefer it for
  the headline sweep; without it, read the perplexity and params columns
  together.
- **Token perplexity is comparable within this study only**, since every model
  shares the 8k tokenizer. A model with a coarser vocabulary predicts fewer,
  harder tokens, so use bits-per-byte for any comparison across tokenizers.
- **The dimension sweep tunes nothing per d.** Learning rate, sigma
  initialisation, and k-WTA width are held fixed across the sweep. A knee could
  in principle be a tuning artifact; the honest follow-up for whichever d wins
  is a short LR sweep at that d before believing the shape of the curve.
- **Memory dynamics must be measured on a trained model.** `--memory` on random
  initial positions characterises the initialization, not the architecture: a
  trained point cloud has attractor structure a random one does not, and that is
  precisely what would change retention. Attempts to identify what bounds the
  horizon from untrained weights produced non-monotonic noise across every knob
  tried (capacity, time constants, propagation rounds). Treat the horizon as an
  empirical property of each checkpoint, not something predictable from config.

## Effective context

The recurrence, not the training window, is what bounds how far back the model
can refer. `python evaluate.py --ckpt <ckpt> --memory` reports the half-life of
a token's footprint in the state, per timescale band, plus how much of the
current token survives into a populated state.

This matters because the two are easily confused. Training at `seq_len 256`
does not mean the model uses 256 tokens of context — if the half-life is 15,
it is using about 6% of its window, and generated text will be locally fluent
while losing entities across a sentence. Check the horizon before concluding
anything from a sample that reads incoherent.

On a banded model read `band_entry`, not the pooled `current_token_entry`. Slow
bands decline the current token by design, so pooling them with the fast bands
makes a healthy model look like it has stopped listening. A healthy banded model
is monotonic in both: entry falls and half-life rises as you go from the fast
band to the slow one.

Checkpoints trained before the injection fix load automatically with
`legacy_injection=True` and print a notice. Their metrics then describe the
model as it was trained rather than new dynamics wearing old weights.

The `ref2` preset is `ref` with the injection scale fixed, a spread of time
constants from ~1.5 to ~250 tokens, and the k-WTA budget split across four
timescale bands so that slow neurons are not evicted by fast ones.
