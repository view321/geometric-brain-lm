"""Run the full ablation ladder and report it.

The experiment is comparative, so this is the actual deliverable: a d-sweep of
the geometric model against the four controls that decide whether any of it
means anything.

    python sweep.py --plan --budget-hours 44        # show the schedule, run nothing
    python sweep.py --budget-hours 44               # calibrate, then run everything
    python sweep.py --only control-frozen           # just the cheap sanity check
    python sweep.py --report-only                   # re-render the table

Every model is trained for the SAME number of steps, not the same wall-clock
time. Equal tokens seen is the fair comparison; equal wall time would quietly
hand the fastest architecture an advantage. The budget therefore sets one step
count for the whole ladder, derived from measured throughput.

Order matters. `control-frozen` runs first and is the cheapest: it freezes the
coordinates and trains only the readout, which makes it an echo state network.
If the full model does not clearly beat it, nothing in the geometry is being
learned and the rest of the sweep is a waste of two days.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------

def iso_param_ranks(d_values, n_neurons: int, ref_d: int, vocab: int) -> dict:
    """Readout rank per d that keeps every rung at the reference budget."""
    sys.path.insert(0, HERE)
    from config import PRESETS
    from model import param_count, solve_readout_rank

    ref = PRESETS["ref"]().brain
    ref.n_neurons, ref.d_space, ref.vocab_size = n_neurons, ref_d, vocab
    target = param_count(ref)

    ranks = {}
    for d in d_values:
        cfg = PRESETS["ref"]().brain
        cfg.n_neurons, cfg.d_space, cfg.vocab_size = n_neurons, d, vocab
        ranks[d] = solve_readout_rank(cfg, target)
    return ranks


def build_ladder(d_values, n_neurons: int, ref_d: int, iso_ranks: dict | None = None):
    """Each rung: (name, preset, extra CLI flags, one-line purpose)."""
    # The control has to be the reference architecture with the positions
    # frozen, not a differently-shaped model. Inheriting d from the preset
    # instead of ref_d would make it a control for a run nobody is doing.
    rungs = [
        ("control-frozen", "frozen", ["--d-space", str(ref_d)],
         "echo state network: positions frozen, only the readout learns"),
    ]
    for d in d_values:
        flags = ["--d-space", str(d)]
        note = ""
        if iso_ranks:
            flags += ["--readout-rank", str(iso_ranks[d])]
            note = f", readout rank {iso_ranks[d]} to hold the parameter budget"
        rungs.append((
            f"brain-d{d}", "ref", flags,
            f"geometric model, {d}-dimensional brain space{note}",
        ))
    rungs += [
        ("ablate-freeweight", "freeweight", ["--d-space", str(ref_d)],
         "same topology and dynamics, edge weights are free parameters"),
        ("baseline-gru", "gru", [], "parameter-matched GRU: the fair fight"),
        ("baseline-transformer", "transformer", [],
         "parameter-matched transformer: the ceiling, for context only"),
    ]
    out = []
    for name, preset, flags, why in rungs:
        if preset in ("ref", "frozen", "freeweight"):
            flags = flags + ["--n-neurons", str(n_neurons)]
        out.append((name, preset, flags, why))
    return out


def common_flags(args) -> list[str]:
    flags = ["--data-dir", args.data_dir, "--out-dir", args.out_dir,
             "--device", args.device, "--batch-size", str(args.batch_size),
             "--seq-len", str(args.seq_len)]
    if args.dtype:
        flags += ["--dtype", args.dtype]
    if args.extra:
        flags += args.extra.split()
    return flags


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def run_training(name, preset, flags, args, steps, match_params=0, quiet=False):
    cmd = [PY, os.path.join(HERE, "train.py"), "--preset", preset,
           "--name", name, "--steps", str(steps)] + flags + common_flags(args)
    if match_params:
        cmd += ["--match-params", str(match_params)]
    if quiet:
        # Calibration measures throughput only. Without capping the final
        # evaluation it would score the entire validation split after 30 steps,
        # which costs more than the run being measured.
        cmd += ["--no-samples", "--eval-interval", "10000000",
                "--ckpt-interval", "10000000", "--final-eval-batches", "2"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(cmd, cwd=HERE, env=env,
                          capture_output=quiet, text=True)
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
        raise RuntimeError(f"{name} failed (exit {proc.returncode})\n{tail}")
    return proc


def measure_throughput(name, preset, flags, args, steps=30) -> float:
    """Tokens/second for one rung, from a short real run.

    Measured rather than estimated: the geometric model is launch-bound on
    hundreds of tiny sequential kernels, so its throughput does not follow from
    FLOP counts and varies a lot with N, d, and rounds.
    """
    run_training(f"_calib-{name}", preset, flags, args, steps, quiet=True)
    log = os.path.join(args.out_dir, f"_calib-{name}", "log.jsonl")
    rates = []
    with open(log, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if "tok_per_s" in rec:
                rates.append(rec["tok_per_s"])
    if not rates:
        raise RuntimeError(f"no throughput recorded for {name}")
    return rates[-1]      # last window: warmed up, excludes startup


def brain_param_count(args, n_neurons: int, d: int) -> int:
    """Parameter count of the reference brain model, for matching baselines."""
    sys.path.insert(0, HERE)
    from config import PRESETS
    from data import TokenStream
    from model import param_count

    cfg = PRESETS["ref"]().brain
    cfg.n_neurons = n_neurons
    cfg.d_space = d
    cfg.vocab_size = TokenStream(args.data_dir).vocab_size
    return param_count(cfg)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def collect(out_dir: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("_calib-"):
            continue
        path = os.path.join(out_dir, name, "summary.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                rows.append(json.load(fh))
    return rows


def render(rows: list[dict]) -> str:
    if not rows:
        return "no completed runs found\n"
    rows = sorted(rows, key=lambda r: r["final"]["loss"])
    frozen = next((r for r in rows if r["name"] == "control-frozen"), None)
    gru = next((r for r in rows if r["name"] == "baseline-gru"), None)

    head = ("| run | params | val ppl | bpb | vs GRU | vs frozen | mean K | "
            "uniformity | edge/ambient | h |")
    sep = "|" + "---|" * 11
    lines = [head, sep]
    for r in rows:
        f = r["final"]
        vs_gru = (f"{f['ppl'] / gru['final']['ppl']:.2f}x"
                  if gru and gru["final"]["ppl"] else "-")
        vs_fro = (f"{f['ppl'] / frozen['final']['ppl']:.2f}x"
                  if frozen and frozen["final"]["ppl"] else "-")
        lines.append(
            f"| {r['name']} | {r['params']['total']:,} | {f['ppl']:.2f} | "
            f"{f.get('bpb', float('nan')):.4f} | {vs_gru} | {vs_fro} | "
            f"{f.get('mean_k', float('nan')):.1f} | "
            f"{f.get('usage_uniformity', float('nan')):.3f} | "
            f"{f.get('ratio', float('nan')):.3f} | {r['wall_s'] / 3600:.1f} |"
        )

    out = ["", "## Results", "", *lines, "", "### Reading this table", ""]
    if frozen:
        best = min((r for r in rows if r["name"].startswith("brain-d")),
                   key=lambda r: r["final"]["loss"], default=None)
        if best:
            ratio = best["final"]["ppl"] / frozen["final"]["ppl"]
            if ratio > 0.95:
                out.append(
                    f"- **The frozen control is not being beaten** ({best['name']} is "
                    f"{ratio:.2f}x its perplexity). Training is not moving the geometry "
                    f"anywhere useful; this is a reservoir. Stop here and debug before "
                    f"reading anything else into the sweep."
                )
            else:
                out.append(
                    f"- Best geometric model ({best['name']}) reaches {ratio:.2f}x the "
                    f"frozen control's perplexity, so the learned coordinates are "
                    f"carrying real signal."
                )
    free = next((r for r in rows if r["name"] == "ablate-freeweight"), None)
    best_d = min((r for r in rows if r["name"].startswith("brain-d")),
                 key=lambda r: r["final"]["loss"], default=None)
    if free and best_d:
        ratio = best_d["final"]["ppl"] / free["final"]["ppl"]
        verdict = ("the distance kernel is a useful inductive bias, not just a "
                   "compression scheme" if ratio < 0.98 else
                   "free weights do at least as well, so the geometry is buying "
                   "compression rather than structure")
        out.append(f"- Geometry vs free weights: {ratio:.2f}x -- {verdict}.")

    d_runs = sorted((r for r in rows if r["name"].startswith("brain-d")),
                    key=lambda r: int(r["name"].split("d")[-1]))
    if len(d_runs) >= 3:
        curve = ", ".join(f"d={r['name'].split('d')[-1]}: {r['final']['ppl']:.1f}"
                          for r in d_runs)
        out.append(f"- Dimension sweep: {curve}")
        out.append(
            "  A knee here is the headline finding: if a small d nearly matches a "
            "large one, language's recurrent structure is low-dimensional and "
            "geometric. If perplexity keeps falling with d, the kernel is just an "
            "expensive way to write down a weight matrix. Compare against the "
            "params column -- larger d also buys parameters, so read the two "
            "together."
        )
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget-hours", type=float, default=44.0)
    ap.add_argument("--d-values", default="2,4,8,16,32,64")
    ap.add_argument("--ref-d", type=int, default=16)
    ap.add_argument("--iso-params", action="store_true",
                    help="hold every d at the same parameter budget by trading "
                         "against the readout rank; removes the size confound "
                         "from the dimension sweep")
    ap.add_argument("--n-neurons", type=int, default=32768)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steps", type=int, default=0,
                    help="fixed step count; skips calibration")
    ap.add_argument("--calib-steps", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--only", default="", help="comma-separated rung names")
    ap.add_argument("--skip", default="", help="comma-separated rung names")
    ap.add_argument("--plan", action="store_true", help="print the schedule and exit")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--extra", default="", help="extra flags passed to every train.py")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "REPORT.md")

    if args.report_only:
        text = render(collect(args.out_dir))
        print(text)
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return

    d_values = [int(v) for v in args.d_values.split(",") if v.strip()]

    iso_ranks = None
    if args.iso_params:
        sys.path.insert(0, HERE)
        from data import TokenStream
        vocab = TokenStream(args.data_dir).vocab_size
        iso_ranks = iso_param_ranks(d_values, args.n_neurons, args.ref_d, vocab)
        print("[iso-params] readout rank per d, holding the budget at the "
              f"d={args.ref_d} model:")
        print("  " + "  ".join(f"d={d}:r={r}" for d, r in iso_ranks.items()) + "\n")

    ladder = build_ladder(d_values, args.n_neurons, args.ref_d, iso_ranks)
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        ladder = [r for r in ladder if r[0] in keep]
    if args.skip:
        drop = {s.strip() for s in args.skip.split(",")}
        ladder = [r for r in ladder if r[0] not in drop]
    if not ladder:
        raise SystemExit("ladder is empty after --only/--skip filtering")

    tokens_per_step = args.batch_size * args.seq_len

    # ---- schedule ----------------------------------------------------
    if args.steps:
        steps = args.steps
        rates = {name: None for name, *_ in ladder}
    else:
        print(f"[calibrate] {args.calib_steps} steps per rung to measure throughput\n")
        rates = {}
        for name, preset, flags, _why in ladder:
            t0 = time.time()
            rates[name] = measure_throughput(name, preset, flags, args, args.calib_steps)
            print(f"  {name:<24} {rates[name]:>9,.0f} tok/s   "
                  f"({time.time() - t0:.0f}s to measure)")
        seconds_per_step = sum(tokens_per_step / r for r in rates.values())
        steps = int(args.budget_hours * 3600 / max(seconds_per_step, 1e-9))
        steps = max(200, min(steps, args.max_steps))
        print()

    print(f"[plan] {len(ladder)} runs x {steps:,} steps "
          f"= {steps * tokens_per_step / 1e6:.0f}M tokens each")
    total = 0.0
    measured = True
    for name, _preset, _flags, why in ladder:
        r = rates.get(name)
        if r:
            hours = steps * tokens_per_step / r / 3600
            total += hours
            print(f"  {name:<24} {hours:>6.1f} h   {why}")
        else:
            measured = False
            print(f"  {name:<24} {'  --  ':>6}     {why}")
    if measured:
        print(f"  {'TOTAL':<24} {total:>6.1f} h  (budget {args.budget_hours:.1f} h)")
    else:
        print(f"  {'TOTAL':<24} {'  --  ':>6}     (fixed --steps, no calibration run)")

    if args.plan:
        return

    # ---- match baseline sizes ----------------------------------------
    try:
        target = brain_param_count(args, args.n_neurons, args.ref_d)
        print(f"\n[match] baselines will be sized to {target:,} params "
              f"(brain N={args.n_neurons}, d={args.ref_d})")
    except Exception as exc:                                  # noqa: BLE001
        target = 0
        print(f"\n[match] could not size baselines automatically ({exc}); "
              f"using preset d_model")

    # ---- run ----------------------------------------------------------
    print()
    results = []
    for i, (name, preset, flags, why) in enumerate(ladder, 1):
        print(f"\n{'=' * 72}\n[{i}/{len(ladder)}] {name} -- {why}\n{'=' * 72}")
        t0 = time.time()
        match = target if name.startswith("baseline-") else 0
        try:
            run_training(name, preset, flags, args, steps, match_params=match)
            results.append(name)
        except Exception as exc:                              # noqa: BLE001
            # One rung failing must not throw away the rungs already finished.
            print(f"[FAILED] {name}: {exc}")
        print(f"[{name}] {(time.time() - t0) / 3600:.2f} h")

        text = render(collect(args.out_dir))
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(text)

    print(render(collect(args.out_dir)))
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
