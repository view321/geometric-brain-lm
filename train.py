"""Train one model. One invocation = one rung of the ablation ladder.

    python train.py --preset smoke --data-dir ./data_smoke
    python train.py --preset ref --d-space 16 --name brain-d16
    python train.py --preset frozen          # the echo-state control
    python train.py --preset gru --match-params 11800000

Truncated BPTT: a batch of `seq_len` tokens is split into `tbptt_chunk` pieces,
each backpropagated and the state detached between them, with the gradients
accumulated into a single optimizer step. Full BPTT through 256 tokens x 3
propagation rounds is 768 sequential graph levels, which neither fits nor
trains stably. The transformer baseline ignores this and sees the full window.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from checkpoint import build_model, resume, save_checkpoint
from config import add_config_args, config_from_args
from data import TokenStream, load_tokenizer
from evaluate import evaluate, generate, geometry_diagnostics
from model import BrainLM


# --------------------------------------------------------------------------

def pick_dtype(name: str, device: torch.device):
    if name == "float32":
        return None
    if name in ("bfloat16", "float16"):
        return getattr(torch, name)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def lr_at(step: int, tc) -> float:
    if step < tc.warmup:
        return (step + 1) / max(tc.warmup, 1)
    progress = (step - tc.warmup) / max(tc.steps - tc.warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tc.min_lr_frac + (1.0 - tc.min_lr_frac) * cosine


def detach_state(state):
    if state is None:
        return None
    if isinstance(state, (tuple, list)):
        return type(state)(detach_state(s) for s in state)
    return state.detach()


class Logger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self.fh.write(json.dumps(record) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


# --------------------------------------------------------------------------

def train(cfg, args) -> dict:
    torch.manual_seed(cfg.train.seed)
    device = torch.device(args.device)
    run_dir = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(run_dir, exist_ok=True)

    stream = TokenStream(cfg.data_dir, device=str(device), seed=cfg.train.seed)
    stream.train_limit = cfg.train.train_tokens
    tokenizer = load_tokenizer(cfg.data_dir)
    cfg.brain.vocab_size = stream.vocab_size
    cfg.baseline.vocab_size = stream.vocab_size
    cfg.baseline.max_seq_len = max(cfg.baseline.max_seq_len, cfg.train.seq_len)

    if args.match_params and cfg.model != "brain":
        from baselines import match_params
        cfg.baseline = match_params(cfg.baseline, args.match_params)
        print(f"[match] d_model={cfg.baseline.d_model} for target {args.match_params:,}")

    model = build_model(cfg).to(device)
    counts = model.n_params()
    is_brain = isinstance(model, BrainLM)

    tc = cfg.train
    optimizer = torch.optim.AdamW(
        model.param_groups(tc.lr, tc.lr_positions, tc.weight_decay),
        betas=(0.9, tc.beta2), eps=1e-8,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]

    autocast_dtype = pick_dtype(tc.dtype, device)
    cfg.save(os.path.join(run_dir, "config.json"))
    logger = Logger(os.path.join(run_dir, "log.jsonl"))

    start_step = 0
    ckpt_last = os.path.join(run_dir, "last.pt")
    if args.resume and os.path.exists(ckpt_last):
        start_step = resume(ckpt_last, model, optimizer, str(device))
        print(f"[resume] continuing from step {start_step}")

    print(f"[model] {cfg.model} '{cfg.name}'  params={counts['total']:,} "
          f"(readout {counts['readout']:,}, geometry {counts['geometry']:,})")
    seen = tc.steps * tc.batch_size * tc.seq_len
    pool = min(stream.n_tokens("train"), tc.train_tokens or stream.n_tokens("train"))
    print(f"[data]  vocab={stream.vocab_size} train={stream.n_tokens('train'):,} tok"
          f"  val={stream.n_tokens('val'):,} tok")
    print(f"[data]  sampling from {pool:,} tokens; the run will draw {seen:,} "
          f"({seen / pool:.1f} epochs)"
          + ("  <- single pass: no overfitting pressure, so a constrained "
             "parameterization cannot show an advantage here"
             if seen < pool else ""))
    print(f"[opt]   dtype={autocast_dtype or 'float32'} bs={tc.batch_size} "
          f"seq={tc.seq_len} tbptt={tc.tbptt_chunk} steps={tc.steps}")
    if is_brain:
        print(f"[brain] N={cfg.brain.n_neurons} d={cfg.brain.d_space} "
              f"top_n={cfg.brain.top_n} K={cfg.brain.k_min}-{cfg.brain.k_max} "
              f"rounds={cfg.brain.rounds} "
              f"sigma={float(model.log_sigma.detach().exp()):.4f}")

    best = {"loss": float("inf"), "step": -1}
    tokens_seen = start_step * tc.batch_size * tc.seq_len
    t0 = time.time()
    window_loss, window_n = 0.0, 0

    model.train()
    for step in range(start_step, tc.steps):
        scale = lr_at(step, tc)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale

        x, y = stream.get_batch("train", tc.batch_size, tc.seq_len)
        chunk = tc.tbptt_chunk if cfg.model != "transformer" else tc.seq_len
        n_chunks = max(1, math.ceil(x.shape[1] / chunk))

        optimizer.zero_grad(set_to_none=True)
        state = model.init_state(tc.batch_size, device)
        step_loss, step_balance = 0.0, 0.0

        for start in range(0, x.shape[1], chunk):
            xc = x[:, start:start + chunk]
            yc = y[:, start:start + chunk]
            if autocast_dtype is not None:
                with torch.autocast(device.type, dtype=autocast_dtype):
                    logits, state, stats = model(xc, state, collect_stats=is_brain)
            else:
                logits, state, stats = model(xc, state, collect_stats=is_brain)

            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), yc.reshape(-1)
            )
            total = loss
            if is_brain and tc.balance_coef > 0:
                balance = model.balance_loss(stats)
                total = total + tc.balance_coef * balance
                step_balance += float(balance) / n_chunks

            (total / n_chunks).backward()
            state = detach_state(state)
            step_loss += float(loss) / n_chunks
            if is_brain:
                model.update_usage(stats, tc.usage_ema)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()
        tokens_seen += tc.batch_size * tc.seq_len
        window_loss += step_loss
        window_n += 1

        # ---- geometry maintenance -------------------------------------
        if is_brain:
            reseeded = 0
            if (tc.reseed_interval and step >= tc.reseed_warmup
                    and step % tc.reseed_interval == 0):
                reseeded = model.reseed_dead(tc.reseed_frac, tc.reseed_noise, optimizer)
            # The cached topology is only valid while the positions that
            # produced it are roughly current. Refresh on a schedule, and
            # immediately after any reseed, which teleports points by design.
            refresh = cfg.brain.knn_refresh
            if not cfg.brain.free_weights and (
                int(model.knn_dirty) == 1
                or (refresh and step % refresh == 0 and step > start_step)
            ):
                model.rebuild_knn()

        # ---- logging ---------------------------------------------------
        if step % tc.log_interval == 0 or step == tc.steps - 1:
            mean = window_loss / max(window_n, 1)
            elapsed = time.time() - t0
            tps = (tokens_seen - start_step * tc.batch_size * tc.seq_len) / max(elapsed, 1e-9)
            record = {
                "step": step, "loss": mean, "ppl": math.exp(min(mean, 60.0)),
                "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm),
                "tokens": tokens_seen, "tok_per_s": tps, "elapsed_s": elapsed,
            }
            if is_brain:
                record["balance"] = step_balance
                record["dead_frac"] = float((model.usage == 0).float().mean())
                if stats.get("k_count"):
                    record["mean_k"] = float(stats["k_sum"]) / stats["k_count"]
            logger.write(record)
            # Rate over the whole run, not the logging window's step count over
            # the total elapsed time -- that mixes a ~20-step numerator with a
            # whole-run denominator and overstates the ETA by roughly the step
            # count so far.
            done = step - start_step + 1
            eta = (tc.steps - step - 1) / max(done / max(elapsed, 1e-9), 1e-9)
            print(f"step {step:>6}/{tc.steps}  loss {mean:.4f}  ppl {record['ppl']:>8.2f}"
                  f"  {tps:>7.0f} tok/s"
                  + (f"  K {record.get('mean_k', 0):.1f}"
                     f"  dead {record.get('dead_frac', 0):.3f}" if is_brain else "")
                  + f"  eta {eta / 3600:.1f}h")
            window_loss, window_n = 0.0, 0

        # ---- eval ------------------------------------------------------
        if (step + 1) % tc.eval_interval == 0 or step == tc.steps - 1:
            metrics = evaluate(
                model, stream, batch_size=tc.batch_size, seq_len=tc.seq_len,
                tbptt=chunk, max_batches=tc.eval_batches, tokenizer=tokenizer,
                autocast_dtype=autocast_dtype,
            )
            metrics["step"] = step
            metrics["split"] = "val"
            if is_brain:
                metrics.update(geometry_diagnostics(model))
            logger.write(metrics)
            print(f"  [eval] loss {metrics['loss']:.4f}  ppl {metrics['ppl']:.2f}"
                  f"  bpb {metrics.get('bpb', float('nan')):.4f}"
                  + (f"  uniformity {metrics.get('usage_uniformity', 0):.3f}"
                     f"  edge/ambient {metrics.get('ratio', 0):.3f}" if is_brain else ""))
            if metrics["loss"] < best["loss"]:
                best = {**metrics}
                save_checkpoint(os.path.join(run_dir, "best.pt"), model, optimizer,
                                cfg, step, metrics)

        if (step + 1) % tc.sample_interval == 0 and args.samples:
            text = generate(model, tokenizer, args.prompt, max_new=100,
                            device=str(device), seed=step)
            print(f"  [sample] {text[:400]!r}")
            logger.write({"step": step, "sample": text})

        if (step + 1) % tc.ckpt_interval == 0 or step == tc.steps - 1:
            save_checkpoint(ckpt_last, model, optimizer, cfg, step + 1)

    # ---- final ---------------------------------------------------------
    final = evaluate(
        model, stream, batch_size=tc.batch_size, seq_len=tc.seq_len,
        tbptt=tc.tbptt_chunk if cfg.model != "transformer" else tc.seq_len,
        max_batches=args.final_eval_batches, tokenizer=tokenizer,
        autocast_dtype=autocast_dtype,
    )
    if is_brain:
        final.update(geometry_diagnostics(model, stream))
    summary = {
        "name": cfg.name, "model": cfg.model, "params": counts,
        "final": final, "best": best, "wall_s": time.time() - t0,
        "tokens_seen": tokens_seen,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.write({"summary": summary})
    logger.close()

    print(f"\n[done] {cfg.name}: full-val loss {final['loss']:.4f} "
          f"ppl {final['ppl']:.2f} bpb {final.get('bpb', float('nan')):.4f} "
          f"in {(time.time() - t0) / 3600:.2f} h")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(ap)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--samples", action="store_true", default=True)
    ap.add_argument("--no-samples", dest="samples", action="store_false")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--match-params", type=int, default=0,
                    help="resize a baseline's d_model to hit this parameter count")
    ap.add_argument("--final-eval-batches", type=int, default=0,
                    help="batches in the closing evaluation; 0 = the whole split")
    args = ap.parse_args()

    cfg = config_from_args(args)
    train(cfg, args)


if __name__ == "__main__":
    main()
