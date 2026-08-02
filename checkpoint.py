"""Model construction, saving, and resuming.

Kept separate from `train.py` so that `evaluate.py` and `sweep.py` can rebuild
a model from a checkpoint without importing the training loop.
"""
from __future__ import annotations

import os

import torch

from baselines import build_baseline
from config import RunConfig
from model import BrainLM


def build_model(cfg: RunConfig) -> torch.nn.Module:
    if cfg.model == "brain":
        return BrainLM(cfg.brain)
    cfg.baseline.kind = cfg.model
    return build_baseline(cfg.baseline)


def save_checkpoint(path: str, model, optimizer, cfg: RunConfig, step: int,
                    metrics: dict | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "config": cfg.to_dict(),
            "step": step,
            "metrics": metrics or {},
            "torch_rng": torch.get_rng_state(),
        },
        tmp,
    )
    # Atomic-ish: a crash mid-write leaves the previous checkpoint intact rather
    # than a truncated file that fails to load two days into a sweep.
    os.replace(tmp, path)


def load_model(path: str, device: str = "cpu", strict: bool = True):
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = RunConfig.from_dict(blob["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(blob["model"], strict=strict)
    model.eval()
    return model, cfg


def resume(path: str, model, optimizer, device: str = "cpu") -> int:
    blob = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    if optimizer is not None and blob.get("optimizer") is not None:
        optimizer.load_state_dict(blob["optimizer"])
    if blob.get("torch_rng") is not None:
        torch.set_rng_state(blob["torch_rng"].cpu().to(torch.uint8))
    return int(blob.get("step", 0))
