"""Nearest-neighbour graph over neuron coordinates.

The model's "top K strongest outbound connections" is, for any kernel that
decreases with distance, exactly "K nearest dendrites to this axon". So the
sparse connectome is a kNN graph and never needs an explicit edge list.

Brute force in chunks is enough here and keeps the dependency list at torch.
At N=65k, d=32 a full rebuild is ~275 GFLOP, well under a second on a modern
GPU, and it only happens every few hundred optimizer steps because the
positions move slowly. FAISS would buy nothing and does not install cleanly
on Windows.

Only the *indices* are cached. The *weights* are recomputed from live
coordinates on every forward pass, which is what lets gradients reach the
positions at all.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def build_knn(
    axon: torch.Tensor,
    dend: torch.Tensor,
    k: int,
    chunk: int = 1024,
    exclude_self: bool = True,
) -> torch.Tensor:
    """K nearest dendrite points to each axon point.

    Args:
        axon: [N, d] source coordinates.
        dend: [N, d] target coordinates.
        k: neighbours per neuron.
        chunk: rows scored at once; peak extra memory is chunk * N floats.
        exclude_self: drop the i -> i edge. Self-connections add nothing the
            decay term does not already provide.

    Returns:
        [N, k] int32 neighbour indices, nearest first.
    """
    n = axon.shape[0]
    if k > (n - 1 if exclude_self else n):
        raise ValueError(f"k={k} exceeds the {n} available neurons")

    device = axon.device
    # Score in fp32 regardless of the model dtype; topk ties under bf16 make
    # the graph jitter between rebuilds for no reason.
    a32 = axon.detach().float()
    d32 = dend.detach().float()
    dend_sq = d32.pow(2).sum(-1)                      # [N]

    out = torch.empty(n, k, dtype=torch.int32, device=device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = a32[start:stop]                       # [c, d]
        # ||a - D||^2 = ||a||^2 - 2 a.D + ||D||^2; the ||a||^2 term is constant
        # within a row and cannot change the ordering, so it is dropped.
        scores = dend_sq.unsqueeze(0) - 2.0 * (block @ d32.t())   # [c, N]
        if exclude_self:
            rows = torch.arange(stop - start, device=device)
            scores[rows, rows + start] = float("inf")
        out[start:stop] = scores.topk(k, dim=1, largest=False).indices.to(torch.int32)
    return out


@torch.no_grad()
def graph_stats(knn_idx: torch.Tensor, n_neurons: int) -> dict:
    """Cheap health check on the connectome.

    A collapsed run shows up here long before it shows up in the loss: if a
    handful of neurons absorb every inbound edge, the geometry has folded.
    """
    flat = knn_idx.reshape(-1).long()
    indeg = torch.bincount(flat, minlength=n_neurons).float()
    reachable = (indeg > 0).float().mean().item()
    # Gini over in-degree: 0 = every neuron equally targeted, 1 = one neuron
    # takes everything.
    sorted_deg, _ = indeg.sort()
    idx = torch.arange(1, n_neurons + 1, device=indeg.device, dtype=torch.float32)
    total = sorted_deg.sum()
    gini = 0.0 if total <= 0 else (
        ((2 * idx - n_neurons - 1) * sorted_deg).sum() / (n_neurons * total)
    ).item()
    return {
        "indeg_mean": indeg.mean().item(),
        "indeg_max": indeg.max().item(),
        "indeg_gini": gini,
        "reachable_frac": reachable,
    }


@torch.no_grad()
def edge_length_stats(
    axon: torch.Tensor, dend: torch.Tensor, knn_idx: torch.Tensor, sample: int = 4096
) -> dict:
    """Distance distribution over realised edges, against the ambient scale.

    `ratio` is mean edge length over mean pairwise distance in the cloud. Near
    1.0 means the kNN graph is barely more local than a random graph, i.e. the
    positions are carrying no structure.
    """
    n = axon.shape[0]
    take = min(sample, n)
    rows = torch.randperm(n, device=axon.device)[:take]
    a = axon[rows].float()
    nbr = knn_idx[rows].long()
    lengths = (a.unsqueeze(1) - dend[nbr].float()).pow(2).sum(-1).sqrt()

    ref_rows = torch.randperm(n, device=axon.device)[: min(1024, n)]
    ref = torch.cdist(axon[ref_rows].float(), dend[ref_rows].float()).mean()
    return {
        "edge_len_mean": lengths.mean().item(),
        "edge_len_p90": lengths.flatten().quantile(0.9).item(),
        "ambient_dist_mean": ref.item(),
        "ratio": (lengths.mean() / ref.clamp_min(1e-6)).item(),
    }
