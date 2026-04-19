"""
Evaluate graph classification checkpoints under multiple attack types (generalization):
  (1) Node feature noise (σ = 0.05, 0.1, 0.15, 0.2)
  (2) Edge removal (0–30% — unseen at train)
  (3) Edge addition (0–30% random edges added — unseen at train, can drop B1 a lot)
  (4) Feature dropout at test (0–30% features zeroed)
  (5) Combined attack (noise + edge removal)
Saves eval_<name>.json and plots. B0/B1/E1/E2 compared on same perturbed test set.

  python eval.py --dataset PROTEINS runs/PROTEINS/B0/best.pt runs/PROTEINS/B1/best.pt runs/PROTEINS/E1/best.pt runs/PROTEINS/E2/best.pt
  python eval.py --aggregate runs/evals/eval_*.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from data import get_loaders, DATASETS
from model import get_model


def _console_safe(text: str) -> str:
    """Windows cp1252 consoles cannot print Greek sigma / unicode minus; keep logs readable."""
    return (
        text.replace("\u03c3", "sigma")
        .replace("\u2212", "-")
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u00b1", "+/-")
    )


def _safe_print(text: str) -> None:
    """Always sanitize so Windows cp1252 consoles do not raise on Greek/minus signs."""
    print(_console_safe(text))


# Fixed seed for eval so all models see the *same* noisy test set at each σ (fair comparison).
EVAL_SEED_BASE = 12345


def perturb_node_features(data, noise_std=0.1, device=None, rng=None):
    """If rng is not None, use it for reproducible noise (same noise for all models at same σ)."""
    data_noisy = data.clone()
    if device is not None:
        data_noisy = data_noisy.to(device)
    if data.x.dtype in (torch.float, torch.float32, torch.float16):
        if rng is not None:
            noise = torch.randn(data.x.shape, device=data.x.device, dtype=data.x.dtype, generator=rng)
            data_noisy.x = data.x + noise_std * noise
        else:
            data_noisy.x = data.x + noise_std * torch.randn_like(data.x, device=data.x.device)
    else:
        # Discrete features: map noise_std to dropout rate (sigma 0.1→20%, 0.2→40%, cap 50%)
        drop_rate = min(0.5, 2.0 * noise_std) if noise_std > 0 else 0.0
        if rng is not None:
            mask = (torch.rand(data.x.shape, device=data.x.device, generator=rng) > drop_rate).float()
        else:
            mask = (torch.rand_like(data.x.float(), device=data.x.device) > drop_rate).float()
        data_noisy.x = (data.x.float() * mask).to(data.x.dtype)
    return data_noisy


@torch.no_grad()
def evaluate(model, loader, device, noise_sigma=0.0, seed=None):
    """If noise_sigma > 0 and seed is not None, use deterministic noise so all models see same perturbed test set."""
    model.eval()
    correct, total = 0, 0
    rng = None
    if seed is not None and noise_sigma > 0:
        rng = torch.Generator(device=device).manual_seed(seed)
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        if noise_sigma > 0:
            if rng is not None:
                # Per-batch seed so order is deterministic and same across model runs
                batch_seed = seed + batch_idx * 997
                rng_batch = torch.Generator(device=device).manual_seed(batch_seed)
                data = perturb_node_features(data, noise_std=noise_sigma, device=device, rng=rng_batch)
            else:
                data = perturb_node_features(data, noise_std=noise_sigma, device=device)
        logits = model(data.x, data.edge_index, data.batch)
        pred = logits.argmax(dim=1)
        y = data.y.squeeze(1) if data.y.dim() > 1 else data.y
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total if total else 0.0


def apply_edge_removal(data, drop_frac=0.0, device=None, rng=None):
    """Randomly drop a fraction of edges (structure attack, unseen at train for B0/B1/E1). Same RNG => same edges for all models."""
    if drop_frac <= 0:
        return data.to(device) if device else data
    data_attacked = data.clone()
    if device is not None:
        data_attacked = data_attacked.to(device)
    edge_index = data_attacked.edge_index
    num_edges = edge_index.size(1)
    if rng is not None:
        keep = (torch.rand(num_edges, device=edge_index.device, generator=rng) > drop_frac)
    else:
        keep = (torch.rand(num_edges, device=edge_index.device) > drop_frac)
    data_attacked.edge_index = edge_index[:, keep]
    return data_attacked


@torch.no_grad()
def evaluate_under_edge_removal(model, loader, device, drop_frac=0.0, seed=None):
    """Accuracy when a fraction of edges are randomly removed (unseen attack)."""
    model.eval()
    correct, total = 0, 0
    rng = None
    if seed is not None and drop_frac > 0:
        rng = torch.Generator(device=device).manual_seed(seed)
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        if drop_frac > 0:
            batch_seed = (seed or 0) + batch_idx * 997
            rng_batch = torch.Generator(device=device).manual_seed(batch_seed)
            data = apply_edge_removal(data, drop_frac=drop_frac, device=device, rng=rng_batch)
        logits = model(data.x, data.edge_index, data.batch)
        pred = logits.argmax(dim=1)
        y = data.y.squeeze(1) if data.y.dim() > 1 else data.y
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total if total else 0.0


def apply_feature_dropout(data, drop_frac=0.0, device=None, rng=None):
    """Zero out a fraction of node features (test-time; unseen at train for B0). Same RNG => same mask for all models."""
    if drop_frac <= 0:
        return data.to(device) if device else data
    data_out = data.clone()
    if device is not None:
        data_out = data_out.to(device)
    x = data_out.x
    if rng is not None:
        keep = (torch.rand(x.shape, device=x.device, generator=rng) > drop_frac)
    else:
        keep = (torch.rand(x.shape, device=x.device) > drop_frac)
    scale = 1.0 / max(1e-8, 1.0 - drop_frac)
    out = x.float() * keep.float() * scale
    data_out.x = out.to(x.dtype) if x.dtype != torch.float else out
    return data_out


@torch.no_grad()
def evaluate_under_feature_dropout(model, loader, device, drop_frac=0.0, seed=None):
    """Accuracy when a fraction of node features are zeroed (tests representation stability / missing features)."""
    model.eval()
    correct, total = 0, 0
    rng = None
    if seed is not None and drop_frac > 0:
        rng = torch.Generator(device=device).manual_seed(seed)
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        if drop_frac > 0:
            batch_seed = (seed or 0) + batch_idx * 997
            rng_batch = torch.Generator(device=device).manual_seed(batch_seed)
            data = apply_feature_dropout(data, drop_frac=drop_frac, device=device, rng=rng_batch)
        logits = model(data.x, data.edge_index, data.batch)
        pred = logits.argmax(dim=1)
        y = data.y.squeeze(1) if data.y.dim() > 1 else data.y
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total if total else 0.0


@torch.no_grad()
def evaluate_combined(model, loader, device, noise_sigma=0.1, edge_drop_frac=0.1, seed=None):
    """Accuracy under both node feature noise and edge removal (E1 is trained for both)."""
    model.eval()
    correct, total = 0, 0
    rng = None
    if seed is not None:
        rng = torch.Generator(device=device).manual_seed(seed)
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        batch_seed = (seed or 0) + batch_idx * 997
        rng_batch = torch.Generator(device=device).manual_seed(batch_seed)
        if noise_sigma > 0:
            data = perturb_node_features(data, noise_std=noise_sigma, device=device, rng=rng_batch)
        if edge_drop_frac > 0:
            # New RNG slice for edge removal so both attacks are reproducible
            rng_er = torch.Generator(device=device).manual_seed(batch_seed + 5000)
            data = apply_edge_removal(data, drop_frac=edge_drop_frac, device=device, rng=rng_er)
        logits = model(data.x, data.edge_index, data.batch)
        pred = logits.argmax(dim=1)
        y = data.y.squeeze(1) if data.y.dim() > 1 else data.y
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total if total else 0.0


def apply_edge_addition(data, add_frac=0.0, device=None, rng=None):
    """Add random edges within each graph (unseen at train; hurts structure-relying models e.g. B1). Same RNG => same edges for all models."""
    if add_frac <= 0:
        return data.to(device) if device else data
    data_out = data.clone()
    if device is not None:
        data_out = data_out.to(device)
    edge_index = data_out.edge_index
    batch = data_out.batch
    batch_ids = batch.unique(sorted=True)
    new_edges = []
    for b in batch_ids:
        b = b.item()
        nodes_b = (batch == b).nonzero().squeeze(1)
        n_b = nodes_b.size(0)
        if n_b < 2:
            continue
        mask_b = (batch[edge_index[0]] == b) & (batch[edge_index[1]] == b)
        existing = set()
        for i in range(edge_index.size(1)):
            if mask_b[i]:
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                existing.add((min(u, v), max(u, v)))
        num_add = max(1, int(add_frac * mask_b.sum().item()))
        rng_local = rng
        added = 0
        max_tries = num_add * 20
        for _ in range(max_tries):
            if added >= num_add:
                break
            if rng_local is not None:
                i1 = torch.randint(0, n_b, (1,), device=nodes_b.device, generator=rng_local).item()
                i2 = torch.randint(0, n_b, (1,), device=nodes_b.device, generator=rng_local).item()
            else:
                i1 = torch.randint(0, n_b, (1,), device=nodes_b.device).item()
                i2 = torch.randint(0, n_b, (1,), device=nodes_b.device).item()
            u, v = nodes_b[i1].item(), nodes_b[i2].item()
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in existing:
                continue
            existing.add(key)
            new_edges.append([u, v])
            new_edges.append([v, u])
            added += 1
    if new_edges:
        new_edges_t = torch.tensor(new_edges, device=edge_index.device, dtype=edge_index.dtype).t()
        data_out.edge_index = torch.cat([edge_index, new_edges_t], dim=1)
    return data_out


@torch.no_grad()
def evaluate_under_edge_addition(model, loader, device, add_frac=0.0, seed=None):
    """Accuracy when random edges are added within each graph (unseen at train; can drop B1 a lot)."""
    model.eval()
    correct, total = 0, 0
    rng = None
    if seed is not None and add_frac > 0:
        rng = torch.Generator(device=device).manual_seed(seed)
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        if add_frac > 0:
            batch_seed = (seed or 0) + batch_idx * 997 + 60000
            rng_batch = torch.Generator(device=device).manual_seed(batch_seed)
            data = apply_edge_addition(data, add_frac=add_frac, device=device, rng=rng_batch)
        logits = model(data.x, data.edge_index, data.batch)
        pred = logits.argmax(dim=1)
        y = data.y.squeeze(1) if data.y.dim() > 1 else data.y
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total if total else 0.0


@torch.no_grad()
def evaluate_consistency(model, loader, device, noise_sigma=0.1, K=10, seed=None):
    """Per-graph: run K noise samples; return consistency % (fraction of graphs with same prediction across K) and mean accuracy."""
    model.eval()
    all_preds_by_run = []
    for k in range(K):
        run_seed = (seed or 0) + k * 777
        rng = torch.Generator(device=device).manual_seed(run_seed)
        preds_this_run = []
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            batch_seed = run_seed + batch_idx * 997
            rng_batch = torch.Generator(device=device).manual_seed(batch_seed)
            data = perturb_node_features(data, noise_std=noise_sigma, device=device, rng=rng_batch)
            logits = model(data.x, data.edge_index, data.batch)
            pred = logits.argmax(dim=1)
            preds_this_run.extend(pred.cpu().tolist())
        all_preds_by_run.append(preds_this_run)
    N = len(all_preds_by_run[0])
    consistent = sum(1 for i in range(N) if len(set(all_preds_by_run[k][i] for k in range(K))) == 1)
    consistency_pct = 100.0 * consistent / N if N else 0.0
    correct_per_run = []
    for k in range(K):
        run_seed = (seed or 0) + k * 777
        correct, total = 0, 0
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            batch_seed = run_seed + batch_idx * 997
            rng_batch = torch.Generator(device=device).manual_seed(batch_seed)
            data = perturb_node_features(data, noise_std=noise_sigma, device=device, rng=rng_batch)
            logits = model(data.x, data.edge_index, data.batch)
            pred = logits.argmax(dim=1)
            y = data.y.squeeze(1) if data.y.dim() > 1 else data.y
            correct += (pred == y).sum().item()
            total += y.size(0)
        correct_per_run.append(100.0 * correct / total if total else 0.0)
    mean_acc = float(np.mean(correct_per_run))
    return consistency_pct, mean_acc


def apply_combined_noise_fd(data, noise_sigma=0.1, fd_frac=0.2, device=None, seed=None):
    """Apply both node feature noise and feature dropout. seed (int) used for reproducibility (two offsets)."""
    if device:
        data = data.to(device)
    if noise_sigma > 0:
        rng1 = torch.Generator(device=data.x.device).manual_seed((seed or 0) + 0) if seed is not None else None
        data = perturb_node_features(data, noise_std=noise_sigma, device=device, rng=rng1)
    if fd_frac > 0:
        rng2 = torch.Generator(device=data.x.device).manual_seed((seed or 0) + 1111) if seed is not None else None
        data = apply_feature_dropout(data, drop_frac=fd_frac, device=device, rng=rng2)
    return data


@torch.no_grad()
def evaluate_combined_noise_fd(model, loader, device, noise_sigma=0.1, fd_frac=0.2, seed=None):
    """Accuracy under both node noise and feature dropout (unseen combination)."""
    model.eval()
    correct, total = 0, 0
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        batch_seed = (seed or 0) + batch_idx * 997
        data = apply_combined_noise_fd(data, noise_sigma=noise_sigma, fd_frac=fd_frac, device=device, seed=batch_seed)
        logits = model(data.x, data.edge_index, data.batch)
        pred = logits.argmax(dim=1)
        y = data.y.squeeze(1) if data.y.dim() > 1 else data.y
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total if total else 0.0


NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2]
NOISE_LEVELS_EXTRA = [0.25, 0.3, 0.4]  # extrapolation (higher than train σ=0.1); PMH should degrade more gracefully
EDGE_DROP_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # include 40%, 50% for stronger structure attack
EDGE_ADD_RATES = [0.0, 0.1, 0.2, 0.3]   # fraction of edges added (random within-graph; unseen at train, can hurt B1)
FEATURE_DROP_RATES = [0.0, 0.1, 0.2, 0.3]  # fraction of node features zeroed at eval (tests representation stability)
# Combined attack: (noise_sigma, edge_drop_frac) — E1 trained for both
COMBINED_ATTACKS = [(0.1, 0.1), (0.15, 0.2)]  # (σ, edge_drop)
# Combined noise + feature dropout (unseen mix; tests generalization)
COMBINED_NOISE_FD = [(0.1, 0.2), (0.15, 0.3)]  # (σ, feature_drop_frac)
CONSISTENCY_K = 10  # number of noise samples per graph for prediction consistency
CONSISTENCY_SIGMA = 0.1  # σ used for consistency eval


def _ensure_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _acc_at_sigma(d, s):
    """Get accuracy at noise level s from result dict (mean if multi-seed)."""
    if s == 0:
        return d.get("clean") or d.get("clean_acc")
    key = f"sigma_{s:.2f}"
    return d.get(key) or d.get(f"acc_sigma_{s:.2f}") or d.get(f"acc_sigma_{s:.1f}")


def _std_at_sigma(d, s):
    """Get std at noise level s if multi-seed eval was used."""
    if s == 0:
        return None
    return d.get(f"sigma_{s:.2f}_std")


def _get_ys(name, d):
    """List of accuracies at NOISE_LEVELS for one model."""
    return [_acc_at_sigma(d, s) or 0 for s in NOISE_LEVELS]


def _acc_at_edge_drop(d, f):
    """Get accuracy at edge drop rate f from result dict."""
    if f == 0:
        return d.get("clean") or d.get("clean_acc") or d.get("edge_drop_0.0")
    return d.get(f"edge_drop_{f:.2f}") or d.get(f"edge_drop_{f:.1f}")


def _std_at_edge_drop(d, f):
    if f == 0:
        return None
    return d.get(f"edge_drop_{f:.2f}_std") or d.get(f"edge_drop_{f:.1f}_std")


def _get_ys_edge_drop(d):
    return [_acc_at_edge_drop(d, f) or 0 for f in EDGE_DROP_RATES]


def _acc_at_edge_add(d, f):
    if f == 0:
        return d.get("clean") or d.get("clean_acc") or d.get("edge_add_0.0")
    return d.get(f"edge_add_{f:.2f}") or d.get(f"edge_add_{f:.1f}")


def _std_at_edge_add(d, f):
    if f == 0:
        return None
    return d.get(f"edge_add_{f:.2f}_std") or d.get(f"edge_add_{f:.1f}_std")


def _get_ys_edge_add(d):
    return [_acc_at_edge_add(d, f) or 0 for f in EDGE_ADD_RATES]


def _acc_at_feature_drop(d, f):
    if f == 0:
        return d.get("clean") or d.get("clean_acc") or d.get("feature_drop_0.0")
    return d.get(f"feature_drop_{f:.2f}") or d.get(f"feature_drop_{f:.1f}")


def _std_at_feature_drop(d, f):
    if f == 0:
        return None
    return d.get(f"feature_drop_{f:.2f}_std") or d.get(f"feature_drop_{f:.1f}_std")


def _get_ys_feature_drop(d):
    return [_acc_at_feature_drop(d, f) or 0 for f in FEATURE_DROP_RATES]


def print_summary_table(rows, out_dir=None):
    """Print and optionally save summary: each model at each σ, and E1−B0 improvement."""
    baseline_name = "B0" if any(r[0] == "B0" for r in rows) else (rows[0][0] if rows else None)
    pmh_name = "E1" if any(r[0] == "E1" for r in rows) else (rows[-1][0] if len(rows) >= 2 else None)
    lines = []
    lines.append("\n--- Summary: accuracy under node feature noise ---")
    col_w = 10
    header = f"{'Model':<8} " + " ".join(f"{'Clean' if s == 0 else f'σ={s:.2f}':>{col_w}}" for s in NOISE_LEVELS)
    lines.append(header)
    lines.append("-" * (8 + (col_w + 1) * len(NOISE_LEVELS)))
    for name, d in rows:
        ys = _get_ys(name, d)
        parts = []
        for i, s in enumerate(NOISE_LEVELS):
            std = _std_at_sigma(d, s) if s > 0 else None
            if std is not None and std > 0:
                parts.append(f"{ys[i]:.2f}±{std:.2f}".rjust(col_w))
            else:
                parts.append(f"{ys[i]:>{col_w}.2f}")
        lines.append(f"{name:<8} " + " ".join(parts))
    if pmh_name and baseline_name and pmh_name != baseline_name:
        base_dict = next((d for n, d in rows if n == baseline_name), None)
        pmh_dict = next((d for n, d in rows if n == pmh_name), None)
        if base_dict is not None and pmh_dict is not None:
            base_ys = _get_ys(baseline_name, base_dict)
            pmh_ys = _get_ys(pmh_name, pmh_dict)
            diff = [pmh_ys[i] - base_ys[i] for i in range(len(NOISE_LEVELS))]
            lines.append(f"{pmh_name}−{baseline_name} " + " ".join(f"{x:>+8.2f}" for x in diff))
            lines.append("")
            lines.append("(Positive = PMH beats baseline at that noise level.)")
    # Edge removal section (generalization to unseen attack)
    if any(f"edge_drop_{r:.2f}" in (d or {}) for r in EDGE_DROP_RATES for _, d in rows):
        col_w = 10
        lines.append("")
        lines.append("--- Accuracy under edge removal (unseen at train) ---")
        header_er = f"{'Model':<8} " + " ".join(f"{'0%' if f == 0 else f'{int(f*100)}%':>{col_w}}" for f in EDGE_DROP_RATES)
        lines.append(header_er)
        lines.append("-" * (8 + (col_w + 1) * len(EDGE_DROP_RATES)))
        for name, d in rows:
            ys = _get_ys_edge_drop(d)
            parts = []
            for i, f in enumerate(EDGE_DROP_RATES):
                std = _std_at_edge_drop(d, f)
                if std is not None and std > 0:
                    parts.append(f"{ys[i]:.2f}±{std:.2f}".rjust(col_w))
                else:
                    parts.append(f"{ys[i]:>{col_w}.2f}")
            lines.append(f"{name:<8} " + " ".join(parts))
        if pmh_name and baseline_name and pmh_name != baseline_name:
            base_dict = next((d for n, d in rows if n == baseline_name), None)
            pmh_dict = next((d for n, d in rows if n == pmh_name), None)
            if base_dict and pmh_dict:
                base_ys = _get_ys_edge_drop(base_dict)
                pmh_ys = _get_ys_edge_drop(pmh_dict)
                diff = [pmh_ys[i] - base_ys[i] for i in range(len(EDGE_DROP_RATES))]
                lines.append(f"{pmh_name}−{baseline_name} " + " ".join(f"{x:>+8.2f}" for x in diff))
                lines.append("(Positive = PMH beats baseline under edge removal.)")
    # Edge addition (unseen at train; can drop B1 a lot — adds fake structure)
    if any(f"edge_add_{r:.2f}" in (d or {}) for r in EDGE_ADD_RATES for _, d in rows):
        col_w = 10
        lines.append("")
        lines.append("--- Accuracy under edge addition (unseen at train; can hurt B1) ---")
        header_ea = f"{'Model':<8} " + " ".join(f"{'0%' if f == 0 else f'+{int(f*100)}%':>{col_w}}" for f in EDGE_ADD_RATES)
        lines.append(header_ea)
        lines.append("-" * (8 + (col_w + 1) * len(EDGE_ADD_RATES)))
        for name, d in rows:
            ys = _get_ys_edge_add(d)
            parts = []
            for i, f in enumerate(EDGE_ADD_RATES):
                std = _std_at_edge_add(d, f)
                if std is not None and std > 0:
                    parts.append(f"{ys[i]:.2f}±{std:.2f}".rjust(col_w))
                else:
                    parts.append(f"{ys[i]:>{col_w}.2f}")
            lines.append(f"{name:<8} " + " ".join(parts))
        if pmh_name and baseline_name and pmh_name != baseline_name:
            base_dict = next((d for n, d in rows if n == baseline_name), None)
            pmh_dict = next((d for n, d in rows if n == pmh_name), None)
            if base_dict and pmh_dict:
                base_ys = _get_ys_edge_add(base_dict)
                pmh_ys = _get_ys_edge_add(pmh_dict)
                diff = [pmh_ys[i] - base_ys[i] for i in range(len(EDGE_ADD_RATES))]
                lines.append(f"{pmh_name}−{baseline_name} " + " ".join(f"{x:>+8.2f}" for x in diff))
                lines.append("(Positive = PMH beats baseline under edge addition.)")
    # Feature dropout (test-time missing features; PMH stability can help)
    if any(f"feature_drop_{r:.2f}" in (d or {}) for r in FEATURE_DROP_RATES for _, d in rows):
        col_w = 10
        lines.append("")
        lines.append("--- Accuracy under feature dropout (test-time; representation stability) ---")
        header_fd = f"{'Model':<8} " + " ".join(f"{'0%' if f == 0 else f'{int(f*100)}%':>{col_w}}" for f in FEATURE_DROP_RATES)
        lines.append(header_fd)
        lines.append("-" * (8 + (col_w + 1) * len(FEATURE_DROP_RATES)))
        for name, d in rows:
            ys = _get_ys_feature_drop(d)
            parts = []
            for i, f in enumerate(FEATURE_DROP_RATES):
                std = _std_at_feature_drop(d, f)
                if std is not None and std > 0:
                    parts.append(f"{ys[i]:.2f}±{std:.2f}".rjust(col_w))
                else:
                    parts.append(f"{ys[i]:>{col_w}.2f}")
            lines.append(f"{name:<8} " + " ".join(parts))
        if pmh_name and baseline_name and pmh_name != baseline_name:
            base_dict = next((d for n, d in rows if n == baseline_name), None)
            pmh_dict = next((d for n, d in rows if n == pmh_name), None)
            if base_dict and pmh_dict:
                base_ys = _get_ys_feature_drop(base_dict)
                pmh_ys = _get_ys_feature_drop(pmh_dict)
                diff = [pmh_ys[i] - base_ys[i] for i in range(len(FEATURE_DROP_RATES))]
                lines.append(f"{pmh_name}−{baseline_name} " + " ".join(f"{x:>+8.2f}" for x in diff))
                lines.append("(Positive = PMH beats baseline under feature dropout.)")
    # Combined attack (noise + edge removal; E1 trained for both)
    combo_keys = [f"combined_s{s:.2f}_e{int(ed*100)}" for s, ed in COMBINED_ATTACKS]
    if any(k in (d or {}) for _, d in rows for k in combo_keys):
        col_w = 10
        lines.append("")
        lines.append("--- Accuracy under combined attack (noise σ + edge removal; E1 trained for both) ---")
        header_c = f"{'Model':<8} " + " ".join(f"σ={s:.2f},e{int(ed*100)}%".rjust(col_w) for s, ed in COMBINED_ATTACKS)
        lines.append(header_c)
        lines.append("-" * (8 + (col_w + 1) * len(COMBINED_ATTACKS)))
        for name, d in rows:
            parts = []
            for s, ed in COMBINED_ATTACKS:
                k = f"combined_s{s:.2f}_e{int(ed*100)}"
                v = d.get(k)
                std = d.get(k + "_std")
                if std is not None and std > 0:
                    parts.append(f"{v:.2f}±{std:.2f}".rjust(col_w) if v is not None else "".rjust(col_w))
                else:
                    parts.append(f"{v:>{col_w}.2f}" if v is not None else "".rjust(col_w))
            lines.append(f"{name:<8} " + " ".join(parts))
        if pmh_name and baseline_name and pmh_name != baseline_name:
            base_dict = next((d for n, d in rows if n == baseline_name), None)
            pmh_dict = next((d for n, d in rows if n == pmh_name), None)
            if base_dict and pmh_dict:
                diff = []
                for s, ed in COMBINED_ATTACKS:
                    k = f"combined_s{s:.2f}_e{int(ed*100)}"
                    bv = base_dict.get(k)
                    pv = pmh_dict.get(k)
                    diff.append((pv - bv) if pv is not None and bv is not None else 0)
                lines.append(f"{pmh_name}−{baseline_name} " + " ".join(f"{x:>+8.2f}" for x in diff))
                lines.append("(Positive = PMH beats baseline under combined attack.)")
    # Extrapolation noise (σ > train)
    if any(f"sigma_{s:.2f}" in (d or {}) for s in NOISE_LEVELS_EXTRA for _, d in rows):
        col_w = 10
        lines.append("")
        lines.append("--- Accuracy under extrapolation noise (σ > train; PMH degrades more gracefully) ---")
        header_ex = f"{'Model':<8} " + " ".join(f"σ={s:.2f}".rjust(col_w) for s in NOISE_LEVELS_EXTRA)
        lines.append(header_ex)
        lines.append("-" * (8 + (col_w + 1) * len(NOISE_LEVELS_EXTRA)))
        for name, d in rows:
            parts = []
            for s in NOISE_LEVELS_EXTRA:
                v = d.get(f"sigma_{s:.2f}")
                std = d.get(f"sigma_{s:.2f}_std")
                if std is not None and std > 0:
                    parts.append(f"{v:.2f}±{std:.2f}".rjust(col_w) if v is not None else "".rjust(col_w))
                else:
                    parts.append(f"{v:>{col_w}.2f}" if v is not None else "".rjust(col_w))
            lines.append(f"{name:<8} " + " ".join(parts))
        if pmh_name and baseline_name and pmh_name != baseline_name:
            base_dict = next((d for n, d in rows if n == baseline_name), None)
            pmh_dict = next((d for n, d in rows if n == pmh_name), None)
            if base_dict and pmh_dict:
                diff = [(pmh_dict.get(f"sigma_{s:.2f}") or 0) - (base_dict.get(f"sigma_{s:.2f}") or 0) for s in NOISE_LEVELS_EXTRA]
                lines.append(f"{pmh_name}−{baseline_name} " + " ".join(f"{x:>+8.2f}" for x in diff))
    # Prediction consistency under noise (σ=0.1, K runs per graph)
    if any("consistency_pct" in (d or {}) for _, d in rows):
        lines.append("")
        lines.append("--- Prediction consistency under noise (σ=0.1, K=10; PMH more stable) ---")
        lines.append(f"{'Model':<8} {'Consistency %':>14} {'Mean acc %':>12}")
        lines.append("-" * 36)
        for name, d in rows:
            c = d.get("consistency_pct")
            a = d.get("consistency_mean_acc")
            if c is not None and a is not None:
                lines.append(f"{name:<8} {c:>14.2f} {a:>12.2f}")
        if pmh_name and baseline_name and pmh_name != baseline_name:
            base_dict = next((d for n, d in rows if n == baseline_name), None)
            pmh_dict = next((d for n, d in rows if n == pmh_name), None)
            if base_dict and pmh_dict and base_dict.get("consistency_pct") is not None:
                dc = (pmh_dict.get("consistency_pct") or 0) - (base_dict.get("consistency_pct") or 0)
                lines.append(f"{pmh_name}−{baseline_name} (consistency): {dc:+.2f}%")
    # Combined noise + feature dropout
    nfd_keys = [f"noise_fd_s{ns:.2f}_f{int(fd*100)}" for ns, fd in COMBINED_NOISE_FD]
    if any(k in (d or {}) for _, d in rows for k in nfd_keys):
        col_w = 10
        lines.append("")
        lines.append("--- Combined noise + feature dropout (unseen mix) ---")
        header_nfd = f"{'Model':<8} " + " ".join(f"σ={ns:.2f},f{int(fd*100)}%".rjust(col_w) for ns, fd in COMBINED_NOISE_FD)
        lines.append(header_nfd)
        lines.append("-" * (8 + (col_w + 1) * len(COMBINED_NOISE_FD)))
        for name, d in rows:
            parts = [f"{d.get(k) or 0:>{col_w}.2f}" for k in nfd_keys]
            lines.append(f"{name:<8} " + " ".join(parts))
        if pmh_name and baseline_name and pmh_name != baseline_name:
            base_dict = next((d for n, d in rows if n == baseline_name), None)
            pmh_dict = next((d for n, d in rows if n == pmh_name), None)
            if base_dict and pmh_dict:
                diff = [(pmh_dict.get(k) or 0) - (base_dict.get(k) or 0) for k in nfd_keys]
                lines.append(f"{pmh_name}−{baseline_name} " + " ".join(f"{x:>+8.2f}" for x in diff))
    # Robustness AUC & worst-case
    if any("auc_noise" in (d or {}) for _, d in rows):
        lines.append("")
        lines.append("--- Robustness AUC (noise) & worst-case accuracy ---")
        lines.append(f"{'Model':<8} {'AUC (noise)':>12} {'Worst-case %':>14}")
        lines.append("-" * 36)
        for name, d in rows:
            auc_v = d.get("auc_noise")
            wc = d.get("worst_case_acc")
            auc_s = f"{auc_v:.2f}" if auc_v is not None else ""
            wc_s = f"{wc:.2f}" if wc is not None else ""
            lines.append(f"{name:<8} {auc_s:>12} {wc_s:>14}")
    for line in lines:
        _safe_print(line)
    if out_dir:
        with open(os.path.join(out_dir, "eval_summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nSaved: {os.path.join(out_dir, 'eval_summary.txt')}")


def plot_comparison_curves(rows, save_path, title="Node feature noise"):
    """rows = [(model_name, results_dict)], results has 'clean' and 'sigma_0.05', etc."""
    plt = _ensure_mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    xs = NOISE_LEVELS
    for i, (name, d) in enumerate(rows):
        clean = d.get("clean") or d.get("clean_acc")
        ys = [clean if s == 0 else d.get(f"sigma_{s:.2f}") or d.get(f"acc_sigma_{s:.2f}") or d.get(f"acc_sigma_{s:.1f}") for s in xs]
        ys = [y if y is not None else 0 for y in ys]
        ax.plot(xs, ys, "o-", label=name, linewidth=2, markersize=6, color=colors[i % len(colors)])
    ax.set_xlabel("Node feature noise σ")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_accuracy_bars(rows, save_path):
    """Bar chart: for each noise condition (Clean, σ=0.05, ...), bars for each model. Shows standard vs PMH under noise."""
    plt = _ensure_mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x_labels = ["Clean"] + [f"σ={s:.2f}" for s in NOISE_LEVELS if s > 0]
    n_conditions = len(x_labels)
    n_models = len(rows)
    width = 0.8 / n_models
    x = np.arange(n_conditions)
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    for i, (name, d) in enumerate(rows):
        ys = _get_ys(name, d)
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, ys, width, label=name, color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Node feature noise")
    ax.set_title("Accuracy under noise: baseline (B0) vs PMH (E1)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_edge_removal_bars(rows, save_path):
    """Bar chart: accuracy under edge removal (unseen attack). B1 helps on feature noise; E1 tested for generalization."""
    plt = _ensure_mpl()
    if plt is None or save_path is None:
        return
    if not any(f"edge_drop_{EDGE_DROP_RATES[1]:.2f}" in d for _, d in rows):
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x_labels = [f"{int(f*100)}%" for f in EDGE_DROP_RATES]
    n_conditions = len(EDGE_DROP_RATES)
    n_models = len(rows)
    width = 0.8 / n_models
    x = np.arange(n_conditions)
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    for i, (name, d) in enumerate(rows):
        ys = _get_ys_edge_drop(d)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, ys, width, label=name, color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Edges removed")
    ax.set_title("Accuracy under edge removal (unseen at train)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_edge_addition_bars(rows, save_path):
    """Bar chart: accuracy when random edges are added (unseen at train; can drop B1 a lot)."""
    plt = _ensure_mpl()
    if plt is None or save_path is None:
        return
    if not any(f"edge_add_{EDGE_ADD_RATES[1]:.2f}" in d for _, d in rows):
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x_labels = [f"{'0%' if f == 0 else f'+'+str(int(f*100))+'%'}" for f in EDGE_ADD_RATES]
    n_conditions = len(EDGE_ADD_RATES)
    n_models = len(rows)
    width = 0.8 / n_models
    x = np.arange(n_conditions)
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    for i, (name, d) in enumerate(rows):
        ys = _get_ys_edge_add(d)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, ys, width, label=name, color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Random edges added")
    ax.set_title("Accuracy under edge addition (unseen at train; can hurt B1)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_dropout_bars(rows, save_path):
    """Bar chart: accuracy when a fraction of node features are zeroed (representation stability)."""
    plt = _ensure_mpl()
    if plt is None or save_path is None:
        return
    if not any(f"feature_drop_{FEATURE_DROP_RATES[1]:.2f}" in d for _, d in rows):
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x_labels = [f"{int(f*100)}%" for f in FEATURE_DROP_RATES]
    n_conditions = len(FEATURE_DROP_RATES)
    n_models = len(rows)
    width = 0.8 / n_models
    x = np.arange(n_conditions)
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    for i, (name, d) in enumerate(rows):
        ys = _get_ys_feature_drop(d)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, ys, width, label=name, color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Node features dropped")
    ax.set_title("Accuracy under feature dropout (test-time; representation stability)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined_bars(rows, save_path):
    """Bar chart: accuracy under combined attack (noise + edge removal); E1 trained for both."""
    plt = _ensure_mpl()
    if plt is None or save_path is None:
        return
    combo_keys = [f"combined_s{s:.2f}_e{int(ed*100)}" for s, ed in COMBINED_ATTACKS]
    if not any(k in d for _, d in rows for k in combo_keys):
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x_labels = [f"σ={s:.2f}, {int(ed*100)}% edges" for s, ed in COMBINED_ATTACKS]
    n_conditions = len(COMBINED_ATTACKS)
    n_models = len(rows)
    width = 0.8 / n_models
    x = np.arange(n_conditions)
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    for i, (name, d) in enumerate(rows):
        ys = [d.get(k) or 0 for k in combo_keys]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, ys, width, label=name, color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Combined attack")
    ax.set_title("Accuracy under combined attack (noise + edge removal; E1 trained for both)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_improvement_over_baseline(rows, baseline_name="B0", pmh_name="E1", save_path=None):
    """Plot how much PMH (E1) beats baseline (B0) at each noise level."""
    plt = _ensure_mpl()
    if plt is None or save_path is None or len(rows) < 2:
        return
    base_row = next((r for r in rows if r[0] == baseline_name), None)
    pmh_row = next((r for r in rows if r[0] == pmh_name), None)
    if base_row is None:
        baseline_name = rows[0][0]
        base_row = rows[0]
    if pmh_row is None or pmh_row == base_row:
        pmh_name = rows[-1][0]
        pmh_row = rows[-1]
    if base_row == pmh_row:
        return
    _, base_d = base_row
    _, pmh_d = pmh_row
    base_ys = _get_ys(baseline_name, base_d)
    pmh_ys = _get_ys(pmh_name, pmh_d)
    diff = [pmh_ys[i] - base_ys[i] for i in range(len(NOISE_LEVELS))]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in diff]
    bars = ax.bar([f"Clean" if s == 0 else f"σ={s:.2f}" for s in NOISE_LEVELS], diff, color=colors, alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Accuracy difference ({pmh_name} − {baseline_name}) %")
    ax.set_xlabel("Node feature noise")
    ax.set_title(f"PMH improvement over baseline under noise")
    for b, v in zip(bars, diff):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (1 if v >= 0 else -2), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_extrapolation_curves(rows, save_path):
    """Accuracy vs σ including extrapolation (0.25, 0.3, 0.4); PMH should degrade more gracefully."""
    plt = _ensure_mpl()
    if plt is None or save_path is None:
        return
    xs = [0] + [s for s in NOISE_LEVELS if s > 0] + list(NOISE_LEVELS_EXTRA)
    if not any(f"sigma_{NOISE_LEVELS_EXTRA[0]:.2f}" in d for _, d in rows):
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    for i, (name, d) in enumerate(rows):
        ys = [d.get("clean") or d.get("clean_acc") or 0] + [d.get(f"sigma_{s:.2f}") or 0 for s in xs if s > 0]
        ax.plot(xs, ys, "o-", label=name, linewidth=2, markersize=5, color=colors[i % len(colors)])
    ax.set_xlabel("Node feature noise σ")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Graceful degradation (incl. extrapolation σ > train)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_consistency_bars(rows, save_path):
    """Prediction consistency % under noise (K runs per graph); PMH should be higher."""
    plt = _ensure_mpl()
    if plt is None or save_path is None or not any("consistency_pct" in d for _, d in rows):
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    names = [r[0] for r in rows]
    vals = [r[1].get("consistency_pct") or 0 for r in rows]
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    bars = ax.bar(range(len(names)), vals, color=[colors[i % len(colors)] for i in range(len(names))], alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Consistency % (same prediction over K noise samples)")
    ax.set_title("Prediction consistency under noise (σ=0.1, K=10)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined_noise_fd_bars(rows, save_path):
    """Combined noise + feature dropout (unseen mix)."""
    plt = _ensure_mpl()
    if plt is None or save_path is None:
        return
    nfd_keys = [f"noise_fd_s{ns:.2f}_f{int(fd*100)}" for ns, fd in COMBINED_NOISE_FD]
    if not any(k in d for _, d in rows for k in nfd_keys):
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x_labels = [f"σ={ns:.2f}, fd{int(fd*100)}%" for ns, fd in COMBINED_NOISE_FD]
    n_conditions = len(COMBINED_NOISE_FD)
    n_models = len(rows)
    width = 0.8 / n_models
    x = np.arange(n_conditions)
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    for i, (name, d) in enumerate(rows):
        ys = [d.get(k) or 0 for k in nfd_keys]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, ys, width, label=name, color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Combined noise + feature dropout (unseen mix)")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_heatmap(rows, save_path):
    """Heatmap: models x conditions (accuracy)."""
    plt = _ensure_mpl()
    if plt is None or save_path is None or len(rows) < 2:
        return
    data = []
    for name, d in rows:
        row = []
        row.append(d.get("clean") or d.get("clean_acc") or 0)
        for s in (NOISE_LEVELS + NOISE_LEVELS_EXTRA):
            if s > 0:
                row.append(d.get(f"sigma_{s:.2f}") or 0)
        for f in EDGE_DROP_RATES:
            if f > 0:
                row.append(d.get(f"edge_drop_{f:.2f}") or 0)
        for f in FEATURE_DROP_RATES:
            if f > 0:
                row.append(d.get(f"feature_drop_{f:.2f}") or 0)
        data.append(row)
    if not data or not data[0]:
        return
    data = np.array(data)
    cond_labels = ["Clean"] + [f"σ={s:.2f}" for s in (NOISE_LEVELS + NOISE_LEVELS_EXTRA) if s > 0] + [f"e{int(f*100)}" for f in EDGE_DROP_RATES if f > 0] + [f"f{int(f*100)}" for f in FEATURE_DROP_RATES if f > 0]
    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(cond_labels) * 0.5), max(4, len(rows) * 0.6)))
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=100, cmap="RdYlGn")
    ax.set_xticks(range(len(cond_labels)))
    ax.set_xticklabels(cond_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    plt.colorbar(im, ax=ax, label="Accuracy %")
    ax.set_title("Accuracy heatmap (models × conditions)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_robustness_auc_bars(rows, save_path):
    """Bar chart: robustness AUC (noise) per model."""
    plt = _ensure_mpl()
    if plt is None or save_path is None or not any("auc_noise" in d for _, d in rows):
        return
    names = [r[0] for r in rows]
    vals = [r[1].get("auc_noise") or 0 for r in rows]
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    bars = ax.bar(names, vals, color=[colors[i % len(colors)] for i in range(len(names))], alpha=0.85)
    ax.set_ylabel("AUC (accuracy vs σ)")
    ax.set_title("Robustness AUC over noise range")
    ax.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_scatter_clean_vs_attacked(rows, save_path, attack_key="sigma_0.20", attack_label="σ=0.2"):
    """Scatter: x=clean accuracy, y=attacked accuracy; robust models top-right."""
    plt = _ensure_mpl()
    if plt is None or save_path is None or len(rows) < 2:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    clean_vals = [r[1].get("clean") or r[1].get("clean_acc") or 0 for r in rows]
    attacked_vals = [r[1].get(attack_key) or 0 for r in rows]
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    for i, (name, _) in enumerate(rows):
        ax.scatter(clean_vals[i], attacked_vals[i], s=120, label=name, color=colors[i % len(colors)], alpha=0.85, edgecolors="black", linewidths=0.5)
        ax.annotate(name, (clean_vals[i], attacked_vals[i]), textcoords="offset points", xytext=(5, 5), fontsize=9, ha="left")
    ax.set_xlabel("Clean accuracy %")
    ax.set_ylabel(f"Accuracy under attack ({attack_label}) %")
    ax.set_title("Clean vs attacked accuracy (top-right = accurate & robust)")
    lims = [min(clean_vals + attacked_vals + [0]) - 5, max(clean_vals + attacked_vals + [100]) + 5]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_worst_case_bars(rows, save_path):
    """Worst-case accuracy (min over all conditions) per model."""
    plt = _ensure_mpl()
    if plt is None or save_path is None or not any("worst_case_acc" in d for _, d in rows):
        return
    names = [r[0] for r in rows]
    vals = [r[1].get("worst_case_acc") or 0 for r in rows]
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    bars = ax.bar(names, vals, color=[colors[i % len(colors)] for i in range(len(names))], alpha=0.85)
    ax.set_ylabel("Worst-case accuracy %")
    ax.set_title("Minimum accuracy over all eval conditions")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_dashboard(rows, save_path, baseline_name="B0", pmh_name="E1"):
    """Dashboard: 2×2 or 2×3 panels (curves, edge removal, consistency, AUC, worst-case, improvement)."""
    plt = _ensure_mpl()
    if plt is None or save_path is None or len(rows) < 2:
        return
    n_models = len(rows)
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    # (0,0) Noise curve (main + extra)
    ax = axes[0, 0]
    xs = [0] + [s for s in NOISE_LEVELS if s > 0] + list(NOISE_LEVELS_EXTRA)
    for i, (name, d) in enumerate(rows):
        ys = [d.get("clean") or d.get("clean_acc") or 0] + [d.get(f"sigma_{s:.2f}") or 0 for s in xs if s > 0]
        ax.plot(xs, ys, "o-", label=name, linewidth=1.5, markersize=3, color=colors[i % len(colors)])
    ax.set_xlabel("σ")
    ax.set_ylabel("Acc %")
    ax.set_title("Noise (incl. extrapolation)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    # (0,1) Edge removal
    ax = axes[0, 1]
    x_er = range(len(EDGE_DROP_RATES))
    for i, (name, d) in enumerate(rows):
        ys = _get_ys_edge_drop(d)
        ax.plot(x_er, ys, "o-", label=name, linewidth=1.5, markersize=3, color=colors[i % len(colors)])
    ax.set_xticks(x_er)
    ax.set_xticklabels([f"{int(f*100)}%" for f in EDGE_DROP_RATES])
    ax.set_xlabel("Edge drop")
    ax.set_ylabel("Acc %")
    ax.set_title("Edge removal")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    # (0,2) Consistency
    ax = axes[0, 2]
    if any("consistency_pct" in d for _, d in rows):
        names = [r[0] for r in rows]
        vals = [r[1].get("consistency_pct") or 0 for r in rows]
        ax.bar(names, vals, color=[colors[i % len(colors)] for i in range(len(names))], alpha=0.85)
    ax.set_ylabel("Consistency %")
    ax.set_title("Prediction consistency")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    # (1,0) AUC
    ax = axes[1, 0]
    if any("auc_noise" in d for _, d in rows):
        names = [r[0] for r in rows]
        vals = [r[1].get("auc_noise") or 0 for r in rows]
        ax.bar(names, vals, color=[colors[i % len(colors)] for i in range(len(names))], alpha=0.85)
    ax.set_ylabel("AUC")
    ax.set_title("Robustness AUC (noise)")
    ax.grid(True, alpha=0.3, axis="y")
    # (1,1) Worst-case
    ax = axes[1, 1]
    if any("worst_case_acc" in d for _, d in rows):
        names = [r[0] for r in rows]
        vals = [r[1].get("worst_case_acc") or 0 for r in rows]
        ax.bar(names, vals, color=[colors[i % len(colors)] for i in range(len(names))], alpha=0.85)
    ax.set_ylabel("Acc %")
    ax.set_title("Worst-case accuracy")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    # (1,2) E1−B0 improvement (noise levels)
    ax = axes[1, 2]
    base_row = next((r for r in rows if r[0] == baseline_name), None)
    pmh_row = next((r for r in rows if r[0] == pmh_name), None)
    if base_row and pmh_row and base_row != pmh_row:
        _, base_d = base_row
        _, pmh_d = pmh_row
        base_ys = _get_ys(baseline_name, base_d)
        pmh_ys = _get_ys(pmh_name, pmh_d)
        diff = [pmh_ys[i] - base_ys[i] for i in range(len(NOISE_LEVELS))]
        colors_bar = ["#2ecc71" if d >= 0 else "#e74c3c" for d in diff]
        ax.bar([f"σ={s:.2f}" if s > 0 else "Clean" for s in NOISE_LEVELS], diff, color=colors_bar, alpha=0.85)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_ylabel(f"{pmh_name}−{baseline_name} %")
    ax.set_title("PMH improvement over baseline")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_single_robustness(results, save_path, name=""):
    """Single model: accuracy vs node feature noise σ."""
    plt = _ensure_mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    xs = NOISE_LEVELS
    ys = _get_ys(name, results) if name else [results.get("clean") or results.get("clean_acc") or 0] + [results.get(f"sigma_{s:.2f}") or results.get(f"acc_sigma_{s:.2f}") or 0 for s in xs if s > 0]
    if len(ys) != len(xs):
        ys = [results.get("clean") or results.get("clean_acc") or 0] + [results.get(f"sigma_{s:.2f}") or results.get(f"acc_sigma_{s:.2f}") or 0 for s in xs if s > 0]
    ys = [y if y is not None else 0 for y in ys]
    ax.plot(xs, ys, "o-", linewidth=2, markersize=6, color="#3498db")
    ax.set_xlabel("Node feature noise σ")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness to node feature noise" + (f" ({name})" if name else ""))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="*", default=None, help="one or more .pt checkpoints")
    p.add_argument("--aggregate", nargs="+", default=None, help="glob of eval_*.json or results.json to plot comparison only")
    p.add_argument("--dataset", type=str, default="PROTEINS", choices=list(DATASETS), help="PROTEINS or ENZYMES recommended; MUTAG has ~20 test graphs (high variance)")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default="runs/evals", help="where to save eval JSONs and plots")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128, help="Must match training; default 128 (use 64 for old checkpoints)")
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_noise_seeds", type=int, default=5, help="eval each σ>0 with this many seeds, report mean±std (reduces variance, e.g. MUTAG has ~20 test graphs)")
    p.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Dataset train/val/test split seed; must match replication train.py --seed",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Aggregate: load JSONs and plot comparison only ---
    if args.aggregate:
        files = []
        for x in args.aggregate:
            # On Windows, glob may need path with forward slashes; normalize for cross-platform
            pat = os.path.normpath(x)
            files.extend(glob.glob(pat))
        files = sorted(set(files))
        # If no files and pattern looks like dataset/evals, try default runs/evals (where eval writes by default)
        if not files and args.aggregate:
            fallback = os.path.normpath("runs/evals/eval_*.json")
            files = sorted(glob.glob(fallback))
            if files:
                print(f"No files matched {args.aggregate}; using {fallback} instead.")
        if not files:
            print("No JSON files found for --aggregate.")
            print(f"  Tried: {args.aggregate}")
            print("  If you used default out_dir, try: --aggregate runs/evals/eval_*.json")
            return
        rows = []
        for f in files:
            with open(f) as fp:
                d = json.load(fp)
            name = os.path.splitext(os.path.basename(f))[0].replace("eval_", "").replace("results_", "")
            if "run" in d:
                name = d["run"]
            rows.append((name, d))
        plot_comparison_curves(rows, os.path.join(args.out_dir, "comparison_curves.png"))
        plot_accuracy_bars(rows, os.path.join(args.out_dir, "accuracy_bars.png"))
        plot_edge_removal_bars(rows, os.path.join(args.out_dir, "accuracy_under_edge_removal.png"))
        plot_edge_addition_bars(rows, os.path.join(args.out_dir, "accuracy_under_edge_addition.png"))
        plot_feature_dropout_bars(rows, os.path.join(args.out_dir, "accuracy_under_feature_dropout.png"))
        plot_combined_bars(rows, os.path.join(args.out_dir, "accuracy_under_combined_attack.png"))
        plot_extrapolation_curves(rows, os.path.join(args.out_dir, "accuracy_extrapolation_noise.png"))
        plot_consistency_bars(rows, os.path.join(args.out_dir, "consistency_under_noise.png"))
        plot_combined_noise_fd_bars(rows, os.path.join(args.out_dir, "accuracy_combined_noise_fd.png"))
        plot_heatmap(rows, os.path.join(args.out_dir, "heatmap_accuracy.png"))
        plot_robustness_auc_bars(rows, os.path.join(args.out_dir, "robustness_auc_bars.png"))
        plot_scatter_clean_vs_attacked(rows, os.path.join(args.out_dir, "scatter_clean_vs_sigma02.png"), attack_key="sigma_0.20", attack_label="σ=0.2")
        plot_worst_case_bars(rows, os.path.join(args.out_dir, "worst_case_accuracy.png"))
        plot_dashboard(rows, os.path.join(args.out_dir, "dashboard.png"))
        plot_improvement_over_baseline(rows, baseline_name="B0", pmh_name="E1", save_path=os.path.join(args.out_dir, "improvement_over_baseline.png"))
        print_summary_table(rows, args.out_dir)
        return

    checkpoints = [c for c in (args.checkpoint or []) if c and os.path.isfile(c)]
    if not checkpoints:
        print("Usage: python eval.py --dataset PROTEINS runs/PROTEINS/B0/best.pt runs/PROTEINS/E1/best.pt")
        print("   or: python eval.py --aggregate runs/evals/eval_*.json  (default: saves to runs/evals/)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, info = get_loaders(
        args.dataset, root=args.data_dir, batch_size=args.batch_size, seed=args.split_seed
    )
    num_node_features = info["num_node_features"]
    num_classes = info["num_classes"]

    rows_full = []
    for ckpt in checkpoints:
        stem = os.path.splitext(os.path.basename(ckpt))[0]
        name = stem.replace("best_", "") if "best_" in stem else stem
        if os.path.basename(os.path.dirname(ckpt)) in ("B0", "B1", "VAT", "E1", "E2"):
            name = os.path.basename(os.path.dirname(ckpt))
        print(f"Evaluating {name} ({ckpt})...")
        model = get_model(num_node_features, num_classes, hidden=args.hidden, num_layers=args.num_layers).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        results = {"clean": evaluate(model, test_loader, device, 0.0)}
        for s in NOISE_LEVELS:
            if s > 0:
                base_seed = EVAL_SEED_BASE + int(round(s * 1000))
                accs = []
                for run_idx in range(args.num_noise_seeds):
                    seed_sigma = base_seed + run_idx * 100
                    accs.append(evaluate(model, test_loader, device, s, seed=seed_sigma))
                results[f"sigma_{s:.2f}"] = float(np.mean(accs))
                if args.num_noise_seeds > 1:
                    results[f"sigma_{s:.2f}_std"] = float(np.std(accs))
        for s in NOISE_LEVELS_EXTRA:
            base_seed = EVAL_SEED_BASE + 300000 + int(round(s * 1000))
            accs = []
            for run_idx in range(args.num_noise_seeds):
                accs.append(evaluate(model, test_loader, device, s, seed=base_seed + run_idx * 100))
            results[f"sigma_{s:.2f}"] = float(np.mean(accs))
            if args.num_noise_seeds > 1:
                results[f"sigma_{s:.2f}_std"] = float(np.std(accs))
        seed_cons = EVAL_SEED_BASE + 400000
        cons_pct, cons_acc = evaluate_consistency(model, test_loader, device, noise_sigma=CONSISTENCY_SIGMA, K=CONSISTENCY_K, seed=seed_cons)
        results["consistency_pct"] = float(cons_pct)
        results["consistency_mean_acc"] = float(cons_acc)
        # Edge removal (unseen at train) — generalization to structure attack
        for f in EDGE_DROP_RATES:
            if f == 0:
                results["edge_drop_0.0"] = results["clean"]
            else:
                base_seed = EVAL_SEED_BASE + 50000 + int(round(f * 1000))
                accs = []
                for run_idx in range(args.num_noise_seeds):
                    seed_er = base_seed + run_idx * 100
                    accs.append(evaluate_under_edge_removal(model, test_loader, device, drop_frac=f, seed=seed_er))
                results[f"edge_drop_{f:.2f}"] = float(np.mean(accs))
                if args.num_noise_seeds > 1:
                    results[f"edge_drop_{f:.2f}_std"] = float(np.std(accs))
        # Edge addition (unseen at train; can drop B1 a lot)
        for f in EDGE_ADD_RATES:
            if f == 0:
                results["edge_add_0.0"] = results["clean"]
            else:
                base_seed = EVAL_SEED_BASE + 70000 + int(round(f * 1000))
                accs = []
                for run_idx in range(args.num_noise_seeds):
                    seed_ea = base_seed + run_idx * 100
                    accs.append(evaluate_under_edge_addition(model, test_loader, device, add_frac=f, seed=seed_ea))
                results[f"edge_add_{f:.2f}"] = float(np.mean(accs))
                if args.num_noise_seeds > 1:
                    results[f"edge_add_{f:.2f}_std"] = float(np.std(accs))
        # Feature dropout at test (representation stability / missing features)
        for f in FEATURE_DROP_RATES:
            if f == 0:
                results["feature_drop_0.0"] = results["clean"]
            else:
                base_seed = EVAL_SEED_BASE + 100000 + int(round(f * 1000))
                accs = []
                for run_idx in range(args.num_noise_seeds):
                    seed_fd = base_seed + run_idx * 100
                    accs.append(evaluate_under_feature_dropout(model, test_loader, device, drop_frac=f, seed=seed_fd))
                results[f"feature_drop_{f:.2f}"] = float(np.mean(accs))
                if args.num_noise_seeds > 1:
                    results[f"feature_drop_{f:.2f}_std"] = float(np.std(accs))
        # Combined attack (noise + edge removal; E1 trained for both)
        for (noise_sigma, edge_drop) in COMBINED_ATTACKS:
            key = f"combined_s{noise_sigma:.2f}_e{int(edge_drop*100)}"
            base_seed = EVAL_SEED_BASE + 200000 + int(round(noise_sigma * 1000)) + int(round(edge_drop * 1000))
            accs = []
            for run_idx in range(args.num_noise_seeds):
                seed_c = base_seed + run_idx * 100
                accs.append(evaluate_combined(model, test_loader, device, noise_sigma=noise_sigma, edge_drop_frac=edge_drop, seed=seed_c))
            results[key] = float(np.mean(accs))
            if args.num_noise_seeds > 1:
                results[key + "_std"] = float(np.std(accs))
        for (ns, fd) in COMBINED_NOISE_FD:
            key = f"noise_fd_s{ns:.2f}_f{int(fd*100)}"
            base_seed = EVAL_SEED_BASE + 80000 + int(round(ns * 1000)) + int(round(fd * 1000))
            accs = []
            for run_idx in range(args.num_noise_seeds):
                accs.append(evaluate_combined_noise_fd(model, test_loader, device, noise_sigma=ns, fd_frac=fd, seed=base_seed + run_idx * 100))
            results[key] = float(np.mean(accs))
            if args.num_noise_seeds > 1:
                results[key + "_std"] = float(np.std(accs))
        # Robustness AUC (accuracy vs sigma 0..0.2) and worst-case accuracy
        sigmas_auc = NOISE_LEVELS
        accs_auc = [results.get("clean") or results.get("clean_acc")] + [results.get(f"sigma_{s:.2f}") or 0 for s in sigmas_auc if s > 0]
        if len(accs_auc) == len(sigmas_auc) and sigmas_auc[-1] > sigmas_auc[0]:
            results["auc_noise"] = float(np.trapezoid(accs_auc, sigmas_auc) / (sigmas_auc[-1] - sigmas_auc[0]))
        else:
            results["auc_noise"] = float(np.mean(accs_auc)) if accs_auc else 0.0
        acc_list = [results.get("clean") or results.get("clean_acc")] + [results.get(f"sigma_{s:.2f}") for s in NOISE_LEVELS + NOISE_LEVELS_EXTRA if s > 0] + [results.get(f"edge_drop_{f:.2f}") for f in EDGE_DROP_RATES if f > 0] + [results.get(f"feature_drop_{f:.2f}") for f in FEATURE_DROP_RATES if f > 0]
        acc_list = [a for a in acc_list if a is not None]
        results["worst_case_acc"] = float(min(acc_list)) if acc_list else 0.0
        out_path = os.path.join(args.out_dir, f"eval_{name}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        rows_full.append((name, results))
        er_10 = results.get("edge_drop_0.10", results.get("edge_drop_0.1"))
        _safe_print(
            f"  Clean: {results['clean']:.2f}%  sigma=0.1: {results.get('sigma_0.10', 0):.2f}%  edge_10%: {er_10:.2f}%"
        )
        print(f"  Saved: {out_path}")

    # Plots and table
    if len(rows_full) >= 2:
        plot_comparison_curves(rows_full, os.path.join(args.out_dir, "comparison_curves.png"), title="Accuracy under node feature noise")
        plot_accuracy_bars(rows_full, os.path.join(args.out_dir, "accuracy_bars.png"))
        plot_edge_removal_bars(rows_full, os.path.join(args.out_dir, "accuracy_under_edge_removal.png"))
        plot_edge_addition_bars(rows_full, os.path.join(args.out_dir, "accuracy_under_edge_addition.png"))
        plot_feature_dropout_bars(rows_full, os.path.join(args.out_dir, "accuracy_under_feature_dropout.png"))
        plot_combined_bars(rows_full, os.path.join(args.out_dir, "accuracy_under_combined_attack.png"))
        plot_extrapolation_curves(rows_full, os.path.join(args.out_dir, "accuracy_extrapolation_noise.png"))
        plot_consistency_bars(rows_full, os.path.join(args.out_dir, "consistency_under_noise.png"))
        plot_combined_noise_fd_bars(rows_full, os.path.join(args.out_dir, "accuracy_combined_noise_fd.png"))
        plot_heatmap(rows_full, os.path.join(args.out_dir, "heatmap_accuracy.png"))
        plot_robustness_auc_bars(rows_full, os.path.join(args.out_dir, "robustness_auc_bars.png"))
        plot_scatter_clean_vs_attacked(rows_full, os.path.join(args.out_dir, "scatter_clean_vs_sigma02.png"), attack_key="sigma_0.20", attack_label="σ=0.2")
        plot_worst_case_bars(rows_full, os.path.join(args.out_dir, "worst_case_accuracy.png"))
        plot_dashboard(rows_full, os.path.join(args.out_dir, "dashboard.png"))
        plot_improvement_over_baseline(rows_full, baseline_name="B0", pmh_name="E1", save_path=os.path.join(args.out_dir, "improvement_over_baseline.png"))
        print_summary_table(rows_full, args.out_dir)
    else:
        name = rows_full[0][0]
        plot_single_robustness(rows_full[0][1], os.path.join(args.out_dir, f"robustness_{name}.png"), name)


if __name__ == "__main__":
    main()
