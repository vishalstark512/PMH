"""
Nuisance Subspace Bound — Tightening Weakness 1 and 2
======================================================
Decomposes the total embedding drift D_ERM into:

  D_total  = sigma^2 * E[||J_phi||^2_F]          (all 3072 input dimensions)
  D_top_r  = sigma^2 * sum_{k=1}^r E[||J_phi w_k||^2]  (top-r nuisance directions)

The nuisance directions {w_k} are the top-r right singular vectors of the
input-gradient matrix G, where G[i] = nabla_x CE(f(x_i), y_i). These are the
input-space directions the network is MOST sensitive to for label prediction —
the operational definition of the nuisance subspace.

Tighter multi-directional bound:
  D_top_r >= sigma^2 * r * rho_min^2 * C(P) / L^2
           ~= sigma^2 * I_total(n;y) / L^2

where I_total = sum_k I_k(n_k; y) grows with r.

Key question: what fraction of the 100,000x gap closes when we account for
the full nuisance subspace (r > 1)?
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import get_model

CIFAR10_MEAN = (0.4914, 0.4565, 0.4067)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
N_CLASSES    = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_loader(data_dir, batch_size=128, max_samples=None, num_workers=4):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = CIFAR10(root=data_dir, train=False, download=True, transform=t)
    if max_samples:
        ds = Subset(ds, np.arange(min(max_samples, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def load_model(ckpt_path, device):
    model = get_model(num_classes=N_CLASSES).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1. Collect input-space gradients to identify nuisance subspace
# ---------------------------------------------------------------------------

def collect_input_gradients(model, loader, device, max_batches=None):
    """
    Compute nabla_x CE(f(x), y) for each (x, y) in loader.
    Returns gradient matrix G of shape (N, d_x) where d_x = 3*32*32 = 3072.
    """
    model.eval()
    grads = []
    for bi, (images, labels) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        images = images.to(device).requires_grad_(True)
        labels = labels.to(device)
        logits = model(images)
        loss   = F.cross_entropy(logits, labels, reduction="sum")
        loss.backward()
        g = images.grad.detach().reshape(images.shape[0], -1)  # (B, 3072)
        grads.append(g.cpu().float())
        images.grad = None
    return torch.cat(grads, dim=0)   # (N, 3072)


# ---------------------------------------------------------------------------
# 2. Find top-r nuisance directions via randomized SVD
# ---------------------------------------------------------------------------

def top_nuisance_directions(G: torch.Tensor, r: int) -> torch.Tensor:
    """
    G: (N, d_x) gradient matrix.
    Returns W: (d_x, r) whose columns are the top-r right singular vectors
    of G / sqrt(N) — the input-space directions most responsible for label
    prediction.
    Uses randomized SVD for efficiency when d_x = 3072 and N = 5000.
    """
    # Randomized SVD of G (thin, top-r)
    U, S, Vt = torch.linalg.svd(G / math.sqrt(G.shape[0]), full_matrices=False)
    # Vt: (r, d_x); rows are top right singular vectors
    W = Vt[:r, :].T   # (d_x, r), columns are nuisance directions
    return W, S[:r]


# ---------------------------------------------------------------------------
# 3. Directional drift: E[||J_phi w||^2] via forward differences
# ---------------------------------------------------------------------------

@torch.no_grad()
def directional_drift(model, loader, device, directions: torch.Tensor,
                      sigma: float, eps: float = 1e-3, n_reps: int = 3) -> torch.Tensor:
    """
    Estimate D_k = sigma^2 * E[||J_phi(x) w_k||^2] for each column w_k in directions.
    Uses forward difference: J_phi(x) w_k ~= (phi(x + eps*w_k) - phi(x)) / eps.

    directions: (d_x, r) on CPU.
    Returns: (r,) tensor of directional drift values at the given sigma.
    """
    mean_t = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(CIFAR10_STD,  device=device).view(1, 3, 1, 1)
    d_x, r = directions.shape

    drift_acc = torch.zeros(r)
    n_total   = 0

    for images, _ in loader:
        images = images.to(device)
        B      = images.shape[0]

        # Clean embedding: phi(x)
        feat_clean = model.get_features(images, return_all=False)  # (B, D)

        for k in range(r):
            w_k = directions[:, k].to(device).view(1, 3, 32, 32)   # (1, 3, 32, 32)
            w_k_norm = w_k / (w_k.norm() + 1e-12)

            rep_vals = []
            for _ in range(n_reps):
                # Perturb x in direction w_k in pixel space
                x_pix      = images * std_t + mean_t
                x_perturb  = (x_pix + eps * w_k_norm).clamp(0, 1)
                x_perturb_norm = (x_perturb - mean_t) / std_t
                feat_p = model.get_features(x_perturb_norm, return_all=False)

                # Directional Jacobian: (phi(x + eps w_k) - phi(x)) / eps
                Jw = (feat_p - feat_clean) / eps                     # (B, D)
                # sigma^2 * E[||Jw||^2]
                d_val = (sigma**2) * Jw.pow(2).sum(dim=1).mean()
                rep_vals.append(float(d_val))

            drift_acc[k] += float(np.mean(rep_vals)) * B
        n_total += B

    return drift_acc / n_total


@torch.no_grad()
def total_embedding_drift(model, loader, device, sigma: float, n_reps: int = 3) -> float:
    """E[||phi(x+delta) - phi(x)||^2] at the given sigma."""
    mean_t = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(CIFAR10_STD,  device=device).view(1, 3, 1, 1)
    acc, n = 0.0, 0
    for images, _ in loader:
        images = images.to(device)
        feat_clean = model.get_features(images, return_all=False)
        rep_vals = []
        for _ in range(n_reps):
            x_pix   = images * std_t + mean_t
            x_noisy = (x_pix + sigma * torch.randn_like(x_pix)).clamp(0, 1)
            x_norm  = (x_noisy - mean_t) / std_t
            feat_n  = model.get_features(x_norm, return_all=False)
            rep_vals.append(float((feat_n - feat_clean).pow(2).sum(dim=1).mean()))
        acc += float(np.mean(rep_vals)) * images.shape[0]
        n   += images.shape[0]
    return acc / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir",    default="runs")
    p.add_argument("--runs",        nargs="+", default=["B0", "E1"])
    p.add_argument("--data_dir",    default="./data")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--max_grad_batches", type=int, default=20,
                   help="Batches to use for gradient collection")
    p.add_argument("--r_values",    type=int, nargs="+", default=[1, 5, 10, 20, 50, 100],
                   help="Nuisance subspace dimensions to evaluate")
    p.add_argument("--sigma",       type=float, default=0.1,
                   help="Noise level for drift measurement")
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--n_reps",      type=int, default=3)
    p.add_argument("--out_dir",     default="runs/interp")
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bound values from prior run (check_bound_tightness.py)
    I_ny       = 0.2849   # nats
    L          = 1.4034   # spectral norm of head
    L_sq       = L ** 2
    scalar_bound = (args.sigma**2) * I_ny / L_sq
    print(f"\nSigma = {args.sigma}  |  Scalar bound (r=1): {scalar_bound:.6f}")
    print(f"I(n;y) = {I_ny:.4f} nats  L = {L:.4f}  L^2 = {L_sq:.4f}\n")

    r_max = max(args.r_values)
    all_results = {}

    for run_name in args.runs:
        ckpt_path = Path(args.runs_dir) / run_name / "best.pt"
        if not ckpt_path.exists():
            print(f"Skip {run_name}: {ckpt_path} not found")
            continue

        model  = load_model(ckpt_path, device)
        loader = get_loader(args.data_dir, args.batch_size, args.max_samples)

        print(f"{'='*60}")
        print(f"Run: {run_name}")
        print(f"{'='*60}")

        # --- Total drift ---
        D_total = total_embedding_drift(model, loader, device, args.sigma, args.n_reps)
        print(f"  D_total (sigma={args.sigma}): {D_total:.4f}")
        print(f"  Scalar bound:                {scalar_bound:.6f}")
        print(f"  Total/bound ratio:           {D_total/scalar_bound:.0f}x\n")

        # --- Gradient collection and SVD ---
        print(f"  Collecting input gradients ({args.max_grad_batches} batches)...")
        G = collect_input_gradients(model, loader, device, args.max_grad_batches)
        print(f"  Gradient matrix G: {G.shape}  (N x d_x)")

        W, S = top_nuisance_directions(G, min(r_max, G.shape[0], G.shape[1]))
        print(f"  Top singular values: {S[:10].numpy().round(4).tolist()}")

        # --- Directional drift for increasing r ---
        print(f"\n  Computing directional drift for r in {args.r_values}...")
        print(f"  {'r':>6}  {'D_top_r':>12}  {'D_top_r/D_total':>16}  {'D_top_r/bound':>14}  {'Tightness ratio':>16}")
        print(f"  {'-'*72}")

        # Compute individual directional drifts for all columns at once
        all_dir_drifts = directional_drift(
            model, loader, device, W[:, :min(r_max, W.shape[1])],
            args.sigma, n_reps=args.n_reps
        )  # (r_max,)

        run_rows = []
        for r in args.r_values:
            if r > W.shape[1]:
                break
            D_r = float(all_dir_drifts[:r].sum())
            frac = D_r / D_total if D_total > 0 else 0.0
            ratio_bound = D_r / scalar_bound if scalar_bound > 0 else float("inf")
            tightness   = D_total / D_r if D_r > 0 else float("inf")
            print(f"  {r:>6}  {D_r:>12.4f}  {frac:>15.4f}x  {ratio_bound:>14.0f}x  {tightness:>15.1f}x remaining")
            run_rows.append({
                "r": r, "D_r": D_r, "frac_of_total": frac,
                "D_r_over_bound": ratio_bound,
                "tightness_remaining": tightness,
            })

        print(f"\n  NOTE: at r={W.shape[1]} (full input dim = d_x), D_top_r = D_total by definition.")
        print(f"  The bound gap closes from {D_total/scalar_bound:.0f}x (total)")
        if run_rows:
            print(f"  to {run_rows[0]['D_r_over_bound']:.0f}x (r=1 nuisance direction)")

        all_results[run_name] = {
            "D_total":      D_total,
            "scalar_bound": scalar_bound,
            "total_ratio":  D_total / scalar_bound,
            "singular_values_top10": S[:10].numpy().tolist(),
            "directional_results": run_rows,
        }
        print()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Nuisance subspace tightening")
    print("="*70)
    print(f"{'Run':<10}  {'D_total':>10}  {'Bound':>10}  {'r=1 frac':>10}  {'r=10 frac':>11}")
    print(f"{'-'*60}")
    for run_name, res in all_results.items():
        rows = {r["r"]: r for r in res["directional_results"]}
        r1  = rows.get(1,  {}).get("frac_of_total", 0.0)
        r10 = rows.get(10, {}).get("frac_of_total", 0.0)
        print(f"{run_name:<10}  {res['D_total']:>10.4f}  {res['scalar_bound']:>10.6f}  {r1:>10.4f}  {r10:>11.4f}")

    out_path = out_dir / "nuisance_subspace_bound.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
