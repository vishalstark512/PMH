"""
Bound Tightness Check — Theorem 1 / Corollary 2
================================================
Checks whether the theoretical lower bound on embedding drift is tight:

    D(phi*, sigma) >= sigma^2 * I(n; y) / L^2       [Corollary 2, cross-entropy]

Measurements for Task 04 (ViT-Small / CIFAR-10, B0 model):

  D_ERM(sigma)   — actual expected embedding drift E||phi(x+delta)-phi(x)||^2
  L              — spectral norm (largest singular value) of the final linear head
  I(n; y)        — mutual information between nuisance features n(x) and label y,
                   estimated from patch embeddings (before any transformer block),
                   which capture low-level texture/colour that correlates with label
                   but are semantically irrelevant (the nuisance in Definition 1).

Tightness ratio = D_ERM(sigma) / [sigma^2 * I(n;y) / L^2]
  ~ 1  => bound is tight (theorem is quantitatively informative)
  >> 1 => bound is loose (theorem is directionally correct but numerically slack)

Also computes Anisotropy A(phi) = Jac_Fro / TDI@sigma for each run, confirming
the new Proposition (see THEOREM_EXTENSIONS.md): A(phi) >= 1 for all phi.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import get_model, ViTCIFAR

CIFAR10_MEAN = (0.4914, 0.4565, 0.4067)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
N_CLASSES    = 10


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_test_loader(data_dir, batch_size=256, max_samples=None, num_workers=4):
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
# 1. Spectral norm L of the classification head
# ---------------------------------------------------------------------------

def head_spectral_norm(model: ViTCIFAR) -> float:
    """Largest singular value of head.weight  (shape: num_classes x embed_dim)."""
    W = model.head.weight.detach().float()      # (10, 192)
    s = torch.linalg.svdvals(W)
    return float(s[0])


# ---------------------------------------------------------------------------
# 2. Actual embedding drift D(phi, sigma) = E||phi(x+delta) - phi(x)||^2
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_embedding_drift(model, loader, device, sigmas, n_reps=5):
    """
    Returns dict sigma -> mean_drift.
    phi(x) = model's CLS embedding after norm layer (before head).
    delta ~ N(0, sigma^2 I) added in pixel space before normalisation.
    """
    mean_t = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(CIFAR10_STD,  device=device).view(1, 3, 1, 1)

    drift_acc = {s: [] for s in sigmas}

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        # Clean embeddings: norm(CLS)
        feat_clean = _get_normed_cls(model, images)          # (B, D)

        for sigma in sigmas:
            if sigma == 0.0:
                drift_acc[sigma].append(0.0)
                continue
            rep_drifts = []
            for _ in range(n_reps):
                # Perturb in pixel space
                x_pix   = images * std_t + mean_t
                x_noisy = (x_pix + sigma * torch.randn_like(x_pix)).clamp(0, 1)
                x_norm  = (x_noisy - mean_t) / std_t
                feat_noisy = _get_normed_cls(model, x_norm)  # (B, D)
                drift = (feat_noisy - feat_clean).pow(2).sum(dim=1).mean()
                rep_drifts.append(float(drift))
            drift_acc[sigma].append(float(np.mean(rep_drifts)))

    return {s: float(np.mean(v)) for s, v in drift_acc.items()}


def _get_normed_cls(model, x):
    """Returns the LayerNorm-normalised CLS token (the input to head)."""
    with torch.no_grad():
        feats = model.get_features(x, return_all=False)   # (B, D) — already normed
    return feats


# ---------------------------------------------------------------------------
# 3. Estimate I(n; y) from patch embeddings (nuisance features)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_patch_features(model, loader, device):
    """
    Collect patch-level features BEFORE transformer blocks as n(x):
    average of all patch tokens after patch_embed + pos_embed (no blocks).
    Shape: (N, embed_dim).  Also return labels.
    """
    mean_t = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(CIFAR10_STD,  device=device).view(1, 3, 1, 1)
    feats, labs = [], []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        B = images.shape[0]
        x = model.patch_embed(images)                          # (B, n_patches, D)
        cls = model.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + model.pos_embed    # (B, 1+n_patches, D)
        # Mean-pool patch tokens (exclude CLS, index 0) — texture summary
        patch_mean = x[:, 1:, :].mean(dim=1)                  # (B, D)
        feats.append(patch_mean.cpu().float())
        labs.append(y)
    return torch.cat(feats, 0), torch.cat(labs, 0)


def estimate_mutual_information(features: torch.Tensor, labels: torch.Tensor, n_classes=10) -> dict:
    """
    Estimate I(n; y) via a linear probe trained on features.

    I(n; y) = H(y) - H(y | n)
            = log(K) - cross_entropy_of_best_linear_predictor_of_y_from_n

    We use sklearn logistic regression with L2 reg for the linear probe.
    Returns:
      rho_sq:  ρ² ≈ 2*I(n;y) (valid for small I, Gaussian approximation)
      I_nats:  I(n; y) in nats
      probe_acc: accuracy of linear probe (%)
      probe_ce:  cross-entropy of linear probe (nats)  — approximate H(y|n)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import log_loss

    X = features.numpy()
    y = labels.numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=0)
    # 5-fold cross-validated probabilities to avoid overfitting
    proba = cross_val_predict(clf, X_scaled, y, cv=5, method="predict_proba")
    probe_ce  = float(log_loss(y, proba))            # H(y|n) estimate (nats)
    probe_acc = float((proba.argmax(1) == y).mean() * 100)

    H_y  = math.log(n_classes)                      # H(y) for balanced classes
    I_ny = max(0.0, H_y - probe_ce)                 # I(n; y) in nats

    # Gaussian approximation: I ≈ ρ²/2  =>  ρ² ≈ 2I
    rho_sq = 2.0 * I_ny

    return {
        "H_y_nats":     H_y,
        "probe_ce_nats": probe_ce,
        "I_ny_nats":    I_ny,
        "rho_sq_approx": rho_sq,
        "probe_acc_pct": probe_acc,
    }


# ---------------------------------------------------------------------------
# 4. Anisotropy A(phi) from existing results.json / TDI & Jacobian data
# ---------------------------------------------------------------------------

PAPER_TABLE1 = {
    "B0":       {"TDI_0": 1.074, "Jac_Fro": 34.10},
    "VAT":      {"TDI_0": 1.281, "Jac_Fro": 4.93},
    "E1_no_pmh":{"TDI_0": 1.011, "Jac_Fro": 13.32},
    "E1":       {"TDI_0": 0.858, "Jac_Fro": 10.84},
    "PGD_4_255":{"TDI_0": 1.336, "Jac_Fro": 2.91},
}


def compute_anisotropy():
    rows = []
    for name, d in PAPER_TABLE1.items():
        A = d["Jac_Fro"] / d["TDI_0"]
        rows.append({"run": name, "Jac_Fro": d["Jac_Fro"], "TDI_0": d["TDI_0"], "A": A})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Theorem 1 bound tightness check")
    p.add_argument("--runs_dir",    default="runs")
    p.add_argument("--run",         default="B0",  help="Run to check (B0 is ERM baseline)")
    p.add_argument("--data_dir",    default="./data")
    p.add_argument("--max_samples", type=int, default=5000)
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--sigmas",      type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.15, 0.2])
    p.add_argument("--n_reps",      type=int, default=5,
                   help="Number of noise realisations to average per sample")
    p.add_argument("--out_dir",     default="runs/interp")
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.runs_dir) / args.run / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\nDevice: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Samples:    {args.max_samples}  |  batch: {args.batch_size}")
    print(f"Sigmas:     {args.sigmas}")

    loader = get_test_loader(args.data_dir, args.batch_size, args.max_samples)
    model  = load_model(ckpt_path, device)

    # ------------------------------------------------------------------
    # 1. Spectral norm L
    # ------------------------------------------------------------------
    L = head_spectral_norm(model)
    print(f"\n[1] Spectral norm of head (L): {L:.6f}")

    # ------------------------------------------------------------------
    # 2. Embedding drift D(phi, sigma)
    # ------------------------------------------------------------------
    print(f"\n[2] Measuring embedding drift D(phi, sigma)...")
    drift = measure_embedding_drift(model, loader, device, args.sigmas, n_reps=args.n_reps)
    for s, d in drift.items():
        print(f"    sigma={s:.2f}  D_ERM={d:.6f}")

    # ------------------------------------------------------------------
    # 3. Mutual information I(n; y)
    # ------------------------------------------------------------------
    print(f"\n[3] Estimating I(n; y) from patch embeddings (nuisance features)...")
    patch_feats, labels = collect_patch_features(model, loader, device)
    mi = estimate_mutual_information(patch_feats, labels)
    print(f"    H(y)             = {mi['H_y_nats']:.4f} nats  ({math.exp(mi['H_y_nats']):.1f} = K classes, check)")
    print(f"    Probe CE H(y|n)  = {mi['probe_ce_nats']:.4f} nats")
    print(f"    I(n; y)          = {mi['I_ny_nats']:.4f} nats")
    print(f"    rho_sq (approx)  = {mi['rho_sq_approx']:.4f}")
    print(f"    Linear probe acc = {mi['probe_acc_pct']:.2f}%  (texture->label)")

    # ------------------------------------------------------------------
    # 4. Bound tightness comparison
    # ------------------------------------------------------------------
    print(f"\n[4] Bound tightness: D_ERM vs sigma^2 * I(n;y) / L^2")
    print(f"    L^2 = {L**2:.6f}   I(n;y) = {mi['I_ny_nats']:.6f}")
    print()
    print(f"    {'sigma':>6}  {'D_ERM':>12}  {'Bound':>12}  {'Tightness':>12}  {'Note'}")
    print(f"    {'-'*60}")
    tightness_rows = []
    for s in args.sigmas:
        if s == 0.0:
            continue
        bound = (s**2) * mi["I_ny_nats"] / (L**2)
        d_erm = drift[s]
        ratio = d_erm / bound if bound > 0 else float("inf")
        note  = "TIGHT(<10x)" if ratio < 10 else ("LOOSE(<100x)" if ratio < 100 else "VERY LOOSE")
        print(f"    {s:>6.2f}  {d_erm:>12.6f}  {bound:>12.6f}  {ratio:>12.2f}x  {note}")
        tightness_rows.append({
            "sigma": s, "D_ERM": d_erm, "bound": bound, "tightness_ratio": ratio
        })
    print()

    # ------------------------------------------------------------------
    # 5. Anisotropy A(phi) = Jac_Fro / TDI@0
    # ------------------------------------------------------------------
    print(f"[5] Anisotropy A(phi) = Jac_Fro / TDI@0  (from Table 1)")
    print(f"    Claim: A(phi) >= 1  for all phi  (Proposition, THEOREM_EXTENSIONS.md)")
    print()
    aniso = compute_anisotropy()
    print(f"    {'Run':<14}  {'Jac_Fro':>9}  {'TDI@0':>7}  {'A':>7}  {'A>=1?'}")
    print(f"    {'-'*50}")
    all_ge_1 = True
    for row in aniso:
        ok = "YES" if row["A"] >= 1.0 else "FAIL"
        if row["A"] < 1.0:
            all_ge_1 = False
        print(f"    {row['run']:<14}  {row['Jac_Fro']:>9.2f}  {row['TDI_0']:>7.3f}  {row['A']:>7.2f}  {ok}")
    print(f"\n    Proposition A(phi)>=1 holds for all runs: {all_ge_1}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = {
        "run":             args.run,
        "max_samples":     args.max_samples,
        "spectral_norm_L": L,
        "L_sq":            L**2,
        "mutual_info":     mi,
        "embedding_drift": drift,
        "tightness":       tightness_rows,
        "anisotropy":      aniso,
    }
    out_path = out_dir / "bound_tightness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
