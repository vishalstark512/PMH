"""
Multi-Scale PMH Experiment — Weakness 4: Removing sigma_eval Dependence
========================================================================
Trains an E1_multiscale model where sigma is cycled through
{0.05, 0.08, 0.10, 0.12, 0.15, 0.20} each epoch instead of fixed sigma=0.12.

Hypothesis (T-alignment prediction):
  - Single-scale E1 (sigma=0.12): TDI is minimised at eval sigma=0.12,
    degrades asymmetrically at mismatch (Table 4 pattern).
  - Multi-scale E1: TDI should be more UNIFORM across all eval sigma levels —
    lower worst-case mismatch at the cost of slightly higher best-case TDI.

This directly closes Weakness 4: the practitioner no longer needs to know
sigma_eval before training. Multi-scale PMH gives a "robustly good" encoder
across a range of deployment shifts.

Usage:
  python train_multiscale_pmh.py [--epochs 100] [--data_dir ./data]

Outputs:
  runs/E1_multiscale/best.pt
  runs/E1_multiscale/results.json
  runs/interp/talignment_comparison.json   <- single-scale vs multi-scale
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import get_model

CIFAR10_MEAN = (0.4914, 0.4565, 0.4067)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# T-alignment evaluation parameters — match Table 4
EVAL_SIGMAS  = [0.0, 0.05, 0.10, 0.15, 0.20]
MULTI_SCHEDULE = "0.05,0.08,0.10,0.12,0.15,0.20"   # cycles each epoch


# ---------------------------------------------------------------------------
# T-alignment evaluation (TDI at multiple sigma levels)
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device):
    model = get_model(num_classes=10).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def get_eval_loader(data_dir, max_samples=2000, batch_size=128, num_workers=4):
    t  = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = CIFAR10(root=data_dir, train=False, download=True, transform=t)
    ds = Subset(ds, np.arange(min(max_samples, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def eval_tdi_and_acc(model, loader, device, sigma=0.0, n_reps=3):
    """Returns (TDI, accuracy) at a given noise sigma."""
    mean_t = torch.tensor(CIFAR10_MEAN, device=device).view(1,3,1,1)
    std_t  = torch.tensor(CIFAR10_STD,  device=device).view(1,3,1,1)
    model.eval()
    embs, labs, correct, total = [], [], 0, 0
    for x, y in loader:
        x = x.to(device)
        if sigma > 0:
            x_pix   = x * std_t + mean_t
            x_noisy = (x_pix + sigma * torch.randn_like(x_pix)).clamp(0, 1)
            x = (x_noisy - mean_t) / std_t
        feats  = model.get_features(x, return_all=False)
        logits = model.head(model.norm(feats))
        pred   = logits.argmax(1)
        correct += (pred.cpu() == y).sum().item()
        total   += y.size(0)
        embs.append(feats.cpu().float()); labs.append(y)
    embs = torch.cat(embs).numpy(); labs = torch.cat(labs).numpy()
    # TDI
    intra = []
    for c in range(10):
        X = embs[labs == c]
        if len(X) < 2: continue
        d = np.linalg.norm(X[:, None] - X[None, :], axis=2)
        idx = np.triu_indices(len(X), k=1)
        intra.extend(d[idx].tolist())
    centroids = [embs[labs==c].mean(0) for c in range(10) if (labs==c).sum()>0]
    C = np.stack(centroids)
    inter = [np.linalg.norm(C[i]-C[j]) for i in range(len(C)) for j in range(i+1, len(C))]
    intra_m = float(np.mean(intra)) if intra else 0.
    inter_m = float(np.mean(inter)) if inter else 1.
    tdi = intra_m / inter_m if inter_m > 0 else 0.
    acc = 100.*correct/total if total else 0.
    return round(tdi, 4), round(acc, 2)


def evaluate_talignment(model, loader, device, eval_sigmas=EVAL_SIGMAS) -> dict:
    """Evaluates TDI and acc at all eval sigma levels."""
    results = {}
    for sigma in eval_sigmas:
        tdi, acc = eval_tdi_and_acc(model, loader, device, sigma)
        results[sigma] = {"tdi": tdi, "acc": acc}
        print(f"    sigma={sigma:.2f}  TDI={tdi:.4f}  acc={acc:.2f}%")
    return results


# ---------------------------------------------------------------------------
# Train multi-scale PMH by calling train.py
# ---------------------------------------------------------------------------

def train_multiscale(data_dir, epochs, out_dir, python_exe=None):
    """Calls train.py with noise_schedule to train E1_multiscale."""
    exe = python_exe or sys.executable
    cmd = [
        exe, "train.py",
        "--run",            "E1_multiscale",
        "--epochs",         str(epochs),
        "--data_dir",       data_dir,
        "--out_dir",        out_dir,
        "--noise_schedule", MULTI_SCHEDULE,
        "--noise_sigma",    "0.12",   # fallback if schedule is not used
        "--batch_size",     "512",
        "--warmup_epochs",  "10",
        "--pmh_cap_ratio",  "0.25",
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        raise RuntimeError(f"train.py returned exit code {result.returncode}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--data_dir",    default="./data")
    p.add_argument("--runs_dir",    default="runs")
    p.add_argument("--out_dir",     default="runs/interp")
    p.add_argument("--eval_samples",type=int,   default=2000)
    p.add_argument("--skip_train",  action="store_true",
                   help="Skip training if E1_multiscale/best.pt exists")
    p.add_argument("--python",      default=None,
                   help="Python executable to use (default: current interpreter)")
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    multiscale_ckpt = Path(args.runs_dir) / "E1_multiscale" / "best.pt"

    # --- Train ---
    if not args.skip_train or not multiscale_ckpt.exists():
        print(f"\n{'='*60}")
        print(f"Training E1_multiscale  (schedule: {MULTI_SCHEDULE})")
        print(f"{'='*60}")
        train_multiscale(args.data_dir, args.epochs, args.runs_dir, args.python)
    else:
        print(f"\nSkipping training — using {multiscale_ckpt}")

    # --- T-alignment evaluation ---
    loader = get_eval_loader(args.data_dir, max_samples=args.eval_samples)

    # Models to compare
    run_specs = {
        "E1_sigma_0.05":  Path(args.runs_dir) / "E1_sigma_0.05" / "best.pt",
        "E1 (sigma=0.12)":Path(args.runs_dir) / "E1"            / "best.pt",
        "E1_sigma_0.20":  Path(args.runs_dir) / "E1_sigma_0.20" / "best.pt",
        "E1_multiscale":  multiscale_ckpt,
    }

    all_results = {}
    print(f"\n{'='*60}")
    print("T-ALIGNMENT COMPARISON: single-scale vs multi-scale PMH")
    print(f"{'='*60}")

    for run_name, ckpt_path in run_specs.items():
        if not ckpt_path.exists():
            print(f"\n  Skip {run_name}: {ckpt_path} not found")
            continue
        print(f"\n  {run_name}:")
        model = load_model(ckpt_path, device)
        results = evaluate_talignment(model, loader, device)
        all_results[run_name] = results

    # Print comparison table
    print(f"\n{'='*70}")
    print("SUMMARY: TDI at eval sigma levels")
    print(f"{'='*70}")
    header = f"{'Run':<20}" + "".join(f"  {s:.2f}" for s in EVAL_SIGMAS)
    print(header)
    print("-" * len(header))
    for run_name, res in all_results.items():
        row = f"{run_name:<20}" + "".join(
            f"  {res[s]['tdi']:>5.3f}" if s in res else "   N/A" for s in EVAL_SIGMAS
        )
        print(row)

    # Uniformity metric: std of TDI across eval sigmas (lower = more uniform)
    print(f"\n{'='*70}")
    print("TDI uniformity (std across eval sigmas, lower = more uniform = Weakness 4 solved)")
    print(f"{'='*70}")
    for run_name, res in all_results.items():
        tdis = [res[s]["tdi"] for s in EVAL_SIGMAS if s in res]
        if tdis:
            print(f"  {run_name:<22}  std={np.std(tdis):.4f}  max={max(tdis):.4f}  min={min(tdis):.4f}")

    # Save
    out_path = out_dir / "talignment_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        serialisable = {k: {str(s): v for s, v in res.items()} for k, res in all_results.items()}
        json.dump({"eval_sigmas": EVAL_SIGMAS, "results": serialisable}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
