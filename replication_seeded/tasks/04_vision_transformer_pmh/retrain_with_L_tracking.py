"""
L-Tracking Experiment — Weakness 3: Lipschitz Constant Verification
====================================================================
Retrains B0 (ERM) and E1 (PMH) from scratch while recording at each epoch:

  L_t  = sigma_max(head.weight)  -- spectral norm of classification head
  TDI  = intra_mean / inter_mean (eval on 1000 clean test samples)
  acc  = clean test accuracy

Key questions:
  1. Does L stay approximately constant during training? (retroactively justifies
     the paper's assumption that L is a fixed model property)
  2. Does PMH reduce L vs ERM? (if yes, part of TDI improvement comes from L
     reduction, not just Frobenius regularisation — needs to be reported)
  3. Is L_B0 ≈ L_E1 at convergence? (required for fair TDI comparison under
     the same L in the theoretical bound)

Training hyperparameters: identical to original runs (100 epochs, lr=1e-3,
batch=512, cosine LR, AMP).
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import get_model

CIFAR10_MEAN = (0.4914, 0.4565, 0.4067)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------------------
# Helpers (minimal copies from train.py / topological_distortion_index.py)
# ---------------------------------------------------------------------------

def get_loaders(data_dir, batch_size=512, num_workers=0):
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    train_ds = CIFAR10(root=data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
    kw = dict(batch_size=batch_size, pin_memory=False, num_workers=0)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(test_ds,  shuffle=False, **kw))


def get_tdi_loader(data_dir, max_samples=1000, batch_size=128, num_workers=0):
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ds   = CIFAR10(root=data_dir, train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor(), norm]))
    ds   = Subset(ds, np.arange(min(max_samples, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def head_spectral_norm(model) -> float:
    W = model.head.weight.detach().float()
    return float(torch.linalg.svdvals(W)[0])


@torch.no_grad()
def eval_accuracy(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return 100. * correct / total if total else 0.


@torch.no_grad()
def compute_tdi(model, loader, device) -> float:
    """TDI = intra_mean / inter_mean on embeddings."""
    model.eval()
    embs, labs = [], []
    for x, y in loader:
        x = x.to(device)
        feats = model.get_features(x, return_all=False)
        embs.append(feats.cpu().float()); labs.append(y)
    embs = torch.cat(embs).numpy(); labs = torch.cat(labs).numpy()
    intra_dists = []
    for c in range(10):
        X = embs[labs == c]
        if len(X) < 2: continue
        d = np.linalg.norm(X[:, None] - X[None, :], axis=2)
        idx = np.triu_indices(len(X), k=1)
        intra_dists.extend(d[idx].tolist())
    centroids = [embs[labs == c].mean(0) for c in range(10) if (labs == c).sum() > 0]
    centroids  = np.stack(centroids)
    inter_dists = [np.linalg.norm(centroids[i] - centroids[j])
                   for i in range(len(centroids)) for j in range(i+1, len(centroids))]
    intra_m = float(np.mean(intra_dists)) if intra_dists else 0.
    inter_m = float(np.mean(inter_dists)) if inter_dists else 1.
    return intra_m / inter_m if inter_m > 0 else 0.


# ---------------------------------------------------------------------------
# Training loops (with L tracked per epoch)
# ---------------------------------------------------------------------------

class PMHLoss(nn.Module):
    def __init__(self, block_indices=(3, 4, 5)):
        super().__init__()
        self.block_indices = list(block_indices)

    def forward(self, features_clean, features_noisy):
        loss = 0.
        for i in self.block_indices:
            c = F.normalize(features_clean[i], p=2, dim=1)
            n = F.normalize(features_noisy[i], p=2, dim=1)
            loss = loss + (c - n).pow(2).sum(dim=1).mean()
        return loss / max(len(self.block_indices), 1)


def train_b0_tracked(model, train_loader, test_loader, tdi_loader, device,
                     epochs=100, lr=1e-3, ckpt_path=None):
    """B0 (ERM) with per-epoch L, TDI, acc tracking."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    best_acc = 0.
    epoch_log = []
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0., 0
        t0 = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss   = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update()
            total_loss += loss.item(); n += 1
        sched.step()
        acc = eval_accuracy(model, test_loader, device)
        L   = head_spectral_norm(model)
        tdi = compute_tdi(model, tdi_loader, device)
        elapsed = time.perf_counter() - t0
        entry = {"epoch": epoch+1, "loss": round(total_loss/n, 4),
                 "acc": round(acc, 2), "L": round(L, 6), "TDI": round(tdi, 6),
                 "time_s": round(elapsed, 1)}
        epoch_log.append(entry)
        if ckpt_path and acc > best_acc:
            best_acc = acc; torch.save(model.state_dict(), ckpt_path)
        print(f"  B0 Epoch {epoch+1:3d}  loss={total_loss/n:.4f}  acc={acc:.2f}%"
              f"  L={L:.4f}  TDI={tdi:.4f}  t={elapsed:.0f}s")
    return epoch_log


def train_e1_tracked(model, train_loader, test_loader, tdi_loader, device,
                     epochs=100, lr=1e-3, noise_sigma=0.12, warmup_epochs=10,
                     pmh_cap_ratio=0.3, task_mix=0.25, ckpt_path=None):
    """E1 (PMH) with per-epoch L, TDI, acc tracking."""
    mean_t = torch.tensor(CIFAR10_MEAN, device=device).view(1,3,1,1)
    std_t  = torch.tensor(CIFAR10_STD,  device=device).view(1,3,1,1)
    pmh_fn = PMHLoss()
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    best_acc = 0.
    epoch_log = []
    for epoch in range(epochs):
        w_ep = 0. if epoch < warmup_epochs else min(1., (epoch - warmup_epochs) / 20.)
        model.train()
        total_loss = total_pmh = 0.; n = 0
        t0 = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_pix   = x * std_t + mean_t
            x_noisy = (x_pix + noise_sigma * torch.randn_like(x_pix)).clamp(0, 1)
            x_noisy = (x_noisy - mean_t) / std_t
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits_c = model(x)
                logits_n, feats_n = model(x_noisy, return_features=True)
                loss_task = (1-task_mix)*F.cross_entropy(logits_c, y) + \
                             task_mix  *F.cross_entropy(logits_n, y)
                with torch.no_grad():
                    feats_c = model.get_features(x, return_all=True)
                loss_pmh  = pmh_fn(feats_c, feats_n)
                pmh_term  = w_ep * loss_pmh
                cap       = pmh_cap_ratio * loss_task.detach()
                pmh_term  = torch.minimum(pmh_term, cap)
                loss      = loss_task + pmh_term
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update()
            total_loss += loss.item(); total_pmh += loss_pmh.item(); n += 1
        sched.step()
        acc = eval_accuracy(model, test_loader, device)
        L   = head_spectral_norm(model)
        tdi = compute_tdi(model, tdi_loader, device)
        elapsed = time.perf_counter() - t0
        entry = {"epoch": epoch+1, "loss": round(total_loss/n, 4),
                 "pmh": round(total_pmh/n, 4), "acc": round(acc, 2),
                 "L": round(L, 6), "TDI": round(tdi, 6), "time_s": round(elapsed, 1)}
        epoch_log.append(entry)
        if ckpt_path and acc > best_acc:
            best_acc = acc; torch.save(model.state_dict(), ckpt_path)
        print(f"  E1  Epoch {epoch+1:3d}  loss={total_loss/n:.4f}  pmh={total_pmh/n:.4f}"
              f"  acc={acc:.2f}%  L={L:.4f}  TDI={tdi:.4f}  t={elapsed:.0f}s")
    return epoch_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--runs",      nargs="+", default=["B0", "E1"])
    p.add_argument("--data_dir",  default="./data")
    p.add_argument("--out_dir",   default="runs")
    p.add_argument("--epochs",    type=int,   default=100)
    p.add_argument("--batch_size",type=int,   default=512)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--tdi_samples", type=int, default=1000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
    tdi_loader = get_tdi_loader(args.data_dir, max_samples=args.tdi_samples)

    all_logs = {}
    for run in args.runs:
        run_dir = Path(args.out_dir) / f"{run}_L_tracked"
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / "best.pt"

        print(f"\n{'='*60}\nTraining {run} with L tracking -> {run_dir}\n{'='*60}")
        model = get_model(num_classes=10).to(device)

        if run == "B0":
            log = train_b0_tracked(
                model, train_loader, test_loader, tdi_loader, device,
                epochs=args.epochs, lr=args.lr, ckpt_path=ckpt
            )
        else:
            log = train_e1_tracked(
                model, train_loader, test_loader, tdi_loader, device,
                epochs=args.epochs, lr=args.lr, ckpt_path=ckpt
            )
        all_logs[run] = log

        out_json = run_dir / "L_tracking_log.json"
        with open(out_json, "w") as f:
            json.dump({"run": run, "epochs": args.epochs, "epoch_log": log}, f, indent=2)
        print(f"Saved {out_json}")

    # Summary
    print(f"\n{'='*60}")
    print("L-TRACKING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Run':<6}  {'L_init':>8}  {'L_mid':>8}  {'L_final':>8}  {'L_range':>10}  {'Best_TDI':>10}  {'Best_acc':>10}")
    print(f"{'-'*65}")
    for run, log in all_logs.items():
        Ls = [e["L"] for e in log]
        TDIs = [e["TDI"] for e in log]
        accs = [e["acc"] for e in log]
        mid = len(log) // 2
        print(f"{run:<6}  {Ls[0]:>8.4f}  {Ls[mid]:>8.4f}  {Ls[-1]:>8.4f}"
              f"  {max(Ls)-min(Ls):>10.4f}  {min(TDIs):>10.4f}  {max(accs):>10.2f}%")
    print()

    # Cross-run comparison
    if "B0" in all_logs and "E1" in all_logs:
        L_b0 = all_logs["B0"][-1]["L"]
        L_e1 = all_logs["E1"][-1]["L"]
        print(f"L at convergence:  B0={L_b0:.4f}  E1={L_e1:.4f}")
        if abs(L_b0 - L_e1) / L_b0 < 0.05:
            print("  -> L is approximately equal at convergence (within 5%).")
            print("     TDI improvement from PMH is NOT from L reduction — it comes")
            print("     purely from Frobenius/directional regularisation.")
        else:
            pct = 100*(L_e1 - L_b0)/L_b0
            print(f"  -> L differs by {pct:+.1f}% (B0->E1). Part of TDI improvement")
            print(f"     may come from Lipschitz constant change.")


if __name__ == "__main__":
    main()
