"""
Train B0 (vanilla), B1 (standard+aug), or E1 (PMH) on CIFAR-10 or CIFAR-100.
Aligned with RES/CLASS notebook: one model (ResNet-18 CIFAR), one validation set.

  B0: No augmentation, CE(clean).
  B1: Augmentation (RandomCrop+Flip), CE(clean) — same as notebook "Standard".
  E1: Same aug as B1; task loss on noisy (denorm + σ=0.1 + clamp + renorm); multi-scale PMH.

  python train.py --dataset cifar10 --run B0 --epochs 50
  python train.py --dataset cifar100 --run B0 --epochs 50
  python train.py --run B1 --epochs 50
  python train.py --run E1 --epochs 50   # E1 always runs 20% more epochs (e.g. 50 -> 60)

  Outputs go to <out_dir>/<dataset>/<run>/ (default: runs/cifar10/B0, runs/cifar100/B0, ...).
"""
import argparse
from functools import partial
import json
import math
import random
import time
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from model import get_model

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Per-dataset normalization (same for train/val)
CIFAR10_MEAN = (0.4914, 0.4565, 0.4067)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

def _get_norm(dataset):
    if dataset == "cifar100":
        return CIFAR100_MEAN, CIFAR100_STD
    return CIFAR10_MEAN, CIFAR10_STD

# Cached tensors per (dataset, device) for evaluate()
_norm_cache = {}

def _norm_tensors(device, dataset="cifar10"):
    key = (dataset, device)
    if key not in _norm_cache or _norm_cache[key][0].device != device:
        mean, std = _get_norm(dataset)
        _norm_cache[key] = (
            torch.tensor(mean, device=device).view(1, 3, 1, 1),
            torch.tensor(std, device=device).view(1, 3, 1, 1),
        )
    return _norm_cache[key]


def set_global_seed(seed, deterministic=False):
    """Set all major RNG sources for reproducible training/eval."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False


def _seed_worker(worker_id, base_seed):
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Transforms and data
# ---------------------------------------------------------------------------
def get_transforms(run, dataset, train=True):
    mean, std = _get_norm(dataset)
    if not train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    if run == "B0":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    # B1 and E1: same as notebook "Standard" — RandomCrop + Flip only
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_loaders(run, dataset, data_dir, batch_size, num_workers, pin_memory, seed=None):
    test_tf = get_transforms(run, dataset, train=False)
    train_tf = get_transforms(run, dataset, train=True)
    ds_class = CIFAR100 if dataset == "cifar100" else CIFAR10
    train_ds = ds_class(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds = ds_class(root=data_dir, train=False, download=True, transform=test_tf)
    kw = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        kw["persistent_workers"] = True
    if seed is not None:
        data_gen = torch.Generator().manual_seed(seed)
        kw["generator"] = data_gen
        kw["worker_init_fn"] = partial(_seed_worker, base_seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# PMH loss (notebook-style: multi-scale, normalized MSE, last 3 layers)
# ---------------------------------------------------------------------------
class PMHLoss(nn.Module):
    """Multi-scale PMH with L2-normalized features, MSE(clean, noisy). Last 3 scales."""
    def __init__(self, num_scales=3):
        super().__init__()
        self.num_scales = num_scales

    def forward(self, features_clean, features_noisy):
        # Last num_scales (e.g. layer2, layer3, layer4)
        fc = features_clean[-self.num_scales:]
        fn = features_noisy[-self.num_scales:]
        loss = 0.0
        for c, n in zip(fc, fn):
            if c.dim() == 4:
                c = F.adaptive_avg_pool2d(c, (1, 1)).flatten(1)
                n = F.adaptive_avg_pool2d(n, (1, 1)).flatten(1)
            c = F.normalize(c, p=2, dim=1)
            n = F.normalize(n, p=2, dim=1)
            loss = loss + (c - n).pow(2).sum(dim=1).mean()
        return loss / self.num_scales


# ---------------------------------------------------------------------------
# Evaluation (single validation path)
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, noise_sigma=0.0, use_amp=False, pixel_space=False, dataset="cifar10", seed=None):
    """Val: clean or Gaussian noise. pixel_space=True => denorm, add noise, clamp, renorm. seed for reproducibility."""
    model.eval()
    mean, std = _norm_tensors(device, dataset)
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            if noise_sigma > 0:
                batch_seed = (seed + batch_idx * 997) if seed is not None else None
                gen = torch.Generator(device=device).manual_seed(batch_seed) if batch_seed is not None else None
                def _noise(shape, dtype, dev):
                    kw = {"device": dev, "dtype": dtype}
                    if gen is not None:
                        kw["generator"] = gen
                    return torch.randn(shape, **kw)
                if pixel_space:
                    x = images * std + mean
                    x = x + noise_sigma * _noise(x.shape, x.dtype, device)
                    x = x.clamp(0, 1)
                    images = (x - mean) / std
                else:
                    images = images + noise_sigma * _noise(images.shape, images.dtype, device)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_b0_b1(model, loader, device, epochs, lr=0.1, use_amp=True, log_cb=None, log_line=None):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[25, 40], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    _log = log_line or print
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.perf_counter()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()
        t1 = time.perf_counter()
        avg_loss = total_loss / n if n else 0.0
        _log(f"  Epoch {epoch+1}/{epochs} loss={avg_loss:.4f} time={t1-t0:.1f}s")
        if log_cb:
            log_cb(epoch + 1, {"loss": avg_loss, "time_s": round(t1 - t0, 2)})
    return model


def _cosine_pmh_weight(epoch, warmup_epochs, ramp_epochs=20, max_weight=1.0):
    """Cosine curriculum: w from 0 to max_weight over ramp (matches Task 02)."""
    if epoch < warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return 0.5 * (1.0 - math.cos(math.pi * progress)) * max_weight


def train_e1(model, loader, device, epochs, warmup_epochs=10, pmh_weight=0.5, pmh_cap_ratio=0.3, noise_sigma=0.1, task_mix=0.2, pmh_ramp_epochs=20, lr=0.1, use_amp=True, log_cb=None, log_line=None, dataset="cifar10"):
    """E1: mixed task loss (1-task_mix)*CE(clean) + task_mix*CE(noisy) + multi-scale PMH. Notebook uses 0.8 clean + 0.2 noisy."""
    mean, std = _norm_tensors(device, dataset)
    pmh_fn = PMHLoss(num_scales=3)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # Scale milestones with epoch count (e.g. 50 -> [25,40], 60 -> [30,48])
    m1, m2 = max(1, int(epochs * 0.5)), max(2, int(epochs * 0.8))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m1, m2], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    for epoch in range(epochs):
        model.train()
        total_loss = total_task = total_pmh = total_pmh_eff = 0.0
        n = 0
        t0 = time.perf_counter()
        w = _cosine_pmh_weight(epoch, warmup_epochs, pmh_ramp_epochs, pmh_weight)
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Noisy view: denorm -> add noise -> clamp(0,1) -> renorm (notebook)
            denorm = images * std + mean
            noisy_denorm = denorm + noise_sigma * torch.randn_like(denorm, device=device)
            noisy_denorm = noisy_denorm.clamp(0, 1)
            images_noisy = (noisy_denorm - mean) / std

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                # Mixed task loss: (1-task_mix)*clean + task_mix*noisy (notebook: 0.8 clean + 0.2 noisy)
                logits_clean = model(images)
                logits_noisy, features_noisy = model(images_noisy, return_features=True)
                loss_clean = F.cross_entropy(logits_clean, labels)
                loss_noisy = F.cross_entropy(logits_noisy, labels)
                loss_task = (1.0 - task_mix) * loss_clean + task_mix * loss_noisy
                # PMH: clean (no_grad) vs noisy, last 3 scales
                with torch.no_grad():
                    features_clean = model.get_features(images, return_all=True)
                loss_pmh = pmh_fn(features_clean, features_noisy)
                # Curriculum; cap on weighted term (matches Tasks 03,04,06,07)
                pmh_term = w * loss_pmh
                if pmh_cap_ratio > 0:
                    pmh_term = torch.minimum(pmh_term, pmh_cap_ratio * loss_task.detach())
                loss = loss_task + pmh_term
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            total_loss += loss.item()
            total_task += loss_task.item()
            total_pmh += loss_pmh.item()
            total_pmh_eff += pmh_term.item()
            n += 1
        scheduler.step()
        t1 = time.perf_counter()
        div = n if n > 0 else 1
        avg_task = total_task / div
        avg_pmh_eff = total_pmh_eff / div
        pmh_frac = avg_pmh_eff / (avg_task + avg_pmh_eff) if (avg_task + avg_pmh_eff) > 0 else 0.0
        d = {"loss": total_loss / div, "task": avg_task, "pmh": total_pmh / div, "pmh_eff": avg_pmh_eff, "pmh_frac": round(pmh_frac, 4), "w": round(w, 4), "time_s": round(t1 - t0, 2)}
        _log = log_line or print
        _log(f"  Epoch {epoch+1}/{epochs} loss={d['loss']:.4f} task={avg_task:.4f} pmh={d['pmh']:.4f} pmh_eff={avg_pmh_eff:.4f} frac={pmh_frac:.2f} w={w:.2f} time={t1-t0:.1f}s")
        if log_cb:
            log_cb(epoch + 1, d)
    return model


def _vat_perturbation(model, images, xi=1e-6, eps=2.0, use_amp=False):
    """Compute virtual adversarial perturbation via power iteration."""
    with torch.no_grad():
        ref_logits = model(images)
        p_ref = F.softmax(ref_logits, dim=1)
    d = torch.randn_like(images)
    d = F.normalize(d.view(d.size(0), -1), p=2, dim=1).view_as(d)
    d = d.detach().requires_grad_(True)
    with torch.amp.autocast("cuda", enabled=use_amp and images.device.type == "cuda"):
        adv_logits = model(images + xi * d)
    kl = F.kl_div(F.log_softmax(adv_logits, dim=1), p_ref, reduction='batchmean')
    kl.backward()
    d = F.normalize(d.grad.view(d.grad.size(0), -1), p=2, dim=1).view_as(d.grad).detach()
    return (eps * d).detach()


def train_vat(model, loader, device, epochs, lr=0.1, vat_eps=2.0, vat_xi=1e-6, vat_weight=1.0,
              use_amp=True, dataset="cifar10", log_cb=None, log_line=None):
    """VAT: CE(clean) + KL consistency under virtual adversarial perturbation."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    m1, m2 = max(1, int(epochs * 0.5)), max(2, int(epochs * 0.8))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m1, m2], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    _log = log_line or print
    for epoch in range(epochs):
        model.train()
        total_loss = total_task = total_vat = 0.0
        n = 0
        t0 = time.perf_counter()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            r_adv = _vat_perturbation(model, images, xi=vat_xi, eps=vat_eps, use_amp=(use_amp and device.type == "cuda"))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits_clean = model(images)
                loss_task = F.cross_entropy(logits_clean, labels)
                logits_adv = model(images + r_adv)
                p_ref = F.softmax(logits_clean.detach(), dim=1)
                loss_vat = F.kl_div(F.log_softmax(logits_adv, dim=1), p_ref, reduction='batchmean')
                loss = loss_task + vat_weight * loss_vat
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item(); total_task += loss_task.item(); total_vat += loss_vat.item()
            n += 1
        scheduler.step()
        t1 = time.perf_counter()
        div = max(n, 1)
        _log(f"  Epoch {epoch+1}/{epochs} loss={total_loss/div:.4f} task={total_task/div:.4f} vat={total_vat/div:.4f} time={t1-t0:.1f}s")
        if log_cb: log_cb(epoch + 1, {"loss": total_loss/div, "task": total_task/div, "vat": total_vat/div, "time_s": round(t1-t0,2)})
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"], help="CIFAR-10 or CIFAR-100")
    p.add_argument("--run", type=str, default="E1", choices=["B0", "B1", "VAT", "E1"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="runs", help="Directory for checkpoints and logs (writes to <out_dir>/<dataset>/<run>/)")
    p.add_argument("--batch_size", type=int, default=128, help="Notebook uses 128")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--pmh_weight", type=float, default=0.5, help="E1 PMH weight")
    p.add_argument("--pmh_cap_ratio", type=float, default=0.3, help="Cap PMH term to pmh_cap_ratio * task_loss (0 disables)")
    p.add_argument("--warmup_epochs", type=int, default=10, help="E1 curriculum warmup")
    p.add_argument("--pmh_ramp_epochs", type=int, default=20, help="E1 cosine ramp epochs (after warmup)")
    p.add_argument("--noise_sigma", type=float, default=0.1, help="E1 training noise (notebook 0.1)")
    p.add_argument("--task_mix", type=float, default=0.2, help="E1: fraction of task loss on noisy (1-task_mix on clean). Notebook 0.2 = 80%% clean, 20%% noisy")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate (SGD; default 0.1 for CIFAR)")
    p.add_argument("--vat_eps", type=float, default=2.0)
    p.add_argument("--vat_weight", type=float, default=1.0)
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "cnn"])
    p.add_argument("--seed", type=int, default=42, help="Global seed for training/evaluation reproducibility")
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic cuDNN mode (slower)")
    args = p.parse_args()

    num_classes = 100 if args.dataset == "cifar100" else 10
    run_dir = Path(args.out_dir) / args.dataset / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    results_path = run_dir / "results.json"
    log_path = run_dir / "training_log.json"
    text_log_path = run_dir / "train.log"

    # E1 and VAT: always 20% more epochs
    epochs = args.epochs
    if args.run in ("E1", "VAT"):
        epochs = int(round(args.epochs * 1.2))
        print(f"{args.run}: using 20% more epochs: {args.epochs} -> {epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed, deterministic=args.deterministic)
    use_amp = not args.no_amp and device.type == "cuda"
    pin = device.type == "cuda"
    nw = args.num_workers
    print(f"Dataset={args.dataset} run={args.run} model={args.model} device={device} batch={args.batch_size} workers={nw} amp={use_amp} run_dir={run_dir}")

    # Training log accumulator and file logger
    training_log = []
    log_file = open(text_log_path, "w", encoding="utf-8")
    def log_line(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
    def log_cb(epoch, metrics):
        training_log.append({"epoch": epoch, **metrics})

    # One validation path: same test set for all
    train_loader, test_loader = get_loaders(
        "B1" if args.run in ("E1", "VAT") else args.run,
        args.dataset,
        args.data_dir, args.batch_size, nw, pin, seed=args.seed
    )
    model = get_model(args.model, num_classes=num_classes).to(device)

    t0 = time.perf_counter()
    if args.run in ("B0", "B1"):
        train_b0_b1(model, train_loader, device, epochs, use_amp=use_amp, log_cb=log_cb, log_line=log_line)
    elif args.run == "VAT":
        train_vat(model, train_loader, device, epochs,
                  lr=args.lr, vat_eps=args.vat_eps, vat_weight=args.vat_weight,
                  use_amp=use_amp, dataset=args.dataset, log_cb=log_cb, log_line=log_line)
    else:
        train_e1(model, train_loader, device, epochs,
                 warmup_epochs=args.warmup_epochs, pmh_weight=args.pmh_weight,
                 pmh_cap_ratio=args.pmh_cap_ratio, pmh_ramp_epochs=args.pmh_ramp_epochs,
                 noise_sigma=args.noise_sigma,
                 task_mix=args.task_mix, use_amp=use_amp, log_cb=log_cb, log_line=log_line, dataset=args.dataset)
    train_time = time.perf_counter() - t0

    # Validation: clean + Gaussian (pixel-space like notebook); fixed to run seed for reproducibility
    EVAL_SEED = args.seed
    acc_clean = evaluate(model, test_loader, device, 0.0, use_amp=use_amp, dataset=args.dataset)
    acc_05 = evaluate(model, test_loader, device, 0.05, use_amp=use_amp, pixel_space=True, dataset=args.dataset, seed=EVAL_SEED)
    acc_10 = evaluate(model, test_loader, device, 0.1, use_amp=use_amp, pixel_space=True, dataset=args.dataset, seed=EVAL_SEED)
    acc_15 = evaluate(model, test_loader, device, 0.15, use_amp=use_amp, pixel_space=True, dataset=args.dataset, seed=EVAL_SEED)
    acc_20 = evaluate(model, test_loader, device, 0.2, use_amp=use_amp, pixel_space=True, dataset=args.dataset, seed=EVAL_SEED)
    val_msg = f"Val — Clean: {acc_clean:.2f}%  σ=0.05: {acc_05:.2f}%  σ=0.1: {acc_10:.2f}%  σ=0.15: {acc_15:.2f}%  σ=0.2: {acc_20:.2f}%  time: {train_time:.1f}s"
    log_line(val_msg)

    results = {
        "dataset": args.dataset,
        "run": args.run,
        "model": args.model,
        "epochs_requested": args.epochs,
        "epochs_actual": epochs,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "clean_acc": acc_clean,
        "acc_sigma_0.05": acc_05,
        "acc_sigma_0.1": acc_10,
        "acc_sigma_0.15": acc_15,
        "acc_sigma_0.2": acc_20,
        "train_time_s": round(train_time, 1),
        "checkpoint": "best.pt",
        "training_log": training_log,
    }
    if args.run == "E1":
        results["pmh_weight"] = args.pmh_weight
        results["warmup_epochs"] = args.warmup_epochs
        results["noise_sigma"] = args.noise_sigma
        results["task_mix"] = args.task_mix
    elif args.run == "VAT":
        results["vat_eps"] = args.vat_eps
        results["vat_weight"] = args.vat_weight

    torch.save(model.state_dict(), ckpt_path)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "run": args.run, "epochs": epochs, "epoch_log": training_log}, f, indent=2)
    log_line(f"Saved checkpoint: {ckpt_path}")
    log_line(f"Saved results: {results_path}")
    log_line(f"Saved training log: {log_path}")
    log_file.close()


if __name__ == "__main__":
    main()
