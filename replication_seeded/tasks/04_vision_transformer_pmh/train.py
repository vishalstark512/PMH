"""
Train ViT on CIFAR-10: B0 (no aug), B1 (aug), E1 (task on noisy + multi-scale PMH).
Goal: show E1 (ViT + PMH) clearly beats B0/B1 on corrupted/noisy test.
RTX 4090: AMP, large batch, TF32, fused AdamW, optional torch.compile.

Replication: --seed / --deterministic; train shuffle uses a fixed generator; default num_workers=0 (Windows-safe).
"""
import argparse
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import get_model

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

CIFAR10_MEAN = (0.4914, 0.4565, 0.4067)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_norm_cache = {}


def set_global_seed(seed, deterministic=False):
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
    s = int(base_seed) + int(worker_id)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _norm_tensors(device):
    if device not in _norm_cache:
        _norm_cache[device] = (
            torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1),
            torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1),
        )
    return _norm_cache[device]


def get_transforms(run, train=True):
    t = [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    if train and (run in ("B1", "E1", "VAT") or run.startswith("E1_")):
        t = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + t
    return transforms.Compose(t)


def get_loaders(run, data_dir, batch_size=512, num_workers=0, seed=None):
    train_ds = CIFAR10(root=data_dir, train=True, download=True, transform=get_transforms(run, train=True))
    test_ds = CIFAR10(root=data_dir, train=False, download=True, transform=get_transforms(run, train=False))
    train_kw = dict(batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=True)
    test_kw = dict(batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=False)
    if num_workers > 0:
        train_kw["persistent_workers"] = True
        test_kw["persistent_workers"] = True
    if seed is not None:
        train_kw["generator"] = torch.Generator().manual_seed(int(seed))
        if num_workers > 0:
            train_kw["worker_init_fn"] = partial(_seed_worker, base_seed=int(seed))
            test_kw["worker_init_fn"] = partial(_seed_worker, base_seed=int(seed) + 1)
    train_loader = DataLoader(train_ds, **train_kw)
    test_loader = DataLoader(test_ds, **test_kw)
    return train_loader, test_loader


class PMHLoss(nn.Module):
    """Multi-scale PMH: block outputs at block_indices (0-based), L2-normalized MSE(clean, noisy)."""
    def __init__(self, block_indices=None):
        super().__init__()
        # default: last 3 blocks (indices 3,4,5 for depth=6)
        self.block_indices = block_indices if block_indices is not None else [3, 4, 5]

    def forward(self, features_clean, features_noisy):
        loss = 0.0
        for i in self.block_indices:
            c = F.normalize(features_clean[i], p=2, dim=1)
            n = F.normalize(features_noisy[i], p=2, dim=1)
            loss = loss + (c - n).pow(2).sum(dim=1).mean()
        return loss / max(len(self.block_indices), 1)


def evaluate(model, loader, device, noise_sigma=0.0, pixel_space=True, use_amp=True, seed=None):
    """Evaluate under optional Gaussian noise. seed for reproducibility when noise_sigma > 0."""
    model.eval()
    mean, std = _norm_tensors(device)
    correct, total = 0, 0
    use_amp = use_amp and device.type == "cuda"
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            if noise_sigma > 0:
                batch_seed = (seed + batch_idx * 997) if seed is not None else None
                gen = torch.Generator(device=device).manual_seed(batch_seed) if batch_seed is not None else None
                kw = {"device": device, "dtype": images.dtype}
                if gen is not None:
                    kw["generator"] = gen
                if pixel_space:
                    x = images * std + mean
                    x = x + noise_sigma * torch.randn(x.shape, **kw)
                    x = x.clamp(0, 1)
                    images = (x - mean) / std
                else:
                    images = images + noise_sigma * torch.randn(images.shape, **kw)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total else 0.0


def train_b0_b1(model, train_loader, test_loader, device, epochs, lr=1e-3, run="B0", ckpt_path=None, use_amp=True):
    use_amp = use_amp and device.type == "cuda"
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, fused=(device.type == "cuda"))
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.perf_counter()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()
        acc = evaluate(model, test_loader, device, 0.0, use_amp=use_amp) if test_loader else 0.0
        if ckpt_path and acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt_path)
        print(f"  Epoch {epoch+1}/{epochs} loss={total_loss/n:.4f} val_clean={acc:.2f}% time={time.perf_counter()-t0:.1f}s")
    return model


def _vat_perturb(model, images, epsilon=3.0, xi=1e-6, n_power=1, device=None):
    """Virtual adversarial direction: r that (approx) maximizes KL(p(y|x) || p(y|x+r)), ||r||_2 <= epsilon."""
    model.eval()
    with torch.no_grad():
        logits_orig = model(images)
        p_orig = F.softmax(logits_orig, dim=1)
    d = torch.randn_like(images, device=device)
    d = d / (d.reshape(d.size(0), -1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-12)
    for _ in range(n_power):
        d.requires_grad_(True)
        logits_d = model(images + xi * d)
        log_p_d = F.log_softmax(logits_d, dim=1)
        kl = F.kl_div(log_p_d, p_orig, reduction="batchmean")
        grad = torch.autograd.grad(kl, d)[0]
        d = grad.detach()
        d = d / (d.reshape(d.size(0), -1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-12)
    return epsilon * d


def train_vat(model, train_loader, test_loader, device, epochs, vat_epsilon=3.0, vat_xi=1e-6, vat_weight=1.0, lr=1e-3, ckpt_path=None, use_amp=True):
    """VAT: CE + lambda * KL(p(y|x) || p(y|x+r_vadv))."""
    use_amp = use_amp and device.type == "cuda"
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, fused=(device.type == "cuda"))
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = total_ce = total_kl = 0.0
        n = 0
        t0 = time.perf_counter()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            r = _vat_perturb(model, images, epsilon=vat_epsilon, xi=vat_xi, device=device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits_clean = model(images)
                    loss_ce = F.cross_entropy(logits_clean, labels)
                    logits_adv = model(images + r)
                    log_p_adv = F.log_softmax(logits_adv, dim=1)
                    p_clean = F.softmax(logits_clean.detach(), dim=1)
                    loss_kl = F.kl_div(log_p_adv, p_clean, reduction="batchmean")
                    loss = loss_ce + vat_weight * loss_kl
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits_clean = model(images)
                loss_ce = F.cross_entropy(logits_clean, labels)
                logits_adv = model(images + r)
                log_p_adv = F.log_softmax(logits_adv, dim=1)
                p_clean = F.softmax(logits_clean.detach(), dim=1)
                loss_kl = F.kl_div(log_p_adv, p_clean, reduction="batchmean")
                loss = loss_ce + vat_weight * loss_kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_kl += loss_kl.item()
            n += 1
        scheduler.step()
        acc = evaluate(model, test_loader, device, 0.0, use_amp=use_amp) if test_loader else 0.0
        if ckpt_path and acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt_path)
        print(f"  Epoch {epoch+1}/{epochs} loss={total_loss/n:.4f} ce={total_ce/n:.4f} kl={total_kl/n:.4f} val_clean={acc:.2f}% time={time.perf_counter()-t0:.1f}s")
    return model


def train_e1(model, train_loader, test_loader, device, epochs, warmup_epochs=10, pmh_weight=0.5, noise_sigma=0.1, task_mix=0.2, pmh_cap_ratio=0.3, lr=1e-3, ckpt_path=None, use_amp=True, pmh_block_indices=None, noise_schedule=None, log_cb=None):
    use_amp = use_amp and device.type == "cuda"
    mean, std = _norm_tensors(device)
    pmh_fn = PMHLoss(block_indices=pmh_block_indices if pmh_block_indices is not None else [3, 4, 5])
    sigmas = None
    if noise_schedule:
        sigmas = [float(s) for s in noise_schedule.replace(",", " ").split()]
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, fused=(device.type == "cuda"))
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_acc = 0.0
    for epoch in range(epochs):
        sigma = noise_sigma
        if sigmas:
            sigma = sigmas[epoch % len(sigmas)]
        model.train()
        total_loss = total_task = total_pmh = total_pmh_eff = 0.0
        n = 0
        t0 = time.perf_counter()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            denorm = images * std + mean
            noisy_denorm = denorm + sigma * torch.randn_like(denorm, device=device)
            noisy_denorm = noisy_denorm.clamp(0, 1)
            images_noisy = (noisy_denorm - mean) / std

            opt.zero_grad(set_to_none=True)
            w_ep = 0.0 if epoch < warmup_epochs else min(1.0, (epoch - warmup_epochs) / 20.0)
            if pmh_weight and pmh_weight > 0:
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits_clean = model(images)
                        logits_noisy, features_noisy = model(images_noisy, return_features=True)
                        loss_clean = F.cross_entropy(logits_clean, labels)
                        loss_noisy = F.cross_entropy(logits_noisy, labels)
                        loss_task = (1.0 - task_mix) * loss_clean + task_mix * loss_noisy
                        with torch.no_grad():
                            features_clean = model.get_features(images, return_all=True)
                        loss_pmh = pmh_fn(features_clean, features_noisy)
                        pmh_term = w_ep * pmh_weight * loss_pmh
                        if pmh_cap_ratio is not None and pmh_cap_ratio > 0:
                            cap = pmh_cap_ratio * loss_task.detach()
                            pmh_term = torch.minimum(pmh_term, cap)
                        loss = loss_task + pmh_term
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits_clean = model(images)
                    logits_noisy, features_noisy = model(images_noisy, return_features=True)
                    loss_clean = F.cross_entropy(logits_clean, labels)
                    loss_noisy = F.cross_entropy(logits_noisy, labels)
                    loss_task = (1.0 - task_mix) * loss_clean + task_mix * loss_noisy
                    with torch.no_grad():
                        features_clean = model.get_features(images, return_all=True)
                    loss_pmh = pmh_fn(features_clean, features_noisy)
                    pmh_term = w_ep * pmh_weight * loss_pmh
                    if pmh_cap_ratio is not None and pmh_cap_ratio > 0:
                        cap = pmh_cap_ratio * loss_task.detach()
                        pmh_term = torch.minimum(pmh_term, cap)
                    loss = loss_task + pmh_term
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                total_pmh += loss_pmh.item()
                total_pmh_eff += pmh_term.item()
            else:
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits_clean = model(images)
                        logits_noisy = model(images_noisy)
                        loss_clean = F.cross_entropy(logits_clean, labels)
                        loss_noisy = F.cross_entropy(logits_noisy, labels)
                        loss_task = (1.0 - task_mix) * loss_clean + task_mix * loss_noisy
                        loss = loss_task
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits_clean = model(images)
                    logits_noisy = model(images_noisy)
                    loss_clean = F.cross_entropy(logits_clean, labels)
                    loss_noisy = F.cross_entropy(logits_noisy, labels)
                    loss_task = (1.0 - task_mix) * loss_clean + task_mix * loss_noisy
                    loss = loss_task
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            total_loss += loss.item()
            total_task += loss_task.item()
            n += 1
        scheduler.step()
        w = 0.0 if epoch < warmup_epochs else min(1.0, (epoch - warmup_epochs) / 20.0)
        div = n if n else 1
        acc = evaluate(model, test_loader, device, 0.0, use_amp=use_amp) if test_loader else 0.0
        if ckpt_path and acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt_path)
        avg_task = total_task / div
        avg_pmh = total_pmh / div
        avg_pmh_eff = total_pmh_eff / div
        frac = avg_pmh_eff / (avg_task + avg_pmh_eff) if (avg_task + avg_pmh_eff) > 0 else 0.0
        cap_note = ""
        if pmh_cap_ratio is not None and pmh_cap_ratio > 0 and avg_pmh > avg_task and avg_pmh_eff < avg_pmh * 0.99:
            cap_note = " (capped)"
        print(
            f"  Epoch {epoch+1}/{epochs} loss={total_loss/div:.4f} task={avg_task:.4f} "
            f"pmh={avg_pmh:.4f} pmh_eff={avg_pmh_eff:.4f} frac={frac:.2f} w={w:.2f} val_clean={acc:.2f}% "
            f"time={time.perf_counter()-t0:.1f}s{cap_note}"
        )
        if log_cb:
            log_cb(epoch + 1, {
                "loss": total_loss / div,
                "task": avg_task,
                "pmh": avg_pmh,
                "pmh_eff": avg_pmh_eff,
                "pmh_frac": round(frac, 4),
                "w": round(w, 4),
                "time_s": round(time.perf_counter() - t0, 2),
                "val_clean": acc,
            })
    return model


def _pgd_attack(model, x_norm, y, eps_pixel, alpha_pixel, steps, mean, std, device, use_amp=True, random_start=True):
    """PGD (L_inf) in pixel space [0,1]. Inputs/outputs are normalized tensors."""
    model.eval()
    x_norm = x_norm.to(device).detach()
    y = y.to(device)

    x0_01 = (x_norm * std + mean).detach()
    if random_start:
        x_adv_01 = (x0_01 + torch.empty_like(x0_01).uniform_(-eps_pixel, eps_pixel)).clamp(0.0, 1.0)
    else:
        x_adv_01 = x0_01.clone()

    for _ in range(steps):
        x_adv_01 = x_adv_01.detach().requires_grad_(True)
        x_adv_norm = (x_adv_01 - mean) / std
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(x_adv_norm)
            loss = F.cross_entropy(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            x_adv_01 = x_adv_01 + alpha_pixel * x_adv_01.grad.sign()
            x_adv_01 = torch.max(torch.min(x_adv_01, x0_01 + eps_pixel), x0_01 - eps_pixel)
            x_adv_01 = x_adv_01.clamp(0.0, 1.0)

    return ((x_adv_01 - mean) / std).detach()


def train_pgd(
    model,
    train_loader,
    test_loader,
    device,
    epochs,
    pgd_eps=8 / 255,
    pgd_alpha=2 / 255,
    pgd_steps=7,
    lr=1e-3,
    ckpt_path=None,
    use_amp=True,
    random_start=True,
):
    """PGD adversarial training: CE on PGD-adv examples (pixel-space L_inf)."""
    use_amp = use_amp and device.type == "cuda"
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, fused=(device.type == "cuda"))
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_acc = 0.0
    mean, std = _norm_tensors(device)

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.perf_counter()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            x_adv = _pgd_attack(model, images, labels, pgd_eps, pgd_alpha, pgd_steps, mean, std, device, use_amp, random_start)

            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(x_adv)
                    loss = F.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x_adv)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total_loss += loss.item()
            n += 1

        scheduler.step()
        acc = evaluate(model, test_loader, device, 0.0, use_amp=use_amp) if test_loader else 0.0
        if ckpt_path and acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt_path)
        print(
            f"  Epoch {epoch+1}/{epochs} loss={total_loss/n:.4f} val_clean={acc:.2f}% "
            f"eps={pgd_eps:.4f} alpha={pgd_alpha:.4f} steps={pgd_steps} time={time.perf_counter()-t0:.1f}s"
        )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default="E1", help="B0, B1, E1, VAT, PGD, or E1_gamma_<float> (e.g. E1_gamma_0.01) for gamma-sweep")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--batch_size", type=int, default=512, help="RTX 4090: 512–1024; LR scaled linearly with batch/512")
    p.add_argument("--lr", type=float, default=None, help="Default: 1e-3 * (batch_size/512)")
    p.add_argument("--pmh_weight", type=float, default=0.5)
    p.add_argument("--pmh_cap_ratio", type=float, default=0.3, help="Cap PMH term to pmh_cap_ratio * task_loss (0 disables cap)")
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--noise_sigma", type=float, default=0.12)
    p.add_argument("--noise_schedule", type=str, default=None, help="Multi-level noise for E1: comma-separated e.g. 0.05,0.1,0.15,0.2 (cycled by epoch)")
    p.add_argument("--task_mix", type=float, default=0.25)
    p.add_argument("--num_workers", type=int, default=0, help="0 = Windows-safe; use 4–8 on Linux for faster IO")
    p.add_argument("--seed", type=int, default=42, help="Global RNG + train shuffle + eval noise batches")
    p.add_argument("--deterministic", action="store_true", help="Strict cuDNN determinism (slower)")
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--compile", action="store_true", help="torch.compile model (PyTorch 2+; skip on Windows)")
    p.add_argument("--no_pmh", action="store_true", help="E1 only: disable PMH term (negative control); saves to E1_no_pmh")
    p.add_argument("--pmh_blocks", type=int, nargs="*", default=None, help="Block indices for PMH (0-based, depth 6). Default: 3 4 5. E.g. --pmh_blocks 5 or 4 5 (mech experiment 3).")
    p.add_argument("--vat_epsilon", type=float, default=3.0, help="VAT: L2 radius of adversarial perturbation (normalized input)")
    p.add_argument("--vat_xi", type=float, default=1e-6, help="VAT: step size for power iteration")
    p.add_argument("--vat_weight", type=float, default=1.0, help="VAT: weight for KL term")
    p.add_argument("--pgd_eps", type=float, default=8/255, help="PGD train eps in pixel space [0,1]")
    p.add_argument("--pgd_alpha", type=float, default=2/255, help="PGD train step size in pixel space [0,1]")
    p.add_argument("--pgd_steps", type=int, default=7, help="PGD train steps")
    p.add_argument("--pgd_random_start", action="store_true", help="PGD train random start within eps ball")
    args = p.parse_args()
    set_global_seed(args.seed, deterministic=args.deterministic)

    if args.run == "E1" and args.no_pmh:
        args.run = "E1_no_pmh"
        args.pmh_weight = 0.0
        print("  Negative control: E1 with no PMH term → run_dir=E1_no_pmh")

    pmh_block_indices = args.pmh_blocks
    if pmh_block_indices is not None and sorted(pmh_block_indices) != [3, 4, 5]:
        args.run = "E1_pmh_b" + "".join(map(str, sorted(pmh_block_indices)))
        print(f"  PMH on blocks {sorted(pmh_block_indices)} only → run_dir={args.run}")

    lr_base = 1e-3
    args.lr = args.lr if args.lr is not None else lr_base * (args.batch_size / 512.0)

    run_dir = Path(args.out_dir) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    results_path = run_dir / "results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    use_compile = getattr(args, "compile", False) and hasattr(torch, "compile") and sys.platform != "win32"
    # E1_no_pmh and all E1_* variants use same aug as E1; get_transforms handles startswith("E1_")
    loader_run = "VAT" if args.run == "VAT" else args.run
    train_loader, test_loader = get_loaders(
        loader_run, args.data_dir, args.batch_size, args.num_workers, seed=args.seed
    )
    model = get_model(num_classes=10).to(device)
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"  torch.compile failed: {e}")
            use_compile = False

    training_log = []  # populated per-epoch if train functions support log_cb; empty otherwise

    def _log_cb(epoch, metrics):
        training_log.append({"epoch": epoch, **metrics})

    print(
        f"Task 04 ViT  run={args.run}  epochs={args.epochs}  device={device}  batch={args.batch_size}  "
        f"lr={args.lr:.2e}  amp={use_amp}  compile={use_compile}  seed={args.seed}  num_workers={args.num_workers}"
    )
    if args.run == "VAT":
        print(f"  VAT: epsilon={args.vat_epsilon}  xi={args.vat_xi}  vat_weight={args.vat_weight}")
    elif args.run == "PGD":
        print(f"  PGD: eps={args.pgd_eps}  alpha={args.pgd_alpha}  steps={args.pgd_steps}  random_start={args.pgd_random_start}")
    elif args.run != "B0" and not args.run.startswith("B1"):
        print(f"  E1: task_mix={args.task_mix}  PMH weight={args.pmh_weight}  pmh_cap_ratio={args.pmh_cap_ratio}  noise_sigma={args.noise_sigma}  noise_schedule={args.noise_schedule}  warmup={args.warmup_epochs}  pmh_blocks={pmh_block_indices}")
    t_train_start = time.perf_counter()
    if args.run == "B0":
        train_b0_b1(model, train_loader, test_loader, device, args.epochs, lr=args.lr, run="B0", ckpt_path=ckpt_path, use_amp=use_amp)
    elif args.run == "B1":
        train_b0_b1(model, train_loader, test_loader, device, args.epochs, lr=args.lr, run="B1", ckpt_path=ckpt_path, use_amp=use_amp)
    elif args.run == "VAT":
        train_vat(model, train_loader, test_loader, device, args.epochs, vat_epsilon=args.vat_epsilon, vat_xi=args.vat_xi, vat_weight=args.vat_weight, lr=args.lr, ckpt_path=ckpt_path, use_amp=use_amp)
    elif args.run == "PGD":
        train_pgd(
            model, train_loader, test_loader, device, args.epochs,
            pgd_eps=args.pgd_eps, pgd_alpha=args.pgd_alpha, pgd_steps=args.pgd_steps,
            lr=args.lr, ckpt_path=ckpt_path, use_amp=use_amp, random_start=args.pgd_random_start,
        )
    else:
        train_e1(
            model,
            train_loader,
            test_loader,
            device,
            args.epochs,
            args.warmup_epochs,
            args.pmh_weight,
            args.noise_sigma,
            args.task_mix,
            args.pmh_cap_ratio,
            lr=args.lr,
            ckpt_path=ckpt_path,
            use_amp=use_amp,
            pmh_block_indices=pmh_block_indices,
            noise_schedule=args.noise_schedule,
            log_cb=_log_cb,
        )
    train_time_s = time.perf_counter() - t_train_start

    # Eval best checkpoint at 4 standard noise levels (consistent with all other tasks)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    eval_seed = args.seed
    acc_clean  = evaluate(model, test_loader, device, noise_sigma=0.0)
    acc_n005   = evaluate(model, test_loader, device, noise_sigma=0.05, seed=eval_seed)
    acc_n010   = evaluate(model, test_loader, device, noise_sigma=0.1, seed=eval_seed)
    acc_n015   = evaluate(model, test_loader, device, noise_sigma=0.15, seed=eval_seed)
    acc_n020   = evaluate(model, test_loader, device, noise_sigma=0.2, seed=eval_seed)
    print(f"  Test clean={acc_clean:.2f}%  s=0.05:{acc_n005:.2f}%  s=0.1:{acc_n010:.2f}%  s=0.15:{acc_n015:.2f}%  s=0.2:{acc_n020:.2f}%")

    results = {
        "task": "04", "run": args.run, "architecture": "vit_small",
        "dataset": "cifar10", "epochs": args.epochs,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "num_workers": args.num_workers,
        "clean_acc": acc_clean,
        "acc_sigma_0.05": acc_n005, "acc_sigma_0.1": acc_n010,
        "acc_sigma_0.15": acc_n015, "acc_sigma_0.2": acc_n020,
        "train_time_s": round(train_time_s, 1),
        "checkpoint": "best.pt", "training_log": training_log,
    }
    if args.run in ("E1", "E1_no_pmh") or args.run.startswith("E1_"):
        results.update({"pmh_weight": args.pmh_weight, "pmh_cap_ratio": args.pmh_cap_ratio,
                        "noise_sigma": args.noise_sigma, "warmup_epochs": args.warmup_epochs})
        # Provenance for φ-study: which ViT blocks PMH targets (no matching term for E1_no_pmh)
        if args.run != "E1_no_pmh":
            results["pmh_blocks"] = list(pmh_block_indices) if pmh_block_indices is not None else [3, 4, 5]
    if args.run == "VAT":
        results.update({"vat_epsilon": args.vat_epsilon, "vat_weight": args.vat_weight})
    if args.run == "PGD":
        results.update({"pgd_eps": args.pgd_eps, "pgd_alpha": args.pgd_alpha, "pgd_steps": args.pgd_steps,
                        "pgd_random_start": bool(args.pgd_random_start)})
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save training_log.json separately (consistent with tasks 01/02/06/07)
    log_path = run_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({"task": "04", "run": args.run, "epochs": args.epochs, "epoch_log": training_log}, f, indent=2)

    print(f"Saved {ckpt_path}, {results_path}, {log_path}")


if __name__ == "__main__":
    main()
