"""
Train chest X-ray: B0 (BCE only), E1 (Task 01–aligned PMH).
E1: same aug for all runs; Gaussian noise view in-loop (denorm → noise → clamp → renorm);
multi-scale PMH on last 3 ResNet stages (normalized MSE); cosine PMH curriculum + cap (matches tasks/01 CIFAR E1).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import IMAGENET_MEAN, IMAGENET_STD, get_train_loader, get_transforms, ChestXrayDataset
from model import get_model

try:
    from eval import evaluate
except ImportError:
    evaluate = None

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


def _norm_tensors(device, dtype=None):
    """ImageNet mean/std as (1,3,1,1) tensors on device (Task 01–style denorm for noisy view)."""
    sample_dtype = dtype or torch.float32
    key = (device, str(sample_dtype))
    if key not in _norm_cache:
        mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=sample_dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device, dtype=sample_dtype).view(1, 3, 1, 1)
        _norm_cache[key] = (mean, std)
    return _norm_cache[key]


# ---------------------------------------------------------------------------
# PMH loss — same as Task 01 (CIFAR10/train.py PMHLoss)
# ---------------------------------------------------------------------------
class PMHLoss(nn.Module):
    """Multi-scale PMH: L2-normalized features, MSE(clean, noisy). Last 3 scales (layer2–4)."""

    def __init__(self, num_scales=3):
        super().__init__()
        self.num_scales = num_scales

    def forward(self, features_clean, features_noisy):
        fc = features_clean[-self.num_scales :]
        fn = features_noisy[-self.num_scales :]
        loss = 0.0
        for c, n in zip(fc, fn):
            if c.dim() == 4:
                c = F.adaptive_avg_pool2d(c, (1, 1)).flatten(1)
                n = F.adaptive_avg_pool2d(n, (1, 1)).flatten(1)
            c = F.normalize(c, p=2, dim=1)
            n = F.normalize(n, p=2, dim=1)
            loss = loss + (c - n).pow(2).sum(dim=1).mean()
        return loss / self.num_scales


def _cosine_pmh_weight(epoch, warmup_epochs, ramp_epochs=20, max_weight=1.0):
    """Cosine curriculum: w from 0 to max_weight over ramp (Task 01 E1)."""
    if epoch < warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return 0.5 * (1.0 - math.cos(math.pi * progress)) * max_weight


def _vat_perturb(model, img, epsilon=2.0, xi=1e-6, n_power=1, device=None):
    """Virtual adversarial direction for multi-label: r maximizing KL(sigmoid(logits(x)) || sigmoid(logits(x+r)))."""
    model.eval()
    with torch.no_grad():
        logits_orig = model(img)
        p_orig = torch.sigmoid(logits_orig)
    d = torch.randn_like(img, device=device)
    d = d / (d.reshape(d.size(0), -1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-12)
    for _ in range(n_power):
        d.requires_grad_(True)
        logits_d = model(img + xi * d)
        p_d = torch.sigmoid(logits_d)
        kl = F.binary_cross_entropy(p_d, p_orig, reduction="mean")
        grad = torch.autograd.grad(kl, d)[0]
        d = grad.detach()
        d = d / (d.reshape(d.size(0), -1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-12)
    return epsilon * d


def train_vat(model, loader, device, epochs, vat_epsilon=2.0, vat_xi=1e-6, vat_weight=1.0, use_amp=True, log_cb=None, log_line=None, on_epoch_end=None):
    """VAT: BCE on labels + vat_weight * KL(sigmoid(logits(x)) || sigmoid(logits(x+r_vadv)))."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    m1, m2 = max(1, int(epochs * 0.5)), max(2, int(epochs * 0.75))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m1, m2], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    _log = log_line or print
    for epoch in range(epochs):
        model.train()
        total_loss = total_bce = total_kl = 0.0
        n = 0
        t0 = time.perf_counter()
        for batch in loader:
            img, labels = batch[0], batch[1]
            img = img.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            r = _vat_perturb(model, img, epsilon=vat_epsilon, xi=vat_xi, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits_clean = model(img)
                loss_bce = F.binary_cross_entropy_with_logits(logits_clean, labels)
                logits_adv = model(img + r)
                p_clean = torch.sigmoid(logits_clean.detach())
                loss_kl = F.binary_cross_entropy_with_logits(logits_adv, p_clean)
                loss = loss_bce + vat_weight * loss_kl
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_kl += loss_kl.item()
            n += 1
        scheduler.step()
        t1 = time.perf_counter()
        div = n if n else 1
        _log(f"  Epoch {epoch+1}/{epochs} loss={total_loss/div:.4f} bce={total_bce/div:.4f} kl={total_kl/div:.4f} time={t1-t0:.1f}s")
        if log_cb:
            log_cb(epoch + 1, {"loss": total_loss / div, "bce": total_bce / div, "kl": total_kl / div, "time_s": round(t1 - t0, 2)})
    return model


def train_b0(model, loader, device, epochs, use_amp=True, log_cb=None, log_line=None, on_epoch_end=None):
    """B0: BCE multi-label loss only."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 50], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    _log = log_line or print

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.perf_counter()
        for batch in loader:
            if len(batch) == 2:
                img, labels = batch[0], batch[1]
            else:
                img, labels = batch[0], batch[2]
            img = img.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(img)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            if scaler:
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
        avg = total_loss / n if n else 0.0
        _log(f"  Epoch {epoch+1}/{epochs} loss={avg:.4f} time={t1-t0:.1f}s")
        if log_cb:
            log_cb(epoch + 1, {"loss": avg, "time_s": round(t1 - t0, 2)})
        if on_epoch_end:
            on_epoch_end(epoch + 1, model)
    return model


def train_e1(
    model,
    loader,
    device,
    epochs,
    warmup_epochs=10,
    pmh_weight=0.5,
    pmh_cap_ratio=0.3,
    noise_sigma=0.1,
    task_mix=0.2,
    pmh_ramp_epochs=20,
    lr=0.1,
    use_amp=True,
    log_cb=None,
    log_line=None,
    on_epoch_end=None,
):
    """E1: Task 01–aligned — mixed BCE + multi-scale PMH (last 3 layers); SGD; cosine PMH ramp; grad clip."""
    pmh_fn = PMHLoss(num_scales=3)
    use_pmh = pmh_weight > 0
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    m1, m2 = max(1, int(epochs * 0.5)), max(2, int(epochs * 0.8))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m1, m2], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    _log = log_line or print

    for epoch in range(epochs):
        model.train()
        total_loss = total_task = total_pmh = total_pmh_eff = 0.0
        n = 0
        t0 = time.perf_counter()
        w = _cosine_pmh_weight(epoch, warmup_epochs, pmh_ramp_epochs, pmh_weight) if use_pmh else 0.0

        for batch in loader:
            if len(batch) != 2:
                raise RuntimeError("E1 expects single-view batches (img, labels); use two_view=False in get_train_loader")
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mean, std = _norm_tensors(device, dtype=images.dtype)
            denorm = images * std + mean
            noisy_denorm = denorm + noise_sigma * torch.randn_like(denorm, device=device)
            noisy_denorm = noisy_denorm.clamp(0, 1)
            images_noisy = (noisy_denorm - mean) / std

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits_clean = model(images)
                if w > 0:
                    logits_noisy, features_noisy = model(images_noisy, return_features=True)
                else:
                    logits_noisy = model(images_noisy)
                loss_clean = F.binary_cross_entropy_with_logits(logits_clean, labels)
                loss_noisy = F.binary_cross_entropy_with_logits(logits_noisy, labels)
                loss_task = (1.0 - task_mix) * loss_clean + task_mix * loss_noisy

                if w > 0:
                    with torch.no_grad():
                        features_clean = model.get_features(images, return_all=True)
                    loss_pmh = pmh_fn(features_clean, features_noisy)
                    pmh_term = w * loss_pmh
                    if pmh_cap_ratio and pmh_cap_ratio > 0:
                        pmh_term = torch.minimum(pmh_term, pmh_cap_ratio * loss_task.detach())
                else:
                    loss_pmh = torch.zeros((), device=device, dtype=loss_task.dtype)
                    pmh_term = loss_pmh
                loss = loss_task + pmh_term

            if torch.isnan(loss) or torch.isinf(loss):
                continue
            if scaler:
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
        d = {
            "loss": total_loss / div,
            "task": avg_task,
            "pmh": total_pmh / div,
            "pmh_eff": avg_pmh_eff,
            "pmh_frac": round(pmh_frac, 4),
            "w": round(w, 4),
            "time_s": round(t1 - t0, 2),
        }
        _log(
            f"  Epoch {epoch+1}/{epochs} loss={d['loss']:.4f} task={avg_task:.4f} "
            f"pmh={d['pmh']:.4f} pmh_eff={avg_pmh_eff:.4f} frac={pmh_frac:.2f} w={w:.2f} time={t1-t0:.1f}s"
        )
        if log_cb:
            log_cb(epoch + 1, d)
        if on_epoch_end:
            on_epoch_end(epoch + 1, model)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default="E1", choices=["B0", "E1", "VAT"])
    p.add_argument("--no_pmh", action="store_true",
                   help="E1 only: disable PMH term (negative control); saves to E1_no_pmh")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default 0 for reproducible behavior on Windows)")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--pmh_weight", type=float, default=0.5, help="E1: max PMH coefficient after cosine ramp (Task 01 default)")
    p.add_argument("--pmh_cap_ratio", type=float, default=0.3,
                   help="Cap PMH term to this fraction of task loss (0 disables)")
    p.add_argument("--warmup_epochs", type=int, default=10, help="E1: PMH weight 0 for these epochs (Task 01 default)")
    p.add_argument("--pmh_ramp_epochs", type=int, default=20, help="E1: cosine ramp length after warmup (Task 01 default)")
    p.add_argument("--lr", type=float, default=0.1, help="E1: SGD learning rate (Task 01 default)")
    p.add_argument("--task_mix", type=float, default=0.2)
    p.add_argument("--noise_sigma", type=float, default=0.1, help="E1: Gaussian noise in denormalized pixel space (Task 01 default)")
    p.add_argument("--intensity_scale", type=float, nargs=2, default=[0.85, 1.15],
                   help="Unused for E1 (Task 01 style uses Gaussian only); kept for API compatibility")
    p.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--max_samples", type=int, default=None, help="Cap train samples for quick runs")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", help="cudnn deterministic mode (slower; stricter reproducibility)")
    p.add_argument("--no_download", action="store_true", help="Do not auto-download data; fail with instructions if missing")
    p.add_argument("--dataset", type=str, default="pneumonia", choices=["pneumonia", "nih"],
                    help="pneumonia = small ~1–2 GB, 2 classes (default); nih = full ~45 GB, 14 classes")
    p.add_argument("--pmh_embed_only", action="store_true",
                   help="Save run as E1_embed_only; training matches E1 (Task 01 PMH is multi-scale only, no embed term)")
    p.add_argument("--load_in_memory", action="store_true",
                   help="Preload all train images into RAM (fits in 128GB; no disk I/O during training)")
    p.add_argument("--vat_epsilon", type=float, default=2.0, help="VAT: L2 radius of perturbation")
    p.add_argument("--vat_xi", type=float, default=1e-6, help="VAT: power iteration step")
    p.add_argument("--vat_weight", type=float, default=1.0, help="VAT: weight for KL term")
    p.add_argument("--val_interval", type=int, default=5, help="Run validation every N epochs and save best.pt by AUC")
    args = p.parse_args()

    if args.pmh_embed_only:
        args.run = "E1_embed_only"
    if args.run == "E1" and args.no_pmh:
        args.run = "E1_no_pmh"
        print("  Negative control: E1 with no PMH term -> run_dir=E1_no_pmh")

    run_dir = Path(args.out_dir) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    results_path = run_dir / "results.json"
    log_path = run_dir / "training_log.json"
    text_log_path = run_dir / "train.log"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed, deterministic=args.deterministic)
    use_amp = not args.no_amp and device.type == "cuda"
    # Task 01 style: single augmented view; noisy view built inside train_e1 (not a second dataset column).
    two_view = False

    train_loader, num_classes, val_s, test_s = get_train_loader(
        args.data_dir,
        args.run,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
        two_view=two_view,
        noise_sigma=0.0,
        intensity_scale_range=None,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        auto_download=not args.no_download,
        dataset=args.dataset,
        load_in_memory=args.load_in_memory,
    )

    model = get_model(num_classes, embed_dim=args.embed_dim, pretrained=True).to(device)

    training_log = []
    log_file = open(text_log_path, "w", encoding="utf-8")

    def log_line(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    def log_cb(epoch, metrics):
        training_log.append({"epoch": epoch, **metrics})

    log_line(f"run={args.run} dataset={args.dataset} device={device} num_classes={num_classes} run_dir={run_dir}")
    if args.run == "VAT":
        log_line(f"VAT: epsilon={args.vat_epsilon} xi={args.vat_xi} vat_weight={args.vat_weight}")
    if args.run in ("E1", "E1_embed_only", "E1_no_pmh"):
        log_line(
            f"{args.run}: Task01-style E1 — SGD lr={args.lr}, PMH last-3 stages, "
            f"noise_sigma={args.noise_sigma}, warmup={args.warmup_epochs}, ramp={args.pmh_ramp_epochs}, "
            f"pmh_weight={args.pmh_weight}, cap={args.pmh_cap_ratio}"
        )

    val_loader = None
    if val_s:
        from torch.utils.data import DataLoader
        tf_eval = get_transforms(img_size=tuple(args.img_size), train=False)
        val_ds = ChestXrayDataset(val_s, transform=tf_eval)
        val_kw = dict(batch_size=64, shuffle=False, num_workers=args.num_workers)
        if args.num_workers > 0:
            val_kw["persistent_workers"] = True
        if args.seed is not None:
            val_kw["generator"] = torch.Generator().manual_seed(int(args.seed) + 1)
        val_loader = DataLoader(val_ds, **val_kw)

    # Periodic val every val_interval; save best.pt when val AUC improves (tie-break by accuracy when AUC=1)
    best_auc = [0.0]
    best_acc = [0.0]
    best_saved = [False]

    def on_epoch_end(epoch_1based, m):
        if epoch_1based % args.val_interval != 0:
            return None
        if val_loader is None or evaluate is None:
            return None
        try:
            met = evaluate(m, val_loader, device, use_amp)
            auc = met.get("auc_macro", 0.0)
            acc = met.get("accuracy", 0.0)
            # When AUC hits 1.0 on small val set, keep updating when accuracy improves
            if auc > best_auc[0] or (auc == best_auc[0] and acc > best_acc[0]):
                best_auc[0] = auc
                best_acc[0] = acc
                best_saved[0] = True
                torch.save(m.state_dict(), ckpt_path)
                log_line(f"  [best] AUC (macro)={auc:.4f} Acc={acc:.4f} saved to {ckpt_path.name}")
            return met
        except Exception as e:
            log_line(f"  Val @ epoch {epoch_1based}: {e}")
            return None

    t0 = time.perf_counter()
    epochs_actual = args.epochs
    if args.run == "B0":
        train_b0(model, train_loader, device, args.epochs, use_amp=use_amp, log_cb=log_cb, log_line=log_line, on_epoch_end=on_epoch_end)
    elif args.run == "VAT":
        train_vat(model, train_loader, device, args.epochs, vat_epsilon=args.vat_epsilon, vat_xi=args.vat_xi, vat_weight=args.vat_weight, use_amp=use_amp, log_cb=log_cb, log_line=log_line, on_epoch_end=on_epoch_end)
    else:
        epochs_actual = int(round(args.epochs * 1.2))
        log_line(f"{args.run}: using 20% more epochs: {args.epochs} -> {epochs_actual}")
        pmh_weight = 0.0 if args.run == "E1_no_pmh" else args.pmh_weight
        train_e1(
            model, train_loader, device, epochs_actual,
            warmup_epochs=args.warmup_epochs,
            pmh_weight=pmh_weight,
            pmh_cap_ratio=args.pmh_cap_ratio if args.pmh_cap_ratio > 0 else None,
            noise_sigma=args.noise_sigma,
            task_mix=args.task_mix,
            pmh_ramp_epochs=args.pmh_ramp_epochs,
            lr=args.lr,
            use_amp=use_amp,
            log_cb=log_cb,
            log_line=log_line,
            on_epoch_end=on_epoch_end,
        )
    train_time = time.perf_counter() - t0

    # Load best checkpoint so reported metrics and saved best.pt match
    if best_saved[0]:
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

    val_metrics = None
    if val_loader is not None and evaluate is not None:
        try:
            val_metrics = evaluate(model, val_loader, device, use_amp)
            log_line(f"Val — AUC (macro): {val_metrics.get('auc_macro', 0):.4f}  Acc: {val_metrics.get('accuracy', 0):.4f}  time: {train_time:.1f}s")
        except Exception as e:
            log_line(f"Val failed: {e}")

    results = {
        "run": args.run,
        "num_classes": num_classes,
        "epochs": args.epochs,
        "epochs_actual": epochs_actual,
        "train_time_s": round(train_time, 1),
        "checkpoint": "best.pt",
        "training_log": training_log,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
    }
    if val_metrics is not None:
        results.update(val_metrics)
    if args.run in ("E1", "E1_embed_only", "E1_no_pmh"):
        results["pmh_weight"] = 0.0 if args.run == "E1_no_pmh" else args.pmh_weight
        results["pmh_embed_only"] = (args.run == "E1_embed_only")
        results["pmh_cap_ratio"] = args.pmh_cap_ratio
        results["warmup_epochs"] = args.warmup_epochs
        results["pmh_ramp_epochs"] = args.pmh_ramp_epochs
        results["lr_e1"] = args.lr
        results["task_mix"] = args.task_mix
        results["noise_sigma"] = args.noise_sigma
        results["e1_task01_aligned"] = True

    torch.save(model.state_dict(), ckpt_path)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"run": args.run, "epochs": args.epochs, "epoch_log": training_log}, f, indent=2)
    log_line(f"Saved {ckpt_path} and {results_path}")
    log_file.close()


if __name__ == "__main__":
    main()
