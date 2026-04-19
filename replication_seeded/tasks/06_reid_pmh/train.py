"""
Train Re-ID: B0 (ID loss only), B1 (ID + aug), E1 (PMH aligned with Task 01/07).

Default E1: B1-style single view from the loader; noisy view = denorm + Gaussian + renorm
in the training loop; PMH = normalized MSE on last 3 backbone scales (layer2–4 pooled);
cosine PMH curriculum + task-tied cap + grad clip (same PMH recipe as CIFAR E1).

Optional --identity_pairs: legacy two-stream data (two crops / two IDs) + linear PMH ramp.
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

from data import IMAGENET_MEAN, IMAGENET_STD, get_train_loader, get_eval_loaders, find_market1501_root
from model import get_model

try:
    from eval import evaluate
except ImportError:
    evaluate = None


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


_norm_cache = {}


def _norm_tensors(device, dtype=None):
    sample_dtype = dtype or torch.float32
    key = (device, str(sample_dtype))
    if key not in _norm_cache:
        mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=sample_dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device, dtype=sample_dtype).view(1, 3, 1, 1)
        _norm_cache[key] = (mean, std)
    return _norm_cache[key]


class PMHLoss(nn.Module):
    """Same as Task 01: last K scales, L2-normalized, MSE(clean, noisy)."""

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
    if epoch < warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return 0.5 * (1.0 - math.cos(math.pi * progress)) * max_weight


def train_b0_b1(model, loader, device, epochs, use_amp=True, log_cb=None, log_line=None, on_epoch_end=None):
    """B0/B1: cross-entropy ID loss only. Optional on_epoch_end(epoch_1based, model) for val + save best."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    # Milestones at 50%/75% of training so LR drops meaningfully for longer runs
    m1, m2 = int(epochs * 0.5), int(epochs * 0.75)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m1, m2], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    _log = log_line or print

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.perf_counter()
        for batch in loader:
            if len(batch) == 2:
                img, pid = batch[0], batch[1]
            else:
                img, pid = batch[0], batch[2]  # E1 two_view: (clean, perturb, pid)
            img = img.to(device, non_blocking=True)
            pid = pid.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(img)
                loss = F.cross_entropy(logits, pid)
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
    task_mix=0.2,
    noise_sigma=0.1,
    pmh_ramp_epochs=20,
    identity_pairs=False,
    neg_margin=0.4,
    neg_weight=0.1,
    pmh_backoff_patience=0,
    pmh_backoff_factor=0.7,
    pmh_backoff_min=0.1,
    use_amp=True,
    log_cb=None,
    log_line=None,
    on_epoch_end=None,
):
    """Default: Task 01/07–aligned — single view + in-loop Gaussian; PMH on last 3 scales; cosine ramp; grad clip.

    Use identity_pairs=True for legacy two-stream data (two images per ID) + optional neg guard and PMH backoff.
    """
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    m1, m2 = int(epochs * 0.5), int(epochs * 0.75)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m1, m2], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    _log = log_line or print
    pmh_fn = PMHLoss(num_scales=3)
    pmh_scale = 1.0
    last_rank1 = None
    non_improve = 0
    backoff_on = pmh_backoff_patience > 0 and identity_pairs
    nw = neg_weight if identity_pairs else 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = total_task = total_pmh = total_pmh_eff = 0.0
        n = 0
        t0 = time.perf_counter()
        if identity_pairs:
            w_lin = 0.0 if epoch < warmup_epochs else min(1.0, (epoch - warmup_epochs) / 20.0)
            w_pmh = w_lin * pmh_weight * pmh_scale
        else:
            w_pmh = _cosine_pmh_weight(epoch, warmup_epochs, pmh_ramp_epochs, pmh_weight)

        for batch in loader:
            opt.zero_grad(set_to_none=True)

            if not identity_pairs:
                if len(batch) != 2:
                    raise RuntimeError("Task01-style E1 expects (image, pid); set two_view=False")
                images, pid = batch[0], batch[1]
                images = images.to(device, non_blocking=True)
                pid = pid.to(device, non_blocking=True)
                mean, std = _norm_tensors(device, dtype=images.dtype)
                denorm = images * std + mean
                noisy_denorm = denorm + noise_sigma * torch.randn_like(denorm, device=device)
                noisy_denorm = noisy_denorm.clamp(0, 1)
                images_noisy = (noisy_denorm - mean) / std

                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    logits_clean = model(images)
                    logits_noisy, _emb_n, feats_noisy = model(images_noisy, return_features=True)
                    loss_clean = F.cross_entropy(logits_clean, pid)
                    loss_noisy = F.cross_entropy(logits_noisy, pid)
                    loss_task = (1.0 - task_mix) * loss_clean + task_mix * loss_noisy
                    with torch.no_grad():
                        feats_clean = model.get_features(images)
                    loss_pmh = pmh_fn(feats_clean, feats_noisy)
                    pmh_term = w_pmh * loss_pmh
                    if pmh_cap_ratio is not None and pmh_cap_ratio > 0:
                        pmh_term = torch.minimum(pmh_term, pmh_cap_ratio * loss_task.detach())
                    loss = loss_task + pmh_term
            else:
                img_clean, img_perturb, pid = batch[0], batch[1], batch[2]
                img_clean = img_clean.to(device, non_blocking=True)
                img_perturb = img_perturb.to(device, non_blocking=True)
                pid = pid.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    logits_clean, emb_clean, feats_clean = model(img_clean, return_features=True)
                    logits_perturb, _emb_p, feats_perturb = model(img_perturb, return_features=True)
                    loss_clean = F.cross_entropy(logits_clean, pid)
                    loss_perturb = F.cross_entropy(logits_perturb, pid)
                    loss_task = (1.0 - task_mix) * loss_clean + task_mix * loss_perturb
                    loss_pmh = pmh_fn([fc.detach() for fc in feats_clean], feats_perturb)
                    emb_n = F.normalize(emb_clean, p=2, dim=1)
                    cos = emb_n @ emb_n.t()
                    same = pid[:, None].eq(pid[None, :])
                    neg_mask = ~same
                    if neg_mask.any() and nw > 0:
                        loss_neg = F.relu(cos[neg_mask] - neg_margin).mean()
                    else:
                        loss_neg = torch.zeros((), device=loss_task.device, dtype=loss_task.dtype)
                    pmh_term = w_pmh * loss_pmh
                    if pmh_cap_ratio is not None and pmh_cap_ratio > 0:
                        pmh_term = torch.minimum(pmh_term, pmh_cap_ratio * loss_task.detach())
                    loss = loss_task + pmh_term + nw * loss_neg

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
        div = n if n else 1
        avg_task = total_task / div
        avg_pmh = total_pmh / div
        avg_pmh_eff = total_pmh_eff / div
        pmh_frac = avg_pmh_eff / (avg_task + avg_pmh_eff) if (avg_task + avg_pmh_eff) > 0 else 0.0
        cap_note = " (capped)" if avg_pmh > avg_task and avg_pmh_eff < avg_pmh * 0.99 else ""
        mode = "identity" if identity_pairs else "task01"
        if epoch >= warmup_epochs and (epoch + 1) % 10 == 0:
            if pmh_frac < 0.02:
                _log(f"  [PMH check] pmh_frac={pmh_frac:.3f} — PMH may be too small (target 0.05–0.35)")
            elif pmh_frac > 0.5:
                _log(f"  [PMH check] pmh_frac={pmh_frac:.3f} — PMH may dominate task (consider lower weight/cap)")
        _log(
            f"  Epoch {epoch+1}/{epochs} [{mode}] loss={total_loss/div:.4f} task={avg_task:.4f} "
            f"pmh={avg_pmh:.4f} pmh_eff={avg_pmh_eff:.4f} frac={pmh_frac:.2f} "
            f"w_pmh={w_pmh:.4f} pmh_scale={pmh_scale:.2f} time={t1-t0:.1f}s{cap_note}"
        )
        if log_cb:
            log_cb(
                epoch + 1,
                {
                    "loss": total_loss / div,
                    "task": total_task / div,
                    "pmh": total_pmh / div,
                    "pmh_eff": total_pmh_eff / div,
                    "pmh_frac": round(pmh_frac, 4),
                    "w_pmh": round(float(w_pmh), 6),
                    "pmh_scale": round(pmh_scale, 4),
                    "time_s": round(t1 - t0, 2),
                },
            )
        if on_epoch_end:
            if backoff_on:
                r1 = on_epoch_end(epoch + 1, model)
                if r1 is not None:
                    if last_rank1 is None or r1 > last_rank1 + 1e-6:
                        non_improve = 0
                    else:
                        non_improve += 1
                    last_rank1 = r1
                    if non_improve >= pmh_backoff_patience and pmh_frac > 0.25 and pmh_scale > pmh_backoff_min:
                        old = pmh_scale
                        pmh_scale = max(pmh_backoff_min, pmh_scale * pmh_backoff_factor)
                        non_improve = 0
                        _log(
                            f"  [PMH backoff] rank1 stalled, pmh_frac={pmh_frac:.3f} -> "
                            f"pmh_scale {old:.2f} -> {pmh_scale:.2f}"
                        )
            else:
                on_epoch_end(epoch + 1, model)
    return model


def train_vat(model, loader, device, epochs, warmup_epochs=5, vat_eps=2.0, vat_xi=1e-6,
              vat_weight=1.0, use_amp=True, log_cb=None, log_line=None, on_epoch_end=None):
    """VAT: ID loss on clean + KL consistency under virtual adversarial image perturbation."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    m1, m2 = int(epochs * 0.5), int(epochs * 0.75)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m1, m2], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    _log = log_line or print
    for epoch in range(epochs):
        model.train()
        total_loss = total_task = total_vat = 0.0
        n = 0
        t0 = time.perf_counter()
        for batch in loader:
            img = batch[0] if len(batch) >= 1 else batch
            pid = batch[1] if len(batch) == 2 else batch[2]
            img = img.to(device, non_blocking=True)
            pid = pid.to(device, non_blocking=True)
            # Compute VAT perturbation
            with torch.no_grad():
                ref_logits = model(img)
                p_ref = F.softmax(ref_logits, dim=1)
            d = torch.randn_like(img)
            d = F.normalize(d.view(d.size(0), -1), p=2, dim=1).view_as(d)
            d = d.detach().requires_grad_(True)
            adv_logits = model(img + vat_xi * d)
            kl = F.kl_div(F.log_softmax(adv_logits, dim=1), p_ref, reduction='batchmean')
            kl.backward()
            d = F.normalize(d.grad.view(d.grad.size(0), -1), p=2, dim=1).view_as(d.grad).detach()
            r_adv = (vat_eps * d).detach()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits_clean = model(img)
                loss_task = F.cross_entropy(logits_clean, pid)
                logits_adv = model(img + r_adv)
                p_ref2 = F.softmax(logits_clean.detach(), dim=1)
                loss_vat = F.kl_div(F.log_softmax(logits_adv, dim=1), p_ref2, reduction='batchmean')
                loss = loss_task + vat_weight * loss_vat
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            total_loss += loss.item(); total_task += loss_task.item(); total_vat += loss_vat.item()
            n += 1
        scheduler.step()
        t1 = time.perf_counter()
        div = max(n, 1)
        _log(f"  Epoch {epoch+1}/{epochs} loss={total_loss/div:.4f} task={total_task/div:.4f} vat={total_vat/div:.4f} time={t1-t0:.1f}s")
        if log_cb: log_cb(epoch + 1, {"loss": total_loss/div, "task": total_task/div, "vat": total_vat/div})
        if on_epoch_end:
            on_epoch_end(epoch + 1, model)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default="E1", choices=["B0", "B1", "VAT", "E1"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default 0 for reproducible shuffle on all platforms)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (loader shuffle, Python random in dataset, torch)")
    p.add_argument("--deterministic", action="store_true", help="cudnn deterministic (slower; stricter reproducibility)")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--pmh_weight", type=float, default=0.5)
    p.add_argument("--pmh_cap_ratio", type=float, default=0.3, help="Cap PMH term at this fraction of task loss (0 = no cap); Task 01 default 0.3")
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--pmh_ramp_epochs", type=int, default=20, help="E1 Task01 mode: cosine PMH ramp length after warmup")
    p.add_argument("--task_mix", type=float, default=0.2)
    p.add_argument("--identity_pairs", action="store_true",
                   help="E1: legacy two-stream loader (two images per step). Default is Task01-style single view + in-loop Gaussian.")
    p.add_argument("--neg_margin", type=float, default=0.4, help="E1 identity mode: cosine margin for negative-pair guard")
    p.add_argument("--neg_weight", type=float, default=0.1, help="E1 identity mode: weight for negative-pair guard (ignored in Task01 mode)")
    p.add_argument("--pmh_backoff_patience", type=int, default=0,
                   help="E1 identity mode only: epochs without rank1 improvement before PMH backoff (0 = off)")
    p.add_argument("--pmh_backoff_factor", type=float, default=0.7, help="E1: multiplicative PMH backoff factor")
    p.add_argument("--pmh_backoff_min", type=float, default=0.1, help="E1: minimum PMH scale after adaptive backoff")
    p.add_argument("--noise_sigma", type=float, default=0.1)
    p.add_argument("--occlusion_ratio", type=float, default=0.2)
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument("--img_size", type=int, nargs=2, default=[256, 128])
    p.add_argument("--dummy", action="store_true", help="Use synthetic data (no Market-1501 needed)")
    p.add_argument("--no_identity_pairs", action="store_true",
                   help="With --identity_pairs: use same-image two-views (dataset) instead of two different images per ID")
    p.add_argument("--same_image_frac", type=float, default=0.0,
                   help="E1: fraction of batches with same-image pairs (rest=identity pairs). Aligns with robustness eval (gaussian/occlusion). Default 0.")
    p.add_argument("--vat_eps", type=float, default=2.0)
    p.add_argument("--vat_weight", type=float, default=1.0)
    p.add_argument("--gpu", type=int, default=None, help="GPU index to use (e.g. 0). When set, use cuda:N; else auto.")
    p.add_argument("--load_in_memory", action="store_true",
                   help="Preload all train images into RAM (Market-1501 fits in 128GB; no disk I/O during training)")
    args = p.parse_args()

    run_dir = Path(args.out_dir) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    results_path = run_dir / "results.json"
    log_path = run_dir / "training_log.json"
    text_log_path = run_dir / "train.log"

    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu requested but torch.cuda.is_available() is False. Check CUDA_VISIBLE_DEVICES and PyTorch install.")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed, deterministic=args.deterministic)
    use_amp = not args.no_amp and device.type == "cuda"
    identity_e1 = args.run == "E1" and args.identity_pairs
    two_view = identity_e1
    same_image_frac = 0.0 if args.no_identity_pairs else args.same_image_frac

    train_loader, num_classes = get_train_loader(
        args.data_dir,
        args.run,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
        two_view=two_view,
        use_identity_pairs=(not args.no_identity_pairs) if identity_e1 else False,
        same_image_frac=same_image_frac if identity_e1 else 0.0,
        noise_sigma=args.noise_sigma,
        occlusion_ratio=args.occlusion_ratio,
        dummy=args.dummy,
        load_in_memory=args.load_in_memory,
        seed=args.seed,
    )

    model = get_model(num_classes, embed_dim=args.embed_dim, pretrained=not args.dummy).to(device)

    training_log = []
    log_file = open(text_log_path, "w", encoding="utf-8")

    def log_line(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    def log_cb(epoch, metrics):
        training_log.append({"epoch": epoch, **metrics})

    log_line(f"run={args.run} device={device} num_classes={num_classes} run_dir={run_dir}")
    if args.run == "E1":
        if identity_e1:
            if args.no_identity_pairs:
                log_line("E1 [identity]: same-image two-stream (dataset noise/occlusion)")
            elif same_image_frac > 0:
                log_line(f"E1 [identity]: {same_image_frac:.0%} same-image + {1-same_image_frac:.0%} cross-image pairs")
            else:
                log_line("E1 [identity]: two different images per ID + dataset perturb on view2")
        else:
            log_line(
                f"E1 [Task01]: single view + in-loop Gaussian σ={args.noise_sigma}; "
                f"PMH last-3 scales; cosine ramp {args.warmup_epochs}+{args.pmh_ramp_epochs} ep"
            )

    # Optional: validation loaders for best-by-rank1 (skip when --dummy)
    query_loader = gallery_loader = None
    if evaluate is not None and not args.dummy:
        try:
            root = find_market1501_root(args.data_dir)
            query_loader, gallery_loader, _, _ = get_eval_loaders(
                root,
                batch_size=64,
                num_workers=args.num_workers,
                seed=args.seed,
            )
        except Exception as e:
            log_line(f"Eval loaders skipped: {e}")

    # Periodic val every val_interval; save best.pt when val rank-1 improves
    best_rank1 = [0.0]
    best_saved = [False]

    def on_epoch_end(epoch_1based, m):
        if epoch_1based % args.val_interval != 0:
            return None
        if query_loader is None or gallery_loader is None or evaluate is None:
            return None
        try:
            met = evaluate(m, query_loader, gallery_loader, device, use_amp)
            r1 = met["rank1"]
            if r1 > best_rank1[0]:
                best_rank1[0] = r1
                best_saved[0] = True
                torch.save(m.state_dict(), ckpt_path)
                log_line(f"  [best] rank-1={r1:.2f}% saved to {ckpt_path.name}")
            return r1
        except Exception as e:
            log_line(f"  Val @ epoch {epoch_1based}: {e}")
            return None

    t0 = time.perf_counter()
    if args.run in ("B0", "B1"):
        epochs_actual = args.epochs
        train_b0_b1(
            model, train_loader, device, args.epochs,
            use_amp=use_amp, log_cb=log_cb, log_line=log_line,
            on_epoch_end=on_epoch_end,
        )
    elif args.run == "VAT":
        epochs_actual = int(round(args.epochs * 1.2))
        log_line(f"VAT: using 20% more epochs: {args.epochs} -> {epochs_actual}")
        train_vat(model, train_loader, device, epochs_actual,
                  vat_eps=args.vat_eps, vat_weight=args.vat_weight,
                  use_amp=use_amp, log_cb=log_cb, log_line=log_line,
                  on_epoch_end=on_epoch_end)
    else:
        epochs_actual = int(round(args.epochs * 1.2))
        log_line(f"E1: using 20% more epochs: {args.epochs} -> {epochs_actual}")
        train_e1(
            model, train_loader, device, epochs_actual,
            warmup_epochs=args.warmup_epochs,
            pmh_weight=args.pmh_weight,
            pmh_cap_ratio=args.pmh_cap_ratio if args.pmh_cap_ratio > 0 else None,
            task_mix=args.task_mix,
            noise_sigma=args.noise_sigma,
            pmh_ramp_epochs=args.pmh_ramp_epochs,
            identity_pairs=identity_e1,
            neg_margin=args.neg_margin,
            neg_weight=args.neg_weight,
            pmh_backoff_patience=args.pmh_backoff_patience,
            pmh_backoff_factor=args.pmh_backoff_factor,
            pmh_backoff_min=args.pmh_backoff_min,
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

    # Validation if we have query/gallery
    rank1 = mAP = None
    val_metrics = None
    if query_loader is not None and gallery_loader is not None and evaluate is not None:
        try:
            val_metrics = evaluate(model, query_loader, gallery_loader, device, use_amp)
            rank1, mAP = val_metrics["rank1"], val_metrics["mAP"]
            msg = f"Val — rank-1: {rank1:.2f}%  mAP: {mAP:.2f}%"
            if "rank5" in val_metrics:
                msg += f"  rank-5: {val_metrics['rank5']:.2f}%  rank-10: {val_metrics['rank10']:.2f}%"
            log_line(f"{msg}  time: {train_time:.1f}s")
        except Exception as e:
            log_line(f"Val failed: {e}")

    results = {
        "task": "06", "run": args.run,
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
        # Ensure mAP is always present when val ran (legacy results may miss it)
        if rank1 is not None and "mAP" not in results:
            results["mAP"] = round(float(mAP), 4) if mAP is not None else None
    if args.run == "E1":
        results["pmh_weight"] = args.pmh_weight
        results["pmh_cap_ratio"] = args.pmh_cap_ratio
        results["e1_task01_aligned"] = not identity_e1
        results["identity_pairs_mode"] = identity_e1
        results["dataset_identity_pairs"] = (not args.no_identity_pairs) if identity_e1 else None
        results["same_image_frac"] = same_image_frac if identity_e1 else None
        results["warmup_epochs"] = args.warmup_epochs
        results["pmh_ramp_epochs"] = args.pmh_ramp_epochs
        results["task_mix"] = args.task_mix
        results["neg_margin"] = args.neg_margin
        results["neg_weight"] = args.neg_weight
        results["pmh_backoff_patience"] = args.pmh_backoff_patience
        results["pmh_backoff_factor"] = args.pmh_backoff_factor
        results["pmh_backoff_min"] = args.pmh_backoff_min
        results["noise_sigma"] = args.noise_sigma
        results["occlusion_ratio"] = args.occlusion_ratio

    torch.save(model.state_dict(), ckpt_path)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"run": args.run, "epochs": args.epochs, "epoch_log": training_log}, f, indent=2)
    log_line(f"Saved {ckpt_path} and {results_path}")
    log_file.close()


if __name__ == "__main__":
    main()
