"""
Train baseline (MSE only) or E1 (MSE + PMH feature alignment + geometric consistency) for pose under occlusion.

Replication: --seed / --deterministic; pass loader_seed to COCO DataLoader when shuffle=True.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import get_model
from data import (
    get_pose_dataloader,
    get_coco_pose_dataloader,
    apply_random_occlusion,
    perturb_input,
    DummyPoseDataset,
)
from geometry import geometric_consistency_loss, mpjpe


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


def _relational_stability_simple(c_flat, n_flat, n_pairs=128):
    """Preserve pairwise distance structure: normalized distances (clean) vs (corrupt). Single scale, no curriculum."""
    B = c_flat.shape[0]
    if B < 4:
        return torch.tensor(0.0, device=c_flat.device)
    c = F.normalize(c_flat, p=2, dim=1)
    n = F.normalize(n_flat, p=2, dim=1)
    npairs = min(n_pairs, B * (B - 1) // 2)
    idx_i = torch.randint(0, B, (npairs,), device=c.device)
    idx_j = torch.randint(0, B, (npairs,), device=c.device)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    if idx_i.numel() < 2:
        return torch.tensor(0.0, device=c.device)
    d_c = (c[idx_i] - c[idx_j]).pow(2).sum(dim=1)
    d_n = (n[idx_i] - n[idx_j]).pow(2).sum(dim=1)
    d_c = d_c / (d_c.mean().detach() + 1e-8)
    d_n = d_n / (d_n.mean().detach() + 1e-8)
    return F.mse_loss(d_n, d_c.detach())


class PMHLossPose(nn.Module):
    """Pointwise PMH: multi-scale ||φ(x) - φ(x')||². Optional: + relational term (preserve pairwise structure)."""
    def __init__(self, num_scales=3, use_relational=False, relational_weight=0.25, n_pairs=128):
        super().__init__()
        self.num_scales = num_scales
        self.use_relational = use_relational
        self.relational_weight = relational_weight
        self.n_pairs = n_pairs

    def forward(self, features_clean, features_corrupt):
        fc = features_clean[-self.num_scales:]
        fn = features_corrupt[-self.num_scales:]
        loss_point = 0.0
        loss_rel = 0.0
        for c, n in zip(fc, fn):
            if c.dim() == 4:
                c_flat = F.adaptive_avg_pool2d(c, (1, 1)).flatten(1)
                n_flat = F.adaptive_avg_pool2d(n, (1, 1)).flatten(1)
            else:
                c_flat, n_flat = c, n
            c_n = F.normalize(c_flat, p=2, dim=1)
            n_n = F.normalize(n_flat, p=2, dim=1)
            loss_point = loss_point + (c_n - n_n).pow(2).sum(dim=1).mean()
            if self.use_relational:
                loss_rel = loss_rel + _relational_stability_simple(c_flat, n_flat, self.n_pairs)
        loss_point = loss_point / self.num_scales
        if self.use_relational:
            loss_point = loss_point + self.relational_weight * (loss_rel / self.num_scales)
        return loss_point


class GPUTensorDataset(torch.utils.data.Dataset):
    """Dataset that holds images and poses already on GPU. No transfer per batch."""
    def __init__(self, images, poses):
        assert images.device.type == "cuda" and poses.device.type == "cuda"
        self.images = images
        self.poses = poses

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.poses[idx]


def _loader_to_gpu(loader, device, shuffle_seed=None):
    """Load entire dataset from loader to GPU; return new DataLoader over GPU tensors. Use for small subsets (e.g. 10k)."""
    assert device.type == "cuda"
    print("  Loading full dataset to GPU...", flush=True)
    batches_im, batches_pose = [], []
    for images, poses in loader:
        batches_im.append(images)
        batches_pose.append(poses)
    images_cat = torch.cat(batches_im, dim=0).to(device, non_blocking=True)
    poses_cat = torch.cat(batches_pose, dim=0).to(device, non_blocking=True)
    n, c, h, w = images_cat.shape
    mem_gb = (images_cat.numel() + poses_cat.numel()) * 4 / 1e9
    print(f"  On GPU: {n} samples, {c}x{h}x{w}, ~{mem_gb:.1f} GB", flush=True)
    gpu_ds = GPUTensorDataset(images_cat, poses_cat)
    dl_kw = dict(
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    if shuffle_seed is not None:
        dl_kw["generator"] = torch.Generator().manual_seed(int(shuffle_seed))
    return torch.utils.data.DataLoader(gpu_ds, **dl_kw)


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _pose_loss_2d(pred, gt, vis):
    """Masked MSE on (x,y). vis: (B, 17), 1 where keypoint is labeled."""
    mask = (vis >= 1).float().unsqueeze(-1)
    diff = (pred[:, :, :2] - gt[:, :, :2]) * mask
    n = (mask.sum() * 2).clamp(min=1)
    return (diff * diff).sum() / n


def _pck_2d_batch(pred, gt, vis, threshold=0.05):
    """PCK (fraction correct) on a batch. pred, gt: (B, 17, 3); vis: (B, 17)."""
    mask = (vis >= 1).float()
    diff = (pred[:, :, :2] - gt[:, :, :2]).norm(dim=-1)
    correct = ((diff <= threshold).float() * mask).sum()
    total = mask.sum().clamp(min=1)
    return (correct / total).item()


def _mke_2d_batch(pred, gt, vis):
    """Mean keypoint error (L2 in normalized coords) over visible keypoints."""
    mask = (vis >= 1).float().unsqueeze(-1)
    diff = (pred[:, :, :2] - gt[:, :, :2]) * mask
    n = mask.sum().clamp(min=1)
    return (diff.norm(dim=-1).sum() / n).item()


def train_baseline(model, loader, device, epochs, lr=1e-3, use_amp=True, occlusion_ratio=0.0,
                   is_2d=False, log_cb=None, log_line=None):
    """MSE only; optional occlusion. is_2d: use first 2 coords and visibility mask (COCO)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    _log = log_line or print
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for images, pose_gt in loader:
            images = images.to(device, non_blocking=True)
            pose_gt = pose_gt.to(device, non_blocking=True)
            if occlusion_ratio > 0:
                images = apply_random_occlusion(images, mask_ratio=occlusion_ratio, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                pose_pred = model(images)
                if is_2d:
                    vis = pose_gt[:, :, 2]
                    loss = _pose_loss_2d(pose_pred, pose_gt, vis)
                else:
                    loss = (pose_pred - pose_gt).pow(2).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        metrics = {"loss": round(avg_loss, 6)}
        if is_2d and len(loader.dataset) > 0:
            model.eval()
            with torch.no_grad():
                pck_sum, mke_sum, cnt = 0.0, 0.0, 0
                for images, pose_gt in loader:
                    images = images.to(device, non_blocking=True)
                    pose_gt = pose_gt.to(device, non_blocking=True)
                    pose_pred = model(images)
                    vis = pose_gt[:, :, 2]
                    pck_sum += _pck_2d_batch(pose_pred, pose_gt, vis) * images.size(0)
                    mke_sum += _mke_2d_batch(pose_pred, pose_gt, vis) * images.size(0)
                    cnt += images.size(0)
                if cnt > 0:
                    metrics["pck@0.05"] = round(pck_sum / cnt, 6)
                    metrics["mke"] = round(mke_sum / cnt, 6)
            model.train()
        if log_cb:
            log_cb(epoch, metrics)
        _log(f"epoch {epoch} " + " ".join(f"{k}={v}" for k, v in metrics.items()))


def train_e1(model, loader, device, epochs, lr=1e-3, use_amp=True, occlusion_ratio=0.3,
             gaussian_sigma=0.05, pmh_weight=0.1, pmh_cap_ratio=0.3, geom_weight=0.1,
             task_mix=0.5, warmup_epochs=10, is_2d=False,
             log_cb=None, log_line=None, use_pmh_relational=False):
    """E1: task loss (clean + perturbed mix) + PMH. Perturbation: occlusion + Gaussian (perturb_input)."""
    pmh_loss_fn = PMHLossPose(num_scales=3, use_relational=use_pmh_relational)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    _log = log_line or print
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_task = 0.0
        total_pmh = 0.0
        total_pmh_eff = 0.0
        total_geom = 0.0
        n_batches = 0
        w_pmh = pmh_weight * min(1.0, (epoch + 1) / max(warmup_epochs, 1))
        w_geom = geom_weight * min(1.0, (epoch + 1) / max(warmup_epochs, 1))
        for images, pose_gt in loader:
            images = images.to(device, non_blocking=True)
            pose_gt = pose_gt.to(device, non_blocking=True)
            images_pert = perturb_input(images, occlusion_ratio=occlusion_ratio, gaussian_sigma=gaussian_sigma, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                pose_clean, feats_clean = model(images, return_features=True)
                pose_occ, feats_occ = model(images_pert, return_features=True)
                if is_2d:
                    vis = pose_gt[:, :, 2]
                    task_clean = _pose_loss_2d(pose_clean, pose_gt, vis)
                    task_occ = _pose_loss_2d(pose_occ, pose_gt, vis)
                else:
                    task_clean = (pose_clean - pose_gt).pow(2).mean()
                    task_occ = (pose_occ - pose_gt).pow(2).mean()
                task_loss = (1 - task_mix) * task_clean + task_mix * task_occ
                pmh_loss = pmh_loss_fn(feats_clean, feats_occ)
                pmh_term = w_pmh * pmh_loss
                if pmh_cap_ratio > 0:
                    pmh_term = torch.minimum(pmh_term, pmh_cap_ratio * task_loss.detach())
                geom_loss = torch.tensor(0.0, device=device)
                if not is_2d and w_geom > 0:
                    geom_loss = geometric_consistency_loss(pose_clean, pose_occ)
                loss = task_loss + pmh_term + w_geom * geom_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            total_task += task_loss.item()
            total_pmh += pmh_loss.item()
            total_pmh_eff += pmh_term.item()
            total_geom += geom_loss.item()
            n_batches += 1
        n_b = max(n_batches, 1)
        avg_task = total_task / n_b
        avg_pmh_eff = total_pmh_eff / n_b
        pmh_frac = avg_pmh_eff / (avg_task + avg_pmh_eff) if (avg_task + avg_pmh_eff) > 0 else 0.0
        metrics = {
            "loss": round(total_loss / n_b, 6),
            "task": round(avg_task, 6),
            "pmh": round(total_pmh / n_b, 6),
            "pmh_eff": round(avg_pmh_eff, 6),
            "pmh_frac": round(pmh_frac, 4),
        }
        if not is_2d:
            metrics["geom"] = round(total_geom / n_b, 6)
        if is_2d and len(loader.dataset) > 0:
            model.eval()
            with torch.no_grad():
                pck_sum, mke_sum, cnt = 0.0, 0.0, 0
                for images, pose_gt in loader:
                    images = images.to(device, non_blocking=True)
                    pose_gt = pose_gt.to(device, non_blocking=True)
                    pose_pred = model(images)
                    vis = pose_gt[:, :, 2]
                    pck_sum += _pck_2d_batch(pose_pred, pose_gt, vis) * images.size(0)
                    mke_sum += _mke_2d_batch(pose_pred, pose_gt, vis) * images.size(0)
                    cnt += images.size(0)
                if cnt > 0:
                    metrics["pck@0.05"] = round(pck_sum / cnt, 6)
                    metrics["mke"] = round(mke_sum / cnt, 6)
            model.train()
        if log_cb:
            log_cb(epoch, metrics)
        _log(f"epoch {epoch} " + " ".join(f"{k}={v}" for k, v in metrics.items()))


def _vat_perturbation_pose(model, images, xi=1e-3, eps=0.1, use_amp=False, n_power=4):
    """VAT direction for pose heatmaps: multi-step power iteration (like Miyato VAT).
    Uses float32 + autograd.grad w.r.t. d only (no .backward into model params here).
    """
    images_f32 = images.float()
    with torch.amp.autocast("cuda", enabled=False):
        with torch.no_grad():
            ref_out = model(images_f32).float()
        d = torch.randn_like(images_f32)
        d = F.normalize(d.view(d.size(0), -1), p=2, dim=1).view_as(d).detach()
        for _ in range(max(1, n_power)):
            d = d.requires_grad_(True)
            adv_out = model(images_f32 + xi * d).float()
            cons_loss = F.mse_loss(adv_out, ref_out)
            g = torch.autograd.grad(cons_loss, d, create_graph=False, retain_graph=False)[0]
            if g is None or not torch.isfinite(g).all():
                return torch.zeros_like(images)
            if g.abs().max() < 1e-14:
                return torch.zeros_like(images)
            d = F.normalize(g.view(g.size(0), -1), p=2, dim=1).view_as(g).detach()
    return (eps * d).to(images.dtype).detach()


def train_vat(model, loader, device, epochs, lr=1e-3, vat_eps=0.1, vat_xi=1e-3,
              vat_weight=1.0, use_amp=True, is_2d=False, log_cb=None, log_line=None):
    """VAT: task loss (MSE heatmap) + output consistency under adversarial perturbation."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs//3), gamma=0.5)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    _log = log_line or print
    for epoch in range(epochs):
        model.train()
        total_loss = total_task = total_vat = 0.0
        n = 0
        t0 = time.perf_counter()
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, targets = batch[0], batch[1]
            else:
                continue
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            r_adv = _vat_perturbation_pose(model, images, xi=vat_xi, eps=vat_eps,
                                           use_amp=(use_amp and device.type == "cuda"))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                pred_clean = model(images)
                loss_task = F.mse_loss(pred_clean, targets)
                pred_adv = model(images + r_adv)
            # Huber on output drift: stronger gradients than MSE when |Δ| is tiny (VAT was ~1e-4 vs task ~0.2)
            with torch.amp.autocast("cuda", enabled=False):
                loss_vat = F.smooth_l1_loss(
                    pred_adv.float(), pred_clean.detach().float(), beta=0.02
                )
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
        metrics = {"loss": round(total_loss/div, 6), "task": round(total_task/div, 6),
                   "vat": round(total_vat/div, 6), "time_s": round(t1-t0, 2)}
        # Validation: PCK and MKE (same as train_baseline)
        if is_2d and n > 0:
            model.eval()
            with torch.no_grad():
                pck_sum, mke_sum, cnt = 0.0, 0.0, 0
                for batch in loader:
                    if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                        continue
                    imgs_v, pose_v = batch[0].to(device), batch[1].to(device)
                    pred_v = model(imgs_v)
                    vis_v = pose_v[:, :, 2]
                    pck_sum += _pck_2d_batch(pred_v, pose_v, vis_v) * imgs_v.size(0)
                    mke_sum += _mke_2d_batch(pred_v, pose_v, vis_v) * imgs_v.size(0)
                    cnt += imgs_v.size(0)
                if cnt > 0:
                    metrics["pck@0.05"] = round(pck_sum / cnt, 6)
                    metrics["mke"] = round(mke_sum / cnt, 6)
            model.train()
        _log(f"  Epoch {epoch+1}/{epochs} loss={total_loss/div:.4f} task={total_task/div:.4f} "
             f"vat={total_vat/div:.4f} pck={metrics.get('pck@0.05','?')} time={t1-t0:.1f}s")
        if log_cb: log_cb(epoch + 1, metrics)
    return model


def main():
    p = argparse.ArgumentParser(description="Train baseline or E1 for pose (COCO 2D or 3D).")
    p.add_argument("--run", type=str, default="baseline", choices=["baseline", "B0", "VAT", "E1"])
    p.add_argument("--epochs", type=int, default=None,
                   help="Epochs. Default: 50 for baseline, 30 for E1 (baseline needs more for fair comparison).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (use 4-8 on Linux; 0 on Windows)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--occlusion_ratio", type=float, default=0.3, help="Occlusion mask ratio for E1 / aug")
    p.add_argument("--gaussian_sigma", type=float, default=0.05, help="Gaussian noise std for E1 perturb_input (0 = no noise).")
    p.add_argument("--pmh_weight", type=float, default=0.1)
    p.add_argument("--pmh_cap_ratio", type=float, default=0.3, help="Cap PMH term to pmh_cap_ratio * task_loss (0 disables)")
    p.add_argument("--geom_weight", type=float, default=0.1)
    p.add_argument("--task_mix", type=float, default=0.5)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--vat_eps", type=float, default=0.1,
                   help="VAT: L2 radius of adversarial perturbation (0.05 was too small; 0.1 gives meaningful signal)")
    p.add_argument("--vat_xi", type=float, default=1e-3,
                   help="VAT: step size for power iteration (1e-6 caused gradient underflow)")
    p.add_argument("--vat_weight", type=float, default=1.0)
    p.add_argument("--dummy", action="store_true", help="Use dummy dataset (no real data)")
    p.add_argument("--dataset", type=str, default=None, choices=["dummy", "coco"],
                   help="dummy: synthetic; coco: COCO train2017/val2017 2D keypoints")
    p.add_argument("--pretrained", action="store_true", help="ImageNet-pretrained backbone (recommended for COCO)")
    p.add_argument("--cache_dataset", action="store_true",
                   help="Cache COCO in RAM after first load (faster epoch 2+; needs ~2GB RAM)")
    p.add_argument("--no_cache_dataset", action="store_true",
                   help="Disable caching COCO in RAM (default: cache when --max_train_samples is set)")
    p.add_argument("--data_on_gpu", action="store_true",
                   help="Load full training set to GPU (for small subsets, e.g. 10k; ~8 GB for 10k@256). No data-load overhead.")
    p.add_argument("--image_size", type=int, default=256,
                   help="Input size (square). 256 default; 192 or 128 for faster training.")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Max COCO samples for training. Random subset with --subset_seed.")
    p.add_argument("--subset_seed", type=int, default=42,
                   help="Seed for random train/val subsample when max_train_samples set.")
    p.add_argument("--seed", type=int, default=42, help="Global RNG + COCO train shuffle (loader_seed)")
    p.add_argument("--deterministic", action="store_true", help="Strict cuDNN determinism (slower)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile(model) for faster training (PyTorch 2+).")
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU id to use (default: all). Set 0 or 1 to use one GPU; no flag = use all (DataParallel).")
    p.add_argument("--pmh_relational", action="store_true",
                   help="E1 only: add simple relational term (preserve pairwise structure) on top of pointwise PMH.")
    args = p.parse_args()
    set_global_seed(args.seed, deterministic=args.deterministic)
    if args.run == "B0": args.run = "baseline"  # B0 is alias for baseline

    if args.epochs is None:
        args.epochs = 50 if args.run == "baseline" else 30
    use_coco = args.dataset == "coco"
    use_dummy = args.dummy or args.dataset == "dummy"

    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        use_multi_gpu = False
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_multi_gpu = device.type == "cuda" and torch.cuda.device_count() > 1
    use_amp = not args.no_amp and device.type == "cuda"
    run_dir = Path(args.out_dir) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    results_path = run_dir / "results.json"

    training_log = []
    def log_cb(epoch, metrics):
        training_log.append({"epoch": epoch, **metrics})

    if use_coco:
        cache_images = args.cache_dataset or (args.max_train_samples is not None and not args.no_cache_dataset)
        print("COCO data:", flush=True)
        loader = get_coco_pose_dataloader(
            args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size, max_samples=args.max_train_samples, subset_seed=args.subset_seed,
            imagenet_norm=args.pretrained, cache_images=cache_images, split="train",
            loader_seed=args.seed,
        )
        if cache_images:
            print("  Cache enabled: first epoch loads from disk, later epochs use RAM.", flush=True)
        if args.max_train_samples is not None:
            print(f"  Subset: {args.max_train_samples} samples (random, seed={args.subset_seed}).", flush=True)
        if not cache_images:
            print("  Tip: use --cache_dataset (or subset with max_train_samples) and --batch_size 64 for faster runs.", flush=True)
        if args.data_on_gpu and device.type == "cuda":
            loader = _loader_to_gpu(loader, device, shuffle_seed=args.seed)
            print("  Training from GPU memory (no data-load overhead).", flush=True)
    elif use_dummy:
        train_ds = DummyPoseDataset(num_samples=500, image_size=256)
        loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(int(args.seed)),
        )
    else:
        loader = get_pose_dataloader(args.data_dir, split="train", batch_size=args.batch_size, num_workers=0)
        if loader is None:
            print("No dataset found. Use --dataset dummy or --dataset coco.", flush=True)
            return

    model = get_model(backbone="resnet18", pretrained=args.pretrained).to(device)
    if use_multi_gpu:
        model = nn.DataParallel(model)
        print(f"  Using DataParallel on {torch.cuda.device_count()} GPUs.", flush=True)
    if getattr(args, "compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
        print("  Model: torch.compile enabled (reduce-overhead).", flush=True)
    dataset_name = (args.dataset if args.dataset else ("dummy" if use_dummy else "3d"))
    occ = args.occlusion_ratio
    print(
        f"run={args.run} dataset={dataset_name} device={device} occlusion_ratio={occ} epochs={args.epochs} "
        f"seed={args.seed} subset_seed={args.subset_seed}" + (" (no occlusion)" if occ == 0 else ""),
        flush=True,
    )
    if args.run == "E1":
        print(f"  E1 perturbation: occlusion + Gaussian (sigma={args.gaussian_sigma}).", flush=True)
    if args.run == "E1" and getattr(args, "pmh_relational", False):
        print("  PMH: pointwise + relational (pairwise structure).", flush=True)
    t0 = time.perf_counter()
    if args.run == "baseline":
        train_baseline(model, loader, device, args.epochs, lr=args.lr, use_amp=use_amp,
                      occlusion_ratio=occ, is_2d=use_coco, log_cb=log_cb, log_line=print)
    elif args.run == "VAT":
        train_vat(model, loader, device, args.epochs, lr=args.lr, vat_eps=args.vat_eps,
                  vat_xi=args.vat_xi, vat_weight=args.vat_weight, use_amp=use_amp, is_2d=use_coco,
                  log_cb=log_cb, log_line=print)
    else:
        train_e1(model, loader, device, args.epochs, lr=args.lr, use_amp=use_amp,
                occlusion_ratio=args.occlusion_ratio, gaussian_sigma=args.gaussian_sigma,
                pmh_weight=args.pmh_weight, pmh_cap_ratio=args.pmh_cap_ratio,
                geom_weight=0.0 if use_coco else args.geom_weight,
                task_mix=args.task_mix, warmup_epochs=args.warmup_epochs, is_2d=use_coco,
                log_cb=log_cb, log_line=print, use_pmh_relational=getattr(args, "pmh_relational", False))
    train_time = time.perf_counter() - t0

    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, ckpt_path)

    # Extract best metrics from training_log for results.json
    # Training log keys: "pck@0.05" (or "pck_05"/"pck") and "mke"/"mpjpe"
    best_pck = best_mpjpe = None
    if training_log:
        def _get_pck(e):
            return e.get("pck@0.05") or e.get("pck_05") or e.get("pck")
        def _get_mke(e):
            return e.get("mke") or e.get("mpjpe")
        # Scan all epochs for best values
        for entry in training_log:
            p = _get_pck(entry)
            if p is not None:
                if best_pck is None or p > best_pck:
                    best_pck = p
            m = _get_mke(entry)
            if m is not None:
                if best_mpjpe is None or m < best_mpjpe:
                    best_mpjpe = m

    # Fallback: if pck/mke missing (e.g. VAT trained with dummy) but COCO available, run eval on val
    if best_pck is None and use_coco:
        try:
            val_loader = get_coco_pose_dataloader(
                args.data_dir, batch_size=args.batch_size, num_workers=0,
                image_size=args.image_size, max_samples=args.max_train_samples, subset_seed=args.subset_seed,
                imagenet_norm=args.pretrained, cache_images=False, split="val", shuffle=False,
                loader_seed=None,
            )
            model.eval()
            pck_sum, mke_sum, cnt = 0.0, 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        imgs, pose = batch[0].to(device), batch[1].to(device)
                        pred = model(imgs)
                        vis = pose[:, :, 2]
                        pck_sum += _pck_2d_batch(pred, pose, vis) * imgs.size(0)
                        mke_sum += _mke_2d_batch(pred, pose, vis) * imgs.size(0)
                        cnt += imgs.size(0)
            if cnt > 0:
                best_pck = round(pck_sum / cnt, 6)
                best_mpjpe = round(mke_sum / cnt, 6)
                print(f"  Fallback eval on val: pck@0.05={best_pck:.4f} mke={best_mpjpe:.4f}")
        except Exception as e:
            print(f"  Fallback eval skipped: {e}")

    results = {
        "task": "05", "run": args.run,
        "dataset": dataset_name,
        "epochs": args.epochs,
        "seed": args.seed,
        "subset_seed": args.subset_seed,
        "deterministic": bool(args.deterministic),
        "occlusion_ratio": occ,
        "train_time_s": round(train_time, 1),
        "checkpoint": "best.pt",
        "training_log": training_log,
    }
    if best_pck is not None:
        results["pck_05"] = best_pck
    if best_mpjpe is not None:
        results["mpjpe_mm"] = best_mpjpe
    # Save final-epoch task loss as fallback metric (always available)
    if training_log:
        results["final_task_loss"] = training_log[-1].get("task", training_log[-1].get("loss"))

    if args.run == "E1":
        results.update({"pmh_weight": args.pmh_weight, "pmh_cap_ratio": args.pmh_cap_ratio,
                        "gaussian_sigma": args.gaussian_sigma, "geom_weight": args.geom_weight,
                        "task_mix": args.task_mix,
                        "pmh_relational": getattr(args, "pmh_relational", False)})
    elif args.run == "VAT":
        results.update({"vat_eps": args.vat_eps, "vat_xi": args.vat_xi, "vat_weight": args.vat_weight})
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save training_log.json separately (consistent with tasks 01/02/06/07)
    log_path = run_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"task": "05", "run": args.run, "dataset": dataset_name,
                   "epochs": args.epochs, "epoch_log": training_log}, f, indent=2)
    print(f"Saved {ckpt_path}, {results_path}, {log_path}")


if __name__ == "__main__":
    main()
