"""
Train B0 (MSE only), VAT, and E1 (MSE + PMH with measurement noise on positions).
Aligned with Chemistry.ipynb / binding.ipynb: hidden=128, node_noise=0.1 + pos=0.15, warmup 25%, PMH max weight 1.0, E1 120 epochs.
Saves to runs/<target_name>/<run>/: best.pt, results.json, training_log.json, train.log.

Replication: --seed (default 42) for split, shuffle, post-train eval noise; --deterministic optional.
Paper headline MAE uses trained E1 with node PMH (same as --run E1); see submission tab:crosstask.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import get_loaders
from model import get_model, NUM_TARGETS
from perturb import add_measurement_noise, MEASUREMENT_NOISE_STD, NODE_NOISE_STD
from eval import evaluate


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


def _make_scaler(use_amp: bool):
    """Return a GradScaler if AMP is requested and CUDA is available."""
    if use_amp and torch.cuda.is_available():
        return torch.amp.GradScaler("cuda")
    return None


def _autocast(use_amp: bool):
    """Context manager: autocast when use_amp=True and CUDA available, else no-op."""
    if use_amp and torch.cuda.is_available():
        return torch.amp.autocast("cuda")
    return torch.amp.autocast("cpu", enabled=False)


def train_b0(
    model,
    loader,
    device,
    epochs,
    norm_params=None,
    val_loader=None,
    ckpt_path=None,
    val_interval=5,
    lr=0.001,
    weight_decay=0.0,
    scheduler="step",
    step_size=50,
    step_gamma=0.5,
    plateau_patience=15,
    use_amp=True,
    log_cb=None,
    log_line=None,
):
    """B0: MSE loss only on normalized targets. No PMH, no noise. Saves best by val MAE (like binding.py)."""
    norm_params = norm_params or {}
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=step_gamma)
    elif scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=plateau_patience, min_lr=1e-5
        )
    scaler = _make_scaler(use_amp)
    best_mae = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.perf_counter()
        for data in loader:
            data = data.to(device)
            opt.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                pred = model(data.x, data.pos, data.edge_index, data.batch)
                y = data.y.view(data.num_graphs, -1).float()[:, : pred.shape[1]]
                mean = norm_params["mean"].to(y.device)
                std = norm_params["std"].to(y.device)
                y_norm = (y - mean) / std
                loss = F.mse_loss(pred, y_norm)
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
            n += 1
        avg_loss = total_loss / max(n, 1)
        if scheduler == "step":
            sched.step()
        elif scheduler == "cosine":
            sched.step()
        else:
            sched.step(avg_loss)
        t1 = time.perf_counter()
        d = {"loss": avg_loss, "time_s": round(t1 - t0, 2)}
        if val_loader is not None and ckpt_path is not None and (epoch + 1) % val_interval == 0:
            mae, _ = evaluate(model, val_loader, device, noise_std=0.0, norm_params=norm_params)
            d["val_mae"] = mae
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), ckpt_path)
            (log_line or print)(
                f"  Epoch {epoch+1}/{epochs} loss={d['loss']:.6f} val_mae={mae:.4f} best={best_mae:.4f} lr={opt.param_groups[0]['lr']:.2e} time={t1-t0:.1f}s"
            )
        else:
            (log_line or print)(
                f"  Epoch {epoch+1}/{epochs} loss={d['loss']:.6f} lr={opt.param_groups[0]['lr']:.2e} time={t1-t0:.1f}s"
            )
        if log_cb:
            log_cb(epoch + 1, d)
    if ckpt_path is not None and best_mae == float("inf"):
        torch.save(model.state_dict(), ckpt_path)
    return model


def _cosine_pmh_weight(epoch, warmup_epochs, ramp_epochs, max_weight):
    """Cosine curriculum: w from 0 to max_weight over ramp (same as Task 02 graph classification)."""
    if epoch < warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    cosine_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return cosine_progress * max_weight


def edge_smoothness_loss(node_features, edge_index):
    """
    Penalize large feature differences between connected nodes (Task 01/02 style).
    Encourages smooth node embeddings so representations are stable under noise.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=node_features.device)
    row, col = edge_index[0], edge_index[1]
    edge_diff = (node_features[row] - node_features[col]).pow(2).sum(dim=1)
    return edge_diff.mean()


def train_e1(
    model,
    loader,
    device,
    epochs,
    norm_params=None,
    val_loader=None,
    ckpt_path=None,
    val_interval=5,
    warmup_epochs=20,
    pmh_max_weight=0.7,
    pmh_scale=1.0,
    pmh_cap_ratio=0.3,
    output_pmh_weight=0.3,
    target_norm_pmh=True,
    noise_std=MEASUREMENT_NOISE_STD,
    node_noise_std=NODE_NOISE_STD,
    pmh_ramp_epochs=60,
    lr=0.001,
    weight_decay=0.0,
    scheduler="step",
    step_size=50,
    step_gamma=0.5,
    plateau_patience=15,
    use_amp=True,
    log_cb=None,
    log_line=None,
):
    """
    E1: task loss on clean only; PMH encourages representation stability.
    PMH: 0.6*graph + 0.4*node L2-normalized distance, asymmetric (clean.detach vs noisy).
    Optional output_pmh_weight: MSE(pred_clean.detach(), pred_noisy) — aligns with VAT for regression.
    Cap on weighted term (like Tasks 04, 06, 07): pmh_term = min(w*scale*loss_pmh, cap_ratio*task).
    """
    norm_params = norm_params or {}
    warmup_eff = min(warmup_epochs, max(0, epochs // 4))
    ramp_eff = min(pmh_ramp_epochs, max(1, epochs - warmup_eff))
    if warmup_eff >= epochs and epochs > 0:
        warmup_eff = max(0, epochs - 1)
        ramp_eff = 1
    if log_line:
        log_line(f"PMH curriculum: warmup={warmup_eff} ep, ramp={ramp_eff} ep (w 0 -> {pmh_max_weight})")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=step_gamma)
    elif scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=plateau_patience, min_lr=1e-5
        )
    scaler = _make_scaler(use_amp)
    best_mae = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = total_task = total_pmh = total_pmh_eff = 0.0
        n = 0
        t0 = time.perf_counter()
        w = _cosine_pmh_weight(epoch, warmup_eff, ramp_eff, pmh_max_weight)
        for data in loader:
            data = data.to(device)
            data_noisy = add_measurement_noise(
                data, noise_std=noise_std, node_noise_std=node_noise_std, device=device
            )
            opt.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                pred_clean, node_clean, graph_clean = model(
                    data.x, data.pos, data.edge_index, data.batch, return_embeddings=True,
                )
                pred_noisy, node_noisy, graph_noisy = model(
                    data_noisy.x, data_noisy.pos, data_noisy.edge_index, data_noisy.batch,
                    return_embeddings=True,
                )
                y = data.y.view(data.num_graphs, -1).float()[:, : pred_clean.shape[1]]
                mean = norm_params["mean"].to(y.device)
                std = norm_params["std"].to(y.device)
                y_norm = (y - mean) / std
                loss_task = F.mse_loss(pred_clean, y_norm)
                # PMH: representation (graph + node) + optional output consistency (like VAT)
                g_c = F.normalize(graph_clean.detach().float(), p=2, dim=1)
                g_n = F.normalize(graph_noisy.float(), p=2, dim=1)
                n_c = F.normalize(node_clean.detach().float(), p=2, dim=1)
                n_n = F.normalize(node_noisy.float(), p=2, dim=1)
                l_mol = (g_c - g_n).pow(2).sum(dim=1).mean()
                l_node = (n_c - n_n).pow(2).sum(dim=1).mean()
                loss_pmh_repr = 0.6 * l_mol + 0.4 * l_node
                loss_pmh = loss_pmh_repr
                if output_pmh_weight > 0:
                    # Target-normalized output PMH:
                    # downweight inherently high-variance targets to avoid over-regularization.
                    if target_norm_pmh:
                        std_t = norm_params["std"].to(pred_clean.device, pred_clean.dtype)
                        std_t = std_t[: pred_clean.shape[1]]
                        inv = 1.0 / std_t.clamp_min(1e-6)
                        target_w = (inv / inv.mean()).detach()
                        diff2 = (pred_noisy - pred_clean.detach()).pow(2)
                        loss_pmh_out = (diff2 * target_w.view(1, -1)).mean()
                    else:
                        loss_pmh_out = F.mse_loss(pred_noisy, pred_clean.detach())
                    loss_pmh = (1.0 - output_pmh_weight) * loss_pmh_repr + output_pmh_weight * loss_pmh_out
                pmh_term = w * pmh_scale * loss_pmh
                if pmh_cap_ratio > 0:
                    pmh_term = torch.minimum(pmh_term, (pmh_cap_ratio * loss_task.detach()).to(pmh_term.dtype))
                loss = loss_task + pmh_term
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
        avg_loss = total_loss / max(n, 1)
        avg_task = total_task / max(n, 1)
        avg_pmh = total_pmh / max(n, 1)
        avg_pmh_eff = total_pmh_eff / max(n, 1)
        pmh_frac = avg_pmh_eff / (avg_task + avg_pmh_eff) if (avg_task + avg_pmh_eff) > 0 else 0.0
        if scheduler == "step":
            sched.step()
        elif scheduler == "cosine":
            sched.step()
        else:
            sched.step(avg_loss)
        t1 = time.perf_counter()
        d = {
            "loss": avg_loss,
            "task": avg_task,
            "pmh": avg_pmh,
            "pmh_eff": avg_pmh_eff,
            "pmh_frac": round(pmh_frac, 4),
            "w": round(w, 4),
            "time_s": round(t1 - t0, 2),
        }
        # PMH health check: pmh_frac target 0.05–0.35
        if epoch >= warmup_eff and (epoch + 1) % 20 == 0:
            if pmh_frac < 0.02:
                (log_line or print)(f"  [PMH check] pmh_frac={pmh_frac:.3f} — PMH may be too small (target 0.05–0.35)")
            elif pmh_frac > 0.5:
                (log_line or print)(f"  [PMH check] pmh_frac={pmh_frac:.3f} — PMH may dominate task")
        pmh_fmt = f"{avg_pmh:.6e}" if avg_pmh < 1e-4 else f"{avg_pmh:.6f}"
        if val_loader is not None and ckpt_path is not None and (epoch + 1) % val_interval == 0:
            mae, _ = evaluate(model, val_loader, device, noise_std=0.0, norm_params=norm_params)
            d["val_mae"] = mae
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), ckpt_path)
            (log_line or print)(
                f"  Epoch {epoch+1}/{epochs} loss={d['loss']:.6f} task={avg_task:.6f} pmh={pmh_fmt} pmh_eff={avg_pmh_eff:.6f} frac={pmh_frac:.2f} w={w:.3f} val_mae={mae:.4f} best={best_mae:.4f} lr={opt.param_groups[0]['lr']:.2e} time={t1-t0:.1f}s"
            )
        else:
            (log_line or print)(
                f"  Epoch {epoch+1}/{epochs} loss={d['loss']:.6f} task={avg_task:.6f} pmh={pmh_fmt} pmh_eff={avg_pmh_eff:.6f} frac={pmh_frac:.2f} w={w:.3f} lr={opt.param_groups[0]['lr']:.2e} time={t1-t0:.1f}s"
            )
        if log_cb:
            log_cb(epoch + 1, d)
    if ckpt_path is not None and best_mae == float("inf"):
        torch.save(model.state_dict(), ckpt_path)
    return model


def train_vat(
    model,
    loader,
    device,
    epochs,
    norm_params=None,
    val_loader=None,
    ckpt_path=None,
    val_interval=5,
    vat_eps=0.05,
    vat_xi=1e-6,
    vat_weight=1.0,
    lr=1e-4,
    weight_decay=0.0,
    scheduler="cosine",
    step_size=50,
    step_gamma=0.5,
    plateau_patience=15,
    use_amp=True,
    log_cb=None,
    log_line=None,
):
    """VAT: MSE(clean) + output consistency under virtual adversarial node-feature perturbation.
    Perturbation search runs in float32 to avoid AMP gradient underflow; main backward uses AMP.
    """
    norm_params = norm_params or {}
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=step_gamma)
    elif scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=plateau_patience, min_lr=1e-5)
    scaler = _make_scaler(use_amp)
    best_mae = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = total_task = total_vat = 0.0
        n = 0
        t0 = time.perf_counter()
        for data in loader:
            data = data.to(device)
            x_orig = data.x.float().clone()
            # VAT perturbation search in float32 to avoid gradient underflow under AMP
            with torch.no_grad():
                ref_out = model(x_orig, data.pos, data.edge_index, data.batch).float()
            d = torch.randn_like(x_orig)
            d = F.normalize(d, p=2, dim=1).detach().requires_grad_(True)
            adv_out = model(x_orig + vat_xi * d, data.pos, data.edge_index, data.batch).float()
            F.mse_loss(adv_out, ref_out.detach()).backward()
            d_grad = d.grad
            if d_grad is None or d_grad.abs().max() < 1e-12:
                r_adv = torch.zeros_like(x_orig)
            else:
                r_adv = (vat_eps * F.normalize(d_grad, p=2, dim=1)).detach()
            # Main backward with AMP
            opt.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                pred_clean = model(x_orig, data.pos, data.edge_index, data.batch)
                y = data.y.view(data.num_graphs, -1).float()[:, : pred_clean.shape[1]]
                mean = norm_params["mean"].to(y.device)
                std = norm_params["std"].to(y.device)
                y_norm = (y - mean) / std
                loss_task = F.mse_loss(pred_clean, y_norm)
                pred_adv = model(x_orig + r_adv.to(pred_clean.dtype), data.pos, data.edge_index, data.batch)
                loss_vat = F.mse_loss(pred_adv, pred_clean.detach())
                loss = loss_task + vat_weight * loss_vat
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item(); total_task += loss_task.item(); total_vat += loss_vat.item()
            n += 1
        if val_loader is not None and (epoch + 1) % val_interval == 0:
            mae, _ = evaluate(model, val_loader, device, noise_std=0.0, norm_params=norm_params)
            if mae < best_mae:
                best_mae = mae
                if ckpt_path:
                    torch.save(model.state_dict(), ckpt_path)
        if scheduler in ("step", "cosine"):
            sched.step()
        else:
            sched.step(total_loss / max(n, 1))
        t1 = time.perf_counter()
        div = max(n, 1)
        (log_line or print)(f"  Epoch {epoch+1}/{epochs} loss={total_loss/div:.6f} task={total_task/div:.6f} vat={total_vat/div:.6f} lr={opt.param_groups[0]['lr']:.2e} time={t1-t0:.1f}s")
        if log_cb:
            log_cb(epoch + 1, {"loss": total_loss/div, "task": total_task/div, "vat": total_vat/div})
    if ckpt_path is not None and best_mae == float("inf"):
        torch.save(model.state_dict(), ckpt_path)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run",
        type=str,
        default="E1",
        choices=["B0", "VAT", "E1"],
        help="B0 = MSE only; VAT = MSE + virtual adversarial training; E1 = MSE + PMH (measurement noise on pos)",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Batch size; 128 recommended for GPU (4× larger than old default of 32)")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0=safe on Windows; 2-4 speeds up I/O on Linux)")
    p.add_argument("--load_in_memory", action="store_true",
                   help="Preload all QM9 graphs into RAM (fits in 128GB; no disk I/O during training)")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable AMP (automatic mixed precision); use for debugging")
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use first N samples (e.g. 5000 for fast runs)",
    )
    p.add_argument(
        "--pmh_max_weight",
        type=float,
        default=0.7,
        help="E1: max curriculum weight on PMH (binding.py: 0.7)",
    )
    p.add_argument(
        "--pmh_scale",
        type=float,
        default=1.0,
        help="E1: scale on PMH term (loss = mean squared L2 per embedding). Default 1.0.",
    )
    p.add_argument("--pmh_cap_ratio", type=float, default=0.3,
                   help="Cap PMH term to pmh_cap_ratio * task_loss (cap on weighted term; 0 disables)")
    p.add_argument("--output_pmh_weight", type=float, default=0.3,
                   help="E1: weight of output consistency MSE(pred_clean, pred_noisy) in PMH (0=repr only). Aligns with VAT for regression.")
    p.add_argument("--target_norm_pmh", action="store_true",
                   help="E1: target-normalized output PMH (inverse-target-std weighting).")
    p.add_argument("--no_target_norm_pmh", action="store_true",
                   help="E1: disable target-normalized output PMH.")
    p.add_argument(
        "--node_noise",
        type=float,
        default=NODE_NOISE_STD,
        help="E1: node feature noise std (default 0.05 for regression; use 0.1 for Task-02-style). 0 = position-only",
    )
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=20,
        help="E1: epochs with w=0 (task only); same as Task 02",
    )
    p.add_argument(
        "--pmh_ramp_epochs",
        type=int,
        default=60,
        help="E1: epochs to ramp w from 0 to pmh_max_weight (cosine); same as Task 02",
    )
    p.add_argument(
        "--noise_std",
        type=float,
        default=MEASUREMENT_NOISE_STD,
        help="E1: position noise std (Å); keep small (e.g. 0.001)",
    )
    p.add_argument("--lr", type=float, default=1e-4, help="Default 1e-4 like binding.py")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="Default 1e-5 like binding.py")
    p.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["step", "plateau", "cosine"],
        help="cosine = CosineAnnealingLR (binding.py style)",
    )
    p.add_argument("--step_size", type=int, default=50)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--scheduler_patience", type=int, default=15)
    p.add_argument("--vat_eps", type=float, default=0.05, help="VAT: max norm of adversarial perturbation on node features")
    p.add_argument("--vat_weight", type=float, default=1.0, help="VAT: weight on consistency loss")
    p.add_argument("--hidden", type=int, default=128, help="Hidden dim (128 matches Chemistry.ipynb / binding; 64 was too small)")
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=None, help="GPU index (e.g. 0). When set, use cuda:N; required if CUDA_VISIBLE_DEVICES is set by runner.")
    p.add_argument("--seed", type=int, default=42, help="RNG: split, train shuffle, eval noise (match run_evals.py --split_seed / --eval_seed)")
    p.add_argument("--deterministic", action="store_true", help="Strict cuDNN determinism (slower)")
    args = p.parse_args()
    set_global_seed(args.seed, deterministic=args.deterministic)
    if args.no_target_norm_pmh:
        args.target_norm_pmh = False
    elif not args.target_norm_pmh:
        # Default ON for PMH-v2; keep explicit opt-out for ablations.
        args.target_norm_pmh = True

    run_dir = Path(args.out_dir) / "QM9" / args.run
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
    use_amp = not args.no_amp and torch.cuda.is_available()
    train_loader, val_loader, test_loader, info = get_loaders(
        root=args.data_dir,
        subset=args.subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        load_in_memory=args.load_in_memory,
    )
    num_node_features = info["num_node_features"]
    num_targets = info["num_targets"]

    model = get_model(
        num_node_features,
        num_targets=num_targets,
        hidden=args.hidden,
        num_layers=args.num_layers,
    ).to(device)
    print(
        f"Multi-task {num_targets} targets run={args.run} device={device} "
        f"batch={args.batch_size} amp={use_amp} seed={args.seed} deterministic={args.deterministic} run_dir={run_dir}"
    )

    training_log = []
    log_file = open(text_log_path, "w", encoding="utf-8")

    def log_line(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    def log_cb(epoch, metrics):
        training_log.append({"epoch": epoch, **metrics})

    norm_params = {"mean": info["target_mean"], "std": info["target_std"]}
    log_line(f"Per-target norm: 19 means/stds from train set")

    sched_kw = {
        "scheduler": args.scheduler,
        "step_size": args.step_size,
        "step_gamma": args.gamma,
        "plateau_patience": args.scheduler_patience,
    }
    val_interval = max(1, min(5, args.epochs // 4))
    t0_train = time.perf_counter()
    if args.run == "B0":
        train_b0(
            model,
            train_loader,
            device,
            args.epochs,
            norm_params=norm_params,
            val_loader=val_loader,
            ckpt_path=ckpt_path,
            val_interval=val_interval,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_amp=use_amp,
            log_cb=log_cb,
            log_line=log_line,
            **sched_kw,
        )
    elif args.run == "VAT":
        train_vat(model, train_loader, device, args.epochs, norm_params=norm_params,
                  val_loader=val_loader, ckpt_path=ckpt_path, val_interval=val_interval,
                  vat_eps=args.vat_eps, vat_weight=args.vat_weight, lr=args.lr,
                  weight_decay=args.weight_decay, use_amp=use_amp,
                  log_cb=log_cb, log_line=log_line, **sched_kw)
    else:
        train_e1(
            model,
            train_loader,
            device,
            args.epochs,
            norm_params=norm_params,
            val_loader=val_loader,
            ckpt_path=ckpt_path,
            val_interval=val_interval,
            warmup_epochs=args.warmup_epochs,
            pmh_max_weight=args.pmh_max_weight,
            pmh_scale=args.pmh_scale,
            pmh_cap_ratio=args.pmh_cap_ratio,
            output_pmh_weight=args.output_pmh_weight,
            target_norm_pmh=args.target_norm_pmh,
            noise_std=args.noise_std,
            node_noise_std=args.node_noise,
            use_amp=use_amp,
            pmh_ramp_epochs=args.pmh_ramp_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            log_cb=log_cb,
            log_line=log_line,
            **sched_kw,
        )
    train_time_s = time.perf_counter() - t0_train

    # Report test metrics for the best model (by val MAE), like binding.py
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        log_line(f"Loaded best checkpoint for test eval: {ckpt_path}")

    eval_seed = args.seed
    mae_clean, mse_clean = evaluate(
        model, test_loader, device, noise_std=0.0, node_noise_std=0.0, norm_params=norm_params
    )
    mae_noisy, mse_noisy = evaluate(
        model,
        test_loader,
        device,
        noise_std=MEASUREMENT_NOISE_STD,
        node_noise_std=0.0,
        norm_params=norm_params,
        seed=eval_seed,
    )
    log_line(
        f"Test -- Clean MAE={mae_clean:.4f} MSE={mse_clean:.6f}  Noisy(noise_std={MEASUREMENT_NOISE_STD}) MAE={mae_noisy:.4f} MSE={mse_noisy:.6f}  time={train_time_s:.1f}s"
    )

    results = {
        "num_targets": num_targets,
        "run": args.run,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "epochs": args.epochs,
        "target_mean": norm_params["mean"].cpu().tolist(),
        "target_std": norm_params["std"].cpu().tolist(),
        "clean_mae": mae_clean,
        "clean_mse": mse_clean,
        "noisy_mae": mae_noisy,
        "noisy_mse": mse_noisy,
        "num_train": info["num_train"],
        "num_test": info["num_test"],
        "train_time_s": round(train_time_s, 1),
        "checkpoint": "best.pt",
        "training_log": training_log,
        "test_indices": info.get("test_indices"),
        "dataset_size": info.get("dataset_size"),
    }
    if args.run == "E1":
        results["pmh_max_weight"] = args.pmh_max_weight
        results["warmup_epochs"] = args.warmup_epochs
        results["pmh_cap_ratio"] = args.pmh_cap_ratio
        results["output_pmh_weight"] = args.output_pmh_weight
        results["target_norm_pmh"] = bool(args.target_norm_pmh)
        results["noise_std"] = args.noise_std
        results["node_noise"] = args.node_noise

    torch.save(model.state_dict(), ckpt_path)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {"num_targets": num_targets, "run": args.run, "epochs": args.epochs, "epoch_log": training_log},
            f,
            indent=2,
        )
    log_line(f"Saved checkpoint: {ckpt_path}")
    log_line(f"Saved results: {results_path}")
    log_file.close()


if __name__ == "__main__":
    main()
