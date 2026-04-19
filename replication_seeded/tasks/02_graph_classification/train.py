"""
Train B0 (no aug), B1 (dropout on features), E1 (noise + PMH, task on clean), E2 (noise + PMH, task on clean+noisy).
Saves to runs/<dataset>/<run>/: best.pt, results.json, training_log.json, train.log.

Replication: use --seed (default 42) for split, shuffle, and final eval noise; --deterministic for strict cuDNN.
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

from data import get_loaders, DATASETS
from model import get_model
from eval import evaluate, perturb_node_features


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


def edge_smoothness_loss(node_features, edge_index):
    """Penalize large feature differences between connected nodes (smoothness prior)."""
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=node_features.device)
    row, col = edge_index
    edge_diff = (node_features[row] - node_features[col]).pow(2).sum(dim=1)
    return edge_diff.mean()


def train_b0_b1(model, loader, device, epochs, use_dropout=False, dropout_p=0.1, lr=0.01, weight_decay=0.0,
               scheduler="step", step_size=50, step_gamma=0.5, plateau_patience=15, log_cb=None, log_line=None):
    """B0/B1: CE loss, Adam. Default lr=0.01 + StepLR(50, 0.5) per gnn-comparison / TUDataset standard."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=step_gamma)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=plateau_patience, min_lr=1e-5)
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.perf_counter()
        for data in loader:
            data = data.to(device)
            if use_dropout and data.x.dtype in (torch.float, torch.float32, torch.float16):
                x = data.x * (torch.rand_like(data.x, device=device) > dropout_p).float() / max(1e-8, 1 - dropout_p)
            else:
                x = data.x
            opt.zero_grad(set_to_none=True)
            logits = model(x, data.edge_index, data.batch)
            y = data.y.squeeze(1) if data.y.dim() > 1 else data.y
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
            n += 1
        avg_loss = total_loss / max(n, 1)
        if scheduler == "step":
            sched.step()
        else:
            sched.step(avg_loss)
        t1 = time.perf_counter()
        d = {"loss": avg_loss, "time_s": round(t1 - t0, 2)}
        (log_line or print)(f"  Epoch {epoch+1}/{epochs} loss={d['loss']:.4f} lr={opt.param_groups[0]['lr']:.2e} time={t1-t0:.1f}s")
        if log_cb:
            log_cb(epoch + 1, d)
    return model


def _cosine_pmh_weight(epoch, warmup_epochs, ramp_epochs, max_weight):
    """Cosine curriculum: w from 0 to max_weight (e.g. 0.6) over ramp. Avoids collapse."""
    if epoch < warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    cosine_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return cosine_progress * max_weight


def _gnn_vat_perturbation(model, data, xi=1e-6, eps=0.1, n_iter=1):
    """Compute virtual adversarial perturbation on node features."""
    with torch.no_grad():
        ref_logits = model(data.x, data.edge_index, data.batch)
        p_ref = F.softmax(ref_logits, dim=1).detach()
    d = torch.randn_like(data.x)
    d = F.normalize(d, p=2, dim=1)
    for _ in range(n_iter):
        d = d.detach().requires_grad_(True)
        perturbed_x = data.x + xi * d
        adv_logits = model(perturbed_x, data.edge_index, data.batch)
        kl = F.kl_div(F.log_softmax(adv_logits, dim=1), p_ref, reduction='batchmean')
        kl.backward()
        d = F.normalize(d.grad, p=2, dim=1).detach()
    return (eps * d).detach()


def train_vat(model, loader, device, epochs, lr=0.01, vat_eps=0.1, vat_xi=1e-6,
              vat_weight=1.0, weight_decay=0.0, scheduler="step", step_size=50,
              step_gamma=0.5, log_cb=None, log_line=None):
    """VAT: CE on clean + KL consistency under virtual adversarial node-feature perturbation."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=step_gamma)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15, min_lr=1e-5)
    for epoch in range(epochs):
        model.train()
        total_loss = total_task = total_vat = 0.0
        n = 0
        t0 = time.perf_counter()
        for data in loader:
            data = data.to(device)
            y = data.y.squeeze(1) if data.y.dim() > 1 else data.y
            r_adv = _gnn_vat_perturbation(model, data, xi=vat_xi, eps=vat_eps)
            opt.zero_grad(set_to_none=True)
            logits_clean = model(data.x, data.edge_index, data.batch)
            loss_task = F.cross_entropy(logits_clean, y)
            logits_adv = model(data.x + r_adv, data.edge_index, data.batch)
            p_ref = F.softmax(logits_clean.detach(), dim=1)
            loss_vat = F.kl_div(F.log_softmax(logits_adv, dim=1), p_ref, reduction='batchmean')
            loss = loss_task + vat_weight * loss_vat
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item(); total_task += loss_task.item(); total_vat += loss_vat.item()
            n += 1
        if scheduler == "step":
            sched.step()
        else:
            sched.step(total_loss / max(n, 1))
        t1 = time.perf_counter()
        div = max(n, 1)
        (log_line or print)(f"  Epoch {epoch+1}/{epochs} loss={total_loss/div:.4f} task={total_task/div:.4f} vat={total_vat/div:.4f} lr={opt.param_groups[0]['lr']:.2e} time={t1-t0:.1f}s")
        if log_cb: log_cb(epoch + 1, {"loss": total_loss/div, "task": total_task/div, "vat": total_vat/div})
    return model


def train_e1(model, loader, device, epochs, warmup_epochs=20, pmh_weight=0.6, pmh_cap_ratio=0.3,
             noise_sigma=0.1, task_mix=0.2, pmh_ramp_epochs=60, task_clean_only=False,
             lr=0.01, weight_decay=0.0, scheduler="step", step_size=50, step_gamma=0.5,
             plateau_patience=15, log_cb=None, log_line=None):
    """E1/E2: task + PMH. task_clean_only=True => E1 (task on clean only); False => E2 (mixed task)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=step_gamma)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=plateau_patience, min_lr=1e-5)
    for epoch in range(epochs):
        model.train()
        total_loss = total_task = total_pmh = 0.0
        n = 0
        t0 = time.perf_counter()
        w = _cosine_pmh_weight(epoch, warmup_epochs, pmh_ramp_epochs, pmh_weight)
        for data in loader:
            data = data.to(device)
            data_noisy = perturb_node_features(data, noise_std=noise_sigma, device=device)

            opt.zero_grad(set_to_none=True)
            logits_clean, node_clean, graph_clean, layers_clean = model(
                data.x, data.edge_index, data.batch, return_all_layers=True)
            logits_noisy, node_noisy, graph_noisy, layers_noisy = model(
                data_noisy.x, data_noisy.edge_index, data_noisy.batch, return_all_layers=True)
            y = data.y.squeeze(1) if data.y.dim() > 1 else data.y

            loss_clean = F.cross_entropy(logits_clean, y)
            if task_clean_only:
                loss_task = loss_clean
            else:
                loss_noisy = F.cross_entropy(logits_noisy, y)
                loss_task = (1.0 - task_mix) * loss_clean + task_mix * loss_noisy

            # Multi-scale PMH: 0.5×graph + 0.3×node + 0.2×edge (per README)
            loss_graph = 0.0
            for gc, gn in zip(layers_clean, layers_noisy):
                gc_n = F.normalize(gc.detach(), p=2, dim=1)
                gn_n = F.normalize(gn, p=2, dim=1)
                loss_graph = loss_graph + (gc_n - gn_n).pow(2).sum(dim=1).mean()
            loss_graph = loss_graph / len(layers_clean)
            n_c = F.normalize(node_clean.detach(), p=2, dim=1)
            n_n = F.normalize(node_noisy, p=2, dim=1)
            loss_node = (n_c - n_n).pow(2).sum(dim=1).mean()
            loss_edge = edge_smoothness_loss(node_clean, data.edge_index)
            loss_pmh = 0.5 * loss_graph + 0.3 * loss_node + 0.2 * loss_edge
            pmh_term = w * pmh_weight * loss_pmh
            if pmh_cap_ratio > 0:
                pmh_term = torch.minimum(pmh_term, pmh_cap_ratio * loss_task.detach())
            loss = loss_task + pmh_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
            total_task += loss_task.item()
            total_pmh += loss_pmh.item()
            n += 1
        avg_loss = total_loss / max(n, 1)
        if scheduler == "step":
            sched.step()
        else:
            sched.step(avg_loss)
        t1 = time.perf_counter()
        d = {"loss": avg_loss, "task": total_task / max(n, 1), "pmh": total_pmh / max(n, 1),
             "w": round(w, 4), "time_s": round(t1 - t0, 2)}
        (log_line or print)(f"  Epoch {epoch+1}/{epochs} loss={d['loss']:.4f} task={d['task']:.4f} pmh={d['pmh']:.4f} w={w:.2f} lr={opt.param_groups[0]['lr']:.2e} time={t1-t0:.1f}s")
        if log_cb:
            log_cb(epoch + 1, d)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="PROTEINS", choices=list(DATASETS), help="PROTEINS or ENZYMES recommended; MUTAG is small")
    p.add_argument("--run", type=str, default="E1", choices=["B0", "B1", "VAT", "E1", "E2"], help="E2 = E1 with mixed task (augmentation seen at task + PMH)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--pmh_max_weight", type=float, default=0.6, help="E1/E2: max curriculum weight on PMH")
    p.add_argument("--pmh_cap_ratio", type=float, default=0.3, help="Cap PMH term to pmh_cap_ratio * task_loss (0 disables)")
    p.add_argument("--pmh_ramp_epochs", type=int, default=60, help="E1/E2: epochs to ramp w from 0 to pmh_max_weight (cosine)")
    p.add_argument("--resume", type=str, default="", help="E1/E2: path to checkpoint to continue training (e.g. different pmh_max_weight)")
    p.add_argument("--warmup_epochs", type=int, default=20, help="E1/E2: epochs with w=0 (task only)")
    p.add_argument("--mixed_task", action="store_true", help="E1: use 0.8*CE(clean)+0.2*CE(noisy); default for E1 is CE(clean) only")
    p.add_argument("--noise_sigma", type=float, default=0.1, help="E1/E2: node feature noise std")
    p.add_argument("--task_mix", type=float, default=0.2, help="E1/E2: fraction of task loss on noisy (when not task_clean_only)")
    p.add_argument("--dropout_p", type=float, default=0.1, help="B1: dropout on node features")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate (0.01 = gnn-comparison standard)")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay")
    p.add_argument("--scheduler", type=str, default="step", choices=["step", "plateau"], help="step = StepLR; plateau = ReduceLROnPlateau")
    p.add_argument("--step_size", type=int, default=50, help="StepLR: decay every N epochs")
    p.add_argument("--gamma", type=float, default=0.5, help="StepLR: lr *= gamma each step")
    p.add_argument("--scheduler_patience", type=int, default=15, help="ReduceLROnPlateau patience (only if --scheduler plateau)")
    p.add_argument("--vat_eps", type=float, default=0.1)
    p.add_argument("--vat_weight", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=128, help="Hidden dim (match eval.py --hidden)")
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42, help="RNG seed: split, train shuffle, final eval noise")
    p.add_argument("--deterministic", action="store_true", help="Strict cuDNN determinism (slower)")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 recommended on Windows)")
    args = p.parse_args()

    set_global_seed(args.seed, deterministic=args.deterministic)

    run_dir = Path(args.out_dir) / args.dataset / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    results_path = run_dir / "results.json"
    log_path = run_dir / "training_log.json"
    text_log_path = run_dir / "train.log"

    epochs = args.epochs
    if args.run in ("E1", "E2"):
        epochs = int(round(args.epochs * 1.2))
        print(f"{args.run}: using 20% more epochs: {args.epochs} -> {epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, info = get_loaders(
        args.dataset,
        root=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    num_node_features = info["num_node_features"]
    num_classes = info["num_classes"]

    model = get_model(num_node_features, num_classes, hidden=args.hidden, num_layers=args.num_layers).to(device)
    if args.resume and args.run in ("E1", "E2"):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt, strict=True)
        print(f"Resumed {args.run} from {args.resume}; continuing with pmh_max_weight={args.pmh_max_weight}")
    print(
        f"Dataset={args.dataset} run={args.run} device={device} batch={args.batch_size} "
        f"seed={args.seed} deterministic={args.deterministic} num_workers={args.num_workers} run_dir={run_dir}"
    )

    training_log = []
    log_file = open(text_log_path, "w", encoding="utf-8")
    def log_line(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
    def log_cb(epoch, metrics):
        training_log.append({"epoch": epoch, **metrics})

    t0_train = time.perf_counter()
    sched_kw = {"scheduler": args.scheduler, "step_size": args.step_size, "step_gamma": args.gamma, "plateau_patience": args.scheduler_patience}
    if args.run == "B0":
        train_b0_b1(model, train_loader, device, epochs, use_dropout=False, lr=args.lr, weight_decay=args.weight_decay, log_cb=log_cb, log_line=log_line, **sched_kw)
    elif args.run == "B1":
        train_b0_b1(model, train_loader, device, epochs, use_dropout=True, dropout_p=args.dropout_p, lr=args.lr, weight_decay=args.weight_decay, log_cb=log_cb, log_line=log_line, **sched_kw)
    elif args.run == "VAT":
        epochs = int(round(args.epochs * 1.2))
        train_vat(model, train_loader, device, epochs, lr=args.lr, vat_eps=args.vat_eps,
                  vat_weight=args.vat_weight, weight_decay=args.weight_decay,
                  scheduler=args.scheduler, step_size=args.step_size, step_gamma=args.gamma,
                  log_cb=log_cb, log_line=log_line)
    else:
        task_clean_only = (args.run == "E1") and not args.mixed_task
        train_e1(model, train_loader, device, epochs,
                 warmup_epochs=args.warmup_epochs, pmh_weight=args.pmh_max_weight,
                 pmh_cap_ratio=args.pmh_cap_ratio,
                 pmh_ramp_epochs=args.pmh_ramp_epochs, noise_sigma=args.noise_sigma, task_mix=args.task_mix, task_clean_only=task_clean_only,
                 lr=args.lr, weight_decay=args.weight_decay, log_cb=log_cb, log_line=log_line, **sched_kw)
    train_time_s = time.perf_counter() - t0_train

    # Final metrics on test set (labeled "Val" in logs; val_loader exists but is unused for selection)
    eval_seed = args.seed  # reproducible noise; use same --split_seed in eval.py
    acc_clean = evaluate(model, test_loader, device, 0.0)
    acc_05 = evaluate(model, test_loader, device, 0.05, seed=eval_seed)
    acc_10 = evaluate(model, test_loader, device, 0.1, seed=eval_seed)
    acc_15 = evaluate(model, test_loader, device, 0.15, seed=eval_seed)
    acc_20 = evaluate(model, test_loader, device, 0.2, seed=eval_seed)
    log_line(
        f"Val - Clean: {acc_clean:.2f}%  sigma=0.05: {acc_05:.2f}%  sigma=0.1: {acc_10:.2f}%  "
        f"sigma=0.15: {acc_15:.2f}%  sigma=0.2: {acc_20:.2f}%  time: {train_time_s:.1f}s"
    )

    results = {
        "dataset": args.dataset,
        "run": args.run,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "num_workers": args.num_workers,
        "epochs_requested": args.epochs,
        "epochs_actual": epochs,
        "clean_acc": acc_clean,
        "acc_sigma_0.05": acc_05,
        "acc_sigma_0.1": acc_10,
        "acc_sigma_0.15": acc_15,
        "acc_sigma_0.2": acc_20,
        "num_train": info["num_train"],
        "num_test": info["num_test"],
        "train_time_s": round(train_time_s, 1),
        "checkpoint": "best.pt",
        "training_log": training_log,
    }
    if args.run in ("E1", "E2"):
        results["pmh_max_weight"] = args.pmh_max_weight
        results["warmup_epochs"] = args.warmup_epochs
        results["noise_sigma"] = args.noise_sigma
        results["task_mix"] = args.task_mix
        results["task_clean_only"] = args.run == "E1" and not args.mixed_task

    torch.save(model.state_dict(), ckpt_path)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "run": args.run, "epochs": epochs, "epoch_log": training_log}, f, indent=2)
    log_line(f"Saved checkpoint: {ckpt_path}")
    log_line(f"Saved results: {results_path}")
    log_file.close()


if __name__ == "__main__":
    main()
