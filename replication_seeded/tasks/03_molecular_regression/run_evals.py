"""
Compare B0 vs E1: evaluate at multiple noise levels, print table, save plots and JSON.
Aligned with GNN/graph_reg_class/binding.py (MAE, RMSE, embedding drift, comparison plots).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from data import get_loaders, get_dataset, NUM_TARGETS
from model import get_model
from eval import evaluate_with_drift, DEFAULT_NOISE_LEVELS

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_run_results(run_dir):
    """
    Load results.json from a run directory.
    Returns dict with: norm_params (or None), test_indices, dataset_size, clean_mae (or None).
    """
    run_dir = Path(run_dir)
    results_path = run_dir / "results.json"
    out = {"norm_params": None, "test_indices": None, "dataset_size": None, "clean_mae": None}
    if not results_path.exists():
        return out
    try:
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)
        mean = data.get("target_mean")
        std = data.get("target_std")
        if mean is not None and std is not None:
            out["norm_params"] = {
                "mean": torch.tensor(mean, dtype=torch.float32),
                "std": torch.tensor(std, dtype=torch.float32),
            }
        out["test_indices"] = data.get("test_indices")
        out["dataset_size"] = data.get("dataset_size")
        out["clean_mae"] = data.get("clean_mae")
    except (json.JSONDecodeError, TypeError):
        pass
    return out


def load_model_only(checkpoint_path, info, device, hidden=128, num_layers=4):
    """Load model from checkpoint; no data loaders. Same test set for B0 and E1 (binding-style)."""
    model = get_model(
        info["num_node_features"],
        num_targets=info["num_targets"],
        hidden=hidden,
        num_layers=num_layers,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def run_evals_all_noise(model, loader, device, noise_levels, node_noise_std, norm_params, seed=42, n_replicates=1):
    """Evaluate at each noise level; return dict noise_level -> {mae, rmse, emb_drift}.
    One prediction per sample per σ (n_replicates=1) so MAE(σ) is comparable and increases with σ.
    Fixed seed so B0 and E1 see the same noisy inputs.
    """
    results = {}
    for eps in noise_levels:
        results[eps] = evaluate_with_drift(
            model, loader, device,
            noise_std=eps,
            node_noise_std=node_noise_std,
            norm_params=norm_params,
            seed=seed,
            n_replicates=n_replicates,
        )
    return results


def print_comparison_table(b0_results, e1_results, noise_levels, node_noise=0.0, vat_results=None):
    """Print table: Noise | B0 MAE | [VAT MAE] | E1 MAE | Δ% (E1−B0) | Drifts.
    MAE = mean over all (sample, target) elements (denorm). Δ% = (E1−B0)/B0×100 (negative = E1 better).
    """
    def _mae(r):
        return r.get("mae_avg", r["mae"])

    has_vat = vat_results is not None
    print("\n" + "=" * (95 if has_vat else 78))
    print("B0 vs E1" + (" vs VAT" if has_vat else "") + " -- Test set (MAE = mean over 19 targets; 1 pred/sample per sigma; fixed seed)")
    print("=" * (95 if has_vat else 78))
    if has_vat:
        print(f"{'Noise(A)':<10} | {'B0 MAE':<10} | {'VAT MAE':<10} | {'E1 MAE':<10} | {'D% (E1-B0)':<12} | {'B0 Drift':<10} | {'VAT Drift':<10} | {'E1 Drift':<10}")
        print("-" * 95)
    else:
        print(f"{'Noise(A)':<10} | {'B0 MAE':<10} | {'E1 MAE':<10} | {'D% (E1-B0)':<12} | {'B0 Drift':<10} | {'E1 Drift':<10}")
        print("-" * 78)
    for eps in noise_levels:
        b0 = b0_results[eps]
        e1 = e1_results[eps]
        b0_m, e1_m = _mae(b0), _mae(e1)
        delta_pct = (e1_m - b0_m) / b0_m * 100 if b0_m > 0 else 0.0
        if has_vat:
            vat = vat_results[eps]
            vat_m = _mae(vat)
            print(
                f"{eps:<10.4f} | {b0_m:<10.4f} | {vat_m:<10.4f} | {e1_m:<10.4f} | {delta_pct:+11.1f}% | "
                f"{b0['emb_drift']:<10.4f} | {vat['emb_drift']:<10.4f} | {e1['emb_drift']:<10.4f}"
            )
        else:
            print(
                f"{eps:<10.4f} | {b0_m:<10.4f} | {e1_m:<10.4f} | {delta_pct:+11.1f}% | "
                f"{b0['emb_drift']:<10.4f} | {e1['emb_drift']:<10.4f}"
            )
    print("=" * (95 if has_vat else 78) + "\n")


def plot_comparison(b0_results, e1_results, noise_levels, save_dir, vat_results=None):
    """MAE vs noise, Embedding drift vs noise, PMH improvement % bar chart (like binding.py)."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available; skipping plots.")
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    colors = {"B0": "#e74c3c", "VAT": "#3498db", "E1": "#2ecc71"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _mae = lambda r: r.get("mae_avg", r["mae"])

    # 1) MAE vs noise (mae_avg = avg over targets, like binding)
    ax = axes[0]
    b0_mae = [_mae(b0_results[e]) for e in noise_levels]
    e1_mae = [_mae(e1_results[e]) for e in noise_levels]
    ax.plot(noise_levels, b0_mae, "o-", color=colors["B0"], lw=2, ms=8, label="B0")
    if vat_results is not None:
        vat_mae = [_mae(vat_results[e]) for e in noise_levels]
        ax.plot(noise_levels, vat_mae, "s-", color=colors["VAT"], lw=2, ms=8, label="VAT")
    ax.plot(noise_levels, e1_mae, "o-", color=colors["E1"], lw=2, ms=8, label="E1 (PMH)")
    ax.set_xlabel("Position noise σ (Å)")
    ax.set_ylabel("MAE (avg over targets)")
    ax.set_title("Prediction error vs noise")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2) Embedding drift vs noise
    ax = axes[1]
    b0_drift = [b0_results[e]["emb_drift"] for e in noise_levels]
    e1_drift = [e1_results[e]["emb_drift"] for e in noise_levels]
    ax.plot(noise_levels, b0_drift, "o-", color=colors["B0"], lw=2, ms=8, label="B0")
    if vat_results is not None:
        vat_drift = [vat_results[e]["emb_drift"] for e in noise_levels]
        ax.plot(noise_levels, vat_drift, "s-", color=colors["VAT"], lw=2, ms=8, label="VAT")
    ax.plot(noise_levels, e1_drift, "o-", color=colors["E1"], lw=2, ms=8, label="E1 (PMH)")
    ax.set_xlabel("Position noise σ (Å)")
    ax.set_ylabel("Embedding drift")
    ax.set_title("Manifold stability")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3) PMH improvement % (bar) — use mae_avg like binding
    ax = axes[2]
    _mae = lambda r: r.get("mae_avg", r["mae"])
    improvements = []
    for e in noise_levels:
        b0_m = _mae(b0_results[e])
        e1_m = _mae(e1_results[e])
        imp = (b0_m - e1_m) / b0_m * 100 if b0_m > 0 else 0
        improvements.append(imp)
    bars = ax.bar(range(len(noise_levels)), improvements, color="#9b59b6", alpha=0.8)
    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([f"{e:.3f}" for e in noise_levels])
    ax.set_xlabel("Noise σ (Å)")
    ax.set_ylabel("PMH MAE improvement (%)")
    ax.set_title("E1 vs B0: MAE improvement")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.grid(alpha=0.3, axis="y")
    for bar, imp in zip(bars, improvements):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h,
            f"{imp:+.1f}%",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=9,
        )

    plt.suptitle("QM9 multi-task (19 targets): B0 vs E1 (PMH)" + (" vs VAT" if vat_results else ""), fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = save_dir / "comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out}")

    # Per-property: MAE vs noise for each target (grid of 19)
    _plot_per_property(b0_results, e1_results, noise_levels, save_dir)
    # Heatmaps: target x noise -> MAE (B0, E1, improvement %)
    _plot_heatmaps(b0_results, e1_results, noise_levels, save_dir)


# QM9 target short names (index -> label)
QM9_TARGET_NAMES = [
    "μ", "α", "ε_HOMO", "ε_LUMO", "Δε", "<R²>", "ZPVE", "U0", "U", "H", "G", "cv",
    "U0_atom", "U_atom", "H_atom", "G_atom", "A", "B", "C",
]


def _plot_per_property(b0_results, e1_results, noise_levels, save_dir):
    """One subplot per target: MAE vs noise (B0 vs E1)."""
    if not HAS_MATPLOTLIB or "mae_per_target" not in (b0_results.get(noise_levels[0]) or {}):
        return
    save_dir = Path(save_dir)
    num_targets = len(b0_results[noise_levels[0]]["mae_per_target"])
    names = QM9_TARGET_NAMES[:num_targets] if len(QM9_TARGET_NAMES) >= num_targets else [str(t) for t in range(num_targets)]
    ncols = 5
    nrows = (num_targets + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.2 * nrows))
    axes = axes.flatten()
    colors = {"B0": "#e74c3c", "E1": "#2ecc71"}
    for t in range(num_targets):
        ax = axes[t]
        b0_mae = [b0_results[e]["mae_per_target"][t] for e in noise_levels]
        e1_mae = [e1_results[e]["mae_per_target"][t] for e in noise_levels]
        ax.plot(noise_levels, b0_mae, "o-", color=colors["B0"], lw=1.5, ms=4, label="B0")
        ax.plot(noise_levels, e1_mae, "s-", color=colors["E1"], lw=1.5, ms=4, label="E1")
        ax.set_title(names[t] if t < len(names) else f"T{t}")
        ax.set_xlabel("σ (Å)")
        ax.set_ylabel("MAE")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(num_targets, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("QM9: MAE vs noise per property (B0 vs E1)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = save_dir / "per_property_mae.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out}")


def _plot_heatmaps(b0_results, e1_results, noise_levels, save_dir):
    """Heatmaps: rows=target, cols=noise; color=MAE (B0, E1) and improvement %."""
    if not HAS_MATPLOTLIB or "mae_per_target" not in (b0_results.get(noise_levels[0]) or {}):
        return
    save_dir = Path(save_dir)
    num_targets = len(b0_results[noise_levels[0]]["mae_per_target"])
    names = QM9_TARGET_NAMES[:num_targets] if len(QM9_TARGET_NAMES) >= num_targets else [str(t) for t in range(num_targets)]
    # Build matrices [num_targets, num_noise]
    b0_mat = [[b0_results[e]["mae_per_target"][t] for e in noise_levels] for t in range(num_targets)]
    e1_mat = [[e1_results[e]["mae_per_target"][t] for e in noise_levels] for t in range(num_targets)]
    imp_mat = []
    for t in range(num_targets):
        row = []
        for i, e in enumerate(noise_levels):
            b0_v = b0_mat[t][i]
            e1_v = e1_mat[t][i]
            imp = (b0_v - e1_v) / b0_v * 100 if b0_v > 0 else 0
            row.append(imp)
        imp_mat.append(row)
    b0_arr = np.array(b0_mat)
    e1_arr = np.array(e1_mat)
    imp_arr = np.array(imp_mat)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    # B0 MAE heatmap
    im0 = axes[0].imshow(b0_arr, aspect="auto", cmap="Reds")
    axes[0].set_yticks(range(num_targets))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xticks(range(len(noise_levels)))
    axes[0].set_xticklabels([f"{e:.3f}" for e in noise_levels])
    axes[0].set_xlabel("Noise σ (Å)")
    axes[0].set_ylabel("Target")
    axes[0].set_title("B0 MAE")
    plt.colorbar(im0, ax=axes[0])
    # E1 MAE heatmap
    im1 = axes[1].imshow(e1_arr, aspect="auto", cmap="Greens")
    axes[1].set_yticks(range(num_targets))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xticks(range(len(noise_levels)))
    axes[1].set_xticklabels([f"{e:.3f}" for e in noise_levels])
    axes[1].set_xlabel("Noise σ (Å)")
    axes[1].set_title("E1 MAE")
    plt.colorbar(im1, ax=axes[1])
    # Improvement % heatmap (diverging: red = worse, green = better)
    im2 = axes[2].imshow(imp_arr, aspect="auto", cmap="RdYlGn", vmin=-20, vmax=20)
    axes[2].set_yticks(range(num_targets))
    axes[2].set_yticklabels(names, fontsize=8)
    axes[2].set_xticks(range(len(noise_levels)))
    axes[2].set_xticklabels([f"{e:.3f}" for e in noise_levels])
    axes[2].set_xlabel("Noise σ (Å)")
    axes[2].set_title("PMH improvement %")
    plt.colorbar(im2, ax=axes[2], label="%")
    plt.suptitle("QM9: MAE per property × noise level", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = save_dir / "heatmap_per_property.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out}")


def main():
    p = argparse.ArgumentParser(description="Compare B0 and E1 (and optionally VAT) at multiple noise levels; table + plots.")
    p.add_argument("--b0", type=str, default="runs/QM9/B0/best.pt", help="B0 checkpoint")
    p.add_argument("--e1", type=str, default="runs/QM9/E1/best.pt", help="E1 checkpoint")
    p.add_argument("--vat", type=str, default=None, help="Optional: VAT checkpoint (e.g. runs/QM9/VAT/best.pt); skipped if missing")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128, help="Must match training (128 = Chemistry/binding)")
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument(
        "--noise_levels",
        type=float,
        nargs="+",
        default=DEFAULT_NOISE_LEVELS,
        help="Position noise levels (Å)",
    )
    p.add_argument("--node_noise", type=float, default=0.0, help="Node noise at eval (0 = pos only)")
    p.add_argument("--n_replicates", type=int, default=1, help="Noise replicates per σ (default 1 = one pred/sample so MAE↑ with σ; B0/E1 same noise via seed)")
    p.add_argument("--out_dir", type=str, default=None, help="Defaults to runs/target_<id>/evals")
    p.add_argument("--no_plot", action="store_true", help="Skip saving comparison plot")
    p.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Fallback get_loaders only: must match train.py --seed used for B0/E1",
    )
    p.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help="Deterministic noise for MAE vs σ (same for B0/VAT/E1)",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or "runs/QM9/evals"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    noise_levels = args.noise_levels

    b0_path = Path(args.b0)
    e1_path = Path(args.e1)
    vat_path = Path(args.vat) if args.vat else None
    if not b0_path.exists():
        raise FileNotFoundError(f"B0 checkpoint not found: {b0_path}")
    if not e1_path.exists():
        raise FileNotFoundError(f"E1 checkpoint not found: {e1_path}")
    if vat_path is not None and not vat_path.exists():
        print(f"VAT checkpoint not found: {vat_path}; skipping VAT in eval.")
        vat_path = None

    # Load B0 run results so we use the exact same test set and norm as training
    b0_run = load_run_results(b0_path.parent)
    norm_params = b0_run["norm_params"]
    test_indices = b0_run["test_indices"]
    dataset_size = b0_run["dataset_size"]
    clean_mae_ref = b0_run["clean_mae"]

    # Use B0's dataset_size when available so we get the same dataset and can use test_indices (same test set as training). Otherwise args.subset can give a different size (e.g. None = full QM9 → wrong test size and MAE).
    subset_for_dataset = (dataset_size if (dataset_size is not None and test_indices is not None) else args.subset)
    dataset = get_dataset(root=args.data_dir, subset=subset_for_dataset)
    n_ds = len(dataset)
    if (
        test_indices is not None
        and dataset_size is not None
        and norm_params is not None
        and n_ds == dataset_size
    ):
        test_ds = Subset(dataset, test_indices)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        sample = dataset[0] if hasattr(dataset, "__getitem__") else dataset.dataset[0]
        num_node_features = sample.x.shape[1] if sample.x.dim() > 1 else sample.x.numel()
        info = {
            "num_node_features": num_node_features,
            "num_targets": NUM_TARGETS,
            "num_test": len(test_indices),
        }
        print(f"Using test set from B0 run ({len(test_indices)} samples, same indices).")
    else:
        _, _, test_loader, info = get_loaders(
            root=args.data_dir,
            subset=args.subset,
            batch_size=args.batch_size,
            seed=args.split_seed,
        )
        if test_indices is None or dataset_size is None:
            print("B0 results.json has no test_indices/dataset_size; using current split (may differ from training).")
        else:
            print(f"Dataset size {n_ds} != B0 dataset_size {dataset_size}; using current split (may differ).")

    if norm_params is None:
        norm_params = {"mean": info["target_mean"], "std": info["target_std"]}
        print("Using norm_params from current data split (no B0 results.json or invalid).")
    else:
        print(f"Using norm_params from B0 run: {b0_path.parent / 'results.json'}")

    model_b0 = load_model_only(b0_path, info, device, args.hidden, args.num_layers)
    print(f"Loaded B0: {b0_path}")
    b0_results = run_evals_all_noise(
        model_b0, test_loader, device, noise_levels, args.node_noise, norm_params,
        seed=args.eval_seed, n_replicates=args.n_replicates,
    )
    # Sanity check: B0 MAE at σ=0 should match clean_mae from training
    b0_mae_0 = b0_results[noise_levels[0]].get("mae_avg", b0_results[noise_levels[0]]["mae"])
    if clean_mae_ref is not None and noise_levels[0] == 0.0:
        rel = abs(b0_mae_0 - clean_mae_ref) / max(clean_mae_ref, 1e-8)
        if rel > 0.05:
            print(f"Warning: B0 MAE at σ=0 ({b0_mae_0:.4f}) differs from training clean_mae ({clean_mae_ref:.4f}); check data_dir/subset match.")

    model_e1 = load_model_only(e1_path, info, device, args.hidden, args.num_layers)
    print(f"Loaded E1: {e1_path}")
    e1_results = run_evals_all_noise(
        model_e1, test_loader, device, noise_levels, args.node_noise, norm_params,
        seed=args.eval_seed, n_replicates=args.n_replicates,
    )

    vat_results = None
    if vat_path is not None:
        model_vat = load_model_only(vat_path, info, device, args.hidden, args.num_layers)
        print(f"Loaded VAT: {vat_path}")
        vat_results = run_evals_all_noise(
            model_vat, test_loader, device, noise_levels, args.node_noise, norm_params,
            seed=args.eval_seed, n_replicates=args.n_replicates,
        )

    print_comparison_table(b0_results, e1_results, noise_levels, args.node_noise, vat_results=vat_results)

    if not args.no_plot:
        plot_comparison(b0_results, e1_results, noise_levels, out_dir, vat_results=vat_results)

    # Save JSON
    summary = {
        "num_targets": info["num_targets"],
        "noise_levels": noise_levels,
        "node_noise_eval": args.node_noise,
        "B0": {str(k): v for k, v in b0_results.items()},
        "E1": {str(k): v for k, v in e1_results.items()},
        "num_test": info["num_test"],
        "split_seed": args.split_seed,
        "eval_seed": args.eval_seed,
        "n_replicates": args.n_replicates,
    }
    if vat_results is not None:
        summary["VAT"] = {str(k): v for k, v in vat_results.items()}
    json_path = Path(out_dir) / "eval_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
