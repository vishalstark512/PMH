"""
Embedding and prediction analysis for QM9 regression: MAE doesn't tell the full story.
- Embedding stability: mean ||emb_noisy - emb_clean|| (B0 vs E1; PMH should be lower).
- t-SNE of graph embeddings colored by a target (e.g. U0) to see structure.
- Prediction stability: per-molecule std of prediction under noise (B0 vs E1).
- Pred vs true scatter for key properties.

Run: python embedding_analysis.py --b0 runs/QM9/B0/best.pt --e1 runs/QM9/E1/best.pt --out runs/QM9/embedding_analysis
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
from perturb import add_measurement_noise

SEED = 42
NOISE_POS = 0.01
NOISE_NODE = 0.05
N_REPLICATES = 5  # for prediction stability

QM9_TARGET_NAMES = [
    "μ", "α", "ε_HOMO", "ε_LUMO", "Δε", "<R²>", "ZPVE", "U0", "U", "H", "G", "cv",
    "U0_atom", "U_atom", "H_atom", "G_atom", "A", "B", "C",
]


def _ensure_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _norm_tensors(norm_params, device):
    mean = norm_params.get("mean", 0.0)
    std = norm_params.get("std", 1.0)
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=torch.float32, device=device)
    else:
        mean = mean.to(device).float()
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=torch.float32, device=device)
    else:
        std = std.to(device).float()
    return mean, std


@torch.no_grad()
def collect_embeddings_and_predictions(model, loader, device, norm_params, noise_std=0.0, node_noise_std=0.0, seed=None):
    """Collect graph embeddings and denormalized predictions (and targets) for test set."""
    model.eval()
    embs_list, preds_list, targets_list = [], [], []
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    for data in loader:
        data = data.to(device)
        if noise_std > 0 or node_noise_std > 0:
            data = add_measurement_noise(data, noise_std=noise_std, node_noise_std=node_noise_std, device=device)
        mean, std = _norm_tensors(norm_params, data.x.device)
        pred_norm, _, graph_emb = model(
            data.x, data.pos, data.edge_index, data.batch, return_embeddings=True
        )
        pred_denorm = pred_norm * std + mean
        y = data.y.view(pred_denorm.shape[0], -1).float()[:, : pred_denorm.shape[1]]
        embs_list.append(graph_emb.cpu().numpy())
        preds_list.append(pred_denorm.cpu().numpy())
        targets_list.append(y.cpu().numpy())
    return (
        np.concatenate(embs_list, axis=0),
        np.concatenate(preds_list, axis=0),
        np.concatenate(targets_list, axis=0),
    )


def embedding_stability(embs_clean, embs_noisy):
    """Mean L2 distance between clean and noisy embeddings (per graph). Lower = more stable."""
    diff = embs_noisy - embs_clean
    return float(np.sqrt((diff ** 2).sum(axis=1)).mean())


@torch.no_grad()
def collect_prediction_replicates(model, loader, device, norm_params, noise_std, node_noise_std, n_replicates, seed=SEED):
    """For each test sample, run n_replicates noisy forwards; return preds [N, n_replicates, 19] (denorm)."""
    model.eval()
    all_preds = []
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        mean, std = _norm_tensors(norm_params, data.x.device)
        preds_batch = []
        for r in range(n_replicates):
            torch.manual_seed(seed + batch_idx * 1000 + r)
            data_r = add_measurement_noise(data, noise_std=noise_std, node_noise_std=node_noise_std, device=device)
            pred_norm = model(data_r.x, data_r.pos, data_r.edge_index, data_r.batch)
            pred_denorm = pred_norm * std + mean
            preds_batch.append(pred_denorm.cpu().numpy())
        all_preds.append(np.stack(preds_batch, axis=1))
    return np.concatenate(all_preds, axis=0), None


def prediction_stability_per_sample(preds_replicates):
    """preds_replicates: [N, n_replicates, 19]. Return per-sample std (mean over 19 targets of std across replicates)."""
    # [N, 19] = std across replicates
    std_per_target = np.std(preds_replicates, axis=1)  # [N, 19]
    return np.mean(std_per_target, axis=1)  # [N]


def load_run_results(run_dir):
    """Load results.json; return norm_params, test_indices, dataset_size (same test set as training)."""
    run_dir = Path(run_dir)
    p = run_dir / "results.json"
    out = {"norm_params": None, "test_indices": None, "dataset_size": None}
    if not p.exists():
        return out
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if "target_mean" in d and "target_std" in d:
            out["norm_params"] = {
                "mean": torch.tensor(d["target_mean"], dtype=torch.float32),
                "std": torch.tensor(d["target_std"], dtype=torch.float32),
            }
        out["test_indices"] = d.get("test_indices")
        out["dataset_size"] = d.get("dataset_size")
    except (KeyError, TypeError, json.JSONDecodeError):
        pass
    return out


def plot_stability_bars(names, values, save_path):
    plt = _ensure_mpl()
    if plt is None or save_path is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    colors = ["#e74c3c", "#2ecc71"]
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=[colors[i % 2] for i in range(len(names))], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean ‖emb_noisy − emb_clean‖")
    ax.set_title("Embedding stability under noise (lower = more stable; E1/PMH should be lower)")
    ax.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_tsne_by_target(embs, target_values, target_name, save_path, perplexity=30):
    """t-SNE of embeddings colored by continuous target (one property)."""
    plt = _ensure_mpl()
    if plt is None or save_path is None:
        return
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("sklearn not found; skipping t-SNE.")
        return
    if embs.shape[0] < 10:
        print("Too few samples for t-SNE; skipping.")
        return
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(perplexity, embs.shape[0] - 1))
    X = tsne.fit_transform(embs)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sc = ax.scatter(X[:, 0], X[:, 1], c=target_values, cmap="viridis", alpha=0.7, s=15)
    plt.colorbar(sc, ax=ax, label=target_name)
    ax.set_title(f"t-SNE of graph embeddings (colored by {target_name})")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_tsne_grid_b0_e1(embs_b0, embs_e1, target_values, target_name, save_path, perplexity=30):
    """Two panels: B0 and E1 t-SNE, same coloring by target."""
    plt = _ensure_mpl()
    if plt is None or save_path is None:
        return
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return
    n = embs_b0.shape[0]
    if n < 10:
        return
    perp = min(perplexity, n - 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, embs, title in zip(axes, [embs_b0, embs_e1], ["B0", "E1 (PMH)"]):
        X = TSNE(n_components=2, random_state=SEED, perplexity=perp).fit_transform(embs)
        sc = ax.scatter(X[:, 0], X[:, 1], c=target_values, cmap="viridis", alpha=0.7, s=15)
        ax.set_title(title)
        ax.set_axis_off()
    plt.colorbar(sc, ax=axes, label=target_name, shrink=0.6)
    plt.suptitle(f"t-SNE of graph embeddings (colored by {target_name})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_prediction_stability(b0_std, e1_std, save_path):
    """Histogram or scatter of per-sample prediction std under noise: B0 vs E1."""
    plt = _ensure_mpl()
    if plt is None or save_path is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(b0_std, bins=50, alpha=0.7, color="#e74c3c", label="B0", density=True)
    axes[0].hist(e1_std, bins=50, alpha=0.7, color="#2ecc71", label="E1", density=True)
    axes[0].set_xlabel("Per-molecule pred std (over noisy replicates)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Prediction stability under noise")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].scatter(b0_std, e1_std, alpha=0.3, s=10, c="#333")
    axes[1].plot([0, max(b0_std.max(), e1_std.max())], [0, max(b0_std.max(), e1_std.max())], "k--", lw=1, label="y=x")
    axes[1].set_xlabel("B0 pred std")
    axes[1].set_ylabel("E1 pred std")
    axes[1].set_title("Below diagonal = E1 more stable")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pred_vs_true(pred_b0, pred_e1, true_y, target_idx, target_name, save_path):
    """Pred vs true for one target: two panels B0 and E1."""
    plt = _ensure_mpl()
    if plt is None or save_path is None:
        return
    p_b0 = pred_b0[:, target_idx]
    p_e1 = pred_e1[:, target_idx]
    t = true_y[:, target_idx]
    lim_lo = min(t.min(), p_b0.min(), p_e1.min())
    lim_hi = max(t.max(), p_b0.max(), p_e1.max())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, pred, title in zip(axes, [p_b0, p_e1], ["B0", "E1 (PMH)"]):
        ax.scatter(t, pred, alpha=0.3, s=8)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="y=x")
        ax.set_xlabel(f"True {target_name}")
        ax.set_ylabel(f"Pred {target_name}")
        ax.set_title(title)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle(f"Pred vs true: {target_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    p = argparse.ArgumentParser(description="Embedding and prediction analysis for QM9 (B0 vs E1)")
    p.add_argument("--b0", type=str, default="runs/QM9/B0/best.pt", help="B0 checkpoint")
    p.add_argument("--e1", type=str, default="runs/QM9/E1/best.pt", help="E1 checkpoint")
    p.add_argument("--out_dir", type=str, default="runs/QM9/embedding_analysis", help="Output directory")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--subset", type=int, default=None, help="Subset dataset size (default full); use e.g. 5000 for faster run")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--noise_pos", type=float, default=NOISE_POS, help="Position noise σ (Å) for stability")
    p.add_argument("--noise_node", type=float, default=NOISE_NODE, help="Node noise for stability")
    p.add_argument("--n_replicates", type=int, default=N_REPLICATES, help="Noisy replicates for prediction stability")
    p.add_argument("--target_idx", type=int, default=7, help="Target index for t-SNE color and pred-vs-true (7=U0)")
    p.add_argument("--max_tsne", type=int, default=2000, help="Max points for t-SNE (subsample if test set larger)")
    p.add_argument("--hidden", type=int, default=128, help="Must match training (128 = Chemistry/binding)")
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Fallback get_loaders only: must match train.py --seed",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Noise / t-SNE / subsample RNG (default 42)",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b0_path = Path(args.b0)
    b0_run = load_run_results(b0_path.parent)
    norm_params = b0_run["norm_params"]
    test_indices = b0_run["test_indices"]
    dataset_size = b0_run["dataset_size"]

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
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        sample = dataset[0] if hasattr(dataset, "__getitem__") else dataset.dataset[0]
        num_node_features = sample.x.shape[1] if sample.x.dim() > 1 else sample.x.numel()
        info = {
            "num_node_features": num_node_features,
            "num_targets": NUM_TARGETS,
            "num_test": len(test_indices),
        }
        print(f"Using test set from B0 run ({len(test_indices)} samples, same as training).")
    else:
        _, _, test_loader, info = get_loaders(
            root=args.data_dir,
            subset=args.subset,
            batch_size=args.batch_size,
            seed=args.split_seed,
        )
        if test_indices is None or dataset_size is None:
            print("B0 results.json has no test_indices/dataset_size; using current split.")
        else:
            print(f"Dataset size {n_ds} != B0 dataset_size {dataset_size}; using current split.")

    n_test = info["num_test"]
    print(f"Test samples: {n_test}")
    if norm_params is None:
        norm_params = {"mean": info["target_mean"], "std": info["target_std"]}

    target_name = QM9_TARGET_NAMES[args.target_idx] if args.target_idx < len(QM9_TARGET_NAMES) else f"T{args.target_idx}"

    results = {}
    for name, ckpt_path in [("B0", args.b0), ("E1", args.e1)]:
        path = Path(ckpt_path)
        if not path.exists():
            print(f"Skip {name}: not found {path}")
            continue
        model = get_model(
            info["num_node_features"],
            num_targets=info["num_targets"],
            hidden=args.hidden,
            num_layers=args.num_layers,
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()

        embs_clean, pred_clean, targets = collect_embeddings_and_predictions(
            model, test_loader, device, norm_params, noise_std=0.0, node_noise_std=0.0
        )
        embs_noisy, pred_noisy, _ = collect_embeddings_and_predictions(
            model, test_loader, device, norm_params,
            noise_std=args.noise_pos, node_noise_std=args.noise_node, seed=args.seed
        )
        stability = embedding_stability(embs_clean, embs_noisy)
        results[name] = {
            "embedding_stability": stability,
            "embs_clean": embs_clean,
            "embs_noisy": embs_noisy,
            "pred_clean": pred_clean,
            "targets": targets,
        }
        print(f"{name}: embedding stability = {stability:.4f}")

        pred_repl, _ = collect_prediction_replicates(
            model, test_loader, device, norm_params,
            args.noise_pos, args.noise_node, args.n_replicates, seed=args.seed
        )
        pred_std = prediction_stability_per_sample(pred_repl)
        results[name]["pred_std_per_sample"] = pred_std
        results[name]["mean_pred_std"] = float(np.mean(pred_std))
        print(f"{name}: mean prediction std (over noisy replicates) = {results[name]['mean_pred_std']:.4f}")

    if len(results) < 2:
        print("Need both B0 and E1 checkpoints.")
        return

    # Report
    report_path = out_dir / "embedding_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("QM9 Embedding & prediction analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"Position noise σ = {args.noise_pos} Å, node noise = {args.noise_node}\n")
        f.write(f"Replicates for pred stability = {args.n_replicates}\n\n")
        f.write(f"{'Model':<6} {'Emb stability':>14} {'Mean pred std':>14}\n")
        f.write("-" * 38 + "\n")
        for name in ["B0", "E1"]:
            if name in results:
                r = results[name]
                f.write(f"{name:<6} {r['embedding_stability']:>14.4f} {r['mean_pred_std']:>14.4f}\n")
        f.write("\nEmbedding stability = mean ‖emb_noisy − emb_clean‖ (lower = more stable).\n")
        f.write("Mean pred std = mean over molecules of std(pred) over noisy replicates (lower = more stable predictions).\n")
    print(f"Saved: {report_path}")

    plt = _ensure_mpl()
    if plt is not None:
        plot_stability_bars(
            list(results.keys()),
            [results[n]["embedding_stability"] for n in results],
            out_dir / "embedding_stability.png",
        )
        # t-SNE colored by target (subsample if large for speed)
        embs_b0 = results["B0"]["embs_clean"]
        embs_e1 = results["E1"]["embs_clean"]
        target_vals = results["B0"]["targets"][:, args.target_idx]
        n_tsne = min(embs_b0.shape[0], args.max_tsne)
        if n_tsne < embs_b0.shape[0]:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(embs_b0.shape[0], n_tsne, replace=False)
            embs_b0 = embs_b0[idx]
            embs_e1 = embs_e1[idx]
            target_vals = target_vals[idx]
        plot_tsne_grid_b0_e1(embs_b0, embs_e1, target_vals, target_name, out_dir / "tsne_by_target.png")
        plot_prediction_stability(
            results["B0"]["pred_std_per_sample"],
            results["E1"]["pred_std_per_sample"],
            out_dir / "prediction_stability.png",
        )
        plot_pred_vs_true(
            results["B0"]["pred_clean"],
            results["E1"]["pred_clean"],
            results["B0"]["targets"],
            args.target_idx,
            target_name,
            out_dir / "pred_vs_true.png",
        )
    else:
        print("matplotlib not available; skipping plots.")


if __name__ == "__main__":
    main()
