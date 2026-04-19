"""
Universal Topological Distortion Index (TDI) utility.

TDI = intra-class mean distance / inter-class mean distance in latent space.
TDI ~= 1   -> classes well-separated, noise does not mix them (stable topology)
TDI >> 1  -> classes collapse under noise (topological blindness)

For regression tasks, use embedding_drift() (no class labels required).
"""
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def compute_tdi(embeddings: np.ndarray, labels: np.ndarray, max_per_class: int = 200, seed: int = 42) -> float:
    """Compute TDI from pre-extracted embeddings and integer labels."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    if len(classes) < 2:
        return float("nan")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb = embeddings / np.maximum(norms, 1e-8)

    per_class = {}
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        per_class[c] = emb[idx]

    intra_dists = []
    for feats in per_class.values():
        if len(feats) < 2:
            continue
        diff = feats[:, None, :] - feats[None, :, :]
        sq = (diff ** 2).sum(axis=2)
        triu = sq[np.triu_indices(len(feats), k=1)]
        intra_dists.append(float(np.sqrt(triu.mean())))
    intra = float(np.mean(intra_dists)) if intra_dists else 0.0

    centroids = {c: feats.mean(axis=0) for c, feats in per_class.items()}
    inter_dists = []
    keys = list(centroids.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter_dists.append(float(np.linalg.norm(centroids[keys[i]] - centroids[keys[j]])))
    inter = float(np.mean(inter_dists)) if inter_dists else 1e-8
    return intra / max(inter, 1e-8)


def embedding_drift(embs_clean: np.ndarray, embs_noisy: np.ndarray) -> float:
    """Mean normalized embedding displacement ||phi(x+eps)-phi(x)||_2."""
    n = np.linalg.norm(embs_clean, axis=1, keepdims=True)
    c = embs_clean / np.maximum(n, 1e-8)
    nn = np.linalg.norm(embs_noisy, axis=1, keepdims=True)
    p = embs_noisy / np.maximum(nn, 1e-8)
    return float(np.linalg.norm(c - p, axis=1).mean())


def tdi_report(results: dict, out_dir: Path, title: str = "TDI vs noise level") -> None:
    """Save tdi_results.json and optional plot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "tdi_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved TDI results -> {out_dir / 'tdi_results.json'}")

    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"B0": "#e74c3c", "VAT": "#f39c12", "E1": "#27ae60"}
    for run_name, sigma_dict in results.items():
        sigmas = sorted(sigma_dict.keys())
        vals = [sigma_dict[s] for s in sigmas]
        ax.plot(sigmas, vals, marker="o", label=run_name, color=colors.get(run_name))
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="TDI=1 (ideal)")
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("TDI (intra/inter)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "tdi.png", dpi=150)
    plt.close()
    print(f"  Saved TDI plot    -> {out_dir / 'tdi.png'}")
