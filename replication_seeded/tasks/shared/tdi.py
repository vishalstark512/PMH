"""
Shared Topological Distortion Index (TDI) computation.

TDI = intra-class mean pairwise distance / inter-class centroid distance.
Higher TDI means representations of same-class samples are more scattered
relative to the inter-class separation — i.e., the manifold is distorted.

This is architecture- and modality-agnostic: pass any (N, D) embedding matrix
with (N,) integer labels and get back a scalar distortion measure.
"""
from __future__ import annotations

import numpy as np


def compute_tdi(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int | None = None,
    max_per_class: int = 400,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """
    Compute Topological Distortion Index.

    Parameters
    ----------
    embeddings   : (N, D) float array of latent representations.
    labels       : (N,) int array of class indices.
    num_classes  : number of classes; inferred from labels if None.
    max_per_class: subsample each class to this many points for tractable
                   O(n²) pairwise computation. 400 gives stable estimates.
    rng_seed     : for reproducible subsampling.

    Returns
    -------
    (intra_mean, tdi) where tdi = intra_mean / inter_mean.
    """
    embs = np.asarray(embeddings, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.int64)
    if num_classes is None:
        num_classes = int(labs.max()) + 1

    rng = np.random.default_rng(rng_seed)
    intra_dists: list[float] = []
    centroids: list[np.ndarray] = []

    for c in range(num_classes):
        mask = labs == c
        if mask.sum() == 0:
            continue
        X = embs[mask]
        n = X.shape[0]
        if n > max_per_class:
            idx = rng.choice(n, max_per_class, replace=False)
            X = X[idx]
            n = max_per_class
        centroids.append(X.mean(axis=0))
        if n >= 2:
            d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
            triu_i, triu_j = np.triu_indices(n, k=1)
            intra_dists.extend(d[triu_i, triu_j].tolist())

    intra_mean = float(np.mean(intra_dists)) if intra_dists else 0.0

    cents = np.stack(centroids)  # (C, D)
    nc = cents.shape[0]
    inter_dists: list[float] = []
    for i in range(nc):
        for j in range(i + 1, nc):
            inter_dists.append(float(np.linalg.norm(cents[i] - cents[j])))
    inter_mean = float(np.mean(inter_dists)) if inter_dists else 1.0

    tdi = intra_mean / inter_mean if inter_mean > 0 else 0.0
    return intra_mean, tdi
