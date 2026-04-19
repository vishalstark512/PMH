"""
Enhanced PMH Loss for Pose Estimation (and general use).

Builds on the existing PMHLossPose with three principled additions:
  1. Relational Stability  - preserves pairwise structure under perturbation
  2. Geometric Quality      - prevents collapse, ensures capacity usage
  3. Multi-scale with Lipschitz normalization

The core idea: pointwise stability (||φ(x) - φ(x')||²) is necessary but
insufficient. We also need the *relationships between samples* to be stable,
and the representation to have good geometric properties.

Drop-in compatible with existing train.py — just replace PMHLossPose.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Component 1: Pointwise Stability (what you already have, refined)
# ──────────────────────────────────────────────────────────────────────

def pointwise_stability(feat_clean, feat_corrupt):
    """
    ||φ(x) - φ(x')||²  per sample, averaged.

    Features are L2-normalized so the loss is in [0, 4] regardless of
    embedding dimension.  This makes the loss scale-invariant across
    backbone layers of different width.
    """
    c = F.normalize(feat_clean, p=2, dim=1)
    n = F.normalize(feat_corrupt, p=2, dim=1)
    return (c - n).pow(2).sum(dim=1).mean()


# ──────────────────────────────────────────────────────────────────────
# Component 2: Relational Stability (NEW — the key addition)
# ──────────────────────────────────────────────────────────────────────

def relational_stability(feat_clean, feat_corrupt, n_pairs=256):
    """
    Preserve pairwise distance *structure* under perturbation.

    Instead of asking "did each point stay put?" we ask "did the
    relative distances between points stay the same?"

    This is fundamentally different:
    - Pointwise: each embedding is individually stable
    - Relational: the TOPOLOGY (who is close to whom) is stable

    Relational stability naturally prevents collapse — if everything
    collapses to one point, clean distances are nonzero but corrupt
    distances are zero, creating a huge loss.

    We normalize distances so the loss is scale-invariant.  This means
    we don't care about uniform scaling or translation of the embedding
    space — only the *structure* matters.
    """
    B = feat_clean.shape[0]
    if B < 4:
        return torch.tensor(0.0, device=feat_clean.device)

    # L2 normalize for consistent scale
    c = F.normalize(feat_clean, p=2, dim=1)
    n = F.normalize(feat_corrupt, p=2, dim=1)

    # Sample random pairs (efficient — O(n_pairs) not O(B²))
    actual_pairs = min(n_pairs, B * (B - 1) // 2)
    idx_i = torch.randint(0, B, (actual_pairs,), device=c.device)
    idx_j = torch.randint(0, B, (actual_pairs,), device=c.device)

    # Avoid self-pairs
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    if idx_i.shape[0] < 2:
        return torch.tensor(0.0, device=feat_clean.device)

    # Pairwise squared distances
    d_clean = (c[idx_i] - c[idx_j]).pow(2).sum(dim=1)
    d_corrupt = (n[idx_i] - n[idx_j]).pow(2).sum(dim=1)

    # Normalize: convert to relative distances (structure, not scale)
    d_clean_norm = d_clean / (d_clean.mean().detach() + 1e-8)
    d_corrupt_norm = d_corrupt / (d_corrupt.mean().detach() + 1e-8)

    # MSE between normalized distance profiles
    # .detach() on clean — clean distances are the "target" structure
    return F.mse_loss(d_corrupt_norm, d_clean_norm.detach())


# ──────────────────────────────────────────────────────────────────────
# Component 3: Geometric Quality (NEW — anti-collapse + capacity)
# ──────────────────────────────────────────────────────────────────────

def effective_dimensionality_loss(embeddings, target_ratio=0.5):
    """
    Encourage the representation to USE its capacity.

    Measures effective dimensionality via participation ratio of the
    singular value spectrum.  If the representation collapses to a
    low-dimensional subspace (e.g. stability pushes everything together),
    this loss pushes back.

    target_ratio: fraction of embedding dims that should be "active".
        0.5 means we want at least half the dimensions to carry signal.
        Too high forces the network to spread signal thin.
        Too low allows collapse.

    Returns a loss that is:
        0 when effective_dim / total_dim >= target_ratio
        positive when effective_dim / total_dim < target_ratio

    This acts as a FLOOR on dimensionality, not a target — it only
    activates when collapse is happening.
    """
    B, D = embeddings.shape
    if B < 4:
        return torch.tensor(0.0, device=embeddings.device)

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)

    # SVD (more stable than eigendecomposition of covariance)
    try:
        _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    except Exception:
        return torch.tensor(0.0, device=embeddings.device)

    # Participation ratio = (Σ σᵢ²)² / Σ σᵢ⁴
    # Equals 1 if one component dominates, D if all equal
    sv_sq = S.pow(2).clamp(min=1e-12)
    participation_ratio = sv_sq.sum().pow(2) / sv_sq.pow(2).sum()

    # Normalize to [0, 1] range
    dim_ratio = participation_ratio / D

    # Hinge loss: only penalize if below target
    loss = F.relu(target_ratio - dim_ratio)

    return loss


# ──────────────────────────────────────────────────────────────────────
# The Complete Enhanced PMH Loss
# ──────────────────────────────────────────────────────────────────────

class EnhancedPMHLoss(nn.Module):
    """
    Complete PMH loss with three principled components:

    1. Pointwise stability:  ||φ(x) - φ(x')||²
       → Individual embeddings don't move under perturbation

    2. Relational stability: ||D(φ(X)) - D(φ(X'))||²  (normalized)
       → Pairwise distance structure is preserved
       → Naturally prevents collapse

    3. Geometric quality: effective dimensionality floor
       → Representation uses its capacity
       → Prevents the stability losses from squishing everything flat

    Multi-scale: operates on last `num_scales` backbone layers.
    Each layer captures different levels of abstraction.

    Curriculum: three-phase introduction
       Phase 1 (0 → warmup_frac):     pointwise only, ramping up
       Phase 2 (warmup_frac → 2x):    + relational, ramping up
       Phase 3 (2x → end):            + geometric quality

    Usage (drop-in replacement for PMHLossPose):
        pmh = EnhancedPMHLoss(num_scales=3)
        loss, components = pmh(features_clean, features_corrupt, epoch, total_epochs)
    """

    def __init__(
        self,
        num_scales: int = 3,
        n_pairs: int = 256,
        dim_target_ratio: float = 0.3,
        # Relative weights of the three components (relational too high can cause instability)
        w_pointwise: float = 1.0,
        w_relational: float = 0.25,
        w_geometric: float = 0.2,
        # Curriculum (phased ramp-in; set epoch=None, total_epochs=None to disable)
        warmup_frac: float = 0.15,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.n_pairs = n_pairs
        self.dim_target_ratio = dim_target_ratio
        self.w_pointwise = w_pointwise
        self.w_relational = w_relational
        self.w_geometric = w_geometric
        self.warmup_frac = warmup_frac

    def forward(self, features_clean, features_corrupt, epoch=None, total_epochs=None):
        """
        Args:
            features_clean:  list of (B, C, H, W) from backbone (clean input)
            features_corrupt: list of (B, C, H, W) from backbone (perturbed input)
            epoch:           current epoch (for curriculum; None = full weight)
            total_epochs:    total training epochs (for curriculum)

        Returns:
            total_loss: scalar
            components: dict with individual loss values (for logging)
        """
        fc = features_clean[-self.num_scales:]
        fn = features_corrupt[-self.num_scales:]

        # Curriculum scheduling
        if epoch is not None and total_epochs is not None and total_epochs > 0:
            progress = epoch / total_epochs
            # Phase 1: pointwise ramps in
            s_point = min(1.0, progress / max(self.warmup_frac, 1e-6))
            # Phase 2: relational ramps in (starts at warmup_frac)
            s_rel = max(0.0, min(1.0, (progress - self.warmup_frac) / max(self.warmup_frac, 1e-6)))
            # Phase 3: geometric ramps in (starts at 2 * warmup_frac)
            s_geo = max(0.0, min(1.0, (progress - 2 * self.warmup_frac) / max(self.warmup_frac, 1e-6)))
        else:
            s_point = s_rel = s_geo = 1.0

        total_pointwise = 0.0
        total_relational = 0.0
        total_geometric = 0.0

        for c_feat, n_feat in zip(fc, fn):
            # Pool spatial dims → (B, C)
            if c_feat.dim() == 4:
                c_flat = F.adaptive_avg_pool2d(c_feat, (1, 1)).flatten(1)
                n_flat = F.adaptive_avg_pool2d(n_feat, (1, 1)).flatten(1)
            else:
                c_flat = c_feat
                n_flat = n_feat

            # Component 1: Pointwise
            total_pointwise += pointwise_stability(c_flat, n_flat)

            # Component 2: Relational
            if s_rel > 0:
                total_relational += relational_stability(
                    c_flat, n_flat, n_pairs=self.n_pairs
                )

            # Component 3: Geometric (on clean embeddings)
            if s_geo > 0:
                total_geometric += effective_dimensionality_loss(
                    c_flat, target_ratio=self.dim_target_ratio
                )

        n = self.num_scales
        total_pointwise = total_pointwise / n
        total_relational = total_relational / n
        total_geometric = total_geometric / n

        # Weighted combination with curriculum
        loss = (
            self.w_pointwise * s_point * total_pointwise
            + self.w_relational * s_rel * total_relational
            + self.w_geometric * s_geo * total_geometric
        )

        components = {
            "pmh_pointwise": total_pointwise.item() if torch.is_tensor(total_pointwise) else total_pointwise,
            "pmh_relational": total_relational.item() if torch.is_tensor(total_relational) else total_relational,
            "pmh_geometric": total_geometric.item() if torch.is_tensor(total_geometric) else total_geometric,
            "schedule_point": round(s_point, 3),
            "schedule_rel": round(s_rel, 3),
            "schedule_geo": round(s_geo, 3),
        }

        return loss, components


# ──────────────────────────────────────────────────────────────────────
# Diagnostics: Measure representation quality (run periodically)
# ──────────────────────────────────────────────────────────────────────

class PMHDiagnostics:
    """
    Compute representation quality metrics.  Run every N epochs to
    track whether the loss is achieving its geometric goals.

    Metrics:
        lipschitz_mean/max: How much embeddings move under perturbation
        knn_preservation:   Fraction of k-nearest neighbors preserved
        effective_dim:      Participation ratio of singular values
        dim_ratio:          effective_dim / total_dim
        class_separation:   Inter-class / intra-class distance ratio
                           (only if labels provided)
    """

    @staticmethod
    @torch.no_grad()
    def compute(emb_clean, emb_corrupt, labels=None, k=10):
        """
        Args:
            emb_clean:   (N, D) clean embeddings
            emb_corrupt: (N, D) perturbed embeddings
            labels:      (N,) optional class labels
            k:           neighborhood size for kNN preservation

        Returns:
            dict of metrics
        """
        metrics = {}
        N, D = emb_clean.shape

        # --- Lipschitz estimate ---
        rep_dist = (emb_clean - emb_corrupt).pow(2).sum(dim=1).sqrt()
        metrics["lipschitz_mean"] = rep_dist.mean().item()
        metrics["lipschitz_max"] = rep_dist.max().item()
        metrics["lipschitz_std"] = rep_dist.std().item()

        # --- Effective dimensionality ---
        centered = emb_clean - emb_clean.mean(dim=0)
        _, S, _ = torch.linalg.svd(centered, full_matrices=False)
        sv_sq = S.pow(2).clamp(min=1e-12)
        pr = (sv_sq.sum().pow(2) / sv_sq.pow(2).sum()).item()
        metrics["effective_dim"] = pr
        metrics["dim_ratio"] = pr / D

        # --- kNN preservation ---
        if N <= 2000:  # full pairwise is tractable
            D_clean = torch.cdist(emb_clean, emb_clean)
            D_corrupt = torch.cdist(emb_corrupt, emb_corrupt)

            actual_k = min(k, N - 1)
            _, knn_c = D_clean.topk(actual_k + 1, largest=False, dim=1)
            _, knn_n = D_corrupt.topk(actual_k + 1, largest=False, dim=1)
            knn_c = knn_c[:, 1:]  # exclude self
            knn_n = knn_n[:, 1:]

            # Per-point Jaccard, averaged
            preservation = 0.0
            for i in range(N):
                set_c = set(knn_c[i].tolist())
                set_n = set(knn_n[i].tolist())
                preservation += len(set_c & set_n) / len(set_c | set_n)
            metrics["knn_preservation"] = preservation / N

        # --- Class separation (if labels available) ---
        if labels is not None:
            unique = labels.unique()
            if len(unique) > 1:
                intra, inter = [], []
                for lbl in unique:
                    mask = labels == lbl
                    cls_emb = emb_clean[mask]
                    other_emb = emb_clean[~mask]
                    if cls_emb.shape[0] > 1:
                        intra.append(torch.cdist(cls_emb, cls_emb).mean())
                    if other_emb.shape[0] > 0:
                        inter.append(torch.cdist(cls_emb, other_emb).mean())
                if intra and inter:
                    metrics["class_separation"] = (
                        torch.stack(inter).mean() /
                        (torch.stack(intra).mean() + 1e-8)
                    ).item()

        return metrics


# ──────────────────────────────────────────────────────────────────────
# Backward-compatible wrapper (matches original PMHLossPose interface)
# ──────────────────────────────────────────────────────────────────────

class PMHLossPose(nn.Module):
    """
    Drop-in replacement for the original PMHLossPose.
    Same call signature: loss = pmh(features_clean, features_corrupt)
    Optional: loss = pmh(features_clean, features_corrupt, epoch, total_epochs) for curriculum.

    Internally uses EnhancedPMHLoss but returns just the scalar loss
    (no components dict) for backward compatibility.

    To get the full interface with components, use EnhancedPMHLoss directly.
    """
    def __init__(self, num_scales=3):
        super().__init__()
        self.enhanced = EnhancedPMHLoss(num_scales=num_scales)

    def forward(self, features_clean, features_corrupt, epoch=None, total_epochs=None):
        loss, _ = self.enhanced(features_clean, features_corrupt, epoch, total_epochs)
        return loss
