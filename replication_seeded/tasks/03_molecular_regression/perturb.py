"""
Perturbation for molecular regression: position noise (measurement-like) and optional node feature noise.
Shared by train.py and eval.py to avoid circular imports.

Like Task 2 (graph classification): PMH needs clean vs noisy *representations* to differ.
Task 2 perturbs all node features (sigma=0.1) so loss_pmh is O(0.01). We can do position-only
(small raw pmh → need large pmh_scale) or add small node noise so raw pmh is larger (Task-2-style).
"""
import torch

# Position noise (Å). 0.01 ≈ 0.7% of typical bond length; property effectively unchanged.
MEASUREMENT_NOISE_STD = 0.01
# Node feature noise. For regression use smaller than classification: 0.05 so perturbation
# is "small" (same molecule) and we don't force invariance to large input changes. 0 = position-only.
NODE_NOISE_STD = 0.05


def add_measurement_noise(data, noise_std=MEASUREMENT_NOISE_STD, node_noise_std=0.0, device=None, generator=None):
    """
    Add Gaussian noise to positions and optionally to node features (both leave label unchanged).
    If generator is not None, use it for reproducible noise (same draw for B0 and E1 at eval).
    """
    data_noisy = data.clone()
    dev = data_noisy.pos.device if hasattr(data_noisy, "pos") and data_noisy.pos is not None else device
    # randn_like doesn't accept generator; use randn(shape, generator=..., device=..., dtype=...) for reproducibility
    def _noise(shape, dtype=torch.float32):
        kwargs = {"device": dev, "dtype": dtype}
        if generator is not None:
            kwargs["generator"] = generator
        return torch.randn(shape, **kwargs)
    if hasattr(data_noisy, "pos") and data_noisy.pos is not None:
        data_noisy.pos = data_noisy.pos + _noise(data_noisy.pos.shape, data_noisy.pos.dtype) * noise_std
    if node_noise_std > 0 and hasattr(data_noisy, "x") and data_noisy.x is not None:
        x = data_noisy.x.float()
        data_noisy.x = x + _noise(x.shape, x.dtype) * node_noise_std
    return data_noisy
