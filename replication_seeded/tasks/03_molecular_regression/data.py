"""
Load QM9 via PyTorch Geometric. Multi-task: keep all 19 targets per molecule.
Per-target normalization (mean, std) from training set for all 19.
Optional load_in_memory: preload all graphs into RAM (fits in 128GB) for faster training.
"""
import os
import random
from functools import partial

import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

NUM_TARGETS = 19
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0.8, 0.1, 0.1
SEED = 42


def _seed_worker(worker_id, base_seed):
    s = int(base_seed) + int(worker_id)
    random.seed(s)
    if np is not None:
        np.random.seed(s)
    torch.manual_seed(s)


class _ListDataset(Dataset):
    """Wrap a list of PyG Data objects for in-memory indexing (no disk in __getitem__)."""
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def _ensure_y_19(data):
    """Ensure data.y is shape [19] (float)."""
    data = data.clone()
    y = data.y.view(-1)
    if y.numel() >= NUM_TARGETS:
        data.y = y[:NUM_TARGETS].float()
    else:
        data.y = y.float()
    return data


def get_dataset(root="./data", subset=None):
    """
    Load QM9. Each sample has data.y shape [19]. No single-target selection.
    """
    path = os.path.join(root, "QM9")
    dataset = QM9(root=path, transform=_ensure_y_19)
    if subset is not None and subset < len(dataset):
        dataset = Subset(dataset, range(subset))
    return dataset


def get_loaders(root="./data", subset=None, batch_size=128, num_workers=0, seed=SEED, load_in_memory=False):
    """
    Returns train_loader, val_loader, test_loader and info dict.
    norm_params: mean [19], std [19] from training set (per-target).
    num_workers=0 is safe on Windows; use 2-4 on Linux for faster I/O.
    load_in_memory: if True, preload all graphs into RAM (no disk I/O during training; fits in 128GB).
    """
    dataset = get_dataset(root=root, subset=subset)
    n = len(dataset)
    g = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(n, generator=g).tolist()
    n_train = int(n * SPLIT_TRAIN)
    n_val = int(n * SPLIT_VAL)
    n_test = n - n_train - n_val

    if load_in_memory:
        # Preload all graphs into a list so __getitem__ never touches disk
        train_ds = _ListDataset([dataset[i] for i in indices[:n_train]])
        val_ds = _ListDataset([dataset[i] for i in indices[n_train : n_train + n_val]])
        test_ds = _ListDataset([dataset[i] for i in indices[n_train + n_val :]])
    else:
        train_ds = Subset(dataset, indices[:n_train])
        val_ds = Subset(dataset, indices[n_train : n_train + n_val])
        test_ds = Subset(dataset, indices[n_train + n_val :])

    # Per-target normalization from training set [19]
    ys = []
    for i in range(len(train_ds)):
        d = train_ds[i]
        y = d.y.view(-1)
        if y.numel() >= NUM_TARGETS:
            y = y[:NUM_TARGETS]
        ys.append(y)
    all_y = torch.stack(ys).float()  # [N_train, 19]
    target_mean = all_y.mean(dim=0)   # [19]
    target_std = all_y.std(dim=0)
    target_std[target_std < 1e-8] = 1.0

    persistent = num_workers > 0
    train_kw = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    if num_workers > 0:
        train_kw["worker_init_fn"] = partial(_seed_worker, base_seed=int(seed))
    val_test_kw = dict(
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=True,
    )
    if num_workers > 0:
        val_test_kw["worker_init_fn"] = partial(_seed_worker, base_seed=int(seed) + 1)
    train_loader = DataLoader(train_ds, **train_kw)
    val_loader = DataLoader(val_ds, **val_test_kw)
    test_loader = DataLoader(test_ds, **val_test_kw)

    sample = dataset[0] if hasattr(dataset, "__getitem__") else dataset.dataset[0]
    num_node_features = sample.x.shape[1] if sample.x.dim() > 1 else sample.x.numel()
    test_indices = indices[n_train + n_val :]

    return train_loader, val_loader, test_loader, {
        "num_node_features": num_node_features,
        "num_targets": NUM_TARGETS,
        "target_mean": target_mean,
        "target_std": target_std,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "test_indices": test_indices,
        "dataset_size": n,
    }
