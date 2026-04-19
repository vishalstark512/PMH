"""
Load MUTAG, PROTEINS, ENZYMES via PyTorch Geometric TUDataset.
Consistent train/val/test split with fixed seed.
"""
import os
import random
from functools import partial

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


DATASETS = ("MUTAG", "PROTEINS", "ENZYMES")
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0.8, 0.1, 0.1
SEED = 42


def _seed_worker(worker_id, base_seed):
    """Picklable worker init for DataLoader (Windows spawn)."""
    s = int(base_seed) + int(worker_id)
    random.seed(s)
    if np is not None:
        np.random.seed(s)
    torch.manual_seed(s)


def get_dataset(name, root="./data"):
    """Load TUDataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name}. Choose from {DATASETS}")
    path = os.path.join(root, "TU", name)
    dataset = TUDataset(path, name=name)
    # 0-indexed graph-level labels (some TU are 1-indexed or -1/1)
    # Use _data to avoid InMemoryDataset warning; fallback to .data for other backends
    data_obj = dataset._data if hasattr(dataset, "_data") else getattr(dataset, "data", None)
    if data_obj is not None and getattr(data_obj, "y", None) is not None:
        y = data_obj.y
        if y.dim() == 2:
            y = y.squeeze(1)
        min_y = y.min().item()
        if min_y != 0:
            data_obj.y = (y - min_y).long()
    return dataset


def get_loaders(name, root="./data", batch_size=32, seed=SEED, num_workers=0):
    """
    Returns train_loader, val_loader, test_loader and info dict.
    Split: 80% train, 10% val, 10% test (by graph). Uses Subset for reproducible splits.
    """
    from torch.utils.data import Subset
    dataset = get_dataset(name, root)
    n = len(dataset)
    g = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(n, generator=g).tolist()
    n_train = int(n * SPLIT_TRAIN)
    n_val = int(n * SPLIT_VAL)
    n_test = n - n_train - n_val
    train_ds = Subset(dataset, indices[:n_train])
    val_ds = Subset(dataset, indices[n_train : n_train + n_val])
    test_ds = Subset(dataset, indices[n_train + n_val :])

    train_kw = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    if num_workers > 0:
        train_kw["persistent_workers"] = True
        train_kw["worker_init_fn"] = partial(_seed_worker, base_seed=int(seed))
    val_test_kw = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if num_workers > 0:
        val_test_kw["persistent_workers"] = True
        val_test_kw["worker_init_fn"] = partial(_seed_worker, base_seed=int(seed) + 1)

    train_loader = DataLoader(train_ds, **train_kw)
    val_loader = DataLoader(val_ds, **val_test_kw)
    test_loader = DataLoader(test_ds, **val_test_kw)

    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes

    return train_loader, val_loader, test_loader, {
        "num_node_features": num_node_features,
        "num_classes": num_classes,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
    }
