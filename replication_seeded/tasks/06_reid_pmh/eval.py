"""
Re-ID evaluation: extract embeddings, match query to gallery, compute rank-1 and mAP.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


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

try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None

from model import get_model
from data import get_eval_loaders, find_market1501_root


def extract_embeddings(model, loader, device, use_amp=True):
    """Extract L2-normalized embeddings and metadata (pid, cam, path) for all samples."""
    model.eval()
    embs, pids, cams, paths = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                img, pid, cam, path = batch
            else:
                img, pid, cam = batch[0], batch[1], batch[2]
                path = [""] * (pid.shape[0] if torch.is_tensor(pid) else len(pid))
            img = img.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                emb = model.get_embedding(img, normalize=True)
            embs.append(emb.cpu().numpy())
            pids.append(pid.numpy() if torch.is_tensor(pid) else np.asarray(pid))
            cams.append(cam.numpy() if torch.is_tensor(cam) else np.asarray(cam))
            path_list = list(path) if isinstance(path, (list, tuple)) else [str(p) for p in path]
            paths.append(path_list)
    if not embs:
        return np.zeros((0, 1)), np.array([]), np.array([]), []
    return (
        np.vstack(embs),
        np.concatenate(pids),
        np.concatenate(cams) if cams else np.array([]),
        [x for sub in paths for x in sub],
    )


def compute_rank1_map(
    query_emb,
    query_pid,
    query_cam,
    gallery_emb,
    gallery_pid,
    gallery_cam,
    gallery_paths=None,
    metric="euclidean",
    exclude_same_cam=True,
    return_rank5_10=False,
):
    """
    Rank gallery by distance and compute rank-1 (and optionally rank-5, rank-10) and mAP.

    Market-1501 protocol: exclude from gallery for each query all images with the same
    (pid, cam) as the query (single-shot: don't match the same capture).
    """
    nq = query_emb.shape[0]
    ng = gallery_emb.shape[0]

    if metric == "cosine":
        dist = cdist(query_emb, gallery_emb, metric="cosine")
    else:
        dist = cdist(query_emb, gallery_emb, metric="euclidean")

    rank1 = rank5 = rank10 = 0
    ap_sum = 0.0

    for i in range(nq):
        qpid = query_pid[i]
        qcam = query_cam[i] if query_cam.size > 0 else None

        # Exclude same identity + same camera from gallery (Market-1501 protocol)
        if exclude_same_cam and query_cam.size > 0 and gallery_cam.size > 0:
            same = (gallery_pid == qpid) & (np.asarray(gallery_cam) == np.asarray(qcam))
            dist[i, same] = np.inf

        order = np.argsort(dist[i])
        # Build valid ranking (skip inf if we excluded same-cam)
        if exclude_same_cam and np.any(np.isinf(dist[i])):
            valid_rank = [j for j in order if dist[i, j] < np.inf]
        else:
            valid_rank = order.tolist()

        if not valid_rank:
            ap_sum += 0.0
            continue

        # Rank-1, Rank-5, Rank-10 (CMC)
        g_pids_ranked = gallery_pid[valid_rank]
        if g_pids_ranked[0] == qpid:
            rank1 += 1
        if return_rank5_10:
            if len(valid_rank) >= 5 and np.any(g_pids_ranked[:5] == qpid):
                rank5 += 1
            if len(valid_rank) >= 10 and np.any(g_pids_ranked[:10] == qpid):
                rank10 += 1

        # mAP over valid gallery only
        matches = (g_pids_ranked == qpid).astype(np.float32)
        num_rel = matches.sum()
        if num_rel == 0:
            ap_sum += 0.0
            continue
        n_valid = len(valid_rank)
        prec_at_k = np.cumsum(matches) / (np.arange(n_valid) + 1.0)
        ap = np.sum(prec_at_k * matches) / num_rel
        ap_sum += ap

    rank1_acc = 100.0 * rank1 / nq if nq else 0.0
    mAP = 100.0 * ap_sum / nq if nq else 0.0
    if return_rank5_10:
        rank5_acc = 100.0 * rank5 / nq if nq else 0.0
        rank10_acc = 100.0 * rank10 / nq if nq else 0.0
        return rank1_acc, mAP, rank5_acc, rank10_acc
    return rank1_acc, mAP


def evaluate(
    model,
    query_loader,
    gallery_loader,
    device,
    use_amp=True,
    exclude_same_cam=True,
    return_rank5_10=True,
):
    """Extract embeddings and return rank-1 (%), mAP (%), and optionally rank-5, rank-10."""
    q_emb, q_pid, q_cam, _ = extract_embeddings(model, query_loader, device, use_amp)
    g_emb, g_pid, g_cam, _ = extract_embeddings(model, gallery_loader, device, use_amp)
    out = compute_rank1_map(
        q_emb, q_pid, q_cam,
        g_emb, g_pid, g_cam,
        exclude_same_cam=exclude_same_cam,
        return_rank5_10=return_rank5_10,
    )
    if return_rank5_10:
        return {"rank1": out[0], "mAP": out[1], "rank5": out[2], "rank10": out[3]}
    return {"rank1": out[0], "mAP": out[1]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint to evaluate")
    p.add_argument("--compare", action="store_true", help="Compare B0, B1, E1 from runs/")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="eval_out")
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for query/gallery")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (workers / any torch randomness)")
    p.add_argument("--deterministic", action="store_true", help="cudnn deterministic mode")
    p.add_argument("--no_exclude_same_cam", action="store_true", help="Disable Market-1501 same-camera exclusion")
    args = p.parse_args()

    if cdist is None:
        raise RuntimeError("scipy is required for eval. pip install scipy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed, deterministic=args.deterministic)
    use_amp = not args.no_amp and device.type == "cuda"

    root = find_market1501_root(args.data_dir)
    query_loader, gallery_loader, query_ds, gallery_ds = get_eval_loaders(
        root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    if len(query_ds) == 0 or len(gallery_ds) == 0:
        raise RuntimeError(
            f"Empty eval set: query={len(query_ds)}, gallery={len(gallery_ds)}. "
            f"Ensure {root}/query/ and {root}/bound_box_test/ contain images."
        )

    def num_classes_from_ckpt(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "classifier.weight" in sd:
            return sd["classifier.weight"].shape[0]
        return 751  # Market-1501 default

    exclude_same_cam = not args.no_exclude_same_cam

    if args.compare:
        runs = ["B0", "VAT", "E1"]
        results = {}
        for run in runs:
            ckpt = Path(args.runs_dir) / run / "best.pt"
            if not ckpt.exists():
                print(f"Skip {run}: {ckpt} not found")
                continue
            num_classes = num_classes_from_ckpt(ckpt)
            model = get_model(num_classes, embed_dim=args.embed_dim, pretrained=False)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model = model.to(device)
            metrics = evaluate(
                model, query_loader, gallery_loader, device, use_amp,
                exclude_same_cam=exclude_same_cam,
            )
            results[run] = metrics
            r1, mAP = metrics["rank1"], metrics["mAP"]
            extra = f"  rank-5 = {metrics['rank5']:.2f}%  rank-10 = {metrics['rank10']:.2f}%" if "rank5" in metrics else ""
            print(f"  {run}: rank-1 = {r1:.2f}%  mAP = {mAP:.2f}%{extra}")

        os.makedirs(args.out_dir, exist_ok=True)
        out_path = Path(args.out_dir) / "compare_results.json"
        payload = {**results, "seed": args.seed, "deterministic": bool(args.deterministic)}
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved {out_path}")
        return

    if not args.ckpt or not os.path.isfile(args.ckpt):
        print("Usage: python eval.py --ckpt runs/E1/best.pt  OR  python eval.py --compare")
        return

    num_classes = num_classes_from_ckpt(args.ckpt)
    model = get_model(num_classes, embed_dim=args.embed_dim, pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model = model.to(device)
    metrics = evaluate(
        model, query_loader, gallery_loader, device, use_amp,
        exclude_same_cam=exclude_same_cam,
    )
    r1, mAP = metrics["rank1"], metrics["mAP"]
    print(f"rank-1: {r1:.2f}%  mAP: {mAP:.2f}%", end="")
    if "rank5" in metrics:
        print(f"  rank-5: {metrics['rank5']:.2f}%  rank-10: {metrics['rank10']:.2f}%")
    else:
        print()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(Path(args.out_dir) / "eval_results.json", "w") as f:
        json.dump(
            {"ckpt": args.ckpt, "seed": args.seed, "deterministic": bool(args.deterministic), **metrics},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
