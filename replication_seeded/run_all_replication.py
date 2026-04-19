"""
Run all nine paper replication tasks (01–09) with one command.

Unified layout under --artifact_root (default: ./artifacts next to this file):
  <artifact_root>/models/<task_slug>/   — checkpoints and train outputs
  <artifact_root>/results/<task_slug>/ — eval JSON, interp, TDI, etc.

After completion, writes <artifact_root>/run_manifest.json with paths and a
flat index of JSON files under each task's results/ tree (capped for size).

Usage (from replication_seeded/):
  python run_all_replication.py --data_dir D:/pmh_data --artifact_root ./artifacts

Subset:
  python run_all_replication.py --data_dir ./data --tasks 01 02 08

Task 08 ignores --data_dir (SST-2 via Hugging Face).

Task 09 (ImageNet ViT): defaults to Hugging Face (``TASK09_USE_HUGGINGFACE`` in
``tasks/run_task.py``). For local CLS-LOC, set that flag False and point
``TASK09_IMAGENET_DATA_DIR`` at an existing directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tasks"))

import run_task as rt  # noqa: E402

_JSON_INDEX_CAP = 500


def _collect_metric_paths(artifact_root: Path, task_ids: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for tid in task_ids:
        slug = rt.TASK_SLUGS[tid]
        res_dir = artifact_root / "results" / slug
        rel_json: list[str] = []
        if res_dir.is_dir():
            for p in sorted(res_dir.rglob("*.json")):
                try:
                    rel_json.append(str(p.relative_to(artifact_root)))
                except ValueError:
                    rel_json.append(str(p))
                if len(rel_json) >= _JSON_INDEX_CAP:
                    rel_json.append(f"... truncated after {_JSON_INDEX_CAP} files")
                    break
        out[tid] = {"slug": slug, "results_json_paths": rel_json}
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run tasks 01–09 into a single artifacts tree (models/ + results/ per task)."
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(HERE / "data"),
        help="Dataset root (each task script uses its own layout under this path)",
    )
    p.add_argument(
        "--artifact_root",
        type=str,
        default=str(HERE / "artifacts"),
        help="Output root: models/<slug>/ and results/<slug>/",
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task ids to run (e.g. 01 04). Default: all 01–09 in sorted order.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Multi-seed mode for tasks 01/04/08 only. Writes under <slug>/seed_<N>/.",
    )
    p.add_argument(
        "--no_sweep",
        action="store_true",
        help="Task 04 only: skip the T-alignment sigma sweep (5 extra E1_sigma_* training runs).",
    )
    args = p.parse_args()

    root = Path(args.artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    order = sorted(rt.TASKS.keys()) if not args.tasks else list(args.tasks)

    started = datetime.now(timezone.utc).isoformat()
    rt.run_all_tasks(args.data_dir, root, tasks=order, seeds=args.seeds, no_sweep=args.no_sweep)
    finished = datetime.now(timezone.utc).isoformat()

    manifest = {
        "started_utc": started,
        "finished_utc": finished,
        "data_dir": str(Path(args.data_dir).resolve()),
        "artifact_root": str(root),
        "tasks": order,
        "seeds": args.seeds,
        "task_slugs": {tid: rt.TASK_SLUGS[tid] for tid in order},
        "layout": {
            "models": f"{root.name}/models/<task_slug>/",
            "results": f"{root.name}/results/<task_slug>/",
            "seed_subfolder": f"{root.name}/{{models,results}}/<task_slug>/seed_<N>/ (if --seeds)",
        },
        "metric_json_index": _collect_metric_paths(root, order),
    }
    manifest_name = "run_manifest.json"
    if args.seeds:
        manifest_name = f"run_manifest_seeds_{'_'.join(str(s) for s in args.seeds)}.json"
    manifest_path = root / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
