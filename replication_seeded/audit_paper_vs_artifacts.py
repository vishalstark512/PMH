#!/usr/bin/env python3
"""Fast spot-check: ICML + arXiv preprint .tex + figures vs replication_seeded JSON.

Typical runtime: well under one second on a local disk.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "replication_seeded" / "artifacts"
MAIN = ROOT / "submission" / "PMH_ICML2025_submission.tex"
FIGDIR = ROOT / "submission" / "figures"
ARXIV_MAIN = ROOT / "submission" / "arxiv_v2" / "PMH_arxiv_preprint.tex"
ARXIV_FIGDIR = ROOT / "submission" / "arxiv_v2" / "figures"


def load(p: Path) -> dict:
    if not p.is_file():
        raise FileNotFoundError(f"Missing artifact: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def r3(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def r2(x: float) -> str:
    return f"{x:.2f}"


def concat_fig_tex(fig_dir: Path) -> str:
    if not fig_dir.is_dir():
        return ""
    parts: list[str] = []
    for name in sorted(p.name for p in fig_dir.glob("*.tex")):
        parts.append((fig_dir / name).read_text(encoding="utf-8"))
    return "".join(parts)


def main() -> int:
    t0 = time.perf_counter()

    if not MAIN.is_file():
        print(f"ERROR: main tex not found: {MAIN}", file=sys.stderr)
        return 2
    if not FIGDIR.is_dir():
        print(f"ERROR: figures dir not found: {FIGDIR}", file=sys.stderr)
        return 2

    icml_tex = MAIN.read_text(encoding="utf-8")
    icml_figtex = concat_fig_tex(FIGDIR)
    icml_scan = icml_tex + icml_figtex

    arxiv_targets: list[tuple[str, str]] = []
    if ARXIV_MAIN.is_file():
        if not ARXIV_FIGDIR.is_dir():
            print(
                f"ERROR: arXiv preprint exists but figures dir missing: {ARXIV_FIGDIR}",
                file=sys.stderr,
            )
            return 2
        arxiv_tex = ARXIV_MAIN.read_text(encoding="utf-8")
        arxiv_figtex = concat_fig_tex(ARXIV_FIGDIR)
        arxiv_targets.append(
            ("arxiv_v2", arxiv_tex + arxiv_figtex),
        )

    fails: list[str] = []

    def need(label: str, s: str, blob: str, doc: str):
        if s not in blob:
            fails.append(f"missing [{label}] ({doc}): {s!r}")

    def forbid(label: str, s: str, blob: str, doc: str):
        if s in blob:
            fails.append(f"stale/forbidden [{label}] ({doc}): found {s!r}")

    forbid("claim", "PMH achieves the highest normalised score on all seven tasks", icml_scan, "ICML")
    forbid("claim", "PMH achieves the highest normalised score on all seven tasks", icml_figtex, "ICML figures")
    forbid("old T06 B0 headline in figures only", "26.81", icml_figtex, "ICML figures")

    for doc_name, blob in arxiv_targets:
        forbid("claim", "PMH achieves the highest normalised score on all seven tasks", blob, doc_name)
        forbid("old T06 B0 headline in figures only", "26.81", concat_fig_tex(ARXIV_FIGDIR), f"{doc_name} figures")

    tdi = load(ART / "results/04_vision_transformer_pmh/interp/tdi_results.json")["tdi"]
    jac = load(ART / "results/04_vision_transformer_pmh/interp/jacobian_norm_results.json")[
        "runs"
    ]
    lp = load(ART / "results/04_vision_transformer_pmh/interp/linear_probe_results.json")
    b01 = load(ART / "models/01_image_classification/cifar10/B0/results.json")
    e03 = load(ART / "models/03_molecular_regression/QM9/E1/results.json")
    ev = load(ART / "results/03_molecular_regression/evals/eval_summary.json")
    pose = load(ART / "results/05_pose_estimation_pmh/eval_out/robustness_comparison.json")
    rid = load(ART / "results/06_reid_pmh/eval_out_robust/compare_results_robust.json")[
        "__summary__"
    ]

    checks: list[tuple[str, str]] = [
        ("TDI B0@0.1", r3(tdi["B0"]["0.1"])),
        ("TDI E1@0.1", r3(tdi["E1"]["0.1"])),
        ("Jac B0", r2(jac["B0"]["jacobian_fro_mean"])),
        ("Jac E1", r2(jac["E1"]["jacobian_fro_mean"])),
        ("L6 E1 @0.1", r2(lp["probe_acc"]["E1"]["0.1"][-1])),
        ("T01 B0 @0.1", r2(b01["acc_sigma_0.1"])),
        ("T03 E1 clean MAE", r2(e03["clean_mae"])),
        ("T03 pos-noise B0 sigma0", r2(ev["B0"]["0.0"]["mae_avg"])),
        ("T05 B0 PCK clean", r2(pose["baseline"]["clean"]["pck@0.05"] * 100)),
        ("T06 VAT avg shift", r2(rid["VAT"]["avg_shift_rank1"])),
        ("figA5 T01 drift B0", "0.692"),
        ("figA7 align 0.12 row bold 59.00", "59.00"),
        ("figA9 T02 edge30 E1", "76.43"),
    ]

    for label, s in checks:
        need(label, s, icml_scan, "ICML")
        for doc_name, blob in arxiv_targets:
            need(label, s, blob, doc_name)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if fails:
        print("AUDIT FAILURES:\n" + "\n".join(fails))
        return 1
    parts = ["ICML manuscript + submission/figures"]
    if arxiv_targets:
        parts.append("arxiv_v2 preprint + arxiv_v2/figures")
    else:
        parts.append("arxiv_v2 skipped (PMH_arxiv_preprint.tex not found)")
    print(f"OK ({elapsed_ms:.0f} ms): {'; '.join(parts)} pass replication spot-checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
