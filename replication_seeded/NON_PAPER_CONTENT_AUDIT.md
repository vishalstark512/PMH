# Non-Paper Content Audit (Cross-Repo Final Check)

This audit now checks **beyond `replication_seeded/`**, as requested:
- source universe from top-level `tasks/` and `experiments/` (excluding generated `runs/`, `out/`, `data/` trees),
- against what is currently kept in `replication_seeded/`,
- with paper alignment from `submission/PMH_ICML2025_submission.tex` and `replication_seeded/PAPER_SCOPE.md`.

## Scope Used

- Paper reference: `submission/PMH_ICML2025_submission.tex`
- Source-code universe (outside replication tree): `tasks/**/*.py` + `experiments/**/*.py` (excluding generated artifact/data folders)
- Replication tree: `replication_seeded/`

## Quantitative Summary

### A) Source universe outside replication tree

- Source Python files considered (`tasks/` + `experiments/`): **80 files**
- Source Python lines considered: **25,114 lines**

### B) Replication code kept

- Replication Python files kept for paper tasks (`replication_seeded/tasks/**/*.py`, excluding `tasks/run_task.py`): **53 files**
- Replication paper-mapped Python lines: **17,353 lines**

### C) Source content intentionally excluded from replication

- Excluded vs source universe: **7,761 lines**  
  (`25,114 - 17,353`)
- These excluded lines are predominantly non-paper/auxiliary content from the broader repo (e.g., extra runners, optional experiment variants, and tooling not needed for reported paper outputs).

### D) Non-paper infrastructure inside replication tree

- `tasks/run_task.py`: **191 lines**
- `.gitignore`: **8 lines**
- `PAPER_SCOPE.md`: **63 lines**
- `NON_PAPER_CONTENT_AUDIT.md`: **48 lines** (this report)
- **Total infra in replication tree:** **310 lines**

## Ratios

- Non-paper infra vs kept paper-mapped Python:
  - **310 / 17,353 = 1.79%**
- Excluded source (outside replication) vs kept paper-mapped Python:
  - **7,761 / 17,353 = 44.72%**

Interpretation: most extra code volume lives outside `replication_seeded` in the original source universe; inside `replication_seeded`, non-paper content is small and operational.

## What counts as "non-paper" in `replication_seeded`

Operational scaffolding only:
1. `tasks/run_task.py` (one-command orchestration, standardized artifact paths)
2. `.gitignore` (prevents model/checkpoint commits)
3. `PAPER_SCOPE.md` and this audit report (documentation)

## Requirements files note

Task-local `requirements.txt` files are packaging metadata for reproducibility (not paper prose/results):
- Count: **8 files**
- Total lines: **57**

## Conclusion

After cross-repo checking (not only `replication_seeded`), the replication tree is tightly paper-scoped.  
Inside `replication_seeded`, the extra non-paper footprint remains small and practical (**~1.8%** of kept implementation-code scale).

