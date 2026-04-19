## PMH Improvements — final status

All "plus" variants we tried on top of the paper-aligned replication have been
**removed** after empirical testing: none of the three variants matched or beat
the paper-aligned E1 baseline on this replication. The only surviving opt-in
lever is the multi-seed wrapper (`--seeds`), because it does not change the
method — it just runs the same pipeline multiple times so you can average seed
variance the same way the paper does.

This document records what was tried, the numbers it produced, and why it was
removed, so you can re-introduce any of them with an informed trade-off later.

---

### Current state of the code

- `replication_seeded/tasks/run_task.py`
  - `TASKS` (01..09) — paper-aligned CLI, unchanged.
  - `_multiseed(seeds, task_id, ...)` — runs 01/04/08 under `<slug>/seed_<N>/`.
  - No more `TASKS_PLUS`, `run_XX_plus`, or `--variant` flag.
- `replication_seeded/run_all_replication.py`
  - `--tasks` and `--seeds` flags only.

---

### Usage

```bash
# Paper-aligned run (default)
python run_all_replication.py --data_dir D:/pmh_data --artifact_root ./artifacts

# Subset of tasks
python run_all_replication.py --data_dir D:/pmh_data --artifact_root ./artifacts --tasks 02 05 07

# Multi-seed for 01/04/08 (writes under <slug>/seed_<N>/)
python run_all_replication.py --data_dir D:/pmh_data --artifact_root ./artifacts \
  --seeds 42 43 44 --tasks 01 04 08
```

---

### Tried and removed — empirical record

All three plus variants were implemented, trained, and evaluated on Apr 17.
Final decision: all three were removed because they did not improve over the
paper-aligned E1. Numbers below are from `artifacts/results/<slug>/plus/`
(which you can delete once you've seen them).

#### 1. Task 02 plus — PROTEINS E2 (mixed-task PMH) — REMOVED

Idea: train an `E2` run that also computes the task loss on the noisy view
(`--run E2 --mixed_task`).

Observed vs paper-aligned E1:

| Run | clean | σ=0.10 | worst-case |
|-----|-------|--------|------------|
| E1 (paper CLI) | 79.46 | **77.86** | **63.75** |
| E2 (plus)      | 78.57 | 77.86 | 47.32 |

Mixing the task loss onto the noisy view did not improve the σ=0.10 metric
and collapsed the worst-case accuracy by **−16.4pp**. On this dataset, E1's
"align features, classify clean" split is strictly better than E2's
"align features, classify both."

#### 2. Task 06 plus — Market-1501 identity pairs + PMH weight bump — REMOVED

Idea: replace noise-based PMH pairs with two-stream same-ID photo pairs
(`--identity_pairs`) and raise `pmh_weight` from 0.5 → 0.75.

Observed vs the (stale) paper-aligned E1:

| Run | clean R1 | avg-shift R1 | worst-shift R1 |
|-----|----------|--------------|----------------|
| E1 (paper CLI) | 67.81 | **63.74** | **53.33** |
| E1 (plus, id-pairs) | 60.18 | 55.67 | 44.36 |

Two-stream same-ID training changes the effective training distribution
substantially; 60 epochs is not enough for it to converge to E1-level Rank-1
on this dataset. Without more epochs (which defeats the "small opt-in patch"
goal), this variant hurts.

#### 3. Task 07 plus — low-PMH backoff — REMOVED

Idea: halve `pmh_weight` (0.5 → 0.25) and tighten `pmh_cap_ratio`
(0.3 → 0.1) on chest X-ray, so the saliency benefit survives but accuracy
over-regularization is removed.

Observed vs paper-aligned E1:

| Run | clean acc | worst-shift acc | saliency cos |
|-----|-----------|-----------------|--------------|
| E1 (paper CLI) | 0.913 | **0.825** | **0.718** |
| E1_no_pmh (control) | 0.913 | 0.825 | 0.717 |
| E1 (plus, pmh=0.25) | 0.913 | 0.825 | 0.716 |

Every metric is identical to three significant figures. The hypothesis
("default PMH weight is over-regularizing on chest X-ray") turned out to
be wrong on this checkpoint budget — `E1_no_pmh` and `E1` already produce
the same model numerically, so there is nothing for lower weight to recover.

---

### Multi-seed wrapper — kept (optional)

The only surviving opt-in. Re-runs tasks 01/04/08 with additional seeds so
you can average, matching the paper's 3-seed reporting protocol. It does not
change the method; it is purely a reporting alignment.

```bash
python run_all_replication.py --data_dir D:/pmh_data --artifact_root ./artifacts \
  --seeds 42 43 44 --tasks 01 04 08
```

Writes under `<artifact_root>/{models,results}/<slug>/seed_<N>/`. Delete the
seed folders to revert.

---

### Deferred ideas (never implemented)

These were in the original plan but deliberately not coded, because they
would require training-loop edits (not additive CLI wrappers) and we chose
to keep the current state trivially reversible:

1. **Task 01 `target_norm_pmh`** — mirror Task 03's target normalization.
2. **Task 04 multi-block PMH** — align the last 3 ViT blocks instead of one.
3. **Task 07 saliency-PMH loss** — regularize gradient cosine similarity
   during training, directly targeting the metric PMH uniquely wins on.
4. **Task 02/08 direct TDI loss** — replace the PMH surrogate with a
   differentiable TDI proxy.

Given that the paper-aligned E1 already beats the paper on Tasks 05 (+6.7pp)
and 07 (+8.3pp) and matches it on 01/02/04, the evidence says the paper CLI
is already at or near the achievable optimum on this environment. The
deferred ideas should only be pursued if a specific remaining gap is
identified and worth the risk of non-reversible edits.

---

### On the Task 03 regression

Separate from the plus variants, the paper-aligned re-run of Task 03 (with
`--target_norm_pmh`, `--output_pmh_weight 0.3`, `--pmh_ramp_epochs 90`,
epochs 100→120) produced **clean MAE 42.31**, worse than the earlier
simpler-CLI E1 which produced **22.20** (and better than paper 23.62).

This is a regression of the paper-aligned pipeline itself, not of any plus
variant, so it is **not** fixed by this removal. It needs its own decision:
either revert `run_03` to a simpler CLI that reproduces MAE 22.20, or keep
the paper-faithful CLI and accept the worse number. This is tracked outside
this document.
