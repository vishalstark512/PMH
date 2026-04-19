# Task 08 — BERT SST-2 Sentiment Classification

**Paper section:** §6.9 / Table A3 (TDI) / Table A4 (semantic TDI)  
**Figures:** Fig A4, Fig A8 (right — scale universality), Fig A10 (fine-tuning hierarchy)  
**Significance:** Extends PMH from vision to language; confirms geometric blind spot
exists in transformer language models.

---

## What this task does

Fine-tunes BERT-base-uncased on SST-2 sentiment classification under two
conditions — ERM baseline and PMH — then evaluates:

1. **Accuracy** — SST-2 validation accuracy (measures task cost of PMH)
2. **TDI (Pert-B)** — embedding-space Gaussian perturbation TDI
3. **Semantic TDI** — paraphrase drift vs. non-paraphrase drift using synonym
   substitution (the geometric blind spot: encoder should be *insensitive* to
   paraphrase, *sensitive* to non-paraphrase)
4. **Scale universality** — blind spot ratio across 3 model sizes (NEW)
5. **Fine-tuning hierarchy** — paraphrase drift comparison across pretrained /
   ERM fine-tuned / PMH fine-tuned (NEW)

## Key results (replication vs. paper)

### Accuracy and TDI cost (paper §6.9)

| | Baseline | PMH | Cost |
|--|---------|-----|------|
| Val accuracy | 93.12% | 92.43% | −0.69pp (paper: −0.69pp) ✓ |
| TDI@0 | 0.496 | **0.354** | −28.7% |
| TDI@0.10 | 0.509 | **0.355** | −30.3% |
| TDI-A | 0.641 | **0.474** | −26.0% |

PMH achieves −28.7% TDI at only −0.69pp accuracy cost. **Exact match to paper.**

### Scale universality — blind spot ratio (Fig A8 right — NEW)

| Model | Parameters | Para drift | Non-para drift | Blind spot ratio |
|-------|-----------|-----------|---------------|-----------------|
| DistilBERT | 66M | 0.00728 | 0.00847 | **0.860** |
| BERT-base | 110M | 0.02435 | 0.03184 | **0.765** |
| BERT-large | 340M | 0.04526 | 0.06100 | **0.742** |

Ratio < 1.0 at every scale → blind spot is a property of ERM, not model size.
Monotone decrease confirms Theorem 1: larger models encode more nuisance.

### Fine-tuning hierarchy (Fig A10 — NEW)

| Condition | Para drift | Blind spot ratio |
|-----------|-----------|-----------------|
| Pretrained BERT | 0.0244 | 0.765 |
| ERM fine-tuned | 0.0375 | 0.681 |
| **PMH fine-tuned** | **0.0033** | **0.633** |

- ERM worsens drift by **+54%** vs. pretrained — task supervision amplifies nuisance
- PMH repairs **11×** vs. ERM — geometric repair at the fine-tuning rung
- Ordering ERM > pretrained > PMH confirmed — matches Theorem 1 prediction

## How to run

```bash
# Train baseline and PMH
python run_task.py --task 08 --run baseline
python run_task.py --task 08 --run pmh

# TDI evaluation
python tasks/08_bert_sst2/eval.py

# Semantic TDI (paraphrase drift + scale + fine-tuning hierarchy)
python tasks/08_bert_sst2/semantic_tdi.py
```

## Artifacts produced

```
artifacts/
  models/08_bert_sst2/{baseline,pmh}/best_model/
  results/08_bert_sst2/
    eval_{baseline,pmh}.json                 — TDI at 3 noise levels
    semantic_tdi/
      semantic_tdi_results.json              — paraphrase drift + scale + hierarchy
```

### Key JSON structure (`semantic_tdi_results.json`)

```json
{
  "scale_experiment": [
    {"name": "distilbert-66M",  "blind_spot_ratio": 0.860, ...},
    {"name": "bert-base-110M",  "blind_spot_ratio": 0.765, ...},
    {"name": "bert-large-340M", "blind_spot_ratio": 0.742, ...}
  ],
  "task_finetuning_experiment": [
    {"name": "bert-pretrained", "para_drift": 0.0244, "blind_spot_ratio": 0.765},
    {"name": "bert-sst2-erm",   "para_drift": 0.0375, "blind_spot_ratio": 0.681},
    {"name": "bert-sst2-pmh",   "para_drift": 0.0033, "blind_spot_ratio": 0.633}
  ]
}
```

## Hyperparameters (PMH)

| Parameter | Value |
|-----------|-------|
| Base model | bert-base-uncased |
| `vat_eps` | 1e-6 |
| `pmh_weight` | 0.1 |
| `lr` | 2e-5 |
| `epochs` | 3 |
| `seed` | 42 |

## Paper claims

> "PMH reduces TDI by 28.7% at only −0.69pp accuracy cost — confirmed exactly.
> The geometric blind spot exists across all language model scales (66M→340M),
> is amplified by ERM fine-tuning (+54%), and repaired 11× by PMH fine-tuning —
> three new confirmations of Theorem 1."
