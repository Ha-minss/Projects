# Phase 2 Extraction Note

This repository contains a Phase 1 Kaggle notebook for the Retroviral Wall Challenge.

The notebook generates Leave-One-Family-Out (LOFO) out-of-fold predictions for the 57 public RTs and writes the final `submission.csv` required by Kaggle.

## Important Distinction

The final Kaggle `submission.csv` is a Phase 1 validation artifact.

It should not be reused directly for Phase 2.

For Phase 2, the modeling approach should be extracted, refit on the full 57 public RT dataset, and applied directly to blind RT candidates with the same feature schema.

## Intended Phase 2 Procedure

For Phase 2, the intended procedure is:

1. Load the full 57 public RT training dataset.
2. Apply the same basic feature engineering used in the notebook.
3. Refit the block-level logistic models on all 57 public RTs.
4. Refit the ESM2 PCA logistic branch on all 57 public RTs.
5. Compute block-level scores for each blind RT candidate.
6. Compute structure-aware gates:
   - canonical-like gate
   - LTR-like gate
   - Retron-like gate
   - false-positive-like penalty gate
7. Compute the structure-aware score.
8. Compute targeted correction components:
   - Retroviral false-positive penalty
   - LTR active correction
   - Retron active rescue
9. Apply the selected targeted correction logic.
10. Output one continuous `predicted_score` per blind RT candidate.

## What Should Be Reused

The extractable modeling components are:

- Step 3: Basic feature engineering helpers
- Step 4-1: Block-level score generation
- Step 4-2: Structure-aware soft scoring
- Step 4-3: Targeted correction components

The LOFO harness and nested validation in Step 4-4 are used for Phase 1 validation and model selection. They are not meant to be applied directly to blind Phase 2 candidates.

## Expected Input Schema

The Phase 2 candidate feature table is expected to follow the same schema as the public competition data.

Required identifiers:

- `rt_name`

For Phase 1 training and validation, the following columns are used:

- `rt_family`
- `active`
- `pe_efficiency_pct`

For Phase 2 blind prediction, labels are not required, but the feature columns should match the public data schema as closely as possible.

The model expects the same types of inputs used in the public dataset:

- handcrafted structural and biophysical features
- FoldSeek similarity features
- active-site / catalytic geometry features
- physicochemical features
- pocket/contact features
- electrostatic features
- thermal stability features
- ESM2 embeddings keyed by `rt_name`

Missing numeric feature values are handled by median imputation inside the scikit-learn pipelines. Some feature-based correction terms also use safe fallback helpers for missing columns.

## Phase 1 vs Phase 2

### Phase 1

The notebook performs:

```text
57 public RTs
→ Leave-One-Family-Out cross-validation
→ 57 out-of-fold predictions
→ submission.csv
→ CLS evaluation
```
