# Retroviral Wall Challenge Submission

This repository contains my solution for the Mandrake Bio Retroviral Wall Challenge.

## Overview

The goal of this competition is to predict which reverse transcriptases (RTs) are active for prime editing and to rank them by prime editing efficiency.

This solution uses a structure-aware targeted correction pipeline evaluated with Leave-One-Family-Out (LOFO) cross-validation.

## Main Notebook

```text
retroviral_wall_challenge_final.ipynb
```

The notebook runs end-to-end in the Kaggle environment and generates:

```text
submission.csv
```

with the required format:

```text
rt_name,predicted_score
```

## Modeling Pipeline

The final pipeline consists of:

1. Data loading and validation
2. Official-style CLS metric implementation
3. Basic feature engineering
4. Block-level LOFO models:
   - global structure
   - canonical retroviral similarity
   - noncanonical similarity
   - active-site geometry
   - thermal stability
   - physicochemical features
   - contacts and pocket features
   - exposure and quality features
   - electrostatics
   - ESM2 PCA embedding score
5. Structure-aware soft scoring
6. Targeted correction:
   - Retroviral false-positive penalty
   - LTR active correction
   - Retron active rescue
7. Nested targeted correction validation
8. Final `submission.csv` generation

## Final Submitted Model

The final submission uses:

```text
nested_targeted_correction_score
```

This is a conservative validation setting where correction weights are selected without using the held-out family.

## Expected Input

The notebook expects the official Kaggle competition input directory:

```text
/kaggle/input/competitions/retroviral-challenge-predict
```

The pipeline expects the same public feature schema used in the competition data, including:

- `rt_name`
- `rt_family`
- `active`
- `pe_efficiency_pct`
- handcrafted structural and biophysical features
- `esm2_embeddings.npz`

For Phase 2, labels are not required for prediction, but the feature schema should match the public competition data. Missing numeric values are handled through imputation or safe feature-access helper functions where applicable.

## Reproducibility

Run the notebook from top to bottom in the Kaggle environment.

Required Python packages:

```text
numpy
pandas
scikit-learn
```

## Phase 2 Extraction Note

The LOFO cross-validation harness is used for Phase 1 evaluation.

For Phase 2, the modeling approach can be extracted as:

1. Train the block-level models on all 57 public RTs
2. Build structure-aware scores
3. Apply targeted correction logic
4. Predict continuous scores for new RT candidates with the same feature schema

The code is organized so the feature engineering, model scoring, and LOFO validation steps are clearly separated.

## Competition Data

Competition input data is not included in this repository. Please use the official Kaggle competition dataset.
