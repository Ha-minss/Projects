# House Prices — Leakage-Free Stacking & Reproducible Kaggle Pipeline

## Overview

This repository refactors the original exploratory notebook into a **reproducible, leakage-aware tabular ML project** for Kaggle's **House Prices: Advanced Regression Techniques** competition.

The original notebook already had strong ingredients:
- mixed-type tabular preprocessing
- domain-aware missing-value handling
- skew correction with log transforms
- multiple base regressors
- stacking and weighted blending

The problem was not the modeling idea itself.  
The problem was the **final blended CV score**: the notebook reported a very strong blended CV RMSE, but the last validation wrapper reused **prefit global models** during cross-validation. That can produce an **optimistically biased estimate**.

This repository fixes that.

---

## What this project does

### Business / modeling question
Predict `SalePrice` for Ames, Iowa houses from structured attributes:
- house size and quality
- lot and neighborhood features
- basement / garage / remodeling information
- sale timing and sale-type indicators
- many sparse categorical descriptors

This is a classic **tabular regression** task with:
- mixed numerical + categorical features
- structured missingness
- skewed continuous variables
- non-linear relationships
- strong interaction effects

### Data science objective
Build a pipeline that is:
1. **reproducible**
2. **honest about validation**
3. **good enough for Kaggle submission**
4. **clean enough for GitHub / portfolio presentation**

---

## Main improvement over the original notebook

### What was wrong before?
In the original notebook, the final `BlendEstimator` used:
- base models already fit on the full training set
- a stacked meta-model derived from globally available predictions
- `cross_val_score` on an estimator whose `fit()` did not retrain the ensemble inside each fold

That means the validation fold was **not fully isolated** from model training at the final blend stage.

### What is fixed here?
This repo uses **leakage-aware OOF generation** and a **nested stacking evaluation** strategy:

- preprocessing statistics are fit using training folds only
- base-model OOF predictions are generated fold-by-fold
- the meta-model is trained only on OOF predictions from the relevant training portion
- the final ensemble OOF estimate is obtained without reusing a globally fit blend object inside CV

This makes the evaluation slower, but much more defensible.

---

## Preprocessing strategy

The Ames dataset has many missing values that do **not** mean "unknown"; they often mean that a feature is absent.

### Domain-aware missing value handling
Examples:
- no garage -> garage-related columns become `"None"` or `0`
- no basement -> basement-related categorical fields become `"None"`
- `Functional` -> `"Typ"`
- `Electrical` -> `"SBrkr"`
- `KitchenQual` -> `"TA"`
- `Exterior1st`, `Exterior2nd`, `SaleType` -> training-fold mode
- `LotFrontage` -> neighborhood median estimated from the training fold only

### Categorical code handling
The following numeric-looking columns are treated as categories:
- `MSSubClass`
- `YrSold`
- `MoSold`

### Skew handling
Right-skewed continuous variables are log-transformed when:
- the variable is non-negative
- it behaves like a continuous quantity
- `abs(skew) > threshold`

### Target transformation
The target is modeled on log scale:

```python
y = np.log1p(SalePrice)
```

Predictions are converted back using:

```python
SalePrice = np.expm1(pred_log)
```

---

## Modeling strategy

Base learners:
- RidgeCV
- SVR
- GradientBoostingRegressor
- RandomForestRegressor
- XGBoost (if available)
- LightGBM (if available)

Final prediction logic:
1. generate base-model OOF predictions
2. train a meta-model on those OOF predictions
3. fit base learners on the full processed training set
4. generate test-set base predictions
5. predict test-set stack output
6. combine base and stack outputs using normalized blend weights

---

## Validation design

There are **two different validation layers** in this project.

### 1) Base-model OOF CV
Used to understand how each model behaves individually.

### 2) Leakage-free nested stacking OOF
Used to estimate the final blended ensemble more honestly.

Why nested?
Because the stacker itself is a learned model.  
If you fit the meta-model on all rows and then score on those same rows, you get a biased estimate for the stacked ensemble.

So this project uses:
- outer folds -> final validation targets
- inner folds -> training-fold OOF features for the meta-model

This is the correct way to evaluate a stacked ensemble if you want the final OOF estimate to be credible.

---

## Repository structure

```text
house-prices-leakage-free/
├─ README.md
├─ README_ko.md
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
├─ data/
│  ├─ raw/
│  │  ├─ train.csv
│  │  └─ test.csv
│  └─ processed/
├─ notebooks/
│  └─ house_prices_kaggle_leakage_free.ipynb
├─ scripts/
│  ├─ train.py
│  └─ predict.py
└─ src/
   └─ house_prices/
      ├─ __init__.py
      ├─ config.py
      ├─ data.py
      ├─ preprocess.py
      ├─ models.py
      ├─ ensemble.py
      └─ utils.py
```

---

## How to reproduce

### 1) Put Kaggle data here

```text
data/raw/train.csv
data/raw/test.csv
```

### 2) Install

```bash
python -m venv .venv
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 3) Run training summary

```bash
python scripts/train.py
```

This will:
- compute base-model OOF results
- compute leakage-free nested ensemble OOF results
- save `artifacts/training_summary.json`

### 4) Create submission

```bash
python scripts/predict.py
```

Output:
```text
artifacts/submission.csv
```

---

## Why this is stronger as a portfolio project

A good data science project is not just "I got a low score."

It should show that you can:
- turn an ad hoc notebook into a reusable project
- separate preprocessing, modeling, and evaluation
- identify suspicious validation design
- explain feature handling decisions
- communicate what is trustworthy and what is not

This repo is stronger than the original notebook because it demonstrates **evaluation discipline**, not only model experimentation.

---

## Recommended way to describe this project in an interview

> This project started as a Kaggle house-price notebook using classical tabular ensemble models.  
> When I refactored it for GitHub, I found that the final blended CV score was likely optimistic because the blend wrapper reused globally fitted models during cross-validation.  
> I rebuilt the project as a leakage-aware pipeline with fold-specific preprocessing, OOF base predictions, and nested evaluation for the stacked ensemble.  
> The main value of the project is not only the final prediction quality, but the fact that I converted exploratory work into a reproducible and defensible ML pipeline.

---

## Notes

- The leakage-free nested evaluation is much slower than a naive notebook-style blend.
- For faster experimentation, you can reduce fold counts in `Config`.
- For Kaggle scoring, the provided notebook trains the final full-data ensemble and exports a submission file.
