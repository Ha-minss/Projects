# Bank Marketing (Profit-Optimized) — CatBoost

This repo turns the original Colab notebook into a **reproducible, CLI-based pipeline**:
- Train a CatBoost model for `y` (subscription: yes/no)
- Compute **OOF (out-of-fold)** probabilities with Stratified K-Fold CV
- Choose a **profit-maximizing decision threshold** (call / no-call)
- Evaluate on a labeled test set (if provided)
- Export `call_target_list.csv` for a call-center style targeting workflow

## 1) Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Data

Place your files here:

```
data/
  train.csv
  test.csv          # optional (if it has `y`, we evaluate; if not, we only predict)
```

Default CSV separator is `;` (as in the original notebook).

## 3) Run

### Train + OOF CV + Profit threshold + (Optional) test evaluation
```bash
python -m scripts.run_experiment   --train_csv data/train.csv   --test_csv  data/test.csv   --sep ";"   --revenue 20   --cost 1   --outdir artifacts
```

### If you only have train.csv (no test labels)
```bash
python -m scripts.run_experiment   --train_csv data/train.csv   --sep ";"   --outdir artifacts
```

## 4) What the pipeline does

### Target and leakage handling
- Target: `y` is mapped to {yes: 1, no: 0}
- `duration` is dropped (common leakage feature in this dataset)
- Additional drops (as in the notebook): `age`, `default`, `day`, `pdays`

### Feature engineering
- Adds:
  - `never_contacted = 1(pdays == -1)`

### Modeling
- CatBoostClassifier with categorical features auto-handled via `cat_features` indices
- OOF predictions via StratifiedKFold (default: 5 folds)
- Metrics:
  - ROC-AUC
  - PR-AUC (Average Precision)

### Profit-based threshold
We treat the decision as:
- If predicted probability >= threshold → **call**
- Else → **don’t call**

Profit definition:
- Each true positive (conversion) gives `revenue`
- Each call costs `cost` (applies to both TP and FP)
- Profit = `TP * revenue - (TP + FP) * cost`

We search thresholds over a grid (0..1) and pick the max-profit threshold on OOF predictions.

## 5) Outputs

All outputs go to `artifacts/`:

- `metrics.json` — main metrics + best threshold summary
- `oof_predictions.csv` — OOF probability per training row
- `call_target_list.csv` — (if test provided) rows with `call_flag=1`
- Plots:
  - `profit_curve_oof.png`
  - `precision_recall_oof.png`
  - `roc_test.png`, `pr_test.png` (only if test has labels)

## 6) Notes / Reproducibility

- Fixed random seed is used across NumPy / CV / CatBoost (`--seed`, default 42).
- Results will still vary slightly across machines if library versions differ;
  pin versions in `requirements.txt` if you need exact matching.

---

If you want, I can also:
- convert your EDA section into a separate `scripts/eda.py` that saves plots,
- or package this as a pip-installable project (pyproject + module entrypoint).
