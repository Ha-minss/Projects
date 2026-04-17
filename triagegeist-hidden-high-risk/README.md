# Apparent Stability Is Not Safety

Recovering hidden high-risk cases among alert emergency department arrivals with an interpretable re-review safety-net.

## Overview

This project studies a narrow but clinically meaningful triage problem: **patients who appear stable at first contact, but still belong to the highest-urgency group**.

Instead of predicting triage severity for every ED arrival, the analysis focuses on patients recorded as **alert at triage** and asks:

> Can hidden high-risk cases be recovered from within the apparently stable triage population?

The modeling task defines **hidden high-risk** as `triage_acuity <= 2` inside the alert subgroup.

The main contribution is not model complexity, but **problem framing** and **operational translation**:
- broad baseline vs. strict baseline comparison
- ablation ladder to test shortcut dependence
- false-negative review for subtle urgent cases
- subgroup robustness checks
- workflow sensitivity analysis for re-review policy

## Why This Matters

A useful triage support tool does not need to replace clinician judgment. A more realistic role is to act as a **secondary safety-net** that helps clinicians decide which apparently stable patients should be re-reviewed first.

This repo shows that, in the competition dataset, much of the hidden-risk signal is already recoverable from basic physiological measurements observed at triage.

## Data

This project uses the **Triagegeist** competition dataset on Kaggle.

The raw competition files are **not included** in this repository. You must obtain access from the competition page and place the files locally under the expected folder structure.

Expected files:
- `train.csv`
- `test.csv`
- `chief_complaints.csv`
- `patient_history.csv`

See [`data/README.md`](data/README.md) for the expected layout.

## Main Approach

1. **Exploratory analysis**
   - verify clinically plausible severity gradients
   - identify hidden high-risk prevalence inside the alert subgroup
   - inspect structured missingness patterns

2. **Leakage and validity audit**
   - confirm no train/test patient overlap
   - exclude direct post-outcome leakage
   - review target-proximal but non-leak features

3. **Primary models**
   - **Broad baseline**: wide structured feature set
   - **Strict baseline**: 12 raw features only
   - **Lightweight tree benchmark**: sensitivity check against a simple non-linear model

4. **Appendix analyses**
   - ablation ladder
   - calibration / threshold policy
   - false-negative deep dive
   - subgroup robustness
   - workflow sensitivity analysis

## Key Results

Representative results from the final notebook:

- **Broad baseline**
  - ROC-AUC: **0.9993**
  - PR-AUC: **0.9871**

- **Strict baseline**
  - ROC-AUC: **0.9985**
  - PR-AUC: **0.9855**

- **Operational ranking view (strict baseline)**
  - Top **3%** re-review captures about **95%** of hidden high-risk cases
  - Top **5%** re-review captures about **99%** of hidden high-risk cases

- **Main interpretation**
  - performance remains extremely strong even after aggressive feature reduction
  - the main signal appears to be concentrated in basic physiology
  - the system is best interpreted as a **re-review prioritization tool**, not a standalone triage replacement

## Repository Structure

```text
triagegeist-hidden-high-risk/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ LICENSE
├─ notebooks/
│  └─ triagegeist_hidden_high_risk.ipynb
├─ data/
│  └─ README.md
├─ docs/
│  └─ project_writeup.md
└─ results/
   └─ README.md
```

## Reproducibility

### Option A: Run on Kaggle

This is the easiest and most reliable path.

1. Open the notebook in Kaggle.
2. Attach the Triagegeist competition data.
3. Run all cells from top to bottom.

### Option B: Run locally

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place the raw competition files under:

```text
data/raw/
  - train.csv
  - test.csv
  - chief_complaints.csv
  - patient_history.csv
```

4. Update the notebook data path if needed.
5. Run the notebook end-to-end.

## Recommended Deployment Posture

If implemented in practice, this system should be used only as a **secondary review prioritization aid** after initial triage.

Recommended interpretations from the notebook:
- **Top 3% review** when review capacity is limited
- **Top 5% review** or **threshold >= 0.50** when minimizing missed urgent cases is the main priority

The model should **not** be used as the sole basis for definitive triage escalation or de-escalation.

## Limitations

- The competition dataset is synthetic.
- Performance may be inflated by unusually clean physiological separation.
- The broad baseline may still benefit from target-proximal summaries and workflow-related missingness patterns.
- Real deployment would require external validation, local recalibration, and clinician workflow testing.

## Public Writeup

A polished project description version is included in [`docs/project_writeup.md`](docs/project_writeup.md).

## License

Competition data remains subject to Kaggle competition terms and is not redistributed here.
