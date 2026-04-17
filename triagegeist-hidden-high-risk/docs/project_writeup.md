### Clinical Problem Statement

Emergency triage is a fast, high-stakes process performed under time pressure, incomplete information, and substantial cognitive load. In practice, the most obviously critical patients are often recognized quickly. A harder and clinically more important problem is under-recognized urgency: patients who appear stable at first contact but still require urgent care.

This project focuses on that narrower and more realistic triage problem. Rather than predicting severity for every emergency department arrival, I focus specifically on patients recorded as alert at triage (`mental_status_triage == "alert"`) and ask a more targeted question:

**Can hidden high-risk cases be recovered from within the apparently stable triage population?**

For this project, hidden high-risk is defined as patients in the alert subgroup whose final triage acuity is still **1 or 2** (`triage_acuity <= 2`). This framing is intentionally narrow because it targets a concrete triage workflow problem rather than a broad generic classification task.

---

### Why This Matters

A clinically useful triage support tool does not need to replace clinician judgment. In many real workflows, a more realistic and safer role is to act as a **secondary safety-net**: helping clinicians decide which apparently stable patients should be re-reviewed first.

This distinction matters because “stable-looking” is not the same as “safe.” If hidden urgent cases can be identified from routine intake signals, then even a simple and interpretable model could support safer prioritization under workload pressure.

The intended role of the system in this project is therefore **queue prioritization for secondary review** among patients who initially appear stable, not automated final triage assignment.

---

### Data and Task Definition

I used the provided Triagegeist competition dataset, which is a synthetic emergency department triage dataset designed for research and education. I merged the structured triage table with chief complaint records and patient history by patient ID to create a patient-level analytical dataset.

The analysis uses:
- structured intake variables
- vital signs
- demographics and arrival context
- chief complaint category and text
- patient history variables

The modeling population is restricted to the alert subgroup. Within this subgroup, the task becomes a binary classification problem: recover urgent cases from inside an apparently stable population.

---

### Exploratory Findings

Exploratory analysis showed that the dataset preserves clinically plausible severity gradients. Higher-acuity patients consistently had lower blood pressure, lower oxygen saturation, lower Glasgow Coma Scale score, and higher heart rate, respiratory rate, temperature, NEWS2, and shock index. This suggests that the labels remain aligned with recognizable physiological severity patterns.

A key EDA finding was that the alert subgroup still contains a small but clinically meaningful hidden high-risk population. This directly supports the project hypothesis that apparent stability at triage does not necessarily imply safety.

I also found that missingness was not fully random. Some missing vital-sign patterns were concentrated in lower-acuity patients, suggesting that missingness may partly encode aspects of triage workflow rather than pure measurement noise. This became important when interpreting broad-model performance.

---

### Leakage Audit and Validity Checks

Before interpreting model results, I performed explicit leakage and validity checks.

I verified that:
- patient IDs are unique within both train and test
- there is no train-test patient overlap
- the train/validation split inside the alert subgroup is clean
- direct leakage variables such as final acuity, post-outcome disposition, and emergency department length of stay are excluded from the model features

I also reviewed whether some broad baseline variables were highly target-proximal rather than truly invalid, especially derived physiological summaries such as NEWS2, shock index, mean arterial pressure, and pulse pressure. These are not direct leakage, but they can make the task easier. This motivated a stricter comparison model.

---

### Methodology

I deliberately used **logistic regression** as the primary model.

This choice was intentional. The goal of this project was not to maximize model complexity, but to test whether hidden urgency is already recoverable from routine intake physiology in a clinically interpretable and reproducible way. Because the competition emphasizes methodological clarity, clinical relevance, and reproducibility, an interpretable baseline was an appropriate starting point.

I built two main models:

#### Broad baseline
The broad baseline used a wide structured feature set from the alert subgroup, excluding identifiers, direct target variables, post-outcome variables, and raw free-text complaint strings. It included demographics, arrival context, vital signs, comorbidity/history variables, selected derived severity summaries, and selected missingness indicators.

#### Strict baseline
The strict baseline was designed as a more conservative test. It used only 12 raw features:
- age
- sex
- arrival mode
- arrival hour
- systolic blood pressure
- diastolic blood pressure
- heart rate
- respiratory rate
- temperature
- oxygen saturation
- Glasgow Coma Scale score
- pain score

This strict design removed derived severity summaries, workflow shortcut flags, and broader engineered features. If the strict model still performed well, that would imply that the main predictive signal is already present in basic triage physiology.

Both models used the same train/validation split, the same preprocessing pipeline, median imputation for numeric variables, one-hot encoding for categorical variables, and class-balanced logistic regression.

As an additional sensitivity check, I also compared the strict logistic baseline to a lightweight tree-based benchmark. The tree model was slightly stronger in overall discrimination, but logistic regression retained higher recall at the default threshold and very similar top-k ranking performance. That result strengthened the case for using logistic regression as the primary model because it preserved practical performance while offering much clearer interpretability.

---

### Results

The broad baseline performed extremely strongly on the alert subgroup:
- **ROC-AUC: 0.9993**
- **PR-AUC: 0.9871**

At the default threshold of 0.50, it recovered **98.2%** of hidden high-risk cases.

The strict baseline performed almost as well:
- **ROC-AUC: 0.9985**
- **PR-AUC: 0.9855**

This small gap is one of the most important findings in the project. The feature count dropped from **64 to 12**, but performance changed only slightly. This suggests that predictive strength is not driven only by engineered summary variables or workflow shortcut features. Instead, most of the signal appears to come directly from basic physiological measurements observed at triage.

Additional ablation testing strengthened this interpretation. Removing derived severity summaries caused only a very small performance drop, and removing missingness indicators produced almost no further change. Even an ultra-strict version using only age and raw vital signs remained highly competitive. This suggests that hidden high-risk cases in this dataset are largely recoverable from basic physiology alone.

At the same time, these extremely strong results should be interpreted cautiously. The synthetic dataset appears to preserve unusually clean physiological separation, which likely makes the task easier than real deployment would be.

---

### Interpretation and Operational Meaning

The strict model’s coefficients were clinically interpretable. Higher temperature, higher pain score, higher respiratory rate, and higher heart rate increased predicted risk, while higher Glasgow Coma Scale score, higher blood pressure, and higher oxygen saturation decreased predicted risk. This suggests that the model is learning recognizable physiological instability patterns rather than relying mainly on opaque correlations.

False negative review was also informative. The small set of missed urgent cases tended to be older, had fully normal Glasgow Coma Scale scores, and showed less extreme abnormalities than the correctly flagged urgent patients. Some belonged to dermatological, cardiovascular, neurological, and musculoskeletal complaint categories, suggesting that certain urgent presentations may be harder to detect when physiology alone does not fully express the underlying risk. Importantly, the broader baseline recovered several of the strict model’s missed cases, indicating that broader structured context may still add value for the hardest borderline presentations.

The ranking-based operating view is especially strong. For the strict baseline:
- reviewing the **top 3%** of alert patients captured about **95%** of all hidden high-risk cases
- reviewing the **top 5%** captured about **99%**

This means the model is most useful not as a standalone triage rule, but as a **re-review prioritization tool**.

A **top 3% review policy** is the strongest recommendation when review capacity is limited, because it keeps the review pool extremely small while still recovering most hidden high-risk cases. A **threshold >= 0.50** policy is a reasonable balanced option. A **top 5% review policy** is more appropriate when minimizing missed urgent cases is the highest priority.

Calibration analysis further supports this deployment framing. The model behaves more convincingly as a ranking tool than as a literal probability estimator, especially in the highest-risk tail. Therefore, the safest interpretation is to use it for prioritization, not as a standalone calibrated risk engine.

---

### Robustness

Subgroup checks across sex, age group, arrival mode, and chief complaint system showed that performance remains strong across most major subgroups rather than being concentrated in one narrow slice of the alert population.

At the same time, several weaker pockets of performance were visible, especially in dermatological presentations and, to a lesser extent, in neurological, cardiovascular, musculoskeletal, and brought-by-family cases. These patterns are consistent with the false negative review and suggest that certain urgent presentations may be less detectable when physiology alone does not fully express the risk.

Because some subgroup counts are small, these checks should be interpreted as robustness analyses rather than definitive fairness claims. Still, the overall picture is reassuring: the model appears broadly stable, with the remaining blind spots concentrated in a limited number of subtler complaint types.

---

### Recommended Deployment Posture

If implemented in practice, this system should be used only as a **secondary review prioritization aid** after initial triage, not as a standalone triage replacement.

A reasonable default recommendation would be:
- **top 3% review** when review capacity is limited
- **top 5% review** or **threshold >= 0.50** when minimizing missed urgent cases is the main priority

The model should **not** be used as the sole basis for definitive triage escalation or de-escalation. Before any deployment, it would require site-specific recalibration, external validation on real emergency department data, and workflow testing with clinicians.

---

### Limitations

This project has several important limitations.

First, the competition dataset is synthetic, so strong internal performance does not automatically imply real-world external validity.

Second, the unusually small performance loss under aggressive feature reduction suggests that the dataset may preserve very clean physiological separation between low-risk and hidden high-risk cases.

Third, the broad baseline may still benefit from target-proximal summaries and workflow-related missingness patterns, even though the strict and ablation analyses reduce concern that these factors alone explain the findings.

Finally, this project evaluates retrospective predictive performance, not prospective clinical utility. Real-world deployment would still require external validation, calibration review, threshold tuning under local workflow constraints, and clinician-facing testing.

---

### Reproducibility

The notebook is designed to run end-to-end on the provided competition files using a fixed random seed and explicit preprocessing steps. All reported results in this submission are generated directly from the public competition dataset and the attached Kaggle notebook.

---

### Conclusion

This project shows that **apparent stability at triage does not always imply safety**. Even among patients recorded as alert, hidden high-risk cases can still be recovered with high accuracy using a simple and interpretable physiology-based model.

More importantly, the strong performance of the strict baseline, the ablation ladder, and the subgroup and operational sensitivity analyses together suggest that the key signal is already present in a small set of basic intake variables and vital signs. In that sense, the most valuable contribution of this work is not model complexity, but problem framing: recovering urgent cases inside the apparently stable triage population and translating that into a realistic re-review workflow.
