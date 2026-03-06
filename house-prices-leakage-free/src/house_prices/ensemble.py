from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from .models import get_base_model_builders, get_meta_model_builder
from .preprocess import AmesPreprocessor


BASE_BLEND_WEIGHTS = {
    "ridge": 0.10,
    "svr": 0.15,
    "gbr": 0.15,
    "rf": 0.10,
    "xgboost": 0.15,
    "lightgbm": 0.15,
}
STACK_WEIGHT = 0.20


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _normalized_weights(active_model_names):
    raw = {name: BASE_BLEND_WEIGHTS.get(name, 0.0) for name in active_model_names}
    total_base = sum(raw.values())
    if total_base <= 0:
        raw = {name: 1.0 for name in active_model_names}
        total_base = sum(raw.values())
    base_target = 1.0 - STACK_WEIGHT
    norm = {name: (value / total_base) * base_target for name, value in raw.items()}
    return norm


def generate_base_oof_predictions(
    raw_X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    builders = get_base_model_builders(random_state=random_state)

    oof = {name: np.zeros(len(raw_X), dtype=float) for name in builders}
    fold_scores = {name: [] for name in builders}

    for train_idx, valid_idx in cv.split(raw_X, y):
        X_tr_raw = raw_X.iloc[train_idx].reset_index(drop=True)
        X_va_raw = raw_X.iloc[valid_idx].reset_index(drop=True)
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        y_va = y.iloc[valid_idx].reset_index(drop=True)

        preprocessor = AmesPreprocessor()
        X_tr = preprocessor.fit_transform(X_tr_raw)
        X_va = preprocessor.transform(X_va_raw)

        for name, builder in builders.items():
            model = builder()
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            oof[name][valid_idx] = pred
            fold_scores[name].append(rmse(y_va, pred))

    scores = {
        name: {
            "fold_rmse": vals,
            "mean_rmse": float(np.mean(vals)),
            "std_rmse": float(np.std(vals)),
        }
        for name, vals in fold_scores.items()
    }
    return pd.DataFrame(oof), scores


def fit_full_base_models(
    raw_X_train: pd.DataFrame,
    y: pd.Series,
    raw_X_test: pd.DataFrame,
    random_state: int = 42,
):
    builders = get_base_model_builders(random_state=random_state)
    preprocessor = AmesPreprocessor()
    X_train = preprocessor.fit_transform(raw_X_train)
    X_test = preprocessor.transform(raw_X_test)

    fitted_models = {}
    test_preds = {}
    for name, builder in builders.items():
        model = builder()
        model.fit(X_train, y)
        fitted_models[name] = model
        test_preds[name] = model.predict(X_test)

    return preprocessor, fitted_models, pd.DataFrame(test_preds)


def evaluate_nested_stacking_ensemble(
    raw_X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 5,
    random_state: int = 42,
):
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    builders = get_base_model_builders(random_state=random_state)
    meta_builder = get_meta_model_builder(random_state=random_state)
    active_names = list(builders.keys())
    base_weights = _normalized_weights(active_names)

    oof_base = {name: np.zeros(len(raw_X), dtype=float) for name in active_names}
    oof_stack = np.zeros(len(raw_X), dtype=float)
    oof_blend = np.zeros(len(raw_X), dtype=float)
    outer_fold_metrics = []

    for fold_no, (train_idx, valid_idx) in enumerate(outer_cv.split(raw_X, y), start=1):
        raw_outer_train = raw_X.iloc[train_idx].reset_index(drop=True)
        raw_outer_valid = raw_X.iloc[valid_idx].reset_index(drop=True)
        y_outer_train = y.iloc[train_idx].reset_index(drop=True)
        y_outer_valid = y.iloc[valid_idx].reset_index(drop=True)

        # inner OOF features for meta-model training
        inner_meta_X, _ = generate_base_oof_predictions(
            raw_outer_train, y_outer_train,
            n_splits=inner_splits,
            random_state=random_state + fold_no,
        )
        meta_model = meta_builder()
        meta_model.fit(inner_meta_X, y_outer_train)

        # fit base models on full outer-train
        outer_preprocessor = AmesPreprocessor()
        X_outer_train = outer_preprocessor.fit_transform(raw_outer_train)
        X_outer_valid = outer_preprocessor.transform(raw_outer_valid)

        valid_base_pred_frame = {}
        for name, builder in builders.items():
            model = builder()
            model.fit(X_outer_train, y_outer_train)
            pred = model.predict(X_outer_valid)
            oof_base[name][valid_idx] = pred
            valid_base_pred_frame[name] = pred

        valid_base_pred_frame = pd.DataFrame(valid_base_pred_frame)
        stack_pred = meta_model.predict(valid_base_pred_frame)
        blend_pred = np.zeros(len(valid_base_pred_frame), dtype=float)

        for name in active_names:
            blend_pred += base_weights[name] * valid_base_pred_frame[name].values
        blend_pred += STACK_WEIGHT * stack_pred

        oof_stack[valid_idx] = stack_pred
        oof_blend[valid_idx] = blend_pred

        outer_fold_metrics.append({
            "fold": fold_no,
            "rmse_stack": rmse(y_outer_valid, stack_pred),
            "rmse_blend": rmse(y_outer_valid, blend_pred),
        })

    summary = {
        "base_weights_normalized": base_weights,
        "stack_weight": STACK_WEIGHT,
        "outer_fold_metrics": outer_fold_metrics,
        "overall_rmse_stack": rmse(y, oof_stack),
        "overall_rmse_blend": rmse(y, oof_blend),
    }
    return {
        "oof_base": pd.DataFrame(oof_base),
        "oof_stack": oof_stack,
        "oof_blend": oof_blend,
        "summary": summary,
    }


def fit_final_ensemble_and_predict(
    raw_X_train: pd.DataFrame,
    y: pd.Series,
    raw_X_test: pd.DataFrame,
    inner_splits: int = 5,
    random_state: int = 42,
):
    builders = get_base_model_builders(random_state=random_state)
    active_names = list(builders.keys())
    base_weights = _normalized_weights(active_names)
    meta_builder = get_meta_model_builder(random_state=random_state)

    train_meta_X, base_scores = generate_base_oof_predictions(
        raw_X_train, y, n_splits=inner_splits, random_state=random_state
    )

    preprocessor, fitted_base_models, test_base_preds = fit_full_base_models(
        raw_X_train, y, raw_X_test, random_state=random_state
    )

    meta_model = meta_builder()
    meta_model.fit(train_meta_X, y)
    stack_test_pred = meta_model.predict(test_base_preds)

    blend_test_pred = np.zeros(len(test_base_preds), dtype=float)
    for name in active_names:
        blend_test_pred += base_weights[name] * test_base_preds[name].values
    blend_test_pred += STACK_WEIGHT * stack_test_pred

    bundle = {
        "preprocessor": preprocessor,
        "base_models": fitted_base_models,
        "meta_model": meta_model,
        "active_model_names": active_names,
        "base_weights": base_weights,
        "stack_weight": STACK_WEIGHT,
        "base_scores": base_scores,
        "train_meta_columns": train_meta_X.columns.tolist(),
    }
    predictions = {
        "base_test_predictions": test_base_preds,
        "stack_test_prediction": stack_test_pred,
        "blend_test_prediction": blend_test_pred,
    }
    return bundle, predictions


def save_submission(artifact_dir: Path, test_ids, pred_log):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": np.expm1(pred_log),
    })
    submission.to_csv(artifact_dir / "submission.csv", index=False)
    return submission


def save_bundle(artifact_dir: Path, bundle: dict, summary: dict):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_dir / "ensemble_bundle.joblib")
    with open(artifact_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
