import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from house_prices.config import Config
from house_prices.data import load_raw_data
from house_prices.preprocess import make_target
from house_prices.ensemble import (
    fit_final_ensemble_and_predict,
    evaluate_nested_stacking_ensemble,
    save_submission,
    save_bundle,
)

def main():
    cfg = Config()
    train_df, test_df = load_raw_data(cfg.raw_dir)

    raw_X_train = train_df.drop(columns=["SalePrice"]).copy()
    raw_X_test = test_df.copy()
    y = make_target(train_df)

    bundle, predictions = fit_final_ensemble_and_predict(
        raw_X_train, y, raw_X_test,
        inner_splits=cfg.inner_folds,
        random_state=cfg.random_state,
    )

    nested = evaluate_nested_stacking_ensemble(
        raw_X_train, y,
        outer_splits=cfg.outer_folds,
        inner_splits=cfg.inner_folds,
        random_state=cfg.random_state,
    )

    submission = save_submission(
        cfg.artifact_dir,
        test_ids=test_df["Id"].tolist(),
        pred_log=predictions["blend_test_prediction"],
    )
    summary = {
        "submission_rows": int(len(submission)),
        "base_scores": bundle["base_scores"],
        "nested_ensemble_summary": nested["summary"],
    }
    save_bundle(cfg.artifact_dir, bundle, summary)
    print(f"Saved submission to: {cfg.artifact_dir / 'submission.csv'}")

if __name__ == "__main__":
    main()
