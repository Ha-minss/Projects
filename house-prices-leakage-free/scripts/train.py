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
    generate_base_oof_predictions,
    evaluate_nested_stacking_ensemble,
    save_bundle,
)

def main():
    cfg = Config()
    train_df, test_df = load_raw_data(cfg.raw_dir)

    raw_X = train_df.drop(columns=["SalePrice"]).copy()
    y = make_target(train_df)

    base_oof, base_scores = generate_base_oof_predictions(
        raw_X, y, n_splits=cfg.inner_folds, random_state=cfg.random_state
    )
    nested = evaluate_nested_stacking_ensemble(
        raw_X, y,
        outer_splits=cfg.outer_folds,
        inner_splits=cfg.inner_folds,
        random_state=cfg.random_state,
    )

    summary = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_raw_features": int(raw_X.shape[1]),
        "base_scores": base_scores,
        "nested_ensemble_summary": nested["summary"],
    }
    save_bundle(cfg.artifact_dir, {"note": "training-only summary run"}, summary)
    print(summary)

if __name__ == "__main__":
    main()
