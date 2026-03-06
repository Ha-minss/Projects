from pathlib import Path
import pandas as pd

def load_raw_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Expected data/raw/train.csv and data/raw/test.csv. "
            "Please place the Kaggle competition files in that location."
        )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df
