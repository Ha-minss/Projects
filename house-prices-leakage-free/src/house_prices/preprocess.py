from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CODE_AS_CATEGORY = ["MSSubClass", "YrSold", "MoSold"]
DISCRETE_EXCLUDE_FOR_LOG = [
    "MSSubClass", "KitchenAbvGr", "BedroomAbvGr", "TotRmsAbvGrd",
    "HalfBath", "FullBath", "BsmtHalfBath", "BsmtFullBath",
    "GarageCars", "Fireplaces", "MiscVal"
]

FIXED_FILL_STRING = {
    "Functional": "Typ",
    "Electrical": "SBrkr",
    "KitchenQual": "TA",
    "PoolQC": "None",
}

MODE_FILL_COLS = ["Exterior1st", "Exterior2nd", "SaleType"]
GARAGE_NUM_COLS = ["GarageYrBlt", "GarageArea", "GarageCars"]
GARAGE_CAT_COLS = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
BSMT_CAT_COLS = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]


class AmesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold: float = 0.8):
        self.skew_threshold = skew_threshold

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])

        for col in CODE_AS_CATEGORY:
            if col in df.columns:
                df[col] = df[col].astype(str)

        self.columns_in_ = df.columns.tolist()

        self.mode_map_ = {}
        for col in MODE_FILL_COLS:
            if col in df.columns:
                mode = df[col].mode(dropna=True)
                self.mode_map_[col] = mode.iloc[0] if len(mode) > 0 else "None"

        self.mszoning_group_mode_ = {}
        self.mszoning_global_mode_ = "None"
        if "MSZoning" in df.columns and "MSSubClass" in df.columns:
            tmp = df[["MSSubClass", "MSZoning"]].copy()
            global_mode = tmp["MSZoning"].mode(dropna=True)
            self.mszoning_global_mode_ = global_mode.iloc[0] if len(global_mode) > 0 else "None"
            for key, sub in tmp.groupby("MSSubClass"):
                mode = sub["MSZoning"].mode(dropna=True)
                self.mszoning_group_mode_[str(key)] = mode.iloc[0] if len(mode) > 0 else self.mszoning_global_mode_

        self.lotfrontage_by_neighborhood_ = {}
        self.lotfrontage_global_median_ = 0.0
        if "LotFrontage" in df.columns:
            self.lotfrontage_global_median_ = float(df["LotFrontage"].median())
        if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
            med = df.groupby("Neighborhood")["LotFrontage"].median()
            self.lotfrontage_by_neighborhood_ = med.to_dict()

        numeric = df.select_dtypes(include=[np.number]).copy()
        for col in DISCRETE_EXCLUDE_FOR_LOG:
            if col in numeric.columns:
                numeric = numeric.drop(columns=[col])

        self.skewed_cols_ = []
        for col in numeric.columns:
            if col == "SalePrice":
                continue
            if numeric[col].dropna().empty:
                continue
            if numeric[col].min(skipna=True) < 0:
                continue
            skew = numeric[col].skew()
            if pd.notna(skew) and abs(float(skew)) > self.skew_threshold:
                self.skewed_cols_.append(col)

        transformed = self._basic_transform(df, fit_mode=True)
        transformed = self._apply_log_transform(transformed)

        dummies = pd.get_dummies(transformed, drop_first=True)
        self.output_columns_ = dummies.columns.tolist()
        self.numeric_fill_values_ = {}
        for col in dummies.columns:
            if pd.api.types.is_numeric_dtype(dummies[col]):
                self.numeric_fill_values_[col] = float(dummies[col].median()) if not dummies[col].dropna().empty else 0.0
        return self

    def _basic_transform(self, df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        out = df.copy()

        for col in CODE_AS_CATEGORY:
            if col in out.columns:
                out[col] = out[col].astype(str)

        for col, fill_value in FIXED_FILL_STRING.items():
            if col in out.columns:
                out[col] = out[col].fillna(fill_value)

        for col in MODE_FILL_COLS:
            if col in out.columns:
                out[col] = out[col].fillna(self.mode_map_.get(col, "None"))

        if "MSZoning" in out.columns and "MSSubClass" in out.columns:
            def fill_zone(row):
                if pd.notna(row["MSZoning"]):
                    return row["MSZoning"]
                key = str(row["MSSubClass"])
                return self.mszoning_group_mode_.get(key, self.mszoning_global_mode_)
            out["MSZoning"] = out.apply(fill_zone, axis=1)

        for col in GARAGE_NUM_COLS:
            if col in out.columns:
                out[col] = out[col].fillna(0)

        for col in GARAGE_CAT_COLS + BSMT_CAT_COLS:
            if col in out.columns:
                out[col] = out[col].fillna("None")

        if "LotFrontage" in out.columns and "Neighborhood" in out.columns:
            out["LotFrontage"] = out.apply(
                lambda row: self._fill_lotfrontage(row["LotFrontage"], row["Neighborhood"]), axis=1
            )
            out["LotFrontage"] = out["LotFrontage"].fillna(self.lotfrontage_global_median_)

        obj_cols = out.select_dtypes(include="object").columns.tolist()
        if obj_cols:
            out[obj_cols] = out[obj_cols].fillna("None")

        num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            if out[col].isna().any():
                fill_value = 0.0 if fit_mode else self.numeric_fill_values_.get(col, 0.0)
                out[col] = out[col].fillna(fill_value)

        return out

    def _fill_lotfrontage(self, value, neighborhood):
        if pd.notna(value):
            return value
        if neighborhood in self.lotfrontage_by_neighborhood_ and pd.notna(self.lotfrontage_by_neighborhood_[neighborhood]):
            return self.lotfrontage_by_neighborhood_[neighborhood]
        return self.lotfrontage_global_median_

    def _apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.skewed_cols_:
            if col in out.columns:
                out[col] = np.log1p(out[col].clip(lower=0))
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])

        for col in self.columns_in_:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.columns_in_].copy()

        df = self._basic_transform(df, fit_mode=False)
        df = self._apply_log_transform(df)

        dummies = pd.get_dummies(df, drop_first=True)
        dummies = dummies.reindex(columns=self.output_columns_, fill_value=0)
        return dummies.astype(float)


def make_target(train_df: pd.DataFrame) -> pd.Series:
    return np.log1p(train_df["SalePrice"]).copy()
