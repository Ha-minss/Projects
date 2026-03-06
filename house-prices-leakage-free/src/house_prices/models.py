from __future__ import annotations

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None


def get_base_model_builders(random_state: int = 42):
    builders = {
        "ridge": lambda: make_pipeline(
            RobustScaler(),
            RidgeCV(
                alphas=[
                    1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4,
                    1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100
                ],
                cv=5,
            ),
        ),
        "svr": lambda: make_pipeline(
            RobustScaler(),
            SVR(C=20, epsilon=0.008, gamma=0.0003),
        ),
        "gbr": lambda: GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=4,
            max_features="sqrt",
            min_samples_leaf=15,
            min_samples_split=10,
            loss="huber",
            random_state=random_state,
        ),
        "rf": lambda: RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if XGBRegressor is not None:
        builders["xgboost"] = lambda: XGBRegressor(
            tree_method="hist",
            learning_rate=0.02,
            n_estimators=1500,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
        )

    if LGBMRegressor is not None:
        builders["lightgbm"] = lambda: LGBMRegressor(
            objective="regression",
            learning_rate=0.02,
            num_leaves=16,
            n_estimators=1500,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.0,
            reg_lambda=0.0,
            min_child_samples=20,
            random_state=random_state,
            verbose=-1,
        )

    return builders


def get_meta_model_builder(random_state: int = 42):
    return lambda: Ridge(alpha=1.0, random_state=random_state)
