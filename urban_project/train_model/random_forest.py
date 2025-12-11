import time
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import lightgbm as lgb


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
FEATURE_COLS = [
    "TOP_CATEGORY",
    "DISTANCE_FROM_HOME",
    "WKT_AREA_SQ_METERS",
    "DEVICE_TYPE",
    "Month",
]

TARGET_COL = "RAW_VISIT_COUNTS"

NUMERIC_FEATURES = ["DISTANCE_FROM_HOME", "WKT_AREA_SQ_METERS"]
CATEGORICAL_FEATURES = ["TOP_CATEGORY", "DEVICE_TYPE", "Month"]


# ---------------------------------------------------------------------
# Preprocessor & metrics
# ---------------------------------------------------------------------
def _make_preprocessor():
    """
    Standardize numeric features, one-hot encode categoricals.
    Sparse output is fine: LightGBM handles scipy.sparse efficiently.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _evaluate_model(model, X_train, y_train, X_eva, y_eva):
    """
    Evaluate a fitted model; y_* are in log1p space.
    Returns RMSE and R^2 in log1p space.
    """
    y_train_pred = model.predict(X_train)
    y_eva_pred = model.predict(X_eva)

    return {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "eva_rmse": float(np.sqrt(mean_squared_error(y_eva, y_eva_pred))),
        "eva_r2": float(r2_score(y_eva, y_eva_pred)),
    }


# ---------------------------------------------------------------------
# Wrapper so .predict(X_df) still works with DataFrames
# ---------------------------------------------------------------------
class RFWithPreprocessor:
    """
    Simple wrapper: applies preprocessor, then LightGBM RF regressor.
    Keeps the same .predict(DataFrame) API you had with sklearn Pipeline.
    """

    def __init__(self, preprocessor: ColumnTransformer, regressor: lgb.LGBMRegressor):
        self.preprocessor = preprocessor
        self.regressor = regressor

    def predict(self, X: pd.DataFrame):
        X_feats = X[FEATURE_COLS].copy()
        X_t = self.preprocessor.transform(X_feats)
        return self.regressor.predict(X_t)


# ---------------------------------------------------------------------
# LightGBM Random Forest training
# ---------------------------------------------------------------------
def train_lightgbm_random_forest(
    train_df: pd.DataFrame,
    eva_df: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: Optional[int] = -1,
    subsample: float = 0.8,         # row sampling (bagging_fraction)
    colsample_bytree: float = 0.8,  # feature_fraction
    random_state: int = 42,
    n_jobs: int = -1,
    verbose_eval: int = 50,         # print progress every N trees
):
    """
    Train a LightGBM *Random Forest* (not boosting).

    - boosting_type='rf' ⇒ true random forest mode
    - bagging (subsample) + feature subsampling
    - multi-core, histogram-based, fast
    - prints progress during training via `verbose_eval`

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with FEATURE_COLS + TARGET_COL
    eva_df : pd.DataFrame
        Evaluation data with FEATURE_COLS + TARGET_COL
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or -1
        Maximum depth of the trees. -1 means no limit.
    subsample : float
        Fraction of rows for each tree (0 < subsample <= 1).
    colsample_bytree : float
        Fraction of features per tree (0 < colsample_bytree <= 1).
    random_state : int
        Random seed.
    n_jobs : int
        Number of threads. -1 = use all cores.
    verbose_eval : int
        Print progress every `verbose_eval` trees.

    Returns
    -------
    model : RFWithPreprocessor
        Fitted model with .predict(DataFrame) method.
    metrics : dict
        RMSE and R^2 on train and eval (log1p space), plus fit time.
    """

    train_df = train_df.copy()
    eva_df = eva_df.copy()

    # Targets in log1p space (same as your original RF)
    y_train = np.log1p(train_df[TARGET_COL].values)
    y_eva = np.log1p(eva_df[TARGET_COL].values)

    # Features
    X_train = train_df[FEATURE_COLS].copy()
    X_eva = eva_df[FEATURE_COLS].copy()

    # Preprocess
    preprocessor = _make_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_eva_t = preprocessor.transform(X_eva)

    # LightGBM Random Forest
    rf = lgb.LGBMRegressor(
        boosting_type="rf",            # <<< RANDOM FOREST MODE
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,           # row sampling (bagging_fraction)
        colsample_bytree=colsample_bytree,  # feature subsampling (feature_fraction)
        bagging_freq=1,                # actually perform bagging
        objective="regression",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # Fit with eval_set so we get progress printed
    t0 = time.time()
    rf.fit(
        X_train_t,
        y_train,
        eval_set=[(X_eva_t, y_eva)],
        eval_metric="rmse",
    )
    fit_time = time.time() - t0

    # Wrap so .predict works on DataFrame and uses the same FEATURE_COLS
    model = RFWithPreprocessor(preprocessor, rf)

    # Metrics in log space (same style as your helper)
    metrics = _evaluate_model(model, X_train, y_train, X_eva, y_eva)
    metrics["fit_time_sec"] = float(fit_time)

    return model, metrics

