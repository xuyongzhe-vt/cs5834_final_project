import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

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

def _make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

def _evaluate_model(model, X_train, y_train, X_eva, y_eva):
    y_train_pred = model.predict(X_train)
    y_eva_pred = model.predict(X_eva)

    return {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "eva_rmse": float(np.sqrt(mean_squared_error(y_eva, y_eva_pred))),
        "eva_r2": float(r2_score(y_eva, y_eva_pred)),
    }

def train_elasticnet_regression(
    train_df: pd.DataFrame,
    eva_df: pd.DataFrame,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
):
    """Train Elastic Net Regression (L1 + L2)."""

    train_df = train_df.copy()
    eva_df = eva_df.copy()

    # Target
    y_train = np.log1p(train_df[TARGET_COL].values)
    y_eva = np.log1p(eva_df[TARGET_COL].values)

    # Features
    X_train = train_df[FEATURE_COLS].copy()
    X_eva = eva_df[FEATURE_COLS].copy()

    preprocessor = _make_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)),
        ]
    )

    model.fit(X_train, y_train)
    metrics = _evaluate_model(model, X_train, y_train, X_eva, y_eva)
    return model, metrics
