import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from urban_project.train_model.linear_regression import train_linear_regression
from urban_project.train_model.lasso_regression import train_lasso_regression
from urban_project.train_model.ridge_regression import train_ridge_regression
from urban_project.train_model.elasticnet_regression import train_elasticnet_regression
from urban_project.train_model.random_forest import train_lightgbm_random_forest

# ONLY keep the required columns
FEATURE_COLS = [
    "TOP_CATEGORY",
    "DISTANCE_FROM_HOME",
    "WKT_AREA_SQ_METERS",
    "DEVICE_TYPE",
    "Month",  # <-- NOTE: Correct case
]

TARGET_COL = "RAW_VISIT_COUNTS"
ID_COL = "PLACEKEY"  # used for grouping in split


def _split_with_all_months(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    max_tries: int = 100,
):
    """Ensure BOTH train & test contain all Month values."""
    if "Month" not in df.columns:
        raise ValueError("Expected column 'Month' not found in dataframe.")

    months_full = set(df["Month"].unique())
    groups = df[ID_COL]

    splitter = GroupShuffleSplit(
        n_splits=max_tries,
        test_size=test_size,
        random_state=random_state,
    )

    for train_idx, test_idx in splitter.split(df, groups=groups):
        train_months = set(df.iloc[train_idx]["Month"].unique())
        test_months = set(df.iloc[test_idx]["Month"].unique())
        if train_months == months_full and test_months == months_full:
            return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    # fallback
    train_idx, test_idx = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        ).split(df, groups=groups)
    )
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _subsample_by_month(
    df: pd.DataFrame,
    max_samples: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Randomly subsample up to max_samples rows, stratified by Month,
    so we keep all months and roughly preserve their proportions.
    """
    if len(df) <= max_samples:
        return df

    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=max_samples,
        random_state=random_state,
    )

    y = df["Month"]
    idx, _ = next(sss.split(df, y))
    return df.iloc[idx].copy()


def train_model_and_compare(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    max_samples: int = 300_000,
):
    """
    Prepare dataframe → split → subsample → train models → print metrics.
    """
    df = df.copy()

    # Ensure necessary columns exist
    required_cols = FEATURE_COLS + [TARGET_COL, ID_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only the required columns
    df = df[required_cols].copy()

    # Drop rows with missing essential fields
    df = df.dropna(subset=[TARGET_COL, ID_COL, "Month"])
    df = df.dropna()  # remove any leftover NaNs

    # Split (grouped by PLACEKEY, full months coverage)
    train_df, test_df = _split_with_all_months(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    # Fair, random, month-stratified subsampling to 100k rows
    train_df = _subsample_by_month(train_df, max_samples=max_samples, random_state=random_state)
    test_df = _subsample_by_month(test_df, max_samples=max_samples, random_state=random_state)

    # Train & evaluate models (uncomment what you need)
    lin_model, linear_regression_metrics = train_linear_regression(train_df, test_df)
    print("Linear Regression:", linear_regression_metrics)

    ridge_model, ridge_metrics = train_ridge_regression(train_df, test_df)
    print("Ridge:", ridge_metrics)

    lasso_model, lasso_metrics = train_lasso_regression(train_df, test_df)
    print("Lasso:", lasso_metrics)

    enet_model, enet_metrics = train_elasticnet_regression(train_df, test_df)
    print("ElasticNet:", enet_metrics)

    rf_model, rf_metrics = train_lightgbm_random_forest(train_df, test_df)
    print("RandomForest:", rf_metrics)

    # ----------------------------------------
    # Combined figure: RMSE + R² horizontally
    # ----------------------------------------
    metrics_dict = {
        "Linear": linear_regression_metrics,
        "Ridge": ridge_metrics,
        "Lasso": lasso_metrics,
        "ElasticNet": enet_metrics,
        "RandomForest": rf_metrics,
    }

    model_names = list(metrics_dict.keys())
    train_rmse = [metrics_dict[m]["train_rmse"] for m in model_names]
    eva_rmse = [metrics_dict[m]["eva_rmse"] for m in model_names]
    train_r2 = [metrics_dict[m]["train_r2"] for m in model_names]
    eva_r2 = [metrics_dict[m]["eva_r2"] for m in model_names]

    x = range(len(model_names))
    width = 0.35

    # ----------------------------------------
    # Create horizontally arranged subplots
    # ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # wide, two-column style

    # ---------------------------
    # Left subplot: RMSE
    # ---------------------------
    ax = axes[0]
    ax.bar([i - width / 2 for i in x], train_rmse, width=width, label="Train RMSE")
    ax.bar([i + width / 2 for i in x], eva_rmse, width=width, label="Eval RMSE")
    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("RMSE (log1p space)")
    ax.set_title("Model Comparison – RMSE")
    ax.legend()
    ax.grid(alpha=0.2)

    # ---------------------------
    # Right subplot: R²
    # ---------------------------
    ax = axes[1]
    ax.bar([i - width / 2 for i in x], train_r2, width=width, label="Train R²")
    ax.bar([i + width / 2 for i in x], eva_r2, width=width, label="Eval R²")
    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("R² (log1p space)")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison – R²")
    ax.legend()
    ax.grid(alpha=0.2)

    # ----------------------------------------
    # Save combined figure
    # ----------------------------------------
    plt.tight_layout()
    plt.savefig("output/model_comparison_combined.png", dpi=300)
    plt.close()
