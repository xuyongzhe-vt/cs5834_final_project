import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

# ======================================================
# CONFIG
# ======================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ID_THRESHOLD = 0.98  # if > 98% of values are unique, treat as ID-like


# ======================================================
# 1. ID-LIKE FEATURE DETECTION
# ======================================================

def find_id_like_features(df, threshold=ID_THRESHOLD, whitelist=()):
    """
    Detect columns that behave like IDs:
    - high cardinality (almost one unique value per row)
    - not in whitelist
    Returns a list of column names to treat as "ID-like".
    """
    n = len(df)
    candidates = []

    for col in df.columns:
        if col in whitelist:
            continue
        unique_ratio = df[col].nunique(dropna=True) / max(n, 1)
        if unique_ratio >= threshold:
            candidates.append(col)

    return candidates


# ======================================================
# 2. FEATURE ENGINEERING
# ======================================================

def engineer_features(df):
    """
    Creates clean/engineered features:
    - Year, Month from DATE_RANGE_START
    - Distance bucket from DISTANCE_FROM_HOME
    Returns a new DataFrame with the extra columns added.
    """
    df = df.copy()

    # --- Parse date ---
    if "DATE_RANGE_START" in df.columns:
        dt = pd.to_datetime(df["DATE_RANGE_START"], errors="coerce")
        dt = dt.fillna(pd.Timestamp("2021-01-01"))
        df["Year"] = dt.dt.year.astype(int)

        # If Month already exists, keep it; otherwise create it
        if "Month" not in df.columns:
            df["Month"] = dt.dt.month.astype(int)

    # --- Distance bucket (Community / City / Regional / Travel) ---
    if "DISTANCE_FROM_HOME" in df.columns:
        dist = df["DISTANCE_FROM_HOME"]

        # Simple imputation for distance
        dist_median = dist.median()
        if pd.isna(dist_median):
            dist_median = 0.0
        df["DISTANCE_FROM_HOME"] = dist.fillna(dist_median)

    return df


# ======================================================
# 3. ENCODING & PREPARATION
# ======================================================

def prepare_X(df, feature_cols, cat_cols=()):
    """
    Create a numeric feature matrix X from df:
    - label-encodes categorical features
    - leaves numeric features as-is
    """
    df = df.copy()
    X = pd.DataFrame(index=df.index)

    for col in feature_cols:
        if col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(df[col].astype(str))
        else:
            X[col] = df[col]

    return X


# ======================================================
# 4. CORRELATION WITH TARGET
# ======================================================

def corr_with_target(df, features, target_col):
    """
    Compute Pearson and Spearman correlation between each numeric feature
    and the target column.
    Assumes the features passed are numeric (or already encoded).
    """
    rows = []
    y = df[target_col]

    for col in features:
        x = df[col]
        mask = x.notna() & y.notna()
        if mask.sum() < 2:
            continue

        try:
            r_pearson, _ = pearsonr(x[mask], y[mask])
        except Exception:
            r_pearson = np.nan
        try:
            r_spearman, _ = spearmanr(x[mask], y[mask])
        except Exception:
            r_spearman = np.nan

        rows.append((col, r_pearson, r_spearman))

    res = pd.DataFrame(rows, columns=["feature", "pearson_r", "spearman_r"])
    res = res.sort_values("pearson_r", key=np.abs, ascending=False)
    return res


# ======================================================
# 5. HIGHLY CORRELATED FEATURE PAIRS
# ======================================================

def highly_correlated_pairs(df, cols, thresh=0.95):
    """
    Among a list of numeric columns, find pairs that are highly correlated
    with each other (|r| >= thresh).
    """
    if len(cols) < 2:
        return pd.DataFrame(columns=["feat1", "feat2", "pearson_r"])

    corr = df[cols].corr()
    pairs = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1 = cols[i]
            c2 = cols[j]
            r = corr.loc[c1, c2]
            if abs(r) >= thresh:
                pairs.append((c1, c2, r))

    result = pd.DataFrame(pairs, columns=["feat1", "feat2", "pearson_r"])
    result = result.sort_values("pearson_r", key=np.abs, ascending=False)
    return result


# ======================================================
# 6. PERMUTATION IMPORTANCE
# ======================================================

def permutation_importance_scores(df, feature_cols, target_col,
                                  cat_cols=(), n_sample=200_000):
    """
    - Optionally subsamples df for speed.
    - Trains a RandomForestRegressor.
    - Computes permutation importance over a validation set.
    Returns a pandas Series of importances (mean drop in score).
    """
    df = df.copy()

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    # Subsample if very large
    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=42)

    X = prepare_X(df, feature_cols, cat_cols)
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    result = permutation_importance(
        model,
        X_valid,
        y_valid,
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    imp = pd.Series(result.importances_mean, index=feature_cols)
    imp = imp.sort_values(ascending=False)
    return imp


def plot_importance(imp, target_col, title_prefix="Permutation Importance"):
    """
    Make a pretty horizontal bar chart for feature importance.
    Saves PNG into OUTPUT_DIR.
    """
    imp = imp.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(imp.index, imp.values)
    ax.set_xlabel("Mean decrease in validation score", fontsize=11)
    ax.set_title(f"{title_prefix}\nTarget: {target_col}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fname = f"perm_importance_{target_col.lower()}.png"
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[saved] {path}")


# ======================================================
# 7. MAIN ANALYSIS WRAPPER
# ======================================================

def run_feature_analysis(df):
    """
    Main function to:
    - engineer features
    - find ID-like columns
    - choose meaningful predictors
    - compute correlations and permutation importance
      for MEDIAN_DWELL and RAW_VISIT_COUNTS (if present)
    """
    # 1) Engineering
    df_fe = engineer_features(df)

    # 2) Define candidate feature set (you can tweak this list)
    candidate_features = []
    for col in ["TOP_CATEGORY", "DISTANCE_FROM_HOME", "Month", "CITY",
                "REGION",
                "POSTAL_CODE",
                "OPEN_HOURS",
                "ENCLOSED",
                "WKT_AREA_SQ_METERS",
                "DEVICE_TYPE"]:
        if col in df_fe.columns:
            candidate_features.append(col)

    # 3) Identify ID-like columns (but whitelist meaningful ones)
    whitelist = {"TOP_CATEGORY", "Year", "Month"}
    id_like_cols = find_id_like_features(df_fe[candidate_features],
                                         threshold=ID_THRESHOLD,
                                         whitelist=whitelist)

    print("ID-like columns (within candidates):", id_like_cols)

    # Final list of features to actually use
    feature_cols = [c for c in candidate_features if c not in id_like_cols]
    print("Using feature columns:", feature_cols)

    # Categorical columns (for encoding)
    # Categorical columns (for encoding)
    cat_cols_base = [
        "TOP_CATEGORY",
        "DISTANCE_BUCKET",
        "CITY",
        "REGION",
        "POSTAL_CODE",
        "OPEN_HOURS",
        "ENCLOSED",
        "DEVICE_TYPE",
    ]

    cat_cols = [c for c in feature_cols if c in cat_cols_base]

    results = {}

    # Targets to analyze (if they exist)
    target_candidates = ["MEDIAN_DWELL", "RAW_VISIT_COUNTS"]
    for target_col in target_candidates:
        if target_col not in df_fe.columns:
            continue

        print("\n" + "=" * 70)
        print(f"ANALYSIS FOR TARGET: {target_col}")
        print("=" * 70)

        # Prepare numeric encodings for correlation
        df_corr = df_fe.copy()
        X_corr = prepare_X(df_corr, feature_cols, cat_cols=cat_cols)
        df_corr = pd.concat([X_corr, df_corr[target_col]], axis=1)

        # 4) Correlation with target
        numeric_for_corr = X_corr.columns.tolist()
        corr_table = corr_with_target(df_corr, numeric_for_corr, target_col)
        print("\n--- Correlation with target (Pearson & Spearman) ---")
        print(corr_table)

        # 5) Highly correlated features among themselves
        high_pairs = highly_correlated_pairs(df_corr, numeric_for_corr, thresh=0.95)
        if not high_pairs.empty:
            print("\n--- Highly correlated feature pairs (|r| >= 0.95) ---")
            print(high_pairs)
        else:
            print("\nNo highly correlated feature pairs with |r| >= 0.95.")

        # 6) Permutation importance
        imp = permutation_importance_scores(
            df_fe,
            feature_cols=feature_cols,
            target_col=target_col,
            cat_cols=cat_cols,
        )
        print("\n--- Permutation importance (mean decrease in score) ---")
        print(imp)

        # 7) Plot importance
        plot_importance(imp, target_col)

        # Store in results
        results[target_col] = {
            "correlation": corr_table,
            "high_corr_pairs": high_pairs,
            "permutation_importance": imp,
        }

    print("\nFeature analysis complete.")
    return results
