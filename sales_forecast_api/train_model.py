"""
train_model.py
==============
Trains the LightGBM sales forecasting model on TRAIN.csv,
evaluates it on a held-out validation split, and serializes
all artifacts into the models/ directory.

Usage
-----
    python train_model.py --data TRAIN.csv

Outputs (models/)
-----------------
    lgb_model.pkl        — trained LightGBM model (joblib)
    feature_names.pkl    — ordered feature list (joblib)
    feature_names.json   — same list in JSON (for the API docs)
    store_stats.json     — per-store trailing statistics for lag inference
    metrics.json         — validation metrics (MAE, RMSE, MAPE, R²)
"""

import argparse
import json
import os
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
SEED       = 42
VAL_START  = pd.Timestamp("2019-04-01")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LAG_COLS   = [
    "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7",
]
BEST_PARAMS = {
    "n_estimators": 212, "learning_rate": 0.1358, "max_depth": 7,
    "num_leaves": 72,    "subsample": 0.662,       "colsample_bytree": 0.662,
    "reg_alpha": 0.00164,"reg_lambda": 1.599,      "min_child_samples": 26,
    "n_jobs": -1,        "verbose": -1,             "random_state": SEED,
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute store-level lag and rolling-window features (leak-free)."""
    df = df.sort_values(["Store_id", "Date"]).reset_index(drop=True)
    grp = df.groupby("Store_id")["Sales"]
    df["sales_lag_7"]     = grp.shift(7)
    df["sales_lag_14"]    = grp.shift(14)
    df["sales_lag_28"]    = grp.shift(28)
    df["rolling_mean_7"]  = grp.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["rolling_mean_28"] = grp.shift(1).rolling(28).mean().reset_index(level=0, drop=True)
    df["rolling_std_7"]   = grp.shift(1).rolling(7).std().reset_index(level=0, drop=True)
    return df.dropna(subset=LAG_COLS).reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline — identical for train and inference."""
    df = df.copy()

    # Time features
    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["DayOfWeek"]   = df["Date"].dt.dayofweek
    df["Quarter"]     = df["Date"].dt.quarter
    df["WeekOfYear"]  = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"]   = df["Date"].dt.dayofyear
    df["IsWeekend"]   = (df["DayOfWeek"] >= 5).astype(int)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"]   = df["Date"].dt.is_month_end.astype(int)

    # Cyclical encodings
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Week_sin"]  = np.sin(2 * np.pi * df["WeekOfYear"] / 52)
    df["Week_cos"]  = np.cos(2 * np.pi * df["WeekOfYear"] / 52)
    df["DOW_sin"]   = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DOW_cos"]   = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    # Categorical encodings
    df["Discount_enc"]      = (df["Discount"] == "Yes").astype(int)
    df["Store_Type_enc"]    = df["Store_Type"].map({"S1": 1, "S2": 2, "S3": 3, "S4": 4})
    df["Location_Type_enc"] = df["Location_Type"].map({"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5})
    df["Region_Code_enc"]   = df["Region_Code"].map({"R1": 1, "R2": 2, "R3": 3, "R4": 4})

    # Interaction features
    df["Holiday_Discount"] = df["Holiday"] * df["Discount_enc"]
    df["Weekend_Discount"] = df["IsWeekend"] * df["Discount_enc"]

    # One-hot encoding
    df = pd.get_dummies(df, columns=["Store_Type", "Location_Type", "Region_Code"], drop_first=False)

    return df


def build_store_stats(df: pd.DataFrame) -> dict:
    """Capture per-store trailing statistics for lag approximation at inference time."""
    tail28 = df.sort_values(["Store_id", "Date"]).groupby("Store_id").tail(28)
    tail7  = df.sort_values(["Store_id", "Date"]).groupby("Store_id").tail(7)

    stats = {}
    for sid in df["Store_id"].unique():
        t28 = tail28[tail28["Store_id"] == sid]["Sales"]
        t7  = tail7[tail7["Store_id"] == sid]["Sales"]
        stats[int(sid)] = {
            "mean28": round(float(t28.mean()), 4),
            "mean7":  round(float(t7.mean()),  4),
            "std7":   round(float(t7.std()),   4),
        }
    return stats


def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2), "R2": round(r2, 4)}


# ── Main ───────────────────────────────────────────────────────────────────────
def main(data_path: str):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load & clean
    print(f"Loading data from  {data_path} ...")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Sales"] > 0].reset_index(drop=True)
    print(f"  Clean rows : {len(df):,}")

    # 2. Lag features
    print("Adding lag features ...")
    df = add_lag_features(df)
    print(f"  After lag drop : {len(df):,}")

    # 3. Feature engineering
    fe = engineer_features(df)
    EXCL     = ["ID", "Date", "Sales", "#Order", "Discount"]
    FEATURES = [c for c in fe.columns if c not in EXCL]
    print(f"  Total features : {len(FEATURES)}")

    # 4. Train / val split
    tr = fe[fe["Date"] <  VAL_START]
    va = fe[fe["Date"] >= VAL_START]
    X_tr, y_tr = tr[FEATURES].astype(float), tr["Sales"].values
    X_va, y_va = va[FEATURES].astype(float), va["Sales"].values
    X_all      = fe[FEATURES].astype(float)
    y_all      = fe["Sales"].values
    print(f"  Train : {X_tr.shape}   Val : {X_va.shape}")

    # 5. Validation run (for metrics)
    print("Training validation model ...")
    val_model = lgb.LGBMRegressor(**BEST_PARAMS)
    val_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    val_metrics = compute_metrics(y_va, val_model.predict(X_va))
    val_metrics.update({"val_period": "Apr–May 2019", "train_rows": int(len(X_all)), "n_features": len(FEATURES)})
    print(f"  Validation  MAE ₹{val_metrics['MAE']:,}  RMSE ₹{val_metrics['RMSE']:,}  "
          f"MAPE {val_metrics['MAPE']}%  R² {val_metrics['R2']}")

    # 6. Final model on full data
    print("Training final model on full dataset ...")
    final_model = lgb.LGBMRegressor(**BEST_PARAMS)
    final_model.fit(X_all, y_all)

    # 7. Store statistics
    store_stats = build_store_stats(df)

    # 8. Persist artifacts
    joblib.dump(final_model, os.path.join(MODELS_DIR, "lgb_model.pkl"))
    joblib.dump(FEATURES,    os.path.join(MODELS_DIR, "feature_names.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(FEATURES, f, indent=2)

    with open(os.path.join(MODELS_DIR, "store_stats.json"), "w") as f:
        json.dump(store_stats, f, indent=2)

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)

    print(f"\nAll artifacts saved to  {MODELS_DIR}/")
    print("  lgb_model.pkl       — trained model")
    print("  feature_names.pkl   — feature list (joblib)")
    print("  feature_names.json  — feature list (JSON)")
    print("  store_stats.json    — per-store lag statistics")
    print("  metrics.json        — validation metrics")
    print("\nDone. Run  `python app.py`  to start the API server.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and serialize the sales forecasting model.")
    parser.add_argument("--data", default="TRAIN.csv", help="Path to TRAIN.csv")
    args = parser.parse_args()
    main(args.data)
