# =======================================
# Step 1. Install/Upgrade dependencies
# =======================================
!pip install -U scikit-learn lightgbm tqdm pandas numpy matplotlib

# =======================================
# Step 2. Imports
# =======================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from tqdm import tqdm
from google.colab import drive

# =======================================
# Step 3. Mount Google Drive
# =======================================
drive.mount('/content/drive')

# ✅ Your working folder inside Drive
DATA_DIR = "/content/drive/MyDrive/market"   # 👈 replace with your actual folder

# Paths
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "submission.csv")   # 🔹 Saved in Drive
MODEL_PATH = os.path.join(DATA_DIR, "lgbm_model.txt")

# =======================================
# Config
# =======================================
RANDOM_STATE = 42
VALIDATION_LAST_N = 180
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "seed": RANDOM_STATE,
    "num_threads": 4,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

# =======================================
# Helpers
# =======================================
def load_data(train_path: str, test_path: str):
    return pd.read_csv(train_path), pd.read_csv(test_path)

def select_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = {
        "date_id", "forward_returns", "risk_free_rate",
        "market_forward_excess_returns", "is_scored",
        "lagged_forward_returns", "lagged_risk_free_rate",
        "lagged_market_forward_excess_returns"
    }
    return [c for c in df.columns if c not in drop_cols]

def fill_missing_with_median(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna("NA")
    return df

# =======================================
# Main
# =======================================
def main():
    print("📂 Loading data...")
    train, test = load_data(TRAIN_CSV, TEST_CSV)

    # Sort
    train = train.sort_values("date_id").reset_index(drop=True)
    test = test.sort_values("date_id").reset_index(drop=True)

    # Features
    feature_cols = [c for c in select_feature_columns(train) if c in test.columns]
    print(f"✅ Using {len(feature_cols)} features: {feature_cols[:10]} ...")

    # Missing values
    train = fill_missing_with_median(train, feature_cols)
    test = fill_missing_with_median(test, feature_cols)

    X, y = train[feature_cols], train["forward_returns"]

    # Train/valid split
    if len(train) <= VALIDATION_LAST_N + 10:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
    else:
        X_train = X.iloc[:-VALIDATION_LAST_N]
        y_train = y.iloc[:-VALIDATION_LAST_N]
        X_valid = X.iloc[-VALIDATION_LAST_N:]
        y_valid = y.iloc[-VALIDATION_LAST_N:]

    # Train LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    print("⚡ Training LightGBM...")
    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    # RMSE (model)
    y_pred_val = model.predict(X_valid, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred_val))
    print(f"📉 Validation RMSE (Model): {rmse:.6f}")

    # ==========================
    # Baseline Comparison
    # ==========================
    y_pred_baseline = np.zeros_like(y_valid)
    baseline_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_baseline))
    print(f"📊 Baseline RMSE (predict 0): {baseline_rmse:.6f}")

    improvement = baseline_rmse - rmse
    if improvement > 0:
        print(f"✅ Model beats baseline by {improvement:.6f}")
    else:
        print(f"⚠️ Model is worse than baseline by {-improvement:.6f}")

    # Save model
    model.save_model(MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

    # Predict test
    preds = model.predict(test[feature_cols], num_iteration=model.best_iteration)
    test_out = test[["date_id"]].copy()
    test_out["predicted_forward_returns"] = preds

    # Save submission in Google Drive
    test_out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Submission saved to Google Drive: {OUTPUT_CSV}")
    print(test_out.head())

# =======================================
# Run
# =======================================
main()
