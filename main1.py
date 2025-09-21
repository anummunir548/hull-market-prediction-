# main.py
import os
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
from src.preprocess import fill_missing_with_median, select_feature_columns

# Paths (assumes CSV files are inside /data/)
TRAIN_CSV = "data/train.csv"
TEST_CSV = "data/test.csv"
OUTPUT_CSV = "submission.csv"
MODEL_PATH = "lgbm_model.txt"

# Config
VALIDATION_LAST_N = 180
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "seed": 42,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

def main():
    print("📂 Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    feature_cols = [c for c in select_feature_columns(train) if c in test.columns]
    train = fill_missing_with_median(train, feature_cols)
    test = fill_missing_with_median(test, feature_cols)

    X, y = train[feature_cols], train["forward_returns"]

    X_train = X.iloc[:-VALIDATION_LAST_N]
    y_train = y.iloc[:-VALIDATION_LAST_N]
    X_valid = X.iloc[-VALIDATION_LAST_N:]
    y_valid = y.iloc[-VALIDATION_LAST_N:]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    print("⚡ Training model...")
    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    y_pred_val = model.predict(X_valid, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred_val))
    print(f"📉 Validation RMSE: {rmse:.6f}")

    model.save_model(MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

    preds = model.predict(test[feature_cols], num_iteration=model.best_iteration)
    test_out = test[["date_id"]].copy()
    test_out["predicted_forward_returns"] = preds
    test_out.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Submission saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
