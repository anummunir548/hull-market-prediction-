# =======================================
# Train Model
# =======================================
feature_cols = [c for c in train.columns if c not in 
                ["date_id", "forward_returns", "risk_free_rate",
                 "market_forward_excess_returns", "is_scored"]]

X, y = train[feature_cols], train["forward_returns"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_valid, label=y_valid)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42
}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(50)]
)

y_pred_val = model.predict(X_valid, num_iteration=model.best_iteration)
model_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_val))
print(f"⚡ Model RMSE: {model_rmse:.6f}")
