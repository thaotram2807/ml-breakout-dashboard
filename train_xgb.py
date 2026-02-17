# train_xgb_tuned.py

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from data_processor import prepare_data


# =========================================
# LOAD DATA
# =========================================
X_train, X_test, y_train, y_test, feature_names = prepare_data(
    r"D:\code\model\dataset_breakout_ml.csv"
)

# =========================================
# TUNED XGBOOST
# =========================================
tuned_weight = 15   # giảm mạnh từ 54 xuống 15

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=tuned_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print(f"Training XGBoost (weight={tuned_weight})...")
model.fit(X_train, y_train)

# =========================================
# PROBABILITY PREDICTION
# =========================================
y_prob = model.predict_proba(X_test)[:, 1]

# =========================================
# THRESHOLD TUNING
# =========================================
print("\n===== THRESHOLD TEST =====")
thresholds = [0.5, 0.6, 0.7, 0.8]

for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    prec = precision_score(y_test, y_pred_thresh, pos_label=1)
    rec = recall_score(y_test, y_pred_thresh, pos_label=1)
    f1 = f1_score(y_test, y_pred_thresh, pos_label=1)
    print(f"Threshold {thresh}: Precision={prec:.2f} | Recall={rec:.2f} | F1={f1:.2f}")

# =========================================
# CHỌN NGƯỠNG 0.7 (tạm)
# =========================================
best_thresh = 0.7
y_final = (y_prob >= best_thresh).astype(int)

import joblib

joblib.dump(model, "xgboost_breakout_best.pkl")
print("Model saved.")
print(f"\n===== FINAL MODEL (Thresh={best_thresh}) =====")
print(classification_report(y_test, y_final))