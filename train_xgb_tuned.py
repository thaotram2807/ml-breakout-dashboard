from data_processor import prepare_data
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# =========================================
# LOAD DATA
# =========================================
X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(
    r"D:\code\model\dataset_breakout_ml.csv"
)

# =========================================
# SAVE SCALER
# =========================================
joblib.dump(scaler, "scaler.pkl")
print("Saved scaler.pkl")

# =========================================
# TRAIN XGBOOST
# =========================================
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=15,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
# =========================================
# SAVE MODEL
# =========================================
joblib.dump(model, "xgboost_breakout_best.pkl")
print("Saved xgboost_breakout_best.pkl")

# =========================================
# SAVE FEATURE LIST (QUAN TRỌNG)
# =========================================
joblib.dump(feature_names, "feature_list.pkl")
print("Saved feature_list.pkl")

# =========================================
# EVALUATION
# =========================================
print("\n===== MODEL REPORT =====")
print(classification_report(y_test, model.predict(X_test)))