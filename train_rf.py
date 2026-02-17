# train_rf.py

from data_processor import prepare_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np


# =========================================
# 1. LOAD DATA ĐÃ SCALE
# =========================================
X_train, X_test, y_train, y_test, feature_names = prepare_data(
    r"D:\code\model\dataset_breakout_ml.csv"
)


# =========================================
# 2. RANDOM FOREST MODEL
# =========================================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",   # xử lý mất cân bằng
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# =========================================
# 3. REPORT
# =========================================
print("\n===== RANDOM FOREST =====")
print(classification_report(y_test, y_pred))


# =========================================
# 4. FEATURE IMPORTANCE
# =========================================
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop Feature Importance:")
print(importance_df.head(10))