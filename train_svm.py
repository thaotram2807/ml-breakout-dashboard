# train_svm.py

from data_processor import prepare_data
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np


# =========================================
# 1. LOAD DATA ĐÃ SCALE
# =========================================
X_train, X_test, y_train, y_test, feature_names = prepare_data(
    r"D:\code\model\dataset_breakout_ml.csv"
)


# =========================================
# 2. SVM MODEL
# =========================================
model = SVC(
    kernel="rbf",            # phi tuyến
    C=1.0,                   # độ phạt
    gamma="scale",           # tự tính
    class_weight="balanced", # BẮT BUỘC với imbalance
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# =========================================
# 3. REPORT
# =========================================
print("\n===== SVM (RBF) =====")
print(classification_report(y_test, y_pred))