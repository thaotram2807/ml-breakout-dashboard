# train_svm_tuned.py

from data_processor import prepare_data
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# Load data
X_train, X_test, y_train, y_test, feature_names = prepare_data(
    r"D:\code\model\dataset_breakout_ml.csv"
)


# Tuned SVM
model = SVC(
    kernel="rbf",
    C=5,
    gamma="scale",
    class_weight={0:1, 1:8},   # giảm thêm
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("\n===== TUNED SVM =====")
print(classification_report(y_test, y_pred))