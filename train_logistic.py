# train_logistic.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def train_logistic(X_train, X_test, y_train, y_test):

    model = LogisticRegression(
        class_weight="balanced",   # bắt buộc vì 1.8%
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n===== Logistic Regression =====")
    print(classification_report(y_test, y_pred))

    return model