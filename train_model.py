import pandas as pd

df = pd.read_csv(r"D:\code\model\dataset_breakout_ml.csv")

features = [
    "Return_1",
    "Return_5",
    "RSI14",
    "BB_width",
    "Trend_strength",
    "Volume_ratio",
    "Position_ratio",
    "High_Low_Range"
]

X = df[features]
y = df["Breakout"]

# Split 80/20 theo thời gian
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test  = X.iloc[split:]

y_train = y.iloc[:split]
y_test  = y.iloc[split:]

#Scale
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Chỉ fit trên train
scaler.fit(X_train)

# Transform train
X_train_scaled = scaler.transform(X_train)

# Transform test bằng scaler của train
X_test_scaled = scaler.transform(X_test)

print("Scale hoàn tất – không có data leakage.")

from sklearn.metrics import accuracy_score

# Dự đoán trên train
train_pred = model.predict(X_train)

# Dự đoán trên test
test_pred = model.predict(X_test)

# Tính accuracy
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)