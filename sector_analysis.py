import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score

# ==========================================
# 1. CONFIG
# ==========================================
DATA_PATH = r"D:\code\model\dataset_breakout_ml.csv"
MODEL_PATH = r"D:\code\model\xgboost_breakout_best.pkl"
SCALER_PATH = r"D:\code\model\scaler.pkl"
FEATURE_PATH = r"D:\code\model\feature_list.pkl"

THRESHOLD = 0.7

# ==========================================
# 2. LOAD DATA
# ==========================================
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# ==========================================
# 3. ONE HOT SECTOR (GIỐNG LÚC TRAIN)
# ==========================================
df["industry_level2"] = df["industry_level2"].fillna("Unknown")

df = pd.get_dummies(
    df,
    columns=["industry_level2"],
    prefix="Sector"
)

# ==========================================
# 4. SPLIT TEST SET (80/20)
# ==========================================
split_index = int(len(df) * 0.8)
test_df = df.iloc[split_index:].copy()

# Lưu sector gốc để group sau
test_df["Sector_Name"] = (
    test_df[[col for col in test_df.columns if col.startswith("Sector_")]]
    .idxmax(axis=1)
    .str.replace("Sector_", "")
)

# ==========================================
# 5. LOAD MODEL COMPONENTS
# ==========================================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURE_PATH)

# ==========================================
# 6. ALIGN FEATURE COLUMNS
# ==========================================
for col in features:
    if col not in test_df.columns:
        test_df[col] = 0

X_test = test_df[features]

# SCALE (giống lúc train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 7. PREDICT
# ==========================================
y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

test_df["y_pred"] = y_pred
test_df["y_true"] = test_df["Breakout"]

# ==========================================
# 8. CALCULATE METRICS BY SECTOR
# ==========================================
results = []

for sector in test_df["Sector_Name"].unique():

    subset = test_df[test_df["Sector_Name"] == sector]

    if len(subset) > 50:  # tránh ngành quá ít mẫu

        prec = precision_score(
            subset["y_true"],
            subset["y_pred"],
            zero_division=0
        )

        rec = recall_score(
            subset["y_true"],
            subset["y_pred"],
            zero_division=0
        )

        acc = accuracy_score(
            subset["y_true"],
            subset["y_pred"]
        )

        results.append({
            "Sector": sector,
            "Precision": prec,
            "Recall": rec,
            "Accuracy": acc,
            "Samples": len(subset)
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Precision", ascending=False)

print("\n===== HIỆU SUẤT THEO NGÀNH =====")
print(results_df)

# ==========================================
# 9. PLOT (MATPLOTLIB ONLY)
# ==========================================
plt.figure(figsize=(12,6))

plt.bar(results_df["Sector"], results_df["Precision"])

plt.axhline(
    y=0.3,
    color="red",
    linestyle="--",
    label="Target 30%"
)

plt.xticks(rotation=45)
plt.ylabel("Precision")
plt.title("Precision theo ngành (XGBoost + Sector)")
plt.legend()
plt.tight_layout()
plt.show()