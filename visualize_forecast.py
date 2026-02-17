# visualize_forecast.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ==========================================
# 1. CONFIG
# ==========================================
DATA_PATH = r"D:\code\model\dataset_breakout_ml.csv"
MODEL_PATH = r"D:\code\model\xgboost_breakout_best.pkl"
SCALER_PATH = r"D:\code\model\scaler.pkl"
FEATURE_PATH = r"D:\code\model\feature_list.pkl"

TICKER = "ACB"      # <-- ĐỔI MÃ Ở ĐÂY
THRESHOLD = 0.7
SPLIT_RATIO = 0.8

# ==========================================
# 2. LOAD MODEL + SCALER + FEATURE LIST
# ==========================================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURE_PATH)

print(f"Model expects {len(feature_names)} features")

# ==========================================
# 3. LOAD DATA
# ==========================================
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# ==========================================
# 4. ONE-HOT SECTOR (giống lúc train)
# ==========================================
if "industry_level2" in df.columns:
    df["industry_level2"] = df["industry_level2"].fillna("Unknown")
    df = pd.get_dummies(df, columns=["industry_level2"], prefix="Sector")

# ==========================================
# 5. CHIA TEST SET (giống lúc train)
# ==========================================
split_index = int(len(df) * SPLIT_RATIO)
df_test = df.iloc[split_index:].copy()

# Lọc 1 mã
df_test = df_test[df_test["Ticker"] == TICKER].copy()

if len(df_test) == 0:
    print("Không tìm thấy mã trong tập test.")
    exit()

# ==========================================
# 6. ĐẢM BẢO ĐỦ FEATURE (CỰC QUAN TRỌNG)
# ==========================================
for col in feature_names:
    if col not in df_test.columns:
        df_test[col] = 0   # nếu ngành không xuất hiện trong test → thêm cột 0

X_input = df_test[feature_names]

# ==========================================
# 7. SCALE
# ==========================================
X_scaled = scaler.transform(X_input)

# ==========================================
# 8. PREDICT
# ==========================================
y_prob = model.predict_proba(X_scaled)[:, 1]
df_test["Probability"] = y_prob
df_test["Signal"] = (y_prob >= THRESHOLD).astype(int)

# ==========================================
# 9. IN BREAKOUT SIGNAL
# ==========================================
signals = df_test[df_test["Signal"] == 1]

print(f"\n===== BREAKOUT SIGNALS: {TICKER} =====")
if len(signals) > 0:
    print(signals[["Date", "Close", "Probability"]].to_string(index=False))
else:
    print("Không có tín hiệu.")

# ==========================================
# 10. PLOT
# ==========================================
plt.figure(figsize=(14,7))

plt.plot(df_test["Date"], df_test["Close"],
         color="black", linewidth=1.5, label="Close Price")

plt.scatter(
    signals["Date"],
    signals["Close"],
    marker="^",
    s=140,
    color="green",
    edgecolor="black",
    label=f"Breakout (Prob ≥ {THRESHOLD})"
)

plt.title(f"{TICKER} - XGBoost Breakout Forecast", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()