# main.py
from load_data import load_data, eda_basic, filter_tickers
from indicators import (
    compute_moving_averages,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_volume_sma,
    compute_dmi
)
from feature_logic import add_relative_features   # ← CHỈ GỌI HÀM

# ==================================================
# 1. LOAD + EDA + FILTER
# ==================================================
df = load_data()
trading_days = eda_basic(df)
df = filter_tickers(df, trading_days)

# ==================================================
# 2. INDICATORS
# ==================================================
df = compute_moving_averages(df)
df = compute_rsi(df)
df = compute_macd(df)
df = compute_bollinger_bands(df)
df = compute_volume_sma(df)
df = compute_dmi(df)

# ==================================================
# 3. FEATURE LOGIC (GỌI FILE RIÊNG)
# ==================================================
df = add_relative_features(df)

# ==================================================
# 4. LABEL BREAKOUT
# ==================================================
LOOKBACK = 20

rolling_high = (
    df.groupby("Ticker")["High"]
      .rolling(LOOKBACK)
      .max()
      .reset_index(level=0, drop=True)
)

df["Breakout"] = (
    (df.groupby("Ticker")["Close"].shift(-1) > rolling_high) &
    (df["Volume"] > 1.5 * df["Volume"].rolling(20).mean()) &
    (df["RSI14"] > 60)
).astype(int)


# ==================================================
# 5. DROP NaN
# ==================================================
features_for_model = [
    "Return_1",
    "Return_5",
    "RSI14",
    "BB_width",
    "Trend_strength",
    "Volume_ratio",
    "Position_ratio",
    "High_Low_Range",
    "Breakout"
]

rows_before = len(df)

df = df.dropna(subset=features_for_model).reset_index(drop=True)

total_rows = len(df)
total_breakout = df["Breakout"].sum()

breakout_ratio = total_breakout / total_rows

print("Tổng số dòng:", total_rows)
print("Số breakout:", total_breakout)
print("Tỷ lệ breakout:", breakout_ratio)
print("Tỷ lệ breakout (%):", breakout_ratio * 100)

# ==================================================
# 6. EXPORT DATASET
# ==================================================
df.to_csv(
    r"D:\code\model\dataset_breakout_ml.csv",
    index=False
)

print("Đã xuất dataset ML-ready")

