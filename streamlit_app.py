import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import numpy as np

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("xgboost_breakout_best.pkl")
scaler = joblib.load("scaler.pkl")
feature_list = joblib.load("feature_list.pkl")

# ===============================
# IMPORT INDICATORS
# ===============================
from indicators import (
    compute_moving_averages,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_volume_sma,
    compute_dmi
)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(layout="wide")
st.title("📈 AI Breakout Detection Dashboard")

# ===============================
# FILE UPLOADER
# ===============================
file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    df_raw = pd.read_csv(file)
    df_raw.columns = df_raw.columns.str.strip().str.lower()
    df_raw["time"] = pd.to_datetime(df_raw["time"], errors="coerce")
    df_raw = df_raw.dropna(subset=["time", "ticker"])

    tickers = sorted(df_raw["ticker"].unique())
    selected_ticker = st.selectbox("Chọn mã cổ phiếu", tickers)

    df_display = df_raw[df_raw["ticker"] == selected_ticker].copy()
    df_display = df_display.sort_values("time")

    # ===============================
    # DATE FILTER
    # ===============================
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Từ ngày", value=df_display["time"].min())
    with col2:
        end_date = st.date_input("Đến ngày", value=df_display["time"].max())

    df_display = df_display[
        (df_display["time"] >= pd.to_datetime(start_date)) &
        (df_display["time"] <= pd.to_datetime(end_date))
    ]

    # Giới hạn 300 phiên cho mượt
    df_display = df_display.tail(300)

    # ===============================
    # COMPUTE INDICATORS
    # ===============================
    df_display = compute_moving_averages(df_display)
    df_display = compute_rsi(df_display, window=14)
    df_display = compute_macd(df_display)
    df_display = compute_bollinger_bands(df_display)
    df_display = compute_volume_sma(df_display, window=9)
    df_display = compute_dmi(df_display)

    df_display = df_display.dropna()

    # ===============================
    # BUILD ML FEATURES
    # ===============================
    df_display["Return_1"] = df_display["close"].pct_change(1)
    df_display["Return_5"] = df_display["close"].pct_change(5)
    df_display["BB_width"] = df_display["bb_upper"] - df_display["bb_lower"]
    df_display["Trend_strength"] = df_display["adx"]
    df_display["Volume_ratio"] = df_display["volume"] / df_display["volume"].rolling(20).mean()
    df_display["Position_ratio"] = (
        (df_display["close"] - df_display["bb_lower"]) /
        (df_display["bb_upper"] - df_display["bb_lower"])
    )
    df_display["High_Low_Range"] = df_display["high"] - df_display["low"]

    df_display = df_display.dropna()

    # đảm bảo đủ feature
    for col in feature_list:
        if col not in df_display.columns:
            df_display[col] = 0

    X_input = df_display[feature_list]
    X_scaled = scaler.transform(X_input)

    y_prob = model.predict_proba(X_scaled)[:, 1]
    df_display["ai_prediction"] = (y_prob >= 0.7).astype(int)

    # ===============================
    # CREATE FIGURE
    # ===============================
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=df_display["time"],
        open=df_display["open"],
        high=df_display["high"],
        low=df_display["low"],
        close=df_display["close"],
        name="Price"
    ))

    # AI breakout marker
    breakout_points = df_display[df_display["ai_prediction"] == 1]

    fig.add_trace(go.Scatter(
        x=breakout_points["time"],
        y=breakout_points["low"] * 0.995,
        mode="markers",
        marker=dict(
            color="#00E676",
            size=12,
            symbol="triangle-up"
        ),
        name="AI Breakout"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dữ liệu")
    st.dataframe(df_display.sort_values("time", ascending=False))