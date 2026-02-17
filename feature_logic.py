import pandas as pd
import numpy as np


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace absolute (non-stationary) variables with relative / ratio features
    suitable for SVM, RF, XGBoost.
    Assumes indicators (MA, BB, RSI, Volume_SMA9) are already computed.
    """

    df = df.sort_values(["Ticker", "Date"]).copy()

    # ==================================================
    # 1. PRICE CONTEXT → RETURN
    # ==================================================
    df["Return_1"] = df.groupby("Ticker")["Close"].pct_change(1)
    df["Return_5"] = df.groupby("Ticker")["Close"].pct_change(5)

    # ==================================================
    # 2. VOLUME → RELATIVE VOLUME
    # ==================================================
    # TẠO Volume_ratio TRƯỚC
    df["Volume_ratio"] = df["Volume"] / df["Volume_SMA9"]

    # Sau đó mới dùng nó
    df["Volume_Up"] = df["Volume_ratio"] * df["Return_1"]

    # ==================================================
    # 3. TREND
    # ==================================================
    df["Trend_strength"] = (df["MA10"] - df["MA20"]) / df["MA20"]

    # ==================================================
    # 4. VOLATILITY
    # ==================================================
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["Close"]
    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]

    # ==================================================
    # 5. POSITION
    # ==================================================
    df["Max_High_20"] = (
        df.groupby("Ticker")["High"]
          .rolling(20)
          .max()
          .reset_index(level=0, drop=True)
    )

    df["Position_ratio"] = df["Close"] / df["Max_High_20"]

    return df