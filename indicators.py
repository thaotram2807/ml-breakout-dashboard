import pandas as pd
import numpy as np


# ==================================================
# COMPUTE MOVING AVERAGES (SMA)
# ==================================================
def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(["ticker", "time"])

    df["ma10"] = (
        df.groupby("ticker")["close"]
        .transform(lambda x: x.rolling(10).mean())
    )

    df["ma20"] = (
        df.groupby("ticker")["close"]
        .transform(lambda x: x.rolling(20).mean())
    )

    df["ma60"] = (
        df.groupby("ticker")["close"]
        .transform(lambda x: x.rolling(60).mean())
    )

    return df


# ==================================================
# COMPUTE RSI (14)
# ==================================================
def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:

    df = df.sort_values(["ticker", "time"])

    def rsi_wilder(close):
        change = close.diff()
        gain = change.clip(lower=0)
        loss = -change.clip(upper=0)

        avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df["rsi14"] = (
        df.groupby("ticker")["close"]
        .transform(rsi_wilder)
    )

    return df


# ==================================================
# COMPUTE MACD
# ==================================================
def compute_macd(df: pd.DataFrame,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:

    df = df.sort_values(["ticker", "time"])

    def macd_series(close):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        return pd.DataFrame({
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line
        })

    macd_df = (
        df.groupby("ticker")["close"]
        .apply(macd_series)
        .reset_index(level=0, drop=True)
    )

    df = pd.concat([df, macd_df], axis=1)
    return df


# ==================================================
# COMPUTE BOLLINGER BANDS
# ==================================================
def compute_bollinger_bands(df: pd.DataFrame,
                            window: int = 20,
                            num_std: int = 2) -> pd.DataFrame:

    df = df.sort_values(["ticker", "time"])

    def bb_series(close):
        sma = close.rolling(window).mean()
        std = close.rolling(window).std(ddof=0)

        return pd.DataFrame({
            "bb_middle": sma,
            "bb_upper": sma + num_std * std,
            "bb_lower": sma - num_std * std
        })

    bb_df = (
        df.groupby("ticker")["close"]
        .apply(bb_series)
        .reset_index(level=0, drop=True)
    )

    df = pd.concat([df, bb_df], axis=1)
    return df


# ==================================================
# COMPUTE VOLUME SMA
# ==================================================
def compute_volume_sma(df: pd.DataFrame, window: int = 9) -> pd.DataFrame:

    df = df.sort_values(["ticker", "time"])

    df["volume_sma9"] = (
        df.groupby("ticker")["volume"]
        .transform(lambda x: x.rolling(window).mean())
    )

    return df


# ==================================================
# COMPUTE DMI / ADX
# ==================================================
def compute_dmi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:

    df = df.sort_values(["ticker", "time"]).copy()

    high = df["high"]
    low = df["low"]
    close = df["close"]

    high_prev = high.groupby(df["ticker"]).shift(1)
    low_prev = low.groupby(df["ticker"]).shift(1)
    close_prev = close.groupby(df["ticker"]).shift(1)

    up_move = high - high_prev
    down_move = low_prev - low

    df["plus_dm"] = np.where(
        (up_move > down_move) & (up_move > 0), up_move, 0.0
    )

    df["minus_dm"] = np.where(
        (down_move > up_move) & (down_move > 0), down_move, 0.0
    )

    tr = np.maximum.reduce([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ])

    df["tr"] = tr

    df["tr_rma"] = (
        df.groupby("ticker")["tr"]
        .transform(lambda x: x.ewm(alpha=1/window, adjust=False).mean())
    )

    df["plus_dm_rma"] = (
        df.groupby("ticker")["plus_dm"]
        .transform(lambda x: x.ewm(alpha=1/window, adjust=False).mean())
    )

    df["minus_dm_rma"] = (
        df.groupby("ticker")["minus_dm"]
        .transform(lambda x: x.ewm(alpha=1/window, adjust=False).mean())
    )

    df["plus_di"] = 100 * df["plus_dm_rma"] / df["tr_rma"]
    df["minus_di"] = 100 * df["minus_dm_rma"] / df["tr_rma"]

    df["dx"] = (
        100 * (df["plus_di"] - df["minus_di"]).abs()
        / (df["plus_di"] + df["minus_di"])
    )

    df["adx"] = (
        df.groupby("ticker")["dx"]
        .transform(lambda x: x.ewm(alpha=1/window, adjust=False).mean())
    )

    df.drop(
        columns=[
            "tr", "plus_dm", "minus_dm",
            "tr_rma", "plus_dm_rma",
            "minus_dm_rma", "dx"
        ],
        inplace=True
    )

    return df