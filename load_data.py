# load_data.py
# load_data.py
import pandas as pd

DATA_PATH = r"D:\code\model\VN100_Historical_Data_2020_2025.csv"
MIN_DAYS = 1000


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)

    # chuẩn hóa tên cột
    df.columns = df.columns.str.strip().str.lower()

    # rename các cột chính
    df = df.rename(columns={
        "time": "Date",
        "ticker": "Ticker",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    # đảm bảo các cột ngành đúng tên (giữ nguyên)
    # company_name
    # industry_level2
    # industry_level3
    # industry_level4
    # company_type

    # xử lý thời gian & sort
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])

    # ✅ GIỮ THÊM 4 CỘT NGÀNH
    df = df[[
        "Date", "Ticker",
        "Open", "High", "Low", "Close", "Volume",
        "company_name",
        "industry_level2",
        "industry_level3",
        "industry_level4",
        "company_type"
    ]]

    df = df.dropna().reset_index(drop=True)

    return df


def eda_basic(df):
    print("Số dòng:", len(df))
    print("Số mã cổ phiếu:", df["Ticker"].nunique())
    print("Khoảng thời gian:", df["Date"].min(), "→", df["Date"].max())

    trading_days = df.groupby("Ticker")["Date"].count()

    print("\nThống kê số phiên / mã:")
    print(trading_days.describe())

    print("\nÍt phiên nhất:")
    print(trading_days.sort_values().head(10))

    print("\nNhiều phiên nhất:")
    print(trading_days.sort_values().tail(10))

    return trading_days


def filter_tickers(df, trading_days, min_days=MIN_DAYS):
    valid_tickers = trading_days[trading_days >= min_days].index
    removed = sorted(set(trading_days.index) - set(valid_tickers))

    df = df[df["Ticker"].isin(valid_tickers)].reset_index(drop=True)

    print("\nNgưỡng phiên:", min_days)
    print("Số mã còn lại:", df["Ticker"].nunique())
    print("Mã bị loại:", removed)

    return df


if __name__ == "__main__":
    df = load_data()
    trading_days = eda_basic(df)
    df = filter_tickers(df, trading_days)

# step_1_lag_features.py
import pandas as pd

def add_lag_features(df):
    df = df.copy()

    # lag giá
    for i in [1, 2, 3]:
        df[f"Close_lag{i}"] = (
            df.groupby("Ticker")["Close"]
              .shift(i)
        )

    # lag High / Low / Volume
    df["High_lag1"] = (
        df.groupby("Ticker")["High"].shift(1)
    )
    df["Low_lag1"] = (
        df.groupby("Ticker")["Low"].shift(1)
    )
    df["Volume_lag1"] = (
        df.groupby("Ticker")["Volume"].shift(1)
    )

    return df


if __name__ == "__main__":
    from load_data import load_data

    df = load_data()
    df = add_lag_features(df)

    # check nhanh 1 mã
    sample = df["Ticker"].iloc[0]
    print(df[df["Ticker"] == sample][
        ["Date", "Close", "Close_lag1", "Close_lag2"]
    ].head(6))

