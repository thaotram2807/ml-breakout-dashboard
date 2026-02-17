# ============================================
# train_evaluate_by_sector.py
# ============================================
# So sánh các mô hình ML theo từng NGÀNH (dùng industry_level2)
# Bài toán: Dự đoán xu hướng tăng/giảm ngày tiếp theo
# ============================================

import pandas as pd
import numpy as np
import joblib  # Để lưu model .pkl
import os      # Để tạo thư mục models/

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

# Thêm import indicators
from indicators import (
    compute_moving_averages,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_volume_sma,
    compute_dmi
)

# XGBoost (nếu có)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed → will skip XGBoost model")

# ============================================
# 1. CONFIG
# ============================================
DATA_PATH = "VN100_Historical_Data_2020_2025.csv"
SPLIT_DATE = "2024-01-01"   # train < date, test >= date
RANDOM_STATE = 42

# Ngưỡng tối thiểu dữ liệu (có thể điều chỉnh)
MIN_TRAIN_SAMPLES = 300
MIN_TEST_SAMPLES = 80

# Cột ngành dùng để nhóm (lowercase sau load)
SECTOR_COL = "industry_level2"

# Tạo thư mục lưu model nếu chưa có
os.makedirs("models", exist_ok=True)
print("[INFO] Đã kiểm tra/tạo thư mục models/")

# ============================================
# 2. LOAD & PREPARE DATA
# ============================================
def load_and_prepare_data(path: str) -> pd.DataFrame:
    print("Đang đọc file dữ liệu...")
    df = pd.read_csv(path)

    # Chuẩn hóa tên cột
    df.columns = df.columns.str.strip().str.lower()

    # Đổi tên cột time → date cho thống nhất (nếu có)
    if "time" in df.columns:
        print("[INFO] Đổi tên cột 'time' thành 'date'")
        df = df.rename(columns={"time": "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"])

    print(f"[DEBUG] Số dòng sau dropna date/ticker/close: {len(df)}")
    print(f"[DEBUG] Các cột hiện có: {list(df.columns)}")

    df = df.sort_values(["ticker", "date"])

    # TÍNH INDICATORS
    print("Đang tính indicators...")
    df = compute_moving_averages(df)
    df = compute_rsi(df, window=14)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    df = compute_volume_sma(df, window=9)
    df = compute_dmi(df, window=14)

    print("[DEBUG] Đã tính xong indicators. Các cột mới: ", [col for col in df.columns if col not in ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']])

    # Target: 1 = tăng, 0 = giảm
    df["close_next"] = df.groupby("ticker")["close"].shift(-1)
    df["y_trend"] = (df["close_next"] > df["close"]).astype(int)

    df = df.dropna(subset=["close_next", "y_trend"])

    print(f"Tổng số dòng sau xử lý (có target): {len(df):,}")
    print(f"Số mã chứng khoán: {df['ticker'].nunique()}")

    return df

# ============================================
# 3. FEATURE SET (khớp với indicators.py)
# ============================================
FEATURES = [
    "open", "high", "low", "close", "volume",
    "ma10", "ma20", "ma60",
    "rsi14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower",
    "adx", "plus_di", "minus_di",
    "volume_sma9"
]

def get_available_features(df):
    available = [f for f in FEATURES if f in df.columns]
    print(f"[INFO] Features khả dụng: {available}")
    return available

# ============================================
# 4. MODELS
# ============================================
def get_models():
    models = {
        "Logistic": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE
            ))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )

    return models

# ============================================
# 5. TRAIN & EVALUATE BY SECTOR
# ============================================
def evaluate_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    available_features = get_available_features(df)
    print(f"Features được sử dụng để train: {available_features}")

    sectors = sorted(df[SECTOR_COL].dropna().unique())
    print(f"[INFO] Các ngành sẽ train: {sectors}")

    models = get_models()

    for sector in sectors:
        print(f"\nXử lý ngành: {sector}")
        df_sector = df[df[SECTOR_COL] == sector].copy()  # ← sửa đúng: == sector

        if df_sector.empty:
            print(f"  → Ngành {sector} rỗng, bỏ qua")
            continue

        train_df = df_sector[df_sector["date"] < SPLIT_DATE]
        test_df  = df_sector[df_sector["date"] >= SPLIT_DATE]

        print(f"  Train size: {len(train_df)}, Test size: {len(test_df)}")

        if len(train_df) < MIN_TRAIN_SAMPLES or len(test_df) < MIN_TEST_SAMPLES:
            print(f"  → Bỏ qua {sector} (train={len(train_df)}, test={len(test_df)})")
            continue

        X_train = train_df[available_features]
        y_train = train_df["y_trend"]
        X_test  = test_df[available_features]
        y_test  = test_df["y_trend"]

        sector_result = {"Sector": sector, "Train_size": len(train_df), "Test_size": len(test_df)}

        best_f1 = -1
        best_model_name = None
        best_model = None

        for model_name, model in models.items():
            try:
                print(f"  Đang train {model_name} cho ngành {sector}...")
                model.fit(X_train, y_train)

                model_path = f"models/{sector}_{model_name}.pkl"
                joblib.dump(model, model_path)
                print(f"  [INFO] Đã lưu {model_name} cho {sector} vào {model_path}")

                y_pred = model.predict(X_test)

                precision = precision_score(y_test, y_pred, zero_division=0)
                recall    = recall_score(y_test, y_pred, zero_division=0)
                f1        = f1_score(y_test, y_pred, zero_division=0)

                sector_result[model_name + "_f1"] = f1
                sector_result[model_name + "_prec"] = precision
                sector_result[model_name + "_rec"] = recall

                if f1 > best_f1:
                    best_f1 = f1
                    best_model_name = model_name
                    best_model = model

            except Exception as e:
                print(f"  Lỗi khi train {model_name} cho {sector}: {e}")
                sector_result[model_name + "_f1"] = np.nan

        # Lưu model tốt nhất
        if best_model_name and best_model:
            best_path = f"models/{sector}_best_{best_model_name}.pkl"
            joblib.dump(best_model, best_path)
            print(f"  [BEST] Ngành {sector}: {best_model_name} với F1 = {best_f1:.4f} → Đã lưu")

        results.append(sector_result)

    result_df = pd.DataFrame(results)
    if result_df.empty:
        print("Không có ngành nào đủ dữ liệu để đánh giá.")
        return pd.DataFrame()

    return result_df

# ============================================
# 6. MAIN
# ============================================
def main():
    df = load_and_prepare_data(DATA_PATH)

    print("\nBắt đầu đánh giá theo ngành...")
    result_df = evaluate_by_sector(df)

    if result_df.empty:
        print("Không có kết quả nào.")
        return

    f1_cols = [col for col in result_df.columns if col.endswith("_f1")]

    print("\n===== F1-SCORE THEO NGÀNH =====")
    print(
        result_df[["Sector", "Train_size", "Test_size"] + f1_cols]
        .round(4)
        .to_string(index=False)
    )

    if f1_cols:
        result_df["Best_Model"] = result_df[f1_cols].idxmax(axis=1).str.replace("_f1", "")
        result_df["Best_F1"] = result_df[f1_cols].max(axis=1)

        print("\n===== MÔ HÌNH TỐT NHẤT THEO NGÀNH =====")
        print(
            result_df[["Sector", "Best_Model", "Best_F1", "Train_size", "Test_size"]]
            .round(4)
            .sort_values("Best_F1", ascending=False)
            .to_string(index=False)
        )

        result_df.to_csv("sector_model_evaluation.csv", index=False)
        print("\nĐã lưu kết quả đánh giá vào sector_model_evaluation.csv")


if __name__ == "__main__":
    main()