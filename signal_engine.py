# signal_engine.py
"""
Module sinh tín hiệu mua theo quy trình:
- Bước 2: Momentum classification (Down / Neutral / Strong Up) → lấy P_strong_up
- Bước 3: Breakout classification (binary) → lấy P_breakout
- Kết hợp sinh Buy Signal theo điều kiện đã định nghĩa
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

BEST_MODELS = {
    "Dịch vụ tài chính": "Logistic",
    "Điện, nước & xăng dầu khí đốt": "Logistic",
    "Y tế": "RandomForest",
    "Bán lẻ": "SVM",
    "Xây dựng và Vật liệu": "Logistic",
    "Du lịch và Giải trí": "Logistic",
    "Tài nguyên Cơ bản": "SVM",
    "Công nghệ Thông tin": "Logistic",
    "Thực phẩm và đồ uống": "RandomForest",
    "Bất động sản": "RandomForest",
    "Hóa chất": "RandomForest",
    "Hàng & Dịch vụ Công nghiệp": "RandomForest",
    "Bảo hiểm": "SVM",
    "Dầu khí": "RandomForest",
    "Ngân hàng": "RandomForest",
    "Hàng cá nhân & Gia dụng": "Logistic"
}

def generate_buy_signals(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ['Date', 'Ticker', 'Sector', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing}")

    result = df.copy().sort_values(['Ticker', 'Date'])

    result['P_trend'] = np.nan
    result['P_strong_up'] = np.nan
    result['P_breakout'] = np.nan
    result['BuySignal'] = False

    result['Return'] = result.groupby('Ticker')['Close'].pct_change()
    result['Volume_SMA_9'] = result.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(window=9, min_periods=5).mean()
    )

    result['High_20'] = result.groupby('Ticker')['High'].transform(
        lambda x: x.rolling(window=21, min_periods=10).max().shift(1)
    )

    # =================================================================
    # Dự đoán cho từng ngành
    # =================================================================
    for sector, group in result.groupby('Sector'):
        if sector not in BEST_MODELS:
            print(f"[WARNING] Không tìm thấy best model cho ngành '{sector}'. Bỏ qua.")
            continue

        model_type = BEST_MODELS[sector]
        print(f"Xử lý ngành {sector} - model tốt nhất: {model_type}")

                # FEATURE COLS PHẢI KHỚP CHÍNH XÁC VỚI LÚC TRAIN (14 feature)
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_60',
            'rsi_14',
            'macd', 'macd_signal',
            'bb_upper', 'bb_lower',
            'adx_14',
            'volume_sma_9'
        ]

        # Lọc feature tồn tại
        available_features = [f for f in feature_cols if f in group.columns]

        # KHÔNG bỏ qua dù thiếu feature → vẫn dự đoán được
        if not available_features:
            print(f"[WARNING] Ngành {sector} không có feature nào khả dụng. Bỏ qua.")
            continue

        print(f"[INFO] Sử dụng {len(available_features)}/{len(feature_cols)} features: {available_features}")
        X = group[available_features].fillna(0)

        # Tiếp tục load và predict...

        # LOAD MODEL TỐT NHẤT
        try:
            model_path = f"models/{sector}_best_{model_type}.pkl"
            loaded_model = joblib.load(model_path)
            print(f"[INFO] Đã load model tốt nhất từ {model_path} cho {sector}")

            trend_model = loaded_model
            momentum_model = loaded_model
            breakout_model = loaded_model

        except FileNotFoundError:
            print(f"[WARNING] Không tìm thấy {model_path} → fit dummy để test")
            from sklearn.ensemble import RandomForestClassifier
            dummy_model = RandomForestClassifier(random_state=42)
            if len(X) > 1:
                y_dummy = (group['Close'].pct_change().shift(-1) > 0).astype(int).dropna()
                X_fit = X.iloc[:len(y_dummy)]
                if len(X_fit) > 0:
                    dummy_model.fit(X_fit, y_dummy)
                    print(f"[INFO] Đã fit dummy model cho {sector}")
                    trend_model = dummy_model
                    momentum_model = dummy_model
                    breakout_model = dummy_model
                else:
                    print(f"[WARNING] Không đủ dữ liệu fit dummy cho {sector}. Bỏ qua.")
                    continue
            else:
                print(f"[WARNING] Nhóm {sector} quá nhỏ, bỏ qua.")
                continue

        except Exception as e:
            print(f"[ERROR] Lỗi load/fit model cho {sector}: {e}")
            continue

        # Dự đoán
        if hasattr(trend_model, "predict_proba"):
            probas = trend_model.predict_proba(X)
            result.loc[group.index, 'P_trend'] = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]

        if hasattr(momentum_model, "predict_proba"):
            probas_mom = momentum_model.predict_proba(X)
            result.loc[group.index, 'P_strong_up'] = probas_mom[:, -1]

        if hasattr(breakout_model, "predict_proba"):
            probas_bo = breakout_model.predict_proba(X)
            result.loc[group.index, 'P_breakout'] = probas_bo[:, 1] if probas_bo.shape[1] > 1 else probas_bo[:, 0]

    # Buy Signal - giữ ngưỡng thấp để test
    result['BuySignal'] = (
        (result['P_trend'] >= 0.45) &
        (result['P_strong_up'] >= 0.45) &
        (result['P_breakout'] >= 0.50) &
        (result['Volume'] > result['Volume_SMA_9']) &
        result['P_trend'].notna()
    )

    output_cols = [
        'Date', 'Ticker', 'Sector', 'Close',
        'P_trend', 'P_strong_up', 'P_breakout',
        'BuySignal'
    ]

    return result[output_cols]

if __name__ == "__main__":
    print("signal_engine.py ready.")
    print("Đã sẵn sàng load model từ thư mục models/")