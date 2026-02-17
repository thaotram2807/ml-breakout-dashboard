import pandas as pd
from sklearn.preprocessing import StandardScaler
def prepare_data_inference(df, scaler, feature_names):

    df = df.copy()

    # One-hot ngành giống lúc train
    if "industry_level2" in df.columns:
        df["industry_level2"] = df["industry_level2"].fillna("Unknown")
        df = pd.get_dummies(df, columns=["industry_level2"], prefix="Sector")

    # 🔥 Đảm bảo đủ feature giống lúc train
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # 🔥 Lấy đúng thứ tự feature
    X = df[feature_names]

    # Scale
    X_scaled = scaler.transform(X)

    return df, X_scaled