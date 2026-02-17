import joblib
import pandas as pd

model = joblib.load("xgboost_breakout_best.pkl")
scaler = joblib.load("scaler.pkl")
feature_list = joblib.load("feature_list.pkl")

df = pd.read_csv("VN100_Historical_Data_2020_2025.csv")
df.columns = df.columns.str.strip().str.lower()

print("Feature list:", feature_list)
print("Columns:", df.columns.tolist())

missing = [f for f in feature_list if f not in df.columns]
print("Missing features:", missing)