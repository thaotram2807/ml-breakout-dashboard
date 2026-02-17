# main_train.py

from data_processor import prepare_data
from train_logistic import train_logistic

path = r"D:\code\model\dataset_breakout_ml.csv"

# Lấy dữ liệu đã scale
X_train, X_test, y_train, y_test = prepare_data(path)

# Train Logistic (Baseline)
log_model = train_logistic(X_train, X_test, y_train, y_test)