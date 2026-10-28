import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """회귀 평가 지표 MAE, RMSE, MAPE를 산출합니다."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    results = {
        "Model": model_name,
        "MAE (MWh)": round(mae, 2),
        "RMSE (MWh)": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
    }
    return results