import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """회귀 평가 지표 MAE, RMSE, MAPE를 산출합니다."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "Model": model_name,
        "MAE (MWh)": round(mae, 2),
        "RMSE (MWh)": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
    }
