import numpy as np
import pandas as pd


def create_features(
    df: pd.DataFrame, target_col="power_demand", horizon=24
) -> pd.DataFrame:
    """[Data Leakage 방지 설계]

    24시간 앞(horizon=24)을 예측하는 문제이므로,
    Lag 및 Rolling 통계량은 최소 '24시간 이전(t-24)'의 과거 데이터만 사용합니다.
    """
    data = df.copy()

    # 1. 캘린더 및 삼각함수 순환 인코딩 (Cyclical Encoding)
    # 23시와 0시, 12월과 1월의 연속성을 보존
    hour = data.index.hour
    dayofweek = data.index.dayofweek
    month = data.index.month

    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)
    data["is_weekend"] = (dayofweek >= 5).astype(int)

    # 2. 도메인 특화 파생 변수 (불쾌지수 DI)
    # DI = 1.8 * T - 0.55 * (1 - RH/100) * (1.8 * T - 26) + 32
    T = data["temperature"]
    RH = data["humidity"] / 100.0
    data["discomfort_index"] = 1.8 * T - 0.55 * (1 - RH) * (1.8 * T - 26) + 32

    # 3. Lag Features (최소 horizon 이상 이전 값만 참조)
    # t 시점 예측 시점 기준: t-24(어제 동시간), t-48(이틀 전), t-168(지난주 동시간)
    data["lag_24"] = data[target_col].shift(horizon)
    data["lag_48"] = data[target_col].shift(horizon + 24)
    data["lag_168"] = data[target_col].shift(horizon + 144)

    # 4. Rolling 통계량 (shift(horizon)을 먼저 하여 미래 데이터 침범 방지)
    base_series = data[target_col].shift(horizon)
    data["rolling_mean_24h"] = base_series.rolling(window=24).mean()
    data["rolling_std_24h"] = base_series.rolling(window=24).std()
    data["rolling_mean_7d"] = base_series.rolling(window=168).mean()

    # 결측치 제거 (Lag 생성으로 인한 초기 NaN 제거)
    data.dropna(inplace=True)

    return data