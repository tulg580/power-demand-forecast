import numpy as np
import pandas as pd


def generate_realistic_energy_data(
    start_date="2021-01-01", end_date="2023-12-31"
):
    """실제 전력 수요 및 기상 관측치와 유사한 시계열 데이터를 생성합니다."""
    date_range = pd.date_range(start=start_date, end=end_date, freq="h")
    n = len(date_range)
    np.random.seed(42)

    df = pd.DataFrame({"datetime": date_range})
    df.set_index("datetime", inplace=True)

    # 1. 기상 데이터 생성 (계절 + 일교차)
    day_of_year = df.index.dayofyear
    hour = df.index.hour
    base_temp = 15 + 15 * np.sin(
        2 * np.pi * (day_of_year - 100) / 365
    )  # 계절성 (여름 30도, 겨울 0도 안팎)
    daily_temp = 5 * np.sin(2 * np.pi * (hour - 8) / 24)  # 일교차
    df["temperature"] = (
        base_temp + daily_temp + np.random.normal(0, 2.0, size=n)
    )
    df["humidity"] = np.clip(
        60 + 20 * np.cos(2 * np.pi * hour / 24) + np.random.normal(0, 10, n),
        10,
        100,
    )

    # 2. 전력 수요(MWh) 생성: 기온과의 비선형 관계(U자형) + 주기성 + 노이즈
    # 기준 온도 18도 전후에서 최저, 극단적 저온/고온에서 냉난방 급증
    temp_effect = np.where(
        df["temperature"] > 22,
        (df["temperature"] - 22) ** 1.8 * 80,
        np.where(
            df["temperature"] < 12, (12 - df["temperature"]) ** 1.5 * 60, 0
        ),
    )

    # 일간 주기(낮 피크), 주간 주기(주말 수요 감소)
    daily_pattern = 1000 * np.sin(2 * np.pi * (hour - 6) / 24)
    weekday_pattern = np.where(
        df.index.dayofweek >= 5, -800, 400
    )  # 주말 감소

    base_load = 50000  # 기본 기저부하
    noise = np.random.normal(0, 300, size=n)

    df["power_demand"] = (
        base_load + temp_effect + daily_pattern + weekday_pattern + noise
    )

    return df