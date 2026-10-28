import numpy as np
import pandas as pd


def _holiday_features(index: pd.DatetimeIndex, extra_holidays=()) -> pd.DataFrame:
    """공휴일/연휴 관련 일 단위 플래그를 시간 단위 인덱스로 매핑.

    is_holiday   : 법정/임시/대체 공휴일 (holidays 패키지 + extra_holidays)
    is_day_off   : 공휴일 또는 주말
    near_holiday : 쉬는 날이 아니지만 '공휴일을 포함한 연속 휴일' 바로 앞/뒤 날 (징검다리·연휴 전후)
    """
    import holidays

    years = range(index.min().year - 1, index.max().year + 2)
    hol = holidays.country_holidays("KR", years=years)
    extra = {pd.Timestamp(d).date() for d in extra_holidays}

    days = pd.date_range(
        index.min().normalize() - pd.Timedelta(days=7),
        index.max().normalize() + pd.Timedelta(days=7),
        freq="D",
    )
    is_hol = pd.Series([(d.date() in hol) or (d.date() in extra) for d in days], index=days)
    is_off = is_hol | (days.dayofweek >= 5)

    block = (is_off != is_off.shift()).cumsum()
    block_has_hol = is_hol.groupby(block).transform("any")
    off_with_hol = is_off & block_has_hol
    near = (~is_off) & (
        off_with_hol.shift(1, fill_value=False) | off_with_hol.shift(-1, fill_value=False)
    )

    day_key = index.normalize()
    return pd.DataFrame(
        {
            "is_holiday": is_hol.reindex(day_key).astype(int).values,
            "is_day_off": is_off.reindex(day_key).astype(int).values,
            "near_holiday": near.reindex(day_key).astype(int).values,
        },
        index=index,
    )


def create_features_real(
    df: pd.DataFrame,
    target_col: str = "power_demand",
    horizon: int = 24,
    extra_holidays=(),
) -> pd.DataFrame:
    """실제 데이터용 특징 생성 (누수 방지 규칙은 features.py와 동일: t-horizon 이전만 사용).

    기상 컬럼(temperature, humidity)이 있으면 기상 피처를 추가하고, 없으면 캘린더/공휴일/Lag만 사용.
    """
    data = df.copy()
    hour, dow, month = data.index.hour, data.index.dayofweek, data.index.month

    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    data["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)
    data["is_weekend"] = (dow >= 5).astype(int)

    data = data.join(_holiday_features(data.index, extra_holidays))

    if {"temperature", "humidity"} <= set(data.columns):
        T, RH = data["temperature"], data["humidity"] / 100.0
        data["discomfort_index"] = 1.8 * T - 0.55 * (1 - RH) * (1.8 * T - 26) + 32

    data["lag_24"] = data[target_col].shift(horizon)
    data["lag_48"] = data[target_col].shift(horizon + 24)
    data["lag_168"] = data[target_col].shift(horizon + 144)

    base = data[target_col].shift(horizon)
    data["rolling_mean_24h"] = base.rolling(24).mean()
    data["rolling_std_24h"] = base.rolling(24).std()
    data["rolling_mean_7d"] = base.rolling(168).mean()

    return data.dropna()
