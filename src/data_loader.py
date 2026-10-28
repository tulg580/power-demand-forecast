import pandas as pd


def load_kpx_hourly_demand(path: str) -> pd.DataFrame:
    """한국전력거래소 '시간별 전국 전력수요량' CSV(wide: 날짜 + 1시~24시)를
    datetime 인덱스의 long 형태(power_demand, 단위 MWh)로 변환.

    - 'N시' 열은 시간 종료 기준(1시=00~01시 구간)으로 보고 N-1시로 매핑함.
      (1시→00:00, 24시→23:00 / 같은 날짜 안에서 시계열이 이어지도록)
    - 끝에 붙은 빈 행은 제거함.
    """
    raw, last_err = None, None
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            raw = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
    if raw is None:
        raise last_err

    date_col = raw.columns[0]
    raw = raw.dropna(subset=[date_col]).copy()
    raw[date_col] = pd.to_datetime(raw[date_col])

    long = raw.melt(
        id_vars=date_col, var_name="hour_label", value_name="power_demand"
    )
    hour_end = long["hour_label"].str.extract(r"(\d+)")[0].astype(int)
    long["datetime"] = long[date_col] + pd.to_timedelta(hour_end - 1, unit="h")

    df = long.set_index("datetime").sort_index()[["power_demand"]].astype(float)

    # 무결성 점검: 중복 / 누락 시각
    assert not df.index.duplicated().any(), "중복된 시각이 있음"
    df = df.asfreq("h")
    n_missing = int(df["power_demand"].isna().sum())
    if n_missing:
        print(f"  [경고] 누락/결측 {n_missing}개 -> 선형 보간(최대 3시간)")
        df["power_demand"] = df["power_demand"].interpolate(limit=3)
        df = df.dropna()
    return df


def load_weather(path: str) -> pd.DataFrame:
    """datetime, temperature, humidity 컬럼을 가진 시간별 기상 CSV 로드."""
    w = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
    return w[["temperature", "humidity"]].sort_index()
