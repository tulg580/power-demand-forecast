"""실제 KPX 시간별 전국 전력수요량 데이터 검증 (Rolling-origin, 월 단위 Expanding Window).

사용:
  python main_real.py                                   # 기상 없이 (캘린더/공휴일/Lag)
  python main_real.py --weather data/weather_2025.csv   # 기상 포함
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

from src.data_loader import load_kpx_hourly_demand, load_weather
from src.evaluate import calculate_metrics
from src.features_real import create_features_real
from src.models_tree import train_baseline_ridge, train_lightgbm
from src.visualize import plot_monthly_mape, plot_range, plot_shap_summary

warnings.filterwarnings("ignore")
TARGET = "power_demand"


def day_type(feat: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(feat["is_holiday"] == 1, "공휴일",
                 np.where(feat["is_weekend"] == 1, "주말", "평일")),
        index=feat.index,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/한국전력거래소_시간별_전국_전력수요량_20251231.csv")
    ap.add_argument("--weather", default=None)
    ap.add_argument("--first-test-month", default="2025-07")
    ap.add_argument("--out", default="outputs_real")
    ap.add_argument("--val-days", type=int, default=28)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 60)
    print("  실제 데이터 검증: KPX 시간별 전국 전력수요량")
    print("=" * 60)

    df = load_kpx_hourly_demand(args.data)
    print(f"\n[Data] {df.index.min()} ~ {df.index.max()} ({len(df)}시간)")
    if args.weather:
        df = df.join(load_weather(args.weather), how="left").dropna()
        print("[Data] 기상 데이터 포함 (temperature, humidity)")
    else:
        print("[Data] 기상 데이터 없음 -> 캘린더/공휴일/Lag 피처만 사용")

    feat = create_features_real(df, TARGET, horizon=24)
    feature_cols = [c for c in feat.columns if c != TARGET]
    print(f"[Features] {len(feature_cols)}개: {feature_cols}")
    print(f"[Features] 사용 가능 구간: {feat.index.min()} ~ {feat.index.max()} ({len(feat)}건)")

    # ---- Rolling-origin: 매 월을 Test로, 그 이전 전부를 Train으로 (Train 끝 N일은 Val) ----
    months = pd.period_range(args.first_test_month, feat.index.max(), freq="M")
    records, last = [], None
    print("\n[Rolling-origin 검증]")
    for m in months:
        t0, t1 = m.start_time, (m + 1).start_time
        train, test = feat[feat.index < t0], feat[(feat.index >= t0) & (feat.index < t1)]
        if len(test) == 0 or len(train) < 24 * 60:
            continue
        cut = train.index.max() - pd.Timedelta(days=args.val_days)
        tr, va = train[train.index <= cut], train[train.index > cut]

        ridge = train_baseline_ridge(train[feature_cols], train[TARGET])
        lgbm = train_lightgbm(tr[feature_cols], tr[TARGET], va[feature_cols], va[TARGET])
        preds = {
            "Naive(lag_24)": test["lag_24"].values,
            "Naive(lag_168)": test["lag_168"].values,
            "Ridge": ridge.predict(test[feature_cols]),
            "LightGBM": lgbm.predict(test[feature_cols]),
        }
        rec = pd.DataFrame(preds, index=test.index)
        rec["y"] = test[TARGET].values
        rec["month"] = str(m)
        rec["day_type"] = day_type(test).values
        records.append(rec)
        last = (lgbm, test)
        print(f"  Test {m}: train {len(train)}건 / test {len(test)}건 / LGBM best_iter {lgbm.best_iteration_}")

    res = pd.concat(records)
    models = ["Naive(lag_24)", "Naive(lag_168)", "Ridge", "LightGBM"]

    def table(group=None):
        rows = []
        for key, g in ([("전체", res)] if group is None else res.groupby(group)):
            for name in models:
                r = calculate_metrics(g["y"], g[name], name)
                r[group or "구간"] = key
                rows.append(r)
        return pd.DataFrame(rows)

    overall = table()
    print("\n" + "=" * 60)
    print(f"  전체 Test 구간 ({res.index.min().date()} ~ {res.index.max().date()}, {len(res)}건)")
    print("=" * 60)
    print(overall.drop(columns="구간").to_markdown(index=False))

    by_month = table("month").pivot(index="month", columns="Model", values="MAPE (%)")[models]
    by_type = table("day_type").pivot(index="day_type", columns="Model", values="MAPE (%)")[models]
    print("\n[월별 MAPE (%)]");   print(by_month.to_markdown())
    print("\n[요일 유형별 MAPE (%)]"); print(by_type.to_markdown())
    print("\n[요일 유형별 샘플 수]"); print(res["day_type"].value_counts().to_string())

    overall.to_csv(f"{args.out}/overall_metrics.csv", index=False)
    by_month.to_csv(f"{args.out}/monthly_mape.csv")
    by_type.to_csv(f"{args.out}/daytype_mape.csv")

    # ---- 시각화 ----
    plot_monthly_mape(by_month, f"{args.out}/monthly_mape.png")
    s, e = pd.Timestamp("2025-09-29"), pd.Timestamp("2025-10-13")
    if res.index.min() <= s and res.index.max() >= e:
        plot_range(
            res["y"], {n: res[n].values for n in ["Naive(lag_168)", "LightGBM"]},
            s, e, f"{args.out}/chuseok_week.png",
            title="2025 Chuseok holidays: Actual vs Predicted",
        )
    lgbm, test = last
    imp = pd.Series(lgbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n[LightGBM 변수 중요도 Top 5 (마지막 fold)]")
    for i, (f, v) in enumerate(imp.head(5).items(), 1):
        print(f"  {i}위: {f:<18} (Gain: {v:.2f})")
    try:
        plot_shap_summary(lgbm, test[feature_cols], f"{args.out}/shap_summary.png")
    except Exception as ex:
        print(f"  (SHAP 생략: {ex})")
    print(f"\n 결과 저장: {args.out}/")


if __name__ == "__main__":
    main()
