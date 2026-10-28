import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data_generator import generate_realistic_energy_data
from src.evaluate import calculate_metrics
from src.features import create_features
from src.models_pytorch import train_pytorch_lstm
from src.models_tree import train_baseline_ridge, train_lightgbm

warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("  시계열 전력 수요 예측 회귀 프로젝트 파이프라인 가동")
    print("=" * 60)

    # 1. 데이터 수집 및 생성
    print("\n[Step 1] 시계열 데이터 적재 중...")
    raw_df = generate_realistic_energy_data("2021-01-01", "2023-12-31")
    print(f"-> 총 샘플 수: {len(raw_df)} 시간 (약 {len(raw_df)//24} 일치)")

    # 2. 특징 엔지니어링 (누수 방지 Lag & Rolling)
    print("\n[Step 2] 누수 방지 특징 엔지니어링 수행 중...")
    df_features = create_features(raw_df, target_col="power_demand", horizon=24)

    target_col = "power_demand"
    feature_cols = [c for c in df_features.columns if c != target_col]

    # 3. TimeSeries Split (시간 축 기준 절대적 분할 -> 누수 차단)
    print("\n[Step 3] 시계열 데이터 분할 (Temporal Split)")
    train_end = "2022-12-31 23:00:00"
    val_end = "2023-12-31 23:00:00"

    train_data = df_features.loc[:train_end]
    val_data = df_features.loc["2023-01-01 00:00:00":val_end]

    X_train, y_train = train_data[feature_cols], train_data[target_col]
    X_val, y_val = val_data[feature_cols], val_data[target_col]

    print(
        f"-> Train 세트 : {X_train.index.min()} ~ {X_train.index.max()} ({len(X_train)}건)"
    )
    print(
        f"-> Val(Test) 세트: {X_val.index.min()} ~ {X_val.index.max()} ({len(X_val)}건)"
    )

    # 4. 모델 1: Scikit-learn Baseline (Ridge)
    print("\n[Step 4-1] Baseline(Ridge Regression) 학습...")
    ridge_model, ridge_preds = train_baseline_ridge(
        X_train, y_train, X_val, y_val
    )
    res_ridge = calculate_metrics(y_val, ridge_preds, model_name="Ridge")

    # 5. 모델 2: GBDT (LightGBM)
    print("[Step 4-2] GBDT(LightGBM) 학습 및 Early Stopping...")
    lgbm_model, lgbm_preds = train_lightgbm(X_train, y_train, X_val, y_val)
    res_lgbm = calculate_metrics(y_val, lgbm_preds, model_name="LightGBM")

    # 6. 모델 3: Deep Learning (PyTorch LSTM)
    print("[Step 4-3] PyTorch LSTM 학습 (신경망 시퀀스 모델링)...")
    # 전처리 누수 방지: Scaler는 반드시 Train 셋만 fit!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lstm_model, lstm_preds, y_val_lstm = train_pytorch_lstm(
        X_train_scaled, y_train, X_val_scaled, y_val, epochs=15
    )
    res_lstm = calculate_metrics(
        y_val_lstm, lstm_preds, model_name="PyTorch-LSTM"
    )

    # 7. 최종 성능 벤치마크 비교표 출력
    print("\n" + "=" * 60)
    print("  모델별 최종 검증 성능 평가 결과표 (Metrics Benchmark)")
    print("=" * 60)
    summary_df = pd.DataFrame([res_ridge, res_lgbm, res_lstm])
    print(summary_df.to_markdown(index=False))

    # 8. Feature Importance (LightGBM)
    print("\n[Step 5] LightGBM 주요 변수 중요도 (Top 5)")
    importances = pd.Series(
        lgbm_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    for rank, (feat, score) in enumerate(importances.head(5).items(), 1):
        print(f"  {rank}위: {feat:<18} (Gain: {score:.2f})")

    print("\n 파이프라인 수행이 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    main()