import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_generator import generate_realistic_energy_data
from src.evaluate import calculate_metrics
from src.features import create_features
from src.models_pytorch import SEQ_LEN, predict_lstm, train_pytorch_lstm
from src.models_tree import train_baseline_ridge, train_lightgbm
from src.visualize import plot_predictions, plot_shap_summary

warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("  시계열 전력 수요 예측 회귀 프로젝트 파이프라인 가동")
    print("=" * 60)

    # 1. 데이터 생성
    print("\n[Step 1] 시계열 데이터 적재 중...")
    raw_df = generate_realistic_energy_data("2021-01-01", "2023-12-31")
    print(f"-> 총 샘플 수: {len(raw_df)} 시간 (약 {len(raw_df)//24} 일치)")

    # 2. 특징 엔지니어링
    print("\n[Step 2] 누수 방지 특징 엔지니어링 수행 중...")
    df_features = create_features(raw_df, target_col="power_demand", horizon=24)
    target_col = "power_demand"
    feature_cols = [c for c in df_features.columns if c != target_col]

    # 3. Train / Val / Test 3분할 (시간축 기준)
    #    Val  : Early Stopping 등 모델 선택 전용
    #    Test : 최종 성능 보고 전용 (학습/튜닝에 일절 사용하지 않음)
    print("\n[Step 3] 시계열 데이터 분할 (Train / Val / Test)")
    train_data = df_features.loc[:"2022-06-30 23:00:00"]
    val_data = df_features.loc["2022-07-01 00:00:00":"2022-12-31 23:00:00"]
    test_data = df_features.loc["2023-01-01 00:00:00":]

    X_train, y_train = train_data[feature_cols], train_data[target_col]
    X_val, y_val = val_data[feature_cols], val_data[target_col]
    X_test, y_test = test_data[feature_cols], test_data[target_col]

    for name, X in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        print(f"-> {name:<5}: {X.index.min()} ~ {X.index.max()} ({len(X)}건)")

    # 4-1. Ridge
    print("\n[Step 4-1] Baseline(Ridge Regression) 학습...")
    ridge_model = train_baseline_ridge(X_train, y_train)
    ridge_preds = ridge_model.predict(X_test)
    res_ridge = calculate_metrics(y_test, ridge_preds, model_name="Ridge")

    # 4-2. LightGBM
    print("[Step 4-2] GBDT(LightGBM) 학습 및 Early Stopping...")
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)
    print(f"   -> best_iteration: {lgbm_model.best_iteration_}")
    lgbm_preds = lgbm_model.predict(X_test)
    res_lgbm = calculate_metrics(y_test, lgbm_preds, model_name="LightGBM")

    # 4-3. LSTM
    print("[Step 4-3] PyTorch LSTM 학습 (신경망 시퀀스 모델링)...")
    scaler = StandardScaler()  # Train만 fit
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    lstm_model, y_scaler = train_pytorch_lstm(
        X_train_s, y_train, X_val_s, y_val, epochs=30
    )
    # Val 마지막 구간을 context로 붙여 Test 전 구간을 동일 y로 평가
    lstm_preds = predict_lstm(
        lstm_model, y_scaler, X_test_s, context=X_val_s, seq_len=SEQ_LEN
    )
    res_lstm = calculate_metrics(y_test, lstm_preds, model_name="PyTorch-LSTM")

    # 5. 결과표
    print("\n" + "=" * 60)
    print("  모델별 최종 Test 성능 평가 결과표 (Metrics Benchmark)")
    print("=" * 60)
    summary_df = pd.DataFrame([res_ridge, res_lgbm, res_lstm])
    print(summary_df.to_markdown(index=False))

    # 6. Feature Importance
    print("\n[Step 5] LightGBM 주요 변수 중요도 (Top 5)")
    importances = pd.Series(
        lgbm_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    for rank, (feat, score) in enumerate(importances.head(5).items(), 1):
        print(f"  {rank}위: {feat:<18} (Gain: {score:.2f})")

    # 7. 시각화 저장 (outputs/)
    print("\n[Step 6] 시각화 저장 (outputs/)")
    plot_predictions(
        y_test,
        {"Ridge": ridge_preds, "LightGBM": lgbm_preds, "LSTM": lstm_preds},
        "outputs/predictions.png",
    )
    print("  -> outputs/predictions.png")
    try:
        if plot_shap_summary(lgbm_model, X_test, "outputs/shap_summary.png"):
            print("  -> outputs/shap_summary.png")
    except Exception as e:  # SHAP 호환 문제로 전체 파이프라인이 죽지 않도록
        print(f"  (SHAP 생략: {e})")

    print("\n 파이프라인 수행이 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    main()
