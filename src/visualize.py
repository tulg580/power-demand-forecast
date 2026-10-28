import os

import matplotlib

matplotlib.use("Agg")  # 창 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt


def plot_predictions(y_true, preds: dict, out_path, days=7):
    """Test 구간 중간 지점부터 N일치 실제값 vs 모델별 예측값 비교 그래프"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    start = len(y_true) // 2
    end = start + 24 * days

    plt.figure(figsize=(14, 5))
    plt.plot(y_true.index[start:end], y_true.values[start:end],
             label="Actual", color="black", linewidth=2)
    for name, p in preds.items():
        plt.plot(y_true.index[start:end], p[start:end], label=name, alpha=0.8)
    plt.title(f"Actual vs Predicted Power Demand ({days} days of Test set)")
    plt.ylabel("Power demand (MWh)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_shap_summary(model, X, out_path, sample=2000):
    """LightGBM SHAP summary plot 저장 (shap 미설치 시 건너뜀)"""
    try:
        import shap
    except ImportError:
        print("  (shap 미설치 -> SHAP 분석 생략)")
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X_s = X.sample(min(sample, len(X)), random_state=42)
    shap_values = shap.TreeExplainer(model).shap_values(X_s)

    plt.figure()
    shap.summary_plot(shap_values, X_s, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True
