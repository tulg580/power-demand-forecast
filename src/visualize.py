import os

import matplotlib

matplotlib.use("Agg")  # 창 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt


def _save(out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_range(y_true, preds: dict, start, end, out_path, title=None):
    """지정 구간의 실제값 vs 모델별 예측값 (y_true: datetime 인덱스 Series, preds: 같은 길이 배열)"""
    mask = (y_true.index >= start) & (y_true.index < end)
    plt.figure(figsize=(14, 5))
    plt.plot(y_true.index[mask], y_true.values[mask], label="Actual", color="black", linewidth=2)
    for name, p in preds.items():
        plt.plot(y_true.index[mask], p[mask], label=name, alpha=0.8)
    plt.title(title or "Actual vs Predicted Power Demand")
    plt.ylabel("Power demand (MWh)")
    plt.legend()
    plt.grid(alpha=0.3)
    _save(out_path)


def plot_predictions(y_true, preds: dict, out_path, days=7):
    """Test 구간 중간 지점부터 N일치 실제값 vs 모델별 예측값 비교 그래프"""
    start = len(y_true) // 2
    end = start + 24 * days
    mask = (y_true.index >= y_true.index[start]) & (y_true.index < y_true.index[end])
    plot_range(
        y_true, preds, y_true.index[start], y_true.index[end], out_path,
        title=f"Actual vs Predicted Power Demand ({days} days of Test set)",
    )


def plot_monthly_mape(table, out_path, title="Monthly MAPE by model"):
    """table: index=월, columns=모델, values=MAPE(%)"""
    ax = table.plot(kind="bar", figsize=(11, 5))
    ax.set_ylabel("MAPE (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    _save(out_path)


def plot_shap_summary(model, X, out_path, sample=2000):
    """LightGBM SHAP summary plot 저장 (shap 미설치 시 건너뜀)"""
    try:
        import shap
    except ImportError:
        print("  (shap 미설치 -> SHAP 분석 생략)")
        return False

    X_s = X.sample(min(sample, len(X)), random_state=42)
    shap_values = shap.TreeExplainer(model).shap_values(X_s)

    plt.figure()
    shap.summary_plot(shap_values, X_s, show=False)
    _save(out_path)
    return True
