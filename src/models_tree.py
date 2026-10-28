import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline_ridge(X_train, y_train, X_val, y_val):
    """Scikit-learn Pipeline을 활용하여 Data Leakage 없이 Scaler를 적용한 Baseline Ridge 모델"""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Train 내에서만 fit 수행
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    pipeline.fit(X_train, y_train)
    val_pred = pipeline.predict(X_val)
    return pipeline, val_pred


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Early Stopping이 적용된 정형 시계열 1티어 LightGBM 모델"""
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        importance_type="gain",
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    val_pred = model.predict(X_val)
    return model, val_pred