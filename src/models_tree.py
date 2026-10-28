import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline_ridge(X_train, y_train):
    """Pipeline으로 Scaler를 Train 내에서만 fit하는 Baseline Ridge 모델"""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Val 셋으로 Early Stopping을 거는 LightGBM (최종 평가는 Test 셋에서 따로 수행)"""
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,  # subsample은 이 값이 1 이상이어야 실제로 적용됨
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
    return model
