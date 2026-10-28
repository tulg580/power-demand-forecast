#  기상 및 캘린더 데이터를 활용한 시간별 전력 수요량 예측 회귀 모델

> **과거 24시간 이전 데이터 및 기상 예보 데이터를 기반으로 향후 24시간 전력 수요량(MWh)을 예측하는 End-to-End 시계열 회귀 프로젝트**

---

##  1. 프로젝트 개요

* **목표**: 일간/계절 주기성과 기온의 비선형적 관계를 반영하여 24시간 뒤의 전력 수요량을 정밀 예측 (회귀 문제)
* **주요 해결 과제**:
  * 시계열 정형 데이터 기반 고도화된 특징 설계 (Feature Engineering)
  * 다중 스텝 시계열 예측 시 필연적으로 발생하는 **데이터 누수(Data Leakage) 원천 차단**
  * `Scikit-learn`, `LightGBM`, `PyTorch(LSTM)` 3개 모델 벤치마크 및 비즈니스 관점 오차 평가
* **기술 스택**: Python, Pandas, Scikit-learn, LightGBM, PyTorch, Matplotlib, SHAP

---

##  2. 프로젝트 구조

```
.
├── main.py                  # 전체 파이프라인 실행 진입점
├── requirements.txt
└── src/
    ├── data_generator.py    # 기상 + 전력 수요 시계열 데이터 생성
    ├── features.py          # 누수 방지 특징 엔지니어링
    ├── models_tree.py       # Ridge(Baseline), LightGBM
    ├── models_pytorch.py    # PyTorch LSTM 모델 및 학습 루프
    └── evaluate.py          # MAE / RMSE / MAPE 산출
```

---

##  3. 파이프라인

| 단계 | 내용 |
|------|------|
| 1. 데이터 생성 | 2021-01-01 ~ 2023-12-31 시간 단위 데이터 (약 26,000 샘플) |
| 2. 특징 엔지니어링 | 순환 인코딩, 불쾌지수, Lag, Rolling 통계량 생성 (전부 t-24 이전 데이터만 사용) |
| 3. 시계열 분할 | 시간축 기준 절대 분할 (Train: 2021~2022 / Val: 2023) |
| 4. 모델 학습 | Ridge → LightGBM → PyTorch LSTM |
| 5. 평가 | MAE, RMSE, MAPE 벤치마크 표 출력 |
| 6. 해석 | LightGBM Feature Importance (Gain 기준 Top 5) |

---

##  4. 데이터

실제 전력 수요 및 기상 관측 패턴을 모사한 **합성 데이터**를 사용함 (`np.random.seed(42)` 고정으로 재현 가능).

* **기온**: 연간 계절성(여름 ≈ 30℃, 겨울 ≈ 0℃) + 일교차 + 가우시안 노이즈
* **습도**: 일주기 패턴 + 노이즈 (10~100% 클리핑)
* **전력 수요**: 기저부하(50,000 MWh) + 기온과의 **비선형 U자형 관계** + 일간 주기 + 주말 감소 효과 + 노이즈
  * 22℃ 초과 시 냉방 수요 급증, 12℃ 미만 시 난방 수요 급증, 그 사이 구간은 최저

---

##  5. 특징 엔지니어링 및 데이터 누수 방지

24시간 뒤를 예측하는 문제라서, 예측 시점 t에서 **실제로 알 수 있는 정보만** 입력으로 사용하도록 설계함.

### 5-1. 특징 목록

| 분류 | 변수 | 설명 |
|------|------|------|
| 기상 | `temperature`, `humidity` | 기상 예보값으로 가정 |
| 캘린더 | `hour_sin/cos`, `month_sin/cos`, `is_weekend` | 삼각함수 순환 인코딩으로 23시↔0시, 12월↔1월 연속성 보존 |
| 도메인 | `discomfort_index` | 기온·습도 기반 불쾌지수(DI) |
| Lag | `lag_24`, `lag_48`, `lag_168` | 어제 동시간 / 이틀 전 / 지난주 동시간 수요 |
| Rolling | `rolling_mean_24h`, `rolling_std_24h`, `rolling_mean_7d` | 최근 24시간 / 7일 통계량 |

### 5-2. 누수 차단 장치

* **Lag**: 최소 `horizon(24h)` 이상 이전 값만 참조
* **Rolling**: `shift(horizon)`을 **먼저** 적용한 뒤 rolling 계산 → 미래 값 침범 차단
* **분할**: 랜덤 분할이 아닌 시간축 기준 Temporal Split
* **전처리**: Scaler는 Train 셋에만 `fit` 후 Val에 `transform` (Ridge는 `Pipeline`으로 캡슐화)

---

##  6. 모델

| 모델 | 역할 | 핵심 설정 |
|------|------|-----------|
| **Ridge Regression** | 선형 Baseline | `StandardScaler` + `Ridge(alpha=1.0)` Pipeline |
| **LightGBM** | 정형 시계열 GBDT | `n_estimators=1000`, `lr=0.03`, `num_leaves=31`, subsample/colsample 0.8, Early Stopping(30 rounds) |
| **PyTorch LSTM** | 시퀀스 딥러닝 | 2-layer LSTM(hidden 64, dropout 0.2) + FC head, 입력 시퀀스 길이 24, Adam(lr=0.003), Gradient Clipping(1.0) |

---

##  7. 평가 지표

비즈니스 관점에서 해석 가능한 3개 지표 사용함.

* **MAE (MWh)**: 평균적으로 몇 MWh 틀리는지
* **RMSE (MWh)**: 큰 오차에 더 큰 페널티 (피크 수요 예측 실패 민감)
* **MAPE (%)**: 수요 규모 대비 오차 비율

### 결과

> 아래 표는 `python main.py` 실행 후 출력된 값으로 채울 것

| Model | MAE (MWh) | RMSE (MWh) | MAPE (%) |
|-------|-----------|------------|----------|
| Ridge | - | - | - |
| LightGBM | - | - | - |
| PyTorch-LSTM | - | - | - |

---

##  8. 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 전체 파이프라인 실행
python main.py
```

* 데이터 생성 → 특징 생성 → 3개 모델 학습 → 성능 비교표 → Feature Importance 순으로 자동 수행됨

---

##  9. 한계 및 향후 개선 방향

* 합성 데이터 기반이라 실제 전력 데이터(공휴일, 특수일, 산업 부하 등) 적용 시 재검증 필요
* Validation 셋이 Early Stopping과 최종 평가에 함께 쓰이고 있어, 별도의 Test 셋 분리 필요
* 단일 시점(t+24) 예측 구조라 향후 24시간 전체를 예측하는 Multi-step 확장 가능
* 공휴일 변수 추가, Optuna 하이퍼파라미터 튜닝, SHAP 기반 변수 해석 추가 예정
