import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SEQ_LEN = 24


class TimeSeriesDataset(Dataset):
    """길이 seq_len 윈도우의 '마지막 행(t)'에 해당하는 타깃 y[t]를 예측.

    features.py에서 row t의 피처는 (t-24 이전 lag/rolling + t시점 기상예보)로 이미
    구성되어 있으므로, 윈도우 마지막 행이 타깃 시점 t와 일치해야 함.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        end = idx + self.seq_len
        return self.X[idx:end], self.y[end - 1]


class LSTMRegressor(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_pytorch_lstm(
    X_train_scaled,
    y_train,
    X_val_scaled,
    y_val,
    epochs=30,
    batch_size=64,
    patience=5,
    seed=42,
    device=None,
):
    """LSTM 학습. Val loss 기준 Early Stopping + 최적 가중치 복원.

    타깃(y)도 Train 기준으로 StandardScaler 적용 (수요값이 5만 단위라 수렴 안정화 목적).
    반환: (model, y_scaler)  -> 예측 시 predict_lstm()으로 원래 단위 복원.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_scaler = StandardScaler()
    y_tr = y_scaler.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()
    y_va = y_scaler.transform(np.asarray(y_val).reshape(-1, 1)).ravel()

    train_ds = TimeSeriesDataset(X_train_scaled, y_tr)
    val_ds = TimeSeriesDataset(X_val_scaled, y_va)
    # 윈도우 단위 샘플이라 train 셔플은 시간 누수 없음 (각 샘플이 독립된 과거 24h 창)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = LSTMRegressor(input_dim=X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * bx.size(0)
        val_loss /= len(val_ds)
        print(f"   epoch {epoch:02d} | val MSE(scaled): {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())  # 얕은 복사 방지
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"   -> Early stopping (epoch {epoch})")
                break

    model.load_state_dict(best_state)
    return model, y_scaler


def predict_lstm(
    model, y_scaler, X_scaled, context=None, seq_len=SEQ_LEN, device=None
):
    """X_scaled의 모든 행에 대한 예측값을 원래 단위(MWh)로 반환.

    context: 직전 구간(예: Val)의 스케일된 피처. 앞쪽 seq_len-1개 행을 붙여서
             Test의 첫 행부터 예측 가능 -> 다른 모델과 동일한 y로 공정 비교.
    """
    device = device or next(model.parameters()).device
    if context is not None:
        X_scaled = np.vstack([context[-(seq_len - 1):], X_scaled])

    ds = TimeSeriesDataset(X_scaled, np.zeros(len(X_scaled)), seq_len)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in loader:
            preds.append(model(bx.to(device)).cpu().numpy())
    preds = np.concatenate(preds).reshape(-1, 1)
    return y_scaler.inverse_transform(preds).ravel()
