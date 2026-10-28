import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 24):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_len],
            self.y[idx + self.seq_len],  # seq_len 직후 시점 예측
        )


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
        # 마지막 타임스텝의 히든 상태를 Dense Layer에 전달
        last_step_out = out[:, -1, :]
        return self.fc(last_step_out).squeeze(-1)


def train_pytorch_lstm(
    X_train_scaled,
    y_train,
    X_val_scaled,
    y_val,
    epochs=20,
    batch_size=64,
    device="cpu",
):
    """PyTorch 학습 및 검증 루프 (학습 중 Leakage 방지 및 Early Stopping 포함)"""
    seq_len = 24
    train_ds = TimeSeriesDataset(X_train_scaled, y_train.values, seq_len)
    val_ds = TimeSeriesDataset(X_val_scaled, y_val.values, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMRegressor(input_dim=X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )  # 그래디언트 클리핑
            optimizer.step()

        # 검증 루프
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                val_loss += criterion(preds, batch_y).item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict().copy()

    model.load_state_dict(best_weights)

    # 최종 Validation 추론값 계산
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            preds = model(batch_x.to(device))
            val_preds.extend(preds.cpu().numpy())

    # 시퀀스 길이로 인해 제외된 앞부분 slice 보정
    aligned_y_val = y_val.values[seq_len:]
    return model, np.array(val_preds), aligned_y_val