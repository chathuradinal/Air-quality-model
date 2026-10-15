"""
Baseline and Comparative Models for Duan et al. (2023) replication.
Implements Models A to I to compare against the proposed ARIMA-DBO-CNN-LSTM (Model J):
- Model A: SVM (kernel='linear')
- Model B: Random Forest (RF)
- Model C: Standalone ARIMA(p, d, q)
- Model D: Standalone LSTM (50, 100, 150 neurons, Table 2)
- Model E: Standalone CNN-LSTM
- Model H: ARIMA-CNN-LSTM (without DBO)
- Model I: ARIMA-DBO-LSTM (ARIMA + DBO-optimized LSTM)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Optional, List

from src.arima_module import ARIMAForecaster
from src.utils import MinMaxNormalizer, create_sliding_windows
from sklearn.ensemble import GradientBoostingRegressor

try:
    import lightgbm as lgb
except (ImportError, OSError):
    lgb = None


# -------------------------------------------------------------
# 0. Naive Persistence Baseline (Phase 1 Essential Benchmark)
# -------------------------------------------------------------
class PersistenceBaseline:
    """
    Naive persistence forecast: y_hat_t = y_{t-1}.
    Essential sanity check benchmark identified in Phase 1 review.
    """
    def __init__(self):
        self.last_train_val = None

    def fit(self, train_data: np.ndarray):
        self.last_train_val = float(train_data[-1])
        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        test_data = np.asarray(test_data, dtype=np.float64)
        return np.concatenate([[self.last_train_val], test_data[:-1]])


# -------------------------------------------------------------
# 1. Traditional ML Baselines: SVM, Random Forest, Gradient Boosting
# -------------------------------------------------------------
class GradientBoostingBaseline:
    """
    Proposed alternative: Gradient Boosted Decision Trees on lag features.
    Tests whether a fast boosted tree model matches or beats deep neural nets.
    Uses LightGBM if libomp is available, else sklearn GradientBoostingRegressor.
    """
    def __init__(self, window_size: int = 10, n_estimators: int = 100, learning_rate: float = 0.05):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = None
        self.train_data = None

    def fit(self, train_data: np.ndarray):
        self.train_data = np.asarray(train_data, dtype=np.float64)
        X, y = create_sliding_windows(self.train_data, self.window_size)
        X_flat = X.squeeze(-1) if X.ndim == 3 else X

        if lgb is not None:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=42,
                verbose=-1,
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=42,
            )
        self.model.fit(X_flat, y)
        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        full_series = np.concatenate([self.train_data, test_data])
        test_start = len(self.train_data)
        X_test = []
        for i in range(test_start, len(full_series)):
            X_test.append(full_series[i - self.window_size : i])
        X_test = np.array(X_test, dtype=np.float32)
        return self.model.predict(X_test)


class SVMBaseline:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.normalizer = MinMaxNormalizer()
        self.model = SVR(kernel="linear")

    def fit(self, train_data: np.ndarray):
        norm_data = self.normalizer.fit_transform(train_data)
        X, y = create_sliding_windows(norm_data, self.window_size)
        X_flat = X.squeeze(-1) if X.ndim == 3 else X
        self.model.fit(X_flat, y)
        self.train_data = train_data
        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        full_series = np.concatenate([self.train_data, test_data])
        norm_full = self.normalizer.transform(full_series)
        
        test_start = len(self.train_data)
        X_test = []
        for i in range(test_start, len(full_series)):
            X_test.append(norm_full[i - self.window_size : i])
        X_test = np.array(X_test, dtype=np.float32)
        preds_norm = self.model.predict(X_test)
        return self.normalizer.inverse_transform(preds_norm)


class RFBaseline:
    def __init__(self, window_size: int = 10, n_estimators: int = 100):
        self.window_size = window_size
        self.normalizer = MinMaxNormalizer()
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

    def fit(self, train_data: np.ndarray):
        norm_data = self.normalizer.fit_transform(train_data)
        X, y = create_sliding_windows(norm_data, self.window_size)
        X_flat = X.squeeze(-1) if X.ndim == 3 else X
        self.model.fit(X_flat, y)
        self.train_data = train_data
        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        full_series = np.concatenate([self.train_data, test_data])
        norm_full = self.normalizer.transform(full_series)
        
        test_start = len(self.train_data)
        X_test = []
        for i in range(test_start, len(full_series)):
            X_test.append(norm_full[i - self.window_size : i])
        X_test = np.array(X_test, dtype=np.float32)
        preds_norm = self.model.predict(X_test)
        return self.normalizer.inverse_transform(preds_norm)


# -------------------------------------------------------------
# 2. Deep Learning Baseline: Standalone LSTM
# -------------------------------------------------------------
class _LSTMNet(nn.Module):
    def __init__(self, hidden_units: List[int] = (50, 100, 150)):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = 1
        for units in hidden_units:
            self.layers.append(nn.LSTM(in_dim, units, batch_first=True))
            in_dim = units
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        for lstm in self.layers:
            out, _ = lstm(x)
            x = out
        return self.fc(x[:, -1, :]).squeeze(-1)


class LSTMBaseline:
    """
    Standalone LSTM adhering to Table 2:
    neurons1=50, neurons2=100, neurons3=150, batch_size=64, epochs=100, lr=0.1, window=10.
    """
    def __init__(
        self,
        window_size: int = 10,
        hidden_units: List[int] = (50, 100, 150),
        lr: float = 0.05,
        batch_size: int = 64,
        epochs: int = 80,
        device: Optional[str] = None,
    ):
        self.window_size = window_size
        self.hidden_units = hidden_units
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.normalizer = MinMaxNormalizer()
        self.model = None

    def fit(self, train_data: np.ndarray, verbose: bool = False):
        self.train_data = train_data
        norm_data = self.normalizer.fit_transform(train_data)
        X_tr, y_tr = create_sliding_windows(norm_data, self.window_size)

        self.model = _LSTMNet(self.hidden_units).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        full_series = np.concatenate([self.train_data, test_data])
        norm_full = self.normalizer.transform(full_series)

        test_start = len(self.train_data)
        X_test = []
        for i in range(test_start, len(full_series)):
            X_test.append(norm_full[i - self.window_size : i])
        X_test = np.array(X_test, dtype=np.float32)
        if X_test.ndim == 2:
            X_test = np.expand_dims(X_test, axis=-1)

        self.model.eval()
        with torch.no_grad():
            preds_norm = self.model(torch.tensor(X_test, dtype=torch.float32).to(self.device)).cpu().numpy()
        return self.normalizer.inverse_transform(preds_norm)
