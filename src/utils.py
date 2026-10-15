"""
Utility functions for ARIMA-DBO-CNN-LSTM project.
Includes seed management, hardware accelerator detection, sliding window generators,
and MinMax scaling based on Equations (22) and (23) of Duan et al. (2023).
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across random, numpy, and torch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pass
    except ImportError:
        pass


def get_device():
    """Returns PyTorch device: 'mps' (Apple Silicon), 'cuda' (Nvidia GPU), or 'cpu'."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    except ImportError:
        return "cpu"


class MinMaxNormalizer:
    """
    MinMax Normalization adhering to paper Equations (22) and (23):
    Eq (22): x_norm = (x - x_min) / (x_max - x_min)
    Eq (23): x_orig = x_norm * (x_max - x_min) + x_min
    """
    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None

    def fit(self, data: np.ndarray) -> "MinMaxNormalizer":
        self.min_val = float(np.nanmin(data))
        self.max_val = float(np.nanmax(data))
        if self.max_val == self.min_val:
            self.max_val += 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        norm = (data - self.min_val) / (self.max_val - self.min_val)
        return norm * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        scaled = (data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return scaled * (self.max_val - self.min_val) + self.min_val


def create_sliding_windows(series: np.ndarray, window_size: int):
    """
    Constructs sequence windows for time-series forecasting.
    Input: [x_t, x_{t+1}, ..., x_{t+W-1}] -> Target: x_{t+W}
    Returns:
        X: np.ndarray of shape (num_samples, window_size, 1)
        y: np.ndarray of shape (num_samples,)
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if X.ndim == 2:
        X = np.expand_dims(X, axis=-1)
    return X, y
