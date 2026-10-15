"""
PyTorch Conv1D-LSTM Neural Network Module for Duan et al. (2023) replication.
Extracts spatial/local features using Conv1D and models long-term dependencies using multi-layer LSTM.
Matches parameters in Table 2: Conv1D (filters, kernel_size, strides) -> 3-layer LSTM (neurons) -> Linear.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, List, Tuple


class CNN_LSTM_Net(nn.Module):
    def __init__(
        self,
        window_size: int = 10,
        filters: int = 512,
        kernel_size: int = 2,
        strides: int = 1,
        lstm_layers: int = 3,
        lstm_neurons: List[int] = (50, 50, 50),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.filters = filters
        self.kernel_size = min(kernel_size, window_size)
        self.strides = strides

        # Conv1D expects shape: (batch_size, in_channels=1, seq_len=window_size)
        # Using padding='same' if stride==1, else manual padding
        if self.strides == 1:
            self.conv1d = nn.Conv1d(
                in_channels=1,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                padding="same",
            )
        else:
            self.conv1d = nn.Conv1d(
                in_channels=1,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                padding=self.kernel_size // 2,
            )

        self.relu = nn.ReLU()

        # Build multi-layer LSTM
        # Input to first LSTM has feature dimension = filters
        self.lstm_layers_list = nn.ModuleList()
        in_dim = self.filters

        for i, units in enumerate(lstm_neurons):
            is_last = (i == len(lstm_neurons) - 1)
            lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=units,
                batch_first=True,
            )
            self.lstm_layers_list.append(lstm)
            in_dim = units

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input shape: (batch_size, window_size, 1) or (batch_size, window_size)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        # Permute for Conv1D: (batch_size, channels=1, seq_len=window_size)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        
        # Permute for LSTM: (batch_size, seq_len_out, channels=filters)
        x = x.permute(0, 2, 1)

        # Pass through stacked LSTM layers
        for i, lstm in enumerate(self.lstm_layers_list):
            out, _ = lstm(x)
            x = out

        # Take last time step
        last_step = x[:, -1, :]
        last_step = self.dropout(last_step)
        out = self.fc(last_step)
        return out.squeeze(-1)


class CNN_LSTM_Regressor:
    """High-level training & prediction wrapper for CNN_LSTM_Net."""
    def __init__(
        self,
        window_size: int = 10,
        filters: int = 512,
        kernel_size: int = 2,
        strides: int = 1,
        lstm_neurons: List[int] = (50, 50, 50),
        learning_rate: float = 0.2,
        batch_size: int = 64,
        epochs: int = 100,
        device: Optional[str] = None,
    ):
        self.window_size = window_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.lstm_neurons = list(lstm_neurons)
        self.learning_rate = learning_rate
        self.batch_size = max(1, batch_size)
        self.epochs = epochs

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        torch.manual_seed(42)
        actual_window = X_train.shape[1]
        self.window_size = actual_window

        self.model = CNN_LSTM_Net(
            window_size=self.window_size,
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            lstm_neurons=self.lstm_neurons,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        tensor_x = torch.tensor(X_train, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {loss.item():.5f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict()")
        self.model.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(tensor_x)
        return preds.cpu().numpy()
