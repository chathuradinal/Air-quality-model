"""
ARIMA-DBO-CNN-LSTM Hybrid Model Orchestrator.
Combines:
1. Linear Modeling via ARIMA(p, d, q) (BIC-selected or preset)
2. Non-linear Residual Extraction: e_t = x_t - L_hat_t
3. Dung Beetle Optimizer (DBO) for hyperparameter tuning
4. Deep Learning Conv1D-LSTM architecture for residual sequence forecasting
5. Additive Recombination: x_hat_t = L_hat_t + e_hat_t
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

from src.arima_module import ARIMAForecaster
from src.cnn_lstm_module import CNN_LSTM_Regressor
from src.dbo_optimizer import DungBeetleOptimizer, decode_hyperparameters
from src.utils import MinMaxNormalizer, create_sliding_windows


class ARIMADBOCNNLSTM:
    def __init__(
        self,
        arima_order: Optional[Tuple[int, int, int]] = None,
        use_dbo: bool = True,
        dbo_pop_size: int = 15,
        dbo_max_iter: int = 10,
        device: Optional[str] = None,
    ):
        self.arima_order = arima_order
        self.use_dbo = use_dbo
        self.dbo_pop_size = dbo_pop_size
        self.dbo_max_iter = dbo_max_iter
        self.device = device

        self.arima = ARIMAForecaster(order=arima_order)
        self.cnn_lstm = None
        self.normalizer = MinMaxNormalizer(feature_range=(0.0, 1.0))
        self.best_params = None
        self.train_data = None
        self.train_residuals = None

    def fit(self, train_data: np.ndarray, verbose: bool = True):
        self.train_data = np.asarray(train_data, dtype=np.float64)

        # 1. Fit ARIMA linear model
        if verbose:
            print("[Step 1/4] Fitting ARIMA model...")
        if self.arima_order is None:
            self.arima.search_best_order_bic(self.train_data)
        self.arima.fit(self.train_data)

        # 2. Extract linear predictions and residuals
        if verbose:
            print("[Step 2/4] Extracting linear predictions and non-linear residuals...")
        train_linear_pred = self.arima.predict_train()
        self.train_residuals = self.arima.extract_residuals(self.train_data, train_linear_pred)

        # Normalize residuals to (0, 1] as described in Eq. 22
        norm_residuals = self.normalizer.fit_transform(self.train_residuals)

        # 3. Hyperparameter search with DBO (or default paper parameters)
        if self.use_dbo:
            if verbose:
                print(f"[Step 3/4] Optimizing CNN-LSTM hyperparameters with DBO (Pop={self.dbo_pop_size}, Iter={self.dbo_max_iter})...")
            
            # Use last 15% of training residuals as validation set for DBO fitness evaluation
            val_split = int(len(norm_residuals) * 0.85)
            r_tr = norm_residuals[:val_split]
            r_va = norm_residuals[val_split:]

            # Bounds for DBO:
            # [0]: sliding_window in [5, 25]
            # [1]: lstm_layer1 in [32, 150]
            # [2]: lstm_layer2 in [32, 150]
            # [3]: lstm_layer3 in [32, 150]
            # [4]: filters in [64, 256]
            # [5]: kernel_size in [2, 5]
            # [6]: strides in [1, 2]
            # [7]: lr in [0.005, 0.05]
            lb = np.array([5.0, 32.0, 32.0, 32.0, 64.0, 2.0, 1.0, 0.005])
            ub = np.array([25.0, 150.0, 150.0, 150.0, 256.0, 5.0, 2.0, 0.05])

            def fitness_func(vec: np.ndarray) -> float:
                try:
                    params = decode_hyperparameters(vec)
                    w = params["sliding_window"]
                    X_tr, y_tr = create_sliding_windows(r_tr, window_size=w)
                    X_va, y_va = create_sliding_windows(r_va, window_size=w)
                    if len(X_tr) < 10 or len(X_va) < 5:
                        return 999.0

                    # Quick proxy training (12 epochs) for DBO fitness evaluation
                    reg = CNN_LSTM_Regressor(
                        window_size=w,
                        filters=params["filters"],
                        kernel_size=params["kernel_size"],
                        strides=params["strides"],
                        lstm_neurons=params["lstm_neurons"],
                        learning_rate=params["learning_rate"],
                        batch_size=64,
                        epochs=12,
                        device=self.device,
                    )
                    reg.fit(X_tr, y_tr, verbose=False)
                    preds = reg.predict(X_va)
                    rmse = float(np.sqrt(np.mean((y_va - preds) ** 2)))
                    return rmse
                except Exception:
                    return 999.0

            dbo = DungBeetleOptimizer(
                objective_func=fitness_func,
                lb=lb,
                ub=ub,
                dim=8,
                pop_size=self.dbo_pop_size,
                max_iter=self.dbo_max_iter,
            )
            best_vec, best_score, _ = dbo.optimize()
            self.best_params = decode_hyperparameters(best_vec)
            if verbose:
                print(f"Optimal Hyperparameters found by DBO (Score: {best_score:.4f}): {self.best_params}")
        else:
            # Paper Table 2 default ARIMA-CNN-LSTM parameters:
            # filters=512, kernel_size=2, strides=1, 3 layers=50, lr=0.2, window=10, batch_size=64, epochs=100
            self.best_params = {
                "sliding_window": 10,
                "filters": 256,
                "kernel_size": 2,
                "strides": 1,
                "lstm_neurons": (50, 50, 50),
                "learning_rate": 0.02,
                "batch_size": 64,
                "epochs": 80,
            }
            if verbose:
                print(f"[Step 3/4] Using preset parameters: {self.best_params}")

        # 4. Train final CNN-LSTM model on complete training residuals
        if verbose:
            print("[Step 4/4] Training final CNN-LSTM network on non-linear residuals...")
        w_final = self.best_params["sliding_window"]
        X_full_train, y_full_train = create_sliding_windows(norm_residuals, window_size=w_final)

        self.cnn_lstm = CNN_LSTM_Regressor(
            window_size=w_final,
            filters=self.best_params["filters"],
            kernel_size=self.best_params["kernel_size"],
            strides=self.best_params["strides"],
            lstm_neurons=self.best_params["lstm_neurons"],
            learning_rate=self.best_params["learning_rate"],
            batch_size=self.best_params.get("batch_size", 64),
            epochs=self.best_params.get("epochs", 80),
            device=self.device,
        )
        self.cnn_lstm.fit(X_full_train, y_full_train, verbose=verbose)
        return self

    def predict(self, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates predictions for test_data.
        Returns:
            final_pred: L_hat + e_hat (aligned to length of valid test predictions)
            linear_pred: L_hat
            residual_pred: e_hat
        """
        test_data = np.asarray(test_data, dtype=np.float64)

        # 1. Linear forecast from ARIMA
        linear_pred_test = self.arima.forecast_test_rolling(test_data)

        # 2. Extract true test residuals up to current time step for sliding window input
        actual_test_residuals = test_data - linear_pred_test
        full_residuals = np.concatenate([self.train_residuals, actual_test_residuals])
        norm_full_residuals = self.normalizer.transform(full_residuals)

        # 3. Construct sliding windows for test set
        w = self.best_params["sliding_window"]
        test_start_idx = len(self.train_residuals)
        
        X_test = []
        for i in range(test_start_idx, len(full_residuals)):
            X_test.append(norm_full_residuals[i - w : i])
        X_test = np.array(X_test, dtype=np.float32)
        if X_test.ndim == 2:
            X_test = np.expand_dims(X_test, axis=-1)

        # 4. Predict residuals with CNN-LSTM
        norm_pred_residuals = self.cnn_lstm.predict(X_test)
        pred_residuals = self.normalizer.inverse_transform(norm_pred_residuals)

        # 5. Additive assembly: x_hat = L_hat + e_hat
        final_pred = linear_pred_test + pred_residuals
        return final_pred, linear_pred_test, pred_residuals
