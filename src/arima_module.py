"""
ARIMA Module for Duan et al. (2023) replication.
Fits ARIMA(p, d, q) using Bayesian Information Criterion (BIC) selection (Eq. 2),
produces the linear predicted component L_hat, and extracts the non-linear residual sequence e_t = x_t - L_hat.
"""

import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, Optional


class ARIMAForecaster:
    def __init__(self, order: Optional[Tuple[int, int, int]] = None):
        self.order = order
        self.fitted_model = None
        self.train_data = None

    def search_best_order_bic(
        self,
        train_data: np.ndarray,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
    ) -> Tuple[int, int, int]:
        """
        Grid searches over p in [1, max_p], d in [1, max_d], q in [1, max_q]
        to find the order minimizing BIC (Eq. 2 of the paper).
        Restricts max p and q to 5 as stated in Section 'Model parameter setting'.
        """
        best_bic = float("inf")
        best_order = (1, 1, 1)

        print(f"Executing ARIMA BIC search (p <= {max_p}, d <= {max_d}, q <= {max_q})...")
        for d in range(1, max_d + 1):
            for p in range(1, max_p + 1):
                for q in range(1, max_q + 1):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ARIMA(train_data, order=(p, d, q))
                            res = model.fit()
                            bic = res.bic
                            if bic < best_bic:
                                best_bic = bic
                                best_order = (p, d, q)
                    except Exception:
                        continue

        print(f"Optimal ARIMA order selected by BIC: {best_order} (BIC: {best_bic:.2f})")
        self.order = best_order
        return best_order

    def fit(self, train_data: np.ndarray, order: Optional[Tuple[int, int, int]] = None):
        """Fits ARIMA on train_data with specified or best order."""
        self.train_data = np.asarray(train_data, dtype=np.float64)
        if order is not None:
            self.order = order
        elif self.order is None:
            self.order = (1, 1, 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(self.train_data, order=self.order)
            self.fitted_model = model.fit()
        return self

    def predict_train(self) -> np.ndarray:
        """In-sample prediction for the training series."""
        if self.fitted_model is None:
            raise RuntimeError("Model must be fitted before calling predict_train()")
        return self.fitted_model.fittedvalues

    def forecast_test_rolling(self, test_data: np.ndarray) -> np.ndarray:
        """
        Generates rolling one-step-ahead forecasts across the test series
        using the fitted ARIMA parameters (incorporating true past observations as they arrive).
        """
        if self.fitted_model is None:
            raise RuntimeError("Model must be fitted before forecasting.")

        test_data = np.asarray(test_data, dtype=np.float64)
        full_history = np.concatenate([self.train_data, test_data])
        
        # Apply fitted parameters to the full series to obtain test-time one-step-ahead forecasts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_res = self.fitted_model.apply(full_history)
            preds = full_res.fittedvalues[len(self.train_data) :]
            
        return np.asarray(preds, dtype=np.float64)

    @staticmethod
    def extract_residuals(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """
        Computes non-linear residual sequence:
        e_t = x_t - L_hat_t (Eq. 21)
        """
        return np.asarray(actual, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
