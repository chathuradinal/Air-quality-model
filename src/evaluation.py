import os
from typing import Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq (24): Root Mean Square Error"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq (25): Mean Absolute Error"""
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq (26): Coefficient of Determination R2"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE)"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true > 0
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def calculate_mase(y_train: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error (MASE) (Hyndman & Koehler, 2006).
    Compares test MAE against the naive in-sample persistence MAE.
    MASE < 1.0 indicates performance better than the naive baseline.
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    mae_test = np.mean(np.abs(y_test - y_pred))
    mae_naive_train = np.mean(np.abs(np.diff(y_train)))
    if mae_naive_train == 0:
        return float("inf")
    return float(mae_test / mae_naive_train)


def calculate_peak_day_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 150.0) -> dict:
    """
    Evaluates forecast errors specifically on critical peak pollution days (e.g., AQI >= 150 or AQI >= 300).
    Addresses regression class imbalance identified in Phase 1 review.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true >= threshold
    count = int(np.sum(mask))
    if count == 0:
        return {"Count": 0, "RMSE": None, "MAE": None}
    return {
        "Count": count,
        "RMSE": round(calculate_rmse(y_true[mask], y_pred[mask]), 3),
        "MAE": round(calculate_mae(y_true[mask], y_pred[mask]), 3),
    }


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy of two forecast error series e1 and e2.
    e1: errors of benchmark (e.g. Persistence or ARIMA) = y_true - y_pred_bench
    e2: errors of proposed model = y_true - y_pred_model
    Loss differential d_t = e1_t^2 - e2_t^2 (or absolute loss).
    Returns DM statistic and two-tailed p-value.
    A positive DM statistic with p < 0.05 indicates e2 has significantly smaller errors than e1.
    """
    from scipy import stats
    d = np.asarray(e1, dtype=np.float64)**2 - np.asarray(e2, dtype=np.float64)**2
    n = len(d)
    mean_d = np.mean(d)
    # Autocovariance adjustment for h-step ahead forecasts
    gamma_0 = np.var(d, ddof=1)
    var_d = gamma_0
    for lag in range(1, h):
        gamma_k = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        var_d += 2.0 * gamma_k

    if var_d <= 0 or np.isnan(var_d):
        se_d = 1e-8
    else:
        se_d = np.sqrt(var_d / n)

    dm_stat = mean_d / se_d
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))
    return {
        "DM_Stat": round(float(dm_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_p05": bool(p_value < 0.05),
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: Optional[np.ndarray] = None) -> dict:
    """Computes full suite of evaluation metrics (both standard and Phase 1 enhanced)."""
    res = {
        "RMSE": round(calculate_rmse(y_true, y_pred), 3),
        "MAE": round(calculate_mae(y_true, y_pred), 3),
        "R2": round(calculate_r2(y_true, y_pred), 3),
        "MAPE": round(calculate_mape(y_true, y_pred), 2),
    }
    if y_train is not None:
        res["MASE"] = round(calculate_mase(y_train, y_true, y_pred), 3)
    return res


def plot_raw_aqi_all_cities(daily_series_dict: dict, save_path: str = "results/figures/fig5_raw_aqi.png"):
    """
    Replicates Figure 5 (Raw data) of the research paper:
    Displays 4 subplots showing the daily AQI series over time (2015 - 2022) for Beijing, Lanzhou/Chengdu, Jiaozuo/Wuhan, Guangzhou.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)
    
    city_display_names = {
        "beijing": "(a) Beijing",
        "chengdu": "(b) Chengdu",
        "wuhan": "(c) Wuhan",
        "guangzhou": "(d) Guangzhou",
    }
    
    for idx, (city_key, series) in enumerate(daily_series_dict.items()):
        ax = axes[idx]
        ax.plot(series.index, series.values, label="AQI", linewidth=1.0, color="#1f77b4")
        title_str = city_display_names.get(city_key.lower(), city_key.capitalize())
        ax.set_title(f"{city_key.capitalize()} AQI", fontsize=14)
        ax.set_xlabel(f"time/day\n{title_str}", fontsize=12)
        ax.set_ylabel("AQI", fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Figure 5 saved to: {save_path}")


def plot_comparison_and_scatter(
    y_true: np.ndarray,
    predictions: dict,
    city_name: str,
    save_path: str,
):
    """
    Replicates Figure 7 / Figure 8 (Model Comparison and Fitting Effect) from the paper:
    - Top: Time-series curve comparison on the test set.
    - Bottom: Individual scatter fitting plots (True Value vs Predicted Value).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_models = len(predictions)
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Top time-series comparison
    ax_top = fig.add_subplot(2, 1, 1)
    ax_top.plot(y_true, label="Actual", color="red", linewidth=1.5)
    
    colors = {
        "RF": "blue",
        "ARIMA": "green",
        "DBO-LSTM": "orange",
        "LSTM": "purple",
        "ARIMA-CNN-LSTM": "brown",
        "ARIMA-DBO-CNN-LSTM": "black",
    }
    
    for name, pred in predictions.items():
        color = colors.get(name, None)
        ls = "--" if "ARIMA" in name else "-"
        ax_top.plot(pred, label=name, linestyle=ls, color=color, alpha=0.85, linewidth=1.2)
        
    ax_top.set_title(f"Comparison of {city_name.capitalize()} AQI Models", fontsize=16)
    ax_top.set_xlabel("time / day", fontsize=13)
    ax_top.set_ylabel("AQI", fontsize=13)
    ax_top.legend(loc="upper left", framealpha=0.9)
    ax_top.grid(True, linestyle="--", alpha=0.5)

    # 2. Scatter fitting plots for selected models
    selected_keys = [k for k in ["RF", "ARIMA", "LSTM", "ARIMA-CNN-LSTM", "ARIMA-DBO-CNN-LSTM"] if k in predictions]
    n_scatter = len(selected_keys)
    cols = min(4, n_scatter)
    rows = (n_scatter + cols - 1) // cols
    
    sub_gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    scatter_gs = sub_gs[1].subgridspec(rows, cols, hspace=0.35, wspace=0.25)
    
    for i, model_key in enumerate(selected_keys):
        r, c = divmod(i, cols)
        ax_scat = fig.add_subplot(scatter_gs[r, c])
        pred_vals = predictions[model_key]
        ax_scat.scatter(y_true, pred_vals, alpha=0.6, s=15, edgecolors="none", color="#1f77b4")
        
        # Diagonal reference line
        min_v = min(np.min(y_true), np.min(pred_vals))
        max_v = max(np.max(y_true), np.max(pred_vals))
        ax_scat.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.0)
        
        ax_scat.set_title(f"{model_key} model {city_name.capitalize()} AQI fitting", fontsize=10)
        ax_scat.set_xlabel("True Value", fontsize=9)
        ax_scat.set_ylabel("Predicted Value", fontsize=9)
        ax_scat.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def plot_metrics_barchart(metrics_df, city_name: str, save_path: str):
    """
    Replicates Figure 7a / 8a (Bar charts of RMSE, MAE, R2 for Models A to J).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = metrics_df["Model"].tolist()
    labels = [chr(65 + i) for i in range(len(models))] # A, B, C, ...
    
    metrics = ["RMSE", "MAE", "R2"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = metrics_df[metric].values
        bars = ax.bar(labels, values, color=colors[idx], width=0.6, edgecolor="black", linewidth=0.8)
        ax.set_title(metric, fontsize=14, weight="bold")
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel("Models", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.suptitle(f"{city_name.capitalize()} City Model Evaluation Metrics (A-J)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Bar chart saved to: {save_path}")
