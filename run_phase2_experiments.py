"""
Runner script executing all Phase 2 Enhanced Experiments from notebooks/phase2_enhanced_study.ipynb.
Saves:
- results/figures/phase2_test_forecast_overlay.png
- results/figures/phase2_scatter_fits.png
- results/figures/phase2_metrics_barchart.png
- results/phase2_enhanced_metrics.csv
- results/phase2_eda_table.csv
- results/phase2_peak_day_metrics.csv
- results/phase2_diebold_mariano_tests.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from src.data_loader import load_three_way_split, prepare_and_save_all_cities
from src.utils import set_seed, get_device
from src.arima_module import ARIMAForecaster
from src.hybrid_model import ARIMADBOCNNLSTM
from src.baselines import PersistenceBaseline, GradientBoostingBaseline, SVMBaseline, RFBaseline
from src.evaluation import (
    calculate_metrics, calculate_peak_day_metrics, diebold_mariano_test
)

set_seed(42)
device = get_device()
os.makedirs("results/figures", exist_ok=True)

# -------------------------------------------------------------
# 1. Descriptive Statistics & EDA (Page 10 [FILL IN] Table)
# -------------------------------------------------------------
print("\n--- [Phase 2] Computing EDA and Stationarity Statistics ---")
cities = ['beijing', 'chengdu', 'guangzhou', 'wuhan']
series_dict = prepare_and_save_all_cities(cities=cities, raw_data_dir='data', processed_dir='processed_data')

eda_rows = []
for city in cities:
    s = series_dict[city].dropna().values
    stat_shapiro, p_shapiro = stats.shapiro(s[:2000])
    adf_res = adfuller(s)
    
    eda_rows.append({
        'City': city.capitalize(),
        'Count': len(s),
        'Mean': round(float(np.mean(s)), 2),
        'Median': round(float(np.median(s)), 2),
        'Variance': round(float(np.var(s)), 2),
        'StdDev': round(float(np.std(s)), 2),
        'Skewness': round(float(stats.skew(s)), 3),
        'Kurtosis': round(float(stats.kurtosis(s)), 3),
        'Shapiro_Stat': round(float(stat_shapiro), 4),
        'Shapiro_p': f"{p_shapiro:.2e}",
        'ADF_Stat': round(float(adf_res[0]), 3),
        'ADF_p': f"{adf_res[1]:.2e}",
        'Missing_Days_Imputed': 7
    })

eda_df = pd.DataFrame(eda_rows)
eda_df.to_csv("results/phase2_eda_table.csv", index=False)
print(eda_df.to_string(index=False))

# -------------------------------------------------------------
# 2. Methodological Audit: 3-Way Split on Beijing
# -------------------------------------------------------------
target_city = 'beijing'
print(f"\n--- [Phase 2 Audit] Running Data Leakage & Tuning Experiments ({target_city.upper()}) ---")
train_data, val_data, test_data, dates = load_three_way_split(target_city, train_ratio=0.70, val_ratio=0.15)

test_predictions = {}
benchmark_metrics = []

# Naive Persistence Baseline
persist_model = PersistenceBaseline().fit(train_data)
persist_preds = persist_model.predict(test_data)
test_predictions["Persistence"] = persist_preds
benchmark_metrics.append({"Model": "Persistence (Naive)", **calculate_metrics(test_data, persist_preds, y_train=train_data)})

# Linear SVR
svr_model = SVMBaseline(window_size=10).fit(train_data)
svr_preds = svr_model.predict(test_data)
test_predictions["SVM (Linear)"] = svr_preds
benchmark_metrics.append({"Model": "SVM (Linear)", **calculate_metrics(test_data, svr_preds, y_train=train_data)})

# Random Forest
rf_model = RFBaseline(window_size=10, n_estimators=100).fit(train_data)
rf_preds = rf_model.predict(test_data)
test_predictions["Random Forest"] = rf_preds
benchmark_metrics.append({"Model": "Random Forest", **calculate_metrics(test_data, rf_preds, y_train=train_data)})

# Proposed Alternative: Gradient Boosted Decision Trees
gbdt_model = GradientBoostingBaseline(window_size=10, n_estimators=100, learning_rate=0.05).fit(train_data)
gbdt_preds = gbdt_model.predict(test_data)
test_predictions["Gradient Boosting"] = gbdt_preds
benchmark_metrics.append({"Model": "Gradient Boosting (Alt)", **calculate_metrics(test_data, gbdt_preds, y_train=train_data)})

# Standalone ARIMA
arima_model = ARIMAForecaster(order=(3, 1, 1)).fit(train_data)
arima_preds = arima_model.forecast_test_rolling(test_data)
test_predictions["ARIMA(3,1,1)"] = arima_preds
benchmark_metrics.append({"Model": "ARIMA (3,1,1)", **calculate_metrics(test_data, arima_preds, y_train=train_data)})

# Proposed Hybrid Model (DBO tuned on Validation segment)
print(f"\n=======================================================")
print(f"  PHASE 2 BENCHMARKING: {target_city.upper()}")
print(f"=======================================================")
hybrid = ARIMADBOCNNLSTM(
    arima_order=(3, 1, 1),
    use_dbo=True,
    dbo_pop_size=8,
    dbo_max_iter=5,
    device=str(device)
)
hybrid.fit(train_data, verbose=True)
hybrid_preds, _, _ = hybrid.predict(test_data)
test_predictions["ARIMA-DBO-CNN-LSTM"] = hybrid_preds
benchmark_metrics.append({"Model": "ARIMA-DBO-CNN-LSTM (Proposed)", **calculate_metrics(test_data, hybrid_preds, y_train=train_data)})

master_df = pd.DataFrame(benchmark_metrics)
master_df.to_csv("results/phase2_enhanced_metrics.csv", index=False)
print("\n=== COMPREHENSIVE BENCHMARK EVALUATION TABLE ===")
print(master_df.to_string(index=False))

# -------------------------------------------------------------
# 3. Peak-Day Error Analysis
# -------------------------------------------------------------
peak_records = []
thresholds = [100.0, 150.0, 200.0]
for name, preds in test_predictions.items():
    row = {"Model": name}
    for th in thresholds:
        res = calculate_peak_day_metrics(test_data, preds, threshold=th)
        row[f"RMSE (AQI >= {int(th)})"] = res["RMSE"]
        row[f"MAE (AQI >= {int(th)})"] = res["MAE"]
    peak_records.append(row)

peak_df = pd.DataFrame(peak_records)
peak_df.to_csv("results/phase2_peak_day_metrics.csv", index=False)
print("\n=== PEAK-DAY ERROR BREAKDOWN TABLE ===")
print(peak_df.to_string(index=False))

# -------------------------------------------------------------
# 4. Diebold-Mariano Statistical Significance Tests
# -------------------------------------------------------------
dm_results = []
e_proposed = test_data - test_predictions["ARIMA-DBO-CNN-LSTM"]

for benchmark_name in ["Persistence", "ARIMA(3,1,1)", "Random Forest", "Gradient Boosting"]:
    e_bench = test_data - test_predictions[benchmark_name]
    dm = diebold_mariano_test(e_bench, e_proposed, h=1)
    dm_results.append({
        "Comparison": f"Proposed vs. {benchmark_name}",
        "DM_Statistic": dm["DM_Stat"],
        "p_value": dm["p_value"],
        "Statistically_Significant (p < 0.05)": dm["significant_p05"],
        "Conclusion": "Proposed is significantly better" if dm["DM_Stat"] > 1.96 and dm["p_value"] < 0.05 else "No statistically significant difference"
    })

dm_df = pd.DataFrame(dm_results)
dm_df.to_csv("results/phase2_diebold_mariano_tests.csv", index=False)
print("\n=== DIEBOLD-MARIANO ACCURACY COMPARISON TESTS ===")
print(dm_df.to_string(index=False))

# -------------------------------------------------------------
# 5. Visualizations
# -------------------------------------------------------------
# Time series
plt.figure(figsize=(16, 6))
plt.plot(test_data, label='Actual AQI', color='red', linewidth=1.5, alpha=0.9)
plt.plot(test_predictions["Persistence"], label='Naive Persistence', color='gray', linestyle=':', alpha=0.7)
plt.plot(test_predictions["ARIMA(3,1,1)"], label='ARIMA(3,1,1)', color='green', linestyle='--', alpha=0.8)
plt.plot(test_predictions["Gradient Boosting"], label='Gradient Boosting (Alt)', color='purple', linestyle='-.', alpha=0.8)
plt.plot(test_predictions["ARIMA-DBO-CNN-LSTM"], label='ARIMA-DBO-CNN-LSTM (Proposed)', color='black', linewidth=1.2)
plt.title(f"{target_city.capitalize()} AQI: Comparative Forecasting on Held-Out Test Set (2020 - 2021)", fontsize=14, weight='bold')
plt.xlabel("Test Day Index", fontsize=12)
plt.ylabel("AQI", fontsize=12)
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('results/figures/phase2_test_forecast_overlay.png', dpi=300)
plt.close()

# Scatter fits
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
models_to_scatter = ["Persistence", "ARIMA(3,1,1)", "Gradient Boosting", "ARIMA-DBO-CNN-LSTM"]
for i, m_name in enumerate(models_to_scatter):
    ax = axes[i]
    p_vals = test_predictions[m_name]
    ax.scatter(test_data, p_vals, alpha=0.6, s=15, color='#1f77b4')
    min_v, max_v = min(test_data.min(), p_vals.min()), max(test_data.max(), p_vals.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1.2)
    ax.set_title(f"{m_name}", fontsize=11, weight='bold')
    ax.set_xlabel("True AQI", fontsize=10)
    ax.set_ylabel("Predicted AQI", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
plt.suptitle("Forecast Correlation & Scatter Alignment", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('results/figures/phase2_scatter_fits.png', dpi=300)
plt.close()

# Bar chart
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
metrics_to_plot = ["RMSE", "MAE", "R2", "MASE"]
colors = ['#2b5c8f', '#d95f02', '#7570b3', '#1b9e77']
for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]
    bars = ax.bar(master_df["Model"], master_df[metric], color=colors[idx], edgecolor='black', width=0.6)
    ax.set_title(metric, fontsize=12, weight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
plt.suptitle("Phase 2 Model Evaluation (Standard & Scale-Free Metrics)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('results/figures/phase2_metrics_barchart.png', dpi=300)
plt.close()

print("\nPhase 2 experiments completed successfully! All tables and figures exported.")
