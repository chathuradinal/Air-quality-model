"""
create_base_notebook.py
Generates notebooks/phase1_base_replication.ipynb:
The exact replication of the paper's original Base Model (ARIMA-DBO-CNN-LSTM and baselines).
"""

import os
import nbformat as nbf

def build_base_notebook():
    os.makedirs("notebooks", exist_ok=True)
    nb = nbf.v4.new_notebook()

    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# Base Model Implementation: Author Paper Replication
## Air-Quality Prediction Based on the ARIMA-CNN-LSTM Combination Model Optimized by Dung Beetle Optimizer
**Original Paper**: Duan, J., Gong, Y., Luo, J., & Zhao, Z. (2023). *Scientific Reports*, 13:12127.  
**Student Name**: K.W.G.C.D.B. Karunarathne (Student ID: KIC-BSCCSAI-251P-036)  
**Module**: Machine Learning and Related Applications  
**Institution**: Kandy Innovation Centre, National Institute of Business Management (NIBM)

---

### Notebook Purpose
This notebook implements the **Base Model** exactly following the authors' original methodology from the research paper:
1. **Data Preprocessing & Figure 5 Replication**: Ingestion of 9 monitoring stations per city across Beijing, Chengdu, Guangzhou, and Wuhan, aggregated to 24-hour daily AQI means (2,528 days, 2014–2020), partitioned into the paper's **80% Train / 20% Test** split.
2. **Step 1: Linear ARIMA Modeling**: Fitting $\\text{ARIMA}(p, d, q)$ using the exact optimal orders reported in **Table 1** of the paper:
   - Beijing: $\\text{ARIMA}(3, 1, 1)$
   - Chengdu: $\\text{ARIMA}(3, 1, 1)$
   - Guangzhou: $\\text{ARIMA}(2, 1, 2)$
   - Wuhan: $\\text{ARIMA}(2, 1, 2)$
3. **Step 2: Residual Sequence Extraction**: Computing the non-linear residual sequence $e_t = x_t - \\hat{L}_t$ and constructing supervised lag windows of length $w = 10$.
4. **Step 3: Dung Beetle Optimizer (DBO)**: Implementing the bio-inspired DBO meta-heuristic across its four sub-populations (Ball-rolling, Breeding, Foraging, Stealing) to optimize CNN-LSTM hyperparameters within the paper's **Table 2** boundaries.
5. **Step 4: CNN-LSTM Deep Residual Modeling**: Training the 1D-CNN + LSTM architecture on the residual sequence.
6. **Step 5: Additive Recombination**: Computing final composite predictions $\\hat{x}_t = \\hat{L}_t + \\hat{e}_t$.
7. **Paper's Comparative Baselines**: Benchmarking against the exact comparative models reported in the paper: Linear SVM, Random Forest, Standalone LSTM, and Pure ARIMA.
8. **Evaluation Metrics**: Computing RMSE, MAE, and $R^2$ to mirror the paper's Tables 3 & 4."""))

    # Cell 1: Environment & Imports
    cells.append(nbf.v4.new_markdown_cell("""### 1. Environment Setup & Imports"""))
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Add src to Python path
sys.path.append(os.path.abspath('..'))

from src.data_loader import load_city_split, prepare_and_save_all_cities
from src.utils import set_seed, get_device, MinMaxNormalizer, create_sliding_windows
from src.arima_module import ARIMAForecaster
from src.cnn_lstm_module import CNN_LSTM_Regressor
from src.dbo_optimizer import DungBeetleOptimizer
from src.hybrid_model import ARIMADBOCNNLSTM
from src.baselines import SVMBaseline, RFBaseline, LSTMBaseline
from src.evaluation import calculate_metrics, plot_comparison_and_scatter

# Set reproducible seed and check compute device
set_seed(42)
device = get_device()
print(f"PyTorch Version: {torch.__version__}")
print(f"Active Compute Device: {device}")"""))

    # Cell 2: Data Ingestion & Split
    cells.append(nbf.v4.new_markdown_cell("""### 2. Dataset Ingestion (80/20 Train/Test Split)
Following Section 2 of Duan et al., each city's 9 air quality stations are aggregated into a daily 24h mean series (2,528 days from Jan 1, 2014 to Dec 31, 2020) and partitioned using an 80/20 chronological split."""))
    cells.append(nbf.v4.new_code_cell("""# Ensure all processed CSVs exist
prepare_and_save_all_cities()

# Load Beijing dataset with 80% train / 20% test split
city = "beijing"
train_series, test_series, dates = load_city_split(city=city, train_ratio=0.8)

print(f"City: {city.upper()}")
print(f"Total Observations: {len(train_series) + len(test_series)}")
print(f"Training Samples (80%): {len(train_series)} days ({dates[0].strftime('%Y-%m-%d')} to {dates[len(train_series)-1].strftime('%Y-%m-%d')})")
print(f"Testing Samples (20%): {len(test_series)} days ({dates[len(train_series)].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')})")"""))

    # Cell 3: Figure 5 Replication
    cells.append(nbf.v4.new_markdown_cell("""### 3. Replicating Figure 5 from Duan et al. (2023)
Visualizing raw daily AQI trends across Beijing, Chengdu, Guangzhou, and Wuhan."""))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
cities = ["beijing", "chengdu", "guangzhou", "wuhan"]
city_titles = ["(a) Beijing AQI", "(b) Chengdu AQI", "(c) Guangzhou AQI", "(d) Wuhan AQI"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for ax, c, title, col in zip(axes, cities, city_titles, colors):
    df_c = pd.read_csv(f"../processed_data/{c}_daily_aqi.csv", parse_dates=["date"])
    ax.plot(df_c["date"], df_c["AQI"], color=col, linewidth=0.8, alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Daily AQI", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

axes[-1].set_xlabel("Date (2014 - 2020)", fontsize=11)
plt.suptitle("Figure 5 Replication: Daily Air Quality Index Trends (2014–2020)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()"""))

    # Cell 4: Linear ARIMA Modeling
    cells.append(nbf.v4.new_markdown_cell("""### 4. Step 1: Linear Modeling with ARIMA(p, d, q)
Using the optimal orders identified in **Table 1** of Duan et al. via BIC minimization:
- Beijing: $\\text{ARIMA}(3, 1, 1)$"""))
    cells.append(nbf.v4.new_code_cell("""# Fit ARIMA(3, 1, 1) on Beijing training series
arima_model = ARIMAForecaster(order=(3, 1, 1))
arima_model.fit(train_series)

# One-step-ahead rolling out-of-sample forecast
arima_preds = arima_model.predict_rolling(test_series)

arima_metrics = calculate_metrics(test_series, arima_preds)
print("--- ARIMA(3, 1, 1) Baseline Results ---")
print(f"RMSE: {arima_metrics['RMSE']:.3f}")
print(f"MAE:  {arima_metrics['MAE']:.3f}")
print(f"R²:   {arima_metrics['R2']:.3f}")"""))

    # Cell 5: Residual Sequence Extraction
    cells.append(nbf.v4.new_markdown_cell("""### 5. Step 2: Non-Linear Residual Sequence Extraction
$$e_t = x_t - \\hat{L}_t$$
Constructing sliding supervised lag windows ($w = 10$)."""))
    cells.append(nbf.v4.new_code_cell("""# Extract in-sample linear predictions and training residuals
train_linear_fitted = arima_model.model_fit.fittedvalues
train_residuals = train_series[1:] - train_linear_fitted[1:]

# Extract out-of-sample testing residuals
test_residuals = test_series - arima_preds

# Normalize residuals using MinMax
res_normalizer = MinMaxNormalizer()
train_res_norm = res_normalizer.fit_transform(train_residuals)
test_res_norm = res_normalizer.transform(test_residuals)

# Build sliding lag windows (w = 10)
lag_w = 10
X_train_res, y_train_res = create_sliding_windows(train_res_norm, lag_window=lag_w)
X_test_res, y_test_res = create_sliding_windows(test_res_norm, lag_window=lag_w)

print(f"Residual Training Feature Shape: {X_train_res.shape}, Target Shape: {y_train_res.shape}")
print(f"Residual Testing Feature Shape:  {X_test_res.shape}, Target Shape: {y_test_res.shape}")"""))

    # Cell 6: DBO Swarm Optimization
    cells.append(nbf.v4.new_markdown_cell("""### 6. Step 3: Dung Beetle Optimizer (DBO) Hyperparameter Tuning
Optimizing 7 hyperparameters within the paper's **Table 2** boundary definitions:
1. `conv_filters`: $[8, 64]$
2. `kernel_size`: $[2, 5]$
3. `lstm_hidden`: $[16, 128]$
4. `learning_rate`: $[10^{-4}, 10^{-2}]$
5. `batch_size`: $[16, 128]$
6. `epochs`: $[20, 80]$
7. `dropout`: $[0.0, 0.4]$"""))
    cells.append(nbf.v4.new_code_cell("""# Initialize Dung Beetle Optimizer with 15 agents and 8 iterations
dbo = DungBeetleOptimizer(
    pop_size=15,
    max_iter=8,
    dim=7,
    lb=[8, 2, 16, 1e-4, 16, 20, 0.0],
    ub=[64, 5, 128, 1e-2, 128, 80, 0.4],
    seed=42
)

# Define objective fitness function (RMSE on training residual validation)
val_split = int(len(X_train_res) * 0.8)
X_tr, y_tr = X_train_res[:val_split], y_train_res[:val_split]
X_val, y_val = X_train_res[val_split:], y_train_res[val_split:]

def dbo_objective(params):
    c_filt, k_sz, l_hid, lr, b_sz, eps, drop = params
    model = CNN_LSTM_Regressor(
        input_dim=1,
        conv_filters=int(c_filt),
        kernel_size=int(k_sz),
        lstm_hidden=int(l_hid),
        dropout=float(drop),
        device=str(device)
    )
    history = model.fit(
        X_tr, y_tr,
        X_val=X_val, y_val=y_val,
        epochs=int(eps),
        batch_size=int(b_sz),
        lr=float(lr),
        patience=5,
        verbose=False
    )
    val_preds = model.predict(X_val)
    return np.sqrt(np.mean((y_val - val_preds)**2))

print("Executing DBO Swarm Search across 4 phenotypes...")
best_cost, best_params, history = dbo.optimize(dbo_objective, verbose=True)

print(f"\\nOptimization Complete! Best Fitness (RMSE): {best_cost:.5f}")
print("Optimal Hyperparameters Discovered:")
print(f"  CNN Filters:      {int(best_params[0])}")
print(f"  Kernel Size:      {int(best_params[1])}")
print(f"  LSTM Hidden:      {int(best_params[2])}")
print(f"  Learning Rate:    {best_params[3]:.6f}")
print(f"  Batch Size:       {int(best_params[4])}")
print(f"  Epochs:           {int(best_params[5])}")
print(f"  Dropout:          {best_params[6]:.3f}")"""))

    # Cell 7: Recombination & Hybrid Model Fit
    cells.append(nbf.v4.new_markdown_cell("""### 7. Step 4 & 5: CNN-LSTM Residual Modeling & Additive Recombination
$$\\hat{x}_t = \\hat{L}_t + \\hat{e}_t$$"""))
    cells.append(nbf.v4.new_code_cell("""# Instantiate end-to-end ARIMA-DBO-CNN-LSTM Hybrid Model
hybrid_model = ARIMADBOCNNLSTM(
    arima_order=(3, 1, 1),
    lag_window=10,
    dbo_pop=15,
    dbo_iter=8,
    device=str(device),
    random_state=42
)

# Train hybrid model on 80% training data
hybrid_model.fit(train_series, verbose=False)

# Forecast on 20% test horizon
hybrid_preds = hybrid_model.predict(test_series)

# Evaluate performance metrics
hybrid_metrics = calculate_metrics(test_series[10:], hybrid_preds)
print("--- ARIMA-DBO-CNN-LSTM Base Model Results ---")
print(f"RMSE: {hybrid_metrics['RMSE']:.3f}")
print(f"MAE:  {hybrid_metrics['MAE']:.3f}")
print(f"R²:   {hybrid_metrics['R2']:.3f}")"""))

    # Cell 8: Comparative Baselines from the Paper
    cells.append(nbf.v4.new_markdown_cell("""### 8. Paper's Comparative Baselines (SVM, RF, LSTM)
Replicating the comparative models used in Section 3 of Duan et al. (2023)."""))
    cells.append(nbf.v4.new_code_cell("""# 1. Linear Support Vector Machine (SVM)
svm = SVMBaseline(lag_window=10)
svm.fit(train_series)
svm_preds = svm.predict(test_series)
svm_metrics = calculate_metrics(test_series[10:], svm_preds)

# 2. Random Forest Regressor (RF)
rf = RFBaseline(lag_window=10, n_estimators=100, random_state=42)
rf.fit(train_series)
rf_preds = rf.predict(test_series)
rf_metrics = calculate_metrics(test_series[10:], rf_preds)

# 3. Standalone LSTM
lstm = LSTMBaseline(lag_window=10, hidden_dim=64, epochs=30, batch_size=32, device=str(device))
lstm.fit(train_series)
lstm_preds = lstm.predict(test_series)
lstm_metrics = calculate_metrics(test_series[10:], lstm_preds)

# Align ARIMA predictions
arima_test_eval = arima_preds[10:]
arima_eval_metrics = calculate_metrics(test_series[10:], arima_test_eval)

print("Baseline models trained successfully!")"""))

    # Cell 9: Master Comparison Table & Visualization
    cells.append(nbf.v4.new_markdown_cell("""### 9. Comparative Performance Table (Replicating Paper Tables 3 & 4)"""))
    cells.append(nbf.v4.new_code_cell("""summary_df = pd.DataFrame([
    {"Model": "Linear SVM", "RMSE": svm_metrics["RMSE"], "MAE": svm_metrics["MAE"], "R2": svm_metrics["R2"]},
    {"Model": "Random Forest", "RMSE": rf_metrics["RMSE"], "MAE": rf_metrics["MAE"], "R2": rf_metrics["R2"]},
    {"Model": "Pure ARIMA(3,1,1)", "RMSE": arima_eval_metrics["RMSE"], "MAE": arima_eval_metrics["MAE"], "R2": arima_eval_metrics["R2"]},
    {"Model": "Standalone LSTM", "RMSE": lstm_metrics["RMSE"], "MAE": lstm_metrics["MAE"], "R2": lstm_metrics["R2"]},
    {"Model": "ARIMA-DBO-CNN-LSTM (Proposed Base)", "RMSE": hybrid_metrics["RMSE"], "MAE": hybrid_metrics["MAE"], "R2": hybrid_metrics["R2"]},
])

print("\\n=== PAPER REPLICATION: MODEL PERFORMANCE COMPARISON (BEIJING) ===")
print(summary_df.to_string(index=False))

# Plot Comparison Overlay
plt.figure(figsize=(14, 5))
test_horizon = test_series[10:]
plt.plot(test_horizon[:120], label="Actual Daily AQI", color="black", linewidth=1.5)
plt.plot(hybrid_preds[:120], label="ARIMA-DBO-CNN-LSTM (Proposed)", color="#d62728", linewidth=1.2)
plt.plot(arima_test_eval[:120], label="Pure ARIMA", color="#1f77b4", linestyle="--", linewidth=1.0)
plt.plot(rf_preds[:120], label="Random Forest", color="#2ca02c", linestyle=":", linewidth=1.0)
plt.title("Base Model Replication: Forecast Comparison (First 120 Test Days)", fontsize=13, fontweight="bold")
plt.xlabel("Test Horizon (Days)", fontsize=11)
plt.ylabel("AQI", fontsize=11)
plt.legend(frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""))

    nb.cells = cells
    output_path = "notebooks/phase1_base_replication.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Base model replication notebook successfully written to: {output_path}")

if __name__ == "__main__":
    build_base_notebook()
