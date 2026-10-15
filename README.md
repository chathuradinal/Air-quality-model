# Air Quality Prediction: ARIMA-DBO-CNN-LSTM Combination Model

This project is a reproduction and production-grade implementation of the research paper:
> **Title**: *Air-quality prediction based on the ARIMA-CNN-LSTM combination model optimized by dung beetle optimizer*  
> **Authors**: Jiahui Duan, Yaping Gong, Jun Luo, and Zhiyao Zhao  
> **Journal**: *Scientific Reports* (Nature Portfolio), 2023, 13:12127  
> **DOI**: [10.1038/s41598-023-36620-4](https://doi.org/10.1038/s41598-023-36620-4)  
> **PDF**: [`s41598-023-36620-4 (2)(1).pdf`](./s41598-023-36620-4%20(2)(1).pdf)

---

## 1. Overview & Architecture

Air quality time series contain both **linear trends/seasonality** and **highly non-linear environmental disturbances**. Single statistical models (e.g., pure ARIMA) fail on nonlinear dynamics, while standard deep neural networks (e.g., pure LSTM) suffer from blind hyperparameter selection and linear trend distortion.

The proposed framework solves this via an **additive hybrid model**:
$$x_t = L_t + N_t$$

1. **Linear Modeling**: An $\text{ARIMA}(p, d, q)$ model fits the linear component $L_t$. Parameters $(p, d, q)$ are determined via the Bayesian Information Criterion (BIC).
2. **Residual Extraction**: The non-linear component is isolated as residuals:
   $$e_t = x_t - \hat{L}_t$$
3. **Hyperparameter Optimization via DBO**: The **Dung Beetle Optimizer (DBO)** (Xue & Shen, 2022) optimizes the CNN-LSTM architecture across 7 continuous/discrete hyperparameters (window size, neurons per layer, filters, kernel size, strides, learning rate) using 4 biological behaviors (ball-rolling, dancing, breeding, foraging, stealing).
4. **Non-linear Sequence Modeling**: A PyTorch **Conv1D-LSTM** network is trained on the residual series to forecast $\hat{e}_t$.
5. **Ensemble Reconstruction**:
   $$\hat{x}_t = \hat{L}_t + \hat{e}_t$$

---

## 2. Directory Structure

```text
ML Assignment/
├── configs/
│   └── config.yaml                     # Model hyperparameters and DBO search bounds
├── data/                               # Raw hourly monitoring stations (4 cities x 9 stations)
│   ├── beijing/
│   ├── chengdu/
│   ├── guangzhou/
│   └── wuhan/
├── processed_data/                     # Aggregated daily AQI series (2,528 days per city)
│   ├── beijing_daily.csv
│   ├── chengdu_daily.csv
│   ├── guangzhou_daily.csv
│   └── wuhan_daily.csv
├── results/                            # Benchmark metrics & figures
│   ├── figures/                        # Generated charts matching paper figures
│   │   ├── fig5_raw_aqi.png            # Paper Figure 5: Raw AQI series
│   │   ├── beijing_model_comparison.png # Paper Figure 7/8: Time series & scatter fitting
│   │   └── beijing_metrics_barchart.png # Paper Figure 7a/8a: Metric bar charts
│   └── beijing_metrics.csv             # Metrics table (RMSE, MAE, R2)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Station reader, daily aggregator, imputation, splits
│   ├── arima_module.py                 # ARIMA BIC selection, linear fitting, residual extraction
│   ├── cnn_lstm_module.py              # PyTorch Conv1D + multi-layer LSTM neural network
│   ├── dbo_optimizer.py                # Pure Python/NumPy Dung Beetle Optimizer (4 behaviors)
│   ├── hybrid_model.py                 # ARIMA-DBO-CNN-LSTM end-to-end pipeline
│   ├── baselines.py                    # SVM, Random Forest, standalone LSTM implementations
│   ├── evaluation.py                   # RMSE, MAE, R2 calculation and plotting utilities
│   └── utils.py                        # Reproducibility seed setup, MinMax scaling, sliding window
├── main.py                             # Single city training & evaluation CLI
├── run_all_cities.py                   # Batch script for all 4 cities
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 3. Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Hardware Accelerator**:
   PyTorch will automatically utilize Apple Silicon GPU acceleration (`mps`), NVIDIA CUDA (`cuda`), or fall back to `cpu`.

---

## 4. Usage & Running Experiments

### Running for a Single City (e.g., Beijing)
```bash
python main.py --city beijing
```

Optional arguments:
- `--dbo_pop 15`: Population size for Dung Beetle Optimizer (default: 15)
- `--dbo_iter 10`: Number of iterations for DBO (default: 10)
- `--device mps`: Manually specify device (`mps`, `cpu`, `cuda`)

### Running for Other Cities
```bash
python main.py --city guangzhou
python main.py --city chengdu
python main.py --city wuhan
```

### Running Batch Benchmark Across All 4 Cities
```bash
python run_all_cities.py
```
This generates `results/master_comparison_all_cities.csv` summarizing comparative metrics across all cities.

---

## 5. Comparative Evaluation Models (A to J)

| Label | Model | Description |
|---|---|---|
| **A** | **SVM** | Support Vector Regression with linear kernel |
| **B** | **RF** | Random Forest Regressor (100 estimators) |
| **C** | **ARIMA** | Pure ARIMA statistical linear model |
| **D** | **LSTM** | Standalone 3-layer LSTM (50-100-150 neurons) |
| **E** | **ARIMA-CNN-LSTM** | ARIMA + default CNN-LSTM on residuals (without DBO) |
| **J** | **ARIMA-DBO-CNN-LSTM** | **Proposed Paper Model** (ARIMA + DBO-optimized CNN-LSTM) |

---

## 6. Target Metrics (Reported in Duan et al., 2023)

| City | ARIMA-DBO-CNN-LSTM RMSE | ARIMA-DBO-CNN-LSTM MAE | ARIMA-DBO-CNN-LSTM $R^2$ |
|---|---|---|---|
| **Beijing** | 14.940 | 10.839 | 0.962 |
| **Guangzhou** | 5.496 | 3.770 | 0.953 |
| **Chengdu / Jiaozuo** | 7.594 | 5.285 | 0.989 |
| **Wuhan / Lanzhou** | 7.841 | 5.120 | 0.953 |
