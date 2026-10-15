# Air Quality Prediction: Base Model Replication & Enhanced Forensic Study

**Coursework Project**: Machine Learning and Related Applications  
**Institution**: National Institute of Business Management (NIBM), Kandy Innovation Centre  
**Student Name**: K.W.G.C.D.B. Karunarathne (Student ID: `KIC-BSCCSAI-251P-036`)  
**Target Research Study**: Duan, J., Gong, Y., Luo, J., & Zhao, Z. (2023). *Air-quality prediction based on the ARIMA-CNN-LSTM combination model optimized by dung beetle optimizer*. *Scientific Reports*, 13(1), Article 12127. [DOI: 10.1038/s41598-023-36620-4](https://doi.org/10.1038/s41598-023-36620-4)

---

## 1. Project Organization: Base Model vs. Enhanced Model

This repository is structured into two clearly demarcated implementations, accompanied by a comprehensive technical review report that compares them:

```text
ML Assignment/
├── notebooks/
│   ├── phase1_base_replication.ipynb      # [BASE MODEL] Exact replication of Duan et al. (80/20 split, Table 1 & 2)
│   └── phase2_enhanced_study.ipynb        # [ENHANCED MODEL] Forensic audit, 3-way split, baselines, DM tests, peak breakdown
├── reports/
│   ├── technical_review_report.pdf        # [PHASE 3 REPORT] Submission-ready 6-page Academic PDF (Coursework deliverable)
│   ├── technical_review_report.docx       # [PHASE 3 REPORT] Editable Microsoft Word Document
│   └── technical_review_report.md         # Master academic text with formulas & citations (~4,500 words)
├── src/                                   # Core modular codebase
│   ├── data_loader.py                     # 9 stations aggregation, 80/20 and 3-way chronological splits
│   ├── arima_module.py                    # ARIMA(p, d, q) Box-Jenkins modeling & residual extraction
│   ├── cnn_lstm_module.py                 # PyTorch 1D-CNN + multi-layer LSTM neural network
│   ├── dbo_optimizer.py                   # Pure Python Dung Beetle Optimizer (all 4 biological behaviors)
│   ├── hybrid_model.py                    # End-to-end ARIMA-DBO-CNN-LSTM combination model
│   ├── baselines.py                       # Naive persistence, GBDT, Random Forest, Linear SVM, Standalone LSTM
│   ├── evaluation.py                      # RMSE, MAE, R², MAPE, MASE, peak error bands, Diebold-Mariano tests
│   └── build_documents.py                 # Script generating the submission PDF and DOCX reports
├── main.py                                # CLI entry point for the Base Model Replication
├── run_phase2_experiments.py              # CLI entry point for the Enhanced Empirical Audit
└── results/                               # Cached metrics CSVs and high-resolution comparison figures
```

---

## 2. Model Implementations Breakdown

### A. Base Model Implementation (Author Replication)
* **Interactive Notebook**: [`notebooks/phase1_base_replication.ipynb`](file:///Users/chathurakarunarathna/Desktop/ML%20Assignment/notebooks/phase1_base_replication.ipynb)
* **Execution Script**: `python main.py --city beijing`
* **Methodology**:
  * Exact 80% Train / 20% Test chronological split across 2,528 daily AQI observations (2014–2020).
  * Figure 5 replication showing daily AQI trajectories across Beijing, Chengdu, Guangzhou, and Wuhan.
  * Linear $\text{ARIMA}(p, d, q)$ modeling using the optimal orders from **Table 1** of the paper:
    * Beijing: $\text{ARIMA}(3, 1, 1)$
    * Chengdu: $\text{ARIMA}(3, 1, 1)$
    * Guangzhou: $\text{ARIMA}(2, 1, 2)$
    * Wuhan: $\text{ARIMA}(2, 1, 2)$
  * Extraction of residual sequence $e_t = x_t - \hat{L}_t$ with sliding lag window $w = 10$.
  * Dung Beetle Optimizer (DBO) searching the 7 hyperparameter dimensions within the boundaries of **Table 2**.
  * 1D-CNN + LSTM deep neural network modeling the non-linear residual sequence.
  * Additive reconstruction: $\hat{x}_t = \hat{L}_t + \hat{e}_t$.
  * Comparative baselines from the paper: Linear SVM, Random Forest, Standalone LSTM, and Pure ARIMA.

### B. Enhanced Model Implementation (Audit & Proposed Enhancements)
* **Interactive Notebook**: [`notebooks/phase2_enhanced_study.ipynb`](file:///Users/chathurakarunarathna/Desktop/ML%20Assignment/notebooks/phase2_enhanced_study.ipynb)
* **Execution Script**: `python run_phase2_experiments.py`
* **Methodology & Enhancements**:
  * **Leak-Free 3-Way Split**: 70% Train, 15% Validation, 15% Test with scalers fitted strictly on training data.
  * **Exploratory Data Analysis Table**: Completed descriptive stats, Shapiro-Wilk normality tests ($p < 10^{-30}$), and ADF stationarity tests ($p < 10^{-5}$) across all 4 cities.
  * **Mandatory Naive Persistence Benchmark**: Implementing $\hat{y}_t = x_{t-1}$ to test whether deep models beat atmospheric inertia.
  * **Proposed Machine Learning Alternative**: Gradient Boosted Decision Trees (GBDT / LightGBM) on 10 lag features.
  * **Scale-Free Metrics**: Evaluation via Mean Absolute Scaled Error (MASE) and Mean Absolute Percentage Error (MAPE).
  * **Peak-Day Extreme Event Breakdown**: Error stratification across $\text{AQI} \ge 100$, $\ge 150$, and $\ge 200$.
  * **Diebold–Mariano Statistical Hypothesis Testing**: Formal $p$-value tests verifying whether performance differences are statistically significant.
  * **Stress-Testing**: Gaussian jitter noise injection ($\sigma \in \{0.05, 0.10, 0.20\}$) and multi-pollutant dimensionality analysis.

---

## 3. Master Empirical Comparison Summary (Beijing Test Horizon)

| Model Architecture | Parameters | Training Time | RMSE | MAE | $R^2$ | MAPE (%) | MASE | Status in Repo |
|---|---|---|---|---|---|---|---|---|
| **Author Claimed Hybrid (Paper)** | $> 110,000$ | *Unreported* | **14.940** | **10.839** | **0.962** | *Unreported* | *Unreported* | Literature Reference |
| **Audited Hybrid (Honest Split)** | $112,450$ | 1,480.2 s | **35.980** | **22.899** | **0.388** | **35.57%** | **0.596** | **Enhanced Model** |
| **Pure ARIMA(3, 1, 1)** | 5 | 0.24 s | 34.633 | 22.935 | 0.433 | 41.80% | 0.597 | Both Models |
| **Naive Persistence ($\hat{y}_t = x_{t-1}$)** | 0 | 0.00 s | 37.473 | 23.361 | 0.336 | 41.83% | 0.608 | **Enhanced Benchmark** |
| **Gradient Boosted Trees (Alt. 1)** | 100 trees | 0.42 s | 36.420 | 25.490 | 0.373 | 54.63% | 0.664 | **Proposed Alternative** |
| **Random Forest Regressor** | 100 trees | 1.18 s | 37.265 | 25.447 | 0.343 | 51.46% | 0.663 | Both Models |
| **Linear Support Vector Machine** | 11 | 0.08 s | 38.939 | 31.110 | 0.283 | 72.63% | 0.810 | Both Models |

### Statistical Significance (Diebold–Mariano Tests)
* **Audited Hybrid vs. Naive Persistence**: $\text{DM} = +0.6255, p = 0.5316 \implies$ The 110k-parameter hybrid is **not** statistically significantly better than trivial persistence.
* **Audited Hybrid vs. Pure ARIMA**: $\text{DM} = -2.6668, p = 0.0077 \implies$ Pure ARIMA is **statistically superior** ($p < 0.01$) to the complex hybrid.

---

## 4. Quickstart: Running the Code

### Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Base Model Replication
```bash
python main.py --city beijing
# Or run across all cities:
python run_all_cities.py
```

### Run Enhanced Audit Experiments
```bash
python run_phase2_experiments.py
```

### Recompile Coursework Reports (PDF & Word)
```bash
python src/build_documents.py
```
Outputs are written to [`reports/technical_review_report.pdf`](file:///Users/chathurakarunarathna/Desktop/ML%20Assignment/reports/technical_review_report.pdf) and [`reports/technical_review_report.docx`](file:///Users/chathurakarunarathna/Desktop/ML%20Assignment/reports/technical_review_report.docx).
