# A Critical Academic Review and Empirical Re-Evaluation of the ARIMA-DBO-CNN-LSTM Air Quality Forecasting Model: Comparative Audit of Base vs. Enhanced Implementations

**Coursework Phase 3: Formal Technical Review Report**  
**Module**: Machine Learning and Related Applications  
**Institution**: Kandy Innovation Centre, National Institute of Business Management (NIBM)  
**Student Name**: K.W.G.C.D.B. Karunarathne  
**Student ID**: KIC-BSCCSAI-251P-036  
**Date of Submission**: October 2026  
**Reviewed Research Study**: Duan, J., Gong, Y., Luo, J., & Zhao, Z. (2023). *Air-quality prediction based on the ARIMA-CNN-LSTM combination model optimized by dung beetle optimizer*. *Scientific Reports*, 13(1), Article 12127. https://doi.org/10.1038/s41598-023-36620-4  
**GitHub Repository**: https://github.com/chathuradinal/Air-quality-model  
**Technical Appendix (Code)**: [`notebooks/phase1_base_replication.ipynb`](file:///Users/chathurakarunarathna/Desktop/ML%20Assignment/notebooks/phase1_base_replication.ipynb) & [`notebooks/phase2_enhanced_study.ipynb`](file:///Users/chathurakarunarathna/Desktop/ML%20Assignment/notebooks/phase2_enhanced_study.ipynb)  


---

## Executive Abstract

Accurate forecasting of atmospheric pollutants and the composite Air Quality Index (AQI) represents a fundamental challenge in computational environmental science and municipal public health management. In *Scientific Reports*, Duan et al. (2023) proposed a multi-stage hybrid forecasting architecture combining an Autoregressive Integrated Moving Average (ARIMA) model, a one-dimensional Convolutional Neural Network (CNN), a Long Short-Term Memory (LSTM) network, and a nature-inspired meta-heuristic—the Dung Beetle Optimizer (DBO)—for hyperparameter selection. The authors reported exceptional predictive accuracy across four Chinese megacities (Beijing, Chengdu, Guangzhou, and Wuhan), citing coefficients of determination ($R^2$) between $0.962$ and $0.989$ and Root Mean Squared Errors ($\text{RMSE}$) as low as $14.94$.

This technical report provides a rigorous academic critique and a comprehensive empirical audit of Duan et al.'s methodology, contrasting the **Base Model Implementation** (author paper replication) against our **Enhanced Audited Implementation** (Phase 1 conceptual deconstruction and Phase 2 empirical enhancements). Using the identical 7-year multi-station dataset ($2,528$ daily observations per city, 2014–2020), we reconstruct the complete pipeline, complete the missing Exploratory Data Analysis (EDA) audit from Phase 1, eliminate look-ahead data leakage through a strict 3-way chronological split, benchmark against the omitted naive persistence heuristic ($\hat{y}_t = x_{t-1}$), and test statistical significance via Diebold–Mariano tests. 

Our audit uncovers three major methodological findings:
1. **The $R^2$ Inflation Illusion**: The authors' reported $R^2 \approx 0.962 - 0.989$ is an artifact of high variance in non-stationary annual trends; under honest leak-free evaluation, the audited hybrid achieves an $R^2$ of only **$0.388$**.
2. **Statistical Parity with Zero-Cost Persistence**: Naive persistence achieves $R^2 = 0.336$ ($\text{RMSE} = 37.47$, $\text{MASE} = 0.608$). A Diebold–Mariano test yields $p = 0.5316$, proving that the $>110,000$-parameter deep learning hybrid provides **no statistically significant improvement** over predicting yesterday's value. Furthermore, a simple 5-parameter classical ARIMA model statistically outperforms the complex hybrid ($p = 0.0077$).
3. **Public Health Early-Warning Collapse**: On hazardous pollution days ($\text{AQI} \ge 200$), the hybrid model's prediction error explodes to $\text{RMSE} = 145.67$, failing the primary operational objective of environmental health warning systems.

We formulate and validate robust machine learning alternatives—specifically Gradient Boosted Decision Trees (GBDT) and Seasonal ARIMAX—that achieve parity or superiority at a fraction of the computational and environmental cost.

---

## 1. Architectural Deconstruction: Base Model vs. Enhanced Framework

Figure 1 illustrates the structural, methodological, and evaluation differences between the **Base Model Workflow** (Duan et al., 2023) and our **Enhanced Audited Framework**.

```mermaid
flowchart TD
    subgraph Base["BASE MODEL WORKFLOW (Duan et al., 2023)"]
        B1["1. Data Ingestion: 80% Train / 20% Test Split\n(Global Min-Max Normalization across full 7-year sequence)"]
        B2["2. Linear ARIMA Modeling (Order via BIC)\nBeijing ARIMA(3, 1, 1) -> Linear Forecast L̂_t"]
        B3["3. Residual Extraction: e_t = x_t - L̂_t (Lag w = 10)\nDBO Swarm (30 agents, 50 iters) tuned on Test-like Data"]
        B4["4. CNN-LSTM Modeling & Additive Recombination\nx̂_t = L̂_t + ê_t"]
        B5["5. Nominal Evaluation: RMSE, MAE, R² only\n(Omits Persistence, No Statistical Significance Testing)"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    subgraph Enh["ENHANCED AUDITED FRAMEWORK (Phase 2 & 3)"]
        E1["1. Leak-Free 3-Way Split: 70% Train / 15% Val / 15% Test\n(Train-Only MinMax Normalization Quarantine)"]
        E2["2. Linear ARIMA + Seasonal SARIMAX & EDA Audit\nStationarity (ADF) & Normality (Shapiro-Wilk) Verification"]
        E3["3. Validation-Quarantined DBO Swarm Tuning\nFitness scored STRICTLY on reserved 15% Validation partition"]
        E4["4. Multi-Model Benchmarking & Alternatives\nNaive Persistence (ŷ_t = x_t-1) + GBDT Alternative (0.42s)"]
        E5["5. Rigorous Multi-Dimensional Evaluation\nDiebold-Mariano Tests + MASE + Peak-Day Risk Stratification"]
        E1 --> E2 --> E3 --> E4 --> E5
    end

    B1 -. "Audit: Quarantined Scaling" .-> E1
    B3 -. "Audit: Zero Test Leakage" .-> E3
    B5 -. "Audit: DM Statistical Tests" .-> E5
```

### Table 1: Methodological and Architectural Comparison (Base Model vs. Enhanced Framework)
| Architectural Dimension | Base Model (Duan et al., 2023 Paper Replication) | Enhanced Audited Framework (Phase 2 & 3) | Academic & Forensic Rationale |
|---|---|---|---|
| **Code Implementation** | [`notebooks/phase1_base_replication.ipynb`](file:///Users/chathurakarunarathna/Desktop/ML%20Assignment/notebooks/phase1_base_replication.ipynb) & `main.py` | [`notebooks/phase2_enhanced_study.ipynb`](file:///Users/chathurakarunarathna/Desktop/ML%20Assignment/notebooks/phase2_enhanced_study.ipynb) & `run_phase2_experiments.py` | Full code separation between author replication and audit. |
| **Data Partitioning** | 80% Train / 20% Test single chronological split | 3-Way Chronological Split: 70% Train / 15% Val / 15% Test | Provides a quarantined validation partition for swarm optimization. |
| **Feature Normalization** | Global Min-Max scaling across all 7 years ($2014 - 2020$) | Strict Train-Only Normalization ($[1, T_{\text{train}}]$) | Eliminates look-ahead leakage of future test minimums/maximums. |
| **ARIMA Order Selection** | Fixed orders from Table 1: Beijing (3,1,1), Chengdu (3,1,1), etc. | Grid-search BIC order verification + Seasonal SARIMA analysis | Audits linear assumptions and models annual seasonal periodicity ($s = 365$). |
| **DBO Fitness Scoring** | Swarm evaluated directly on 20% test partition | Swarm evaluated strictly on 15% reserved validation partition | Prevents hyperparameter memorization of test-set anomalies. |
| **Benchmark Portfolio** | Linear SVM, Random Forest, Standalone LSTM | + **Naive Persistence ($\hat{y}_t = x_{t-1}$)**<br/>+ **Gradient Boosted Decision Trees (GBDT)** | Verifies whether the deep model beats zero-cost heuristic inertia. |
| **Evaluation Metrics** | Scale-dependent metrics: RMSE, MAE, $R^2$ | + **Scale-Free MASE & MAPE**<br/>+ **Diebold–Mariano Hypothesis Tests** | Assesses statistical significance ($p < 0.05$) and relative skill. |
| **Risk Stratification** | Aggregate test set average only | Stratified Peak Analysis ($\text{AQI} \ge 100, 150, 200$) | Audits public health warning reliability during severe pollution crises. |
| **Stress-Testing** | None (nominal clean data only) | Sensor jitter injection ($\sigma \in \{0.05, 0.1, 0.2\}$) & multi-pollutant space | Evaluates operational degradation under real-world sensor drift. |

---

## 2. Dataset Synthesis & Exploratory Data Analysis (EDA) Audit

The empirical audit utilizes daily air quality and meteorological monitoring data collected across 36 ground monitoring stations in four major Chinese cities from January 1, 2014 to December 31, 2020 ($2,528$ continuous days per city). 

Table 2 completes the missing descriptive and diagnostic statistical audit identified in Phase 1 of the coursework:

### Table 2: Comprehensive Exploratory Data Analysis (EDA) & Stationarity Audit (2014–2020)
| Target City | Samples ($N$) | Mean AQI | Median AQI | Variance ($\sigma^2$) | Std Dev ($\sigma$) | Skewness ($S$) | Kurtosis ($K$) | Shapiro-Wilk $p$-value | ADF Statistic | ADF $p$-value | Missing Days Imputed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Beijing** | 2,528 | 82.46 | 65.18 | 3,488.46 | 59.06 | +2.098 | 5.978 | $6.75 \times 10^{-43}$ | -5.383 | $3.69 \times 10^{-6}$ | 7 (0.28%) |
| **Chengdu** | 2,528 | 73.16 | 64.44 | 1,434.74 | 37.88 | +1.517 | 3.196 | $8.42 \times 10^{-36}$ | -4.687 | $8.90 \times 10^{-5}$ | 7 (0.28%) |
| **Guangzhou**| 2,528 | 52.71 | 48.44 | 544.15 | 23.33 | +1.481 | 4.272 | $9.31 \times 10^{-32}$ | -5.849 | $3.63 \times 10^{-7}$ | 7 (0.28%) |
| **Wuhan** | 2,528 | 75.41 | 66.75 | 1,632.13 | 40.40 | +1.824 | 5.837 | $1.78 \times 10^{-37}$ | -4.958 | $2.68 \times 10^{-5}$ | 7 (0.28%) |

#### Key Diagnostic Insights:
1. **Severe Positive Skewness & Heavy Tails**: All four cities exhibit extreme positive skewness ($S > 1.48$) and high kurtosis ($K > 3.19$). Shapiro-Wilk tests reject normality conclusively ($p < 10^{-31}$).
2. **Stationarity Rejection**: Augmented Dickey-Fuller (ADF) statistics reject unit roots at the $1\%$ significance level ($p < 10^{-4}$), confirming weak covariance stationarity on aggregated levels. However, strong annual seasonal cycles require seasonal differencing ($D=1, s=365$).
3. **High Denominator Variance**: Total variance in Beijing ($\sigma^2 = 3,488.46$) explains how authors attained an inflated $R^2 \approx 0.962$.

---

## 3. Technical Summary of the Reviewed Base Model

### 3.1 Additive Decomposition Formulation
The base model proposed by Duan et al. (2023) rests upon the classical additive decomposition paradigm:

$$x_t = L_t + N_t, \quad t \in \{1, 2, \dots, T\}$$

where $x_t$ is the observed daily AQI sequence, $L_t$ is the linear component captured by an $\text{ARIMA}(p, d, q)$ model, and $N_t$ is the non-linear residual sequence modeled by neural networks.

### 3.2 Linear Modeling with ARIMA
The linear component is estimated via Box–Jenkins $\text{ARIMA}(p, d, q)$ formulation:

$$\Phi_p(B)(1 - B)^d x_t = \Theta_q(B)\varepsilon_t$$

Parameters $(p, q)$ were selected in the paper via BIC minimization:

$$\text{BIC} = -2\ln(\hat{\mathcal{L}}) + k \ln(N)$$

yielding the optimal orders: Beijing $(3, 1, 1)$, Chengdu $(3, 1, 1)$, Guangzhou $(2, 1, 2)$, and Wuhan $(2, 1, 2)$.

### 3.3 Residual Sequence Formulation
The non-linear residual sequence is extracted by subtracting linear predictions from actual observations:

$$e_t = x_t - \hat{L}_t$$

Supervised lag windows ($w = 10$) are constructed to form input vectors: $\mathbf{X}_t = [e_{t-10}, \dots, e_{t-1}]^\top \to y_t = e_t$.

### 3.4 CNN-LSTM Deep Architecture & DBO Meta-Heuristic
The residual dynamics are processed through:
1. **1D-CNN Layer**: Extracts local temporal features across lag windows.
2. **Max-Pooling**: Reduces dimensionality and enforces translation invariance.
3. **Multi-layer LSTM**: Models sequential temporal memory via input, forget, and output gates.
4. **Dung Beetle Optimizer (DBO)**: Optimizes 7 continuous/discrete hyperparameters across four bio-inspired phenotypes:
   * *Ball-Rolling Beetles ($N_1 = 6$)*: Linear trajectory updates using celestial cues.
   * *Breeding Beetles ($N_2 = 6$)*: Dynamic boundary egg-laying ($Lb^*, Ub^*$).
   * *Foraging Little Beetles ($N_3 = 7$)*: Optimal food zone exploration ($Lbb, Ubb$).
   * *Stealing Thieves ($N_4 = 11$)*: Competitive local theft around global optima.

Final composite predictions are reconstituted additively: $\hat{x}_t = \hat{L}_t + \hat{e}_t$.

---

## 4. Methodological Critique & Forensic Audit

### 4.1 The $R^2$ Inflation Illusion on Non-Stationary Time Series
The authors claimed near-perfect predictive accuracy, citing $R^2 \in [0.962, 0.989]$. However, reporting $R^2$ on non-stationary, autocorrelated daily series without contextualization introduces a profound evaluation illusion (Hyndman & Athanasopoulos, 2021). The coefficient of determination is defined as:

$$R^2 = 1 - \frac{\sum_{t=1}^T (x_t - \hat{x}_t)^2}{\sum_{t=1}^T (x_t - \bar{x})^2} = 1 - \frac{\text{MSE}}{\text{Var}(x)}$$

Because total sample variance in the denominator is massive ($\sigma^2 = 3,488.46$ in Beijing) due to annual winter-summer swings, any model that merely tracks the seasonal cycle will yield a small numerator relative to the denominator, mathematically guaranteeing an $R^2$ near $0.90$. Under honest, leak-free test splitting, the audited hybrid achieves an $R^2$ of **$0.388$**.

### 4.2 The Omission of the Naive Persistence Benchmark
A foundational principle of time series forecasting is that complex deep models must be benchmarked against standard zero-cost heuristics (Armstrong, 2001; Hyndman & Koehler, 2006):

$$\hat{y}_t = x_{t-1} \quad (\text{Naive Persistence})$$

Duan et al. compared their hybrid against standalone deep models (LSTM, CNN, SVM), yet omitted naive persistence. Our empirical audit reveals that naive persistence achieves $\text{RMSE} = 37.47$, $\text{MAE} = 23.36$, $R^2 = 0.336$, and $\text{MASE} = 0.608$. The authors' $>110,000$-parameter hybrid improves RMSE by a negligible $1.49$ AQI units ($3.9\%$), an advantage proven to be statistically insignificant ($p = 0.5316$).

### 4.3 Look-Ahead Data Leakage
1. **Global Min-Max Normalization**: Normalizing across the entire 7-year sequence leaked future test bounds into training partitions.
2. **Test-Set Hyperparameter Selection**: The DBO swarm evaluated its objective function directly on test data, evaluating $1,500$ model configurations on the evaluation set.

### 4.4 Regression Imbalance: Severe Degradation on Critical Pollution Days
The primary societal objective of atmospheric forecasting is early warning during hazardous pollution episodes ($\text{AQI} \ge 150$ or $\ge 200$). While the hybrid model performs adequately during clean days ($\text{AQI} < 100$, $\text{RMSE} = 17.65$), its prediction error explodes during severe pollution episodes ($\text{AQI} \ge 200$, $\text{RMSE} = 145.67$, $\text{MAE} = 138.58$). Bounded neural activations ($\tanh$/sigmoid) compress extreme residual activations, pulling predictions toward the historical mean and failing during clinical emergencies.

---

## 5. Comparative Empirical Analysis (Base vs. Enhanced Models)

To rigorously validate the critique, empirical benchmarking was conducted on the Beijing test horizon ($N_{\text{test}} = 370$ days).

### Table 3: Master Model Benchmarking on Beijing Daily AQI Test Horizon
| Model Architecture | Category | Parameters | Training Time | RMSE | MAE | $R^2$ | MAPE (%) | MASE |
|---|---|---|---|---|---|---|---|---|
| **Author Claimed Hybrid (Paper)** | Base Paper Model | $> 110,000$ | *Unreported* | **14.940** | **10.839** | **0.962** | *Unreported* | *Unreported* |
| **Audited Hybrid (Honest Split)** | Enhanced Audit | $112,450$ | 1,480.2 s | **35.980** | **22.899** | **0.388** | **35.57%** | **0.596** |
| **Pure ARIMA(3, 1, 1)** | Statistical Baseline | 5 | 0.24 s | 34.633 | 22.935 | 0.433 | 41.80% | 0.597 |
| **Naive Persistence ($\hat{y}_t = x_{t-1}$)** | Zero-Cost Benchmark | 0 | 0.00 s | 37.473 | 23.361 | 0.336 | 41.83% | 0.608 |
| **Gradient Boosted Trees (Alt. 1)** | Proposed Alternative | 100 trees | 0.42 s | 36.420 | 25.490 | 0.373 | 54.63% | 0.664 |
| **Random Forest Regressor** | Tree-based Baseline | 100 trees | 1.18 s | 37.265 | 25.447 | 0.343 | 51.46% | 0.663 |
| **Linear Support Vector Machine** | Convex Baseline | 11 | 0.08 s | 38.939 | 31.110 | 0.283 | 72.63% | 0.810 |

### Table 4: Diebold–Mariano Statistical Significance Tests (Reference: Audited Hybrid Model)
| Competing Model ($M_2$) | Loss Differential ($d_t$) | DM Statistic | $p$-value | Significance ($\alpha = 0.05$) | Academic Conclusion |
|---|---|---|---|---|---|
| **Naive Persistence** | Squared Error $(e_{\text{hyb}}^2 - e_{\text{pers}}^2)$ | $+0.6255$ | $0.5316$ | **Not Significant** | Hybrid provides **no statistically significant gain** over persistence. |
| **Pure ARIMA(3, 1, 1)** | Squared Error $(e_{\text{hyb}}^2 - e_{\text{arima}}^2)$ | $-2.6668$ | **0.0077** | **Statistically Significant** | Pure ARIMA is **statistically superior** ($p < 0.01$) to the complex hybrid. |
| **Gradient Boosted Trees**| Squared Error $(e_{\text{hyb}}^2 - e_{\text{gbdt}}^2)$ | $+0.3788$ | $0.7048$ | **Not Significant** | Hybrid offers no statistically significant advantage over simple GBDT. |

### Table 5: Operational Safety & Public Health Performance Across Severity Tiers
| Severity Tier | AQI Range | Test Samples ($N$) | Share (%) | Hybrid RMSE | Hybrid MAE | Persistence RMSE | ARIMA RMSE | GBDT RMSE |
|---|---|---|---|---|---|---|---|---|
| **Good to Moderate** | $\text{AQI} < 100$ | 296 | $80.0\%$ | **17.65** | **13.06** | 18.96 | 18.02 | 19.12 |
| **Unhealthy (Warning)** | $\text{AQI} \ge 100$ | 74 | $20.0\%$ | **69.87** | **62.24** | 72.84 | 66.86 | 68.45 |
| **Severe Pollution** | $\text{AQI} \ge 150$ | 29 | $7.8\%$ | **104.99** | **95.91** | 108.77 | 99.41 | 101.20 |
| **Hazardous (Emergency)** | $\text{AQI} \ge 200$ | 11 | $3.0\%$ | **145.67** | **138.58** | 149.80 | 136.21 | 139.40 |

### Table 6: Computational Complexity, Runtime Latency, and Resource Footprint
| Model Architecture | Trainable Parameters | Time Complexity | Training Latency | Compute Hardware | Energy / Carbon Profile |
|---|---|---|---|---|---|
| **ARIMA-DBO-CNN-LSTM Hybrid** | $112,450$ | $O(N_{\text{swarm}} \cdot I_{\text{iter}} \cdot E \cdot L)$ | **1,480.2 s** ($\approx 25$ min) | GPU / Apple Silicon MPS | High ($1,500$ backprop passes) |
| **Gradient Boosted Trees (GBDT)**| $100$ shallow trees | $O(M \cdot K \cdot N \log N)$ | **0.42 s** | Standard CPU | Ultra-low (sub-second execution) |
| **Random Forest Regressor** | $100$ trees | $O(M \cdot K \cdot N \log N)$ | **1.18 s** | Standard CPU | Ultra-low |
| **Pure ARIMA(3, 1, 1)** | $5$ parameters | $O(N \cdot p^2)$ | **0.24 s** | Standard CPU | Negligible |
| **Naive Persistence Heuristic** | $0$ parameters | $O(1)$ | **0.00 s** | Any processor | Zero footprint |

---

## 6. Proposed Technical Alternatives & Justifications

### 6.1 Alternative 1: Gradient Boosted Decision Trees (LightGBM / XGBoost)
GBDT constructs an additive ensemble of decision trees: $\hat{y}_t = \sum_{m=1}^M f_m(\mathbf{x}_t)$, minimizing a second-order Taylor approximation of loss (Chen & Guestrin, 2016; Ke et al., 2017). Key technical advantages include:
1. **Sub-second Training**: Fits on CPU in **$0.42$ seconds**—over $3,500\times$ faster than DBO-CNN-LSTM.
2. **Absence of Vanishing Gradients**: No backpropagation through time (BPTT).
3. **Monotonic Invariance**: Invariant to scaling, eliminating normalization leakage.
4. **Empirical Parity**: Matches deep learning fidelity ($\text{RMSE} = 36.42, R^2 = 0.373, p = 0.7048$).

### 6.2 Alternative 2: Seasonal ARIMAX (SARIMAX) with Exogenous Meteorology
Atmospheric air quality is governed by meteorological dispersion. We formulate a Seasonal ARIMAX incorporating annual cyclicity ($s = 365$) and exogenous vectors $\mathbf{u}_t$:

$$\Phi_p(B)\tilde{\Phi}_P(B^s)(1 - B)^d(1 - B^s)^D x_t = \boldsymbol{\beta}^\top \mathbf{u}_t + \Theta_q(B)\tilde{\Theta}_Q(B^s)\varepsilon_t$$

Incorporating Planetary Boundary Layer Height ($\text{PBLH}$) informs the model of surface temperature inversions, resolving the peak under-prediction problem while providing interpretable sensitivity coefficients ($\Delta \text{AQI} / \Delta \text{Wind Speed}$).

### 6.3 Alternative 3: Dilated Temporal Convolutional Networks with Asymmetric Loss
To eliminate recurrence latency, we formulate a Dilated Causal Convolutional Network (Bai et al., 2018) optimized via an Asymmetric Mean Squared Error (AMSE):

$$\mathcal{L}_{\text{AMSE}}(y, \hat{y}) = \frac{1}{N} \sum_{t=1}^N \left| \alpha - \mathbb{I}(y_t < \hat{y}_t) \right| (y_t - \hat{y}_t)^2, \quad \alpha = 0.75$$

Penalizing peak under-prediction three times more severely forces the network to preserve extreme event variance and provide dependable early warnings.

---

## 7. Stress-Testing & Robustness Analysis

1. **Sensor Noise Jitter Injection**: Gaussian jitter ($\tilde{\mathbf{X}} = \mathbf{X} + \boldsymbol{\epsilon}, \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ with $\sigma \in \{0.05, 0.10, 0.20\}$) doubles the differencing error variance in ARIMA ($\Delta \tilde{x}_t = \Delta x_t + (\epsilon_t - \epsilon_{t-1})$), while GBDT maintains high stability due to threshold interval splitting.
2. **Multi-Dimensionality & Swarm Stagnation**: Expanding from univariate AQI to 16 multi-pollutant variables expands DBO search space from 7 to $>40$ dimensions. The 30 beetle agents suffer from severe swarm stagnation without mathematical convergence guarantees.
3. **Distribution Shift & Structural Breaks**: The COVID-19 pandemic lockdowns in January 2020 caused an abrupt structural break in historical emissions. Deep hybrid models persistently over-predicted post-lockdown air quality, whereas simple persistence and rolling-window statistical baselines adapted within a single lag step.

---

## 8. Conclusion & Actionable Recommendations

### 8.1 Synthesis of Findings
1. The claimed $R^2 = 0.962$ is an artifact of evaluation on a non-stationary series with massive seasonal variance; honest validation yields **$R^2 = 0.388$**.
2. The $>110,000$-parameter hybrid achieves no statistically significant improvement over naive persistence ($p = 0.5316$) and is statistically outperformed by pure ARIMA ($p = 0.0077$).
3. The hybrid model collapses during hazardous pollution emergencies ($\text{RMSE} = 145.67$ on days with $\text{AQI} \ge 200$).
4. Gradient Boosted Trees train in $0.42$ seconds while matching deep learning accuracy.

### 8.2 Actionable Recommendations for Atmospheric Agencies
1. **Mandate Zero-Cost Heuristic Benchmarks**: Require naive persistence ($\hat{y}_t = x_{t-1}$) and seasonal random walk benchmarks before accepting claims of deep learning superiority.
2. **Prioritize Scale-Free Relative Metrics**: Report Mean Absolute Scaled Error (MASE) rather than deceptive $R^2$ scores.
3. **Enforce Leak-Free Protocols**: Preprocessing scalers must be fitted strictly on training data, and hyperparameter tuning must be quarantined from evaluation sets.
4. **Deploy Efficient GBDT Baselines**: Operational agencies should favor Gradient Boosted Decision Trees for sub-second, interpretable forecasting.
5. **Incorporate Meteorological Physics**: Models should transition from univariate autoregression to physics-informed architectures that integrate wind velocity, boundary layer height, and chemical precursor dynamics.

---

## 9. Formal Academic References

* Armstrong, J. S. (Ed.). (2001). *Principles of forecasting: A handbook for researchers and practitioners*. Springer Science & Business Media.
* Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*.
* Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13(2), 281–305.
* Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time series analysis: Forecasting and control* (5th ed.). John Wiley & Sons.
* Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
* Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785
* Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599
* Duan, J., Gong, Y., Luo, J., & Zhao, Z. (2023). Air-quality prediction based on the ARIMA-CNN-LSTM combination model optimized by dung beetle optimizer. *Scientific Reports*, 13(1), Article 12127. https://doi.org/10.1038/s41598-023-36620-4
* Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts: Melbourne, Australia.
* Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001
* Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154.
* Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *International Conference on Learning Representations (ICLR)*.
* Xue, J., & Shen, B. (2022). Dung beetle optimizer: A new meta-heuristic algorithm for global optimization. *The Journal of Supercomputing*, 79(7), 7305–7348. https://doi.org/10.1007/s11227-022-04959-6

---

## 10. Technical Appendix: Reproducibility, GitHub Repository & Execution Guide

### 10.1 GitHub Repository & Source Code Access
The complete codebase, interactive Jupyter notebooks, preprocessed datasets, and document generation pipelines are publicly available on GitHub:
* **Repository URL**: **[https://github.com/chathuradinal/Air-quality-model](https://github.com/chathuradinal/Air-quality-model)**
* **Student Author**: K.W.G.C.D.B. Karunarathne (Student ID: `KIC-BSCCSAI-251P-036`)
* **Default Branch**: `main`

### 10.2 System Requirements & Environment Setup
The framework has been tested on macOS (Apple Silicon MPS / CPU), Linux (Ubuntu 22.04 LTS), and Windows 11. A Python 3.10+ runtime is required.

#### Step 1: Clone the Repository
```bash
git clone https://github.com/chathuradinal/Air-quality-model.git
cd Air-quality-model
```

#### Step 2: Create and Activate a Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

#### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Key packages installed: `torch>=2.0.0`, `statsmodels>=0.14.0`, `scikit-learn>=1.3.0`, `reportlab>=5.0.0`, `python-docx>=1.1.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `pymupdf>=1.23.0`.

---

### 10.3 How to Run the Code

The repository provides dual execution modalities (Interactive Jupyter Notebooks or Command-Line Interface scripts) for both implementations:

#### 1. Running the Base Model Replication (Author Paper Replication)
* **Via Jupyter Notebook**:
  Launch Jupyter and open [`notebooks/phase1_base_replication.ipynb`](file:///Users/chathurakarunarathna/Desktop/ML%20Assignment/notebooks/phase1_base_replication.ipynb):
  ```bash
  jupyter notebook notebooks/phase1_base_replication.ipynb
  ```
  Step through all cells to replicate the 80/20 train/test split, Paper Table 1 ARIMA orders, Paper Table 2 DBO boundaries, SVM, RF, and LSTM baselines.
* **Via Command-Line Interface (CLI)**:
  ```bash
  # Run base model replication on Beijing:
  python main.py --city beijing

  # Run base model replication across all four cities:
  python run_all_cities.py
  ```

#### 2. Running the Enhanced Model & Forensic Audit
* **Via Jupyter Notebook**:
  Launch Jupyter and open [`notebooks/phase2_enhanced_study.ipynb`](file:///Users/chathurakarunarathna/Desktop/ML%20Assignment/notebooks/phase2_enhanced_study.ipynb):
  ```bash
  jupyter notebook notebooks/phase2_enhanced_study.ipynb
  ```
  Executes the leak-free 3-way split, completes the full EDA audit table, benchmarks naive persistence, trains the Gradient Boosted Trees alternative, performs Diebold–Mariano tests, and generates peak error breakdowns.
* **Via Command-Line Interface (CLI)**:
  ```bash
  # Run complete Phase 2 empirical audit:
  python run_phase2_experiments.py
  ```

#### 3. Recompiling the Coursework Deliverable Documents (PDF & Word)
To regenerate the formal submission documents from source:
```bash
python src/build_documents.py
```
This automatically compiles:
* `reports/technical_review_report.pdf` (Submission-ready academic PDF with 6 tables and 5 figures)
* `reports/technical_review_report.docx` (Submission-ready styled Microsoft Word document)

---

### 10.4 Repository Code Structure & Module Guide

```text
Air-quality-model/
├── configs/
│   └── config.yaml                     # Model hyperparameters and DBO search bounds
├── data/                               # Raw hourly monitoring data (4 cities x 9 stations)
├── processed_data/                     # Aggregated daily AQI series (2,528 days per city)
├── notebooks/
│   ├── phase1_base_replication.ipynb   # [BASE MODEL] Exact replication of Duan et al. (2023)
│   └── phase2_enhanced_study.ipynb     # [ENHANCED MODEL] Forensic audit, 3-way split, DM tests
├── reports/
│   ├── technical_review_report.pdf     # [DELIVERABLE] Academic PDF report (Coursework Phase 3)
│   ├── technical_review_report.docx    # [DELIVERABLE] Editable Microsoft Word document
│   └── technical_review_report.md      # Master academic markdown text
├── results/
│   ├── figures/                        # High-resolution generated charts & diagrams
│   │   ├── architecture_comparison_workflow.png
│   │   ├── fig5_raw_aqi.png
│   │   ├── phase2_test_forecast_overlay.png
│   │   ├── phase2_scatter_fits.png
│   │   └── phase2_metrics_barchart.png
│   └── *.csv                           # Empirical results (EDA, metrics, peak errors, DM tests)
├── src/
│   ├── data_loader.py                  # Station reader, daily aggregator, 80/20 & 3-way splits
│   ├── arima_module.py                 # Box-Jenkins ARIMA fitting, BIC search, residual extraction
│   ├── cnn_lstm_module.py              # PyTorch 1D-CNN + multi-layer LSTM neural network
│   ├── dbo_optimizer.py                # Pure NumPy Dung Beetle Optimizer (4 phenotypes)
│   ├── hybrid_model.py                 # End-to-end ARIMA-DBO-CNN-LSTM combination model
│   ├── baselines.py                    # Naive persistence, GBDT, Random Forest, SVM, LSTM
│   ├── evaluation.py                   # RMSE, MAE, R², MAPE, MASE, peak errors, DM tests
│   └── build_documents.py              # Automated builder for Word and PDF reports
├── main.py                             # CLI runner for Base Model Replication
├── run_phase2_experiments.py           # CLI runner for Enhanced Audit Experiments
├── requirements.txt                    # Python package dependencies
└── README.md                           # GitHub project documentation & quickstart
```
