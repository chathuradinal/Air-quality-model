"""
generate_architecture_diagram.py
Generates a publication-grade architectural flowchart comparing:
1. Base Model Workflow (Duan et al. 2023)
2. Enhanced Audited Workflow (Karunarathne 2026)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

output_path = Path("/Users/chathurakarunarathna/Desktop/ML Assignment/results/figures/architecture_comparison_workflow.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis("off")

# Title
ax.text(7.5, 8.1, "Architectural Comparison: Base Model (Duan et al., 2023) vs. Enhanced Audited Framework", 
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1B365D")

# Left Column: Base Model
ax.add_patch(patches.FancyBboxPatch((0.4, 0.4), 6.8, 7.3, boxstyle="round,pad=0.2", 
                                    facecolor="#F7FAFC", edgecolor="#CBD5E0", linewidth=1.5))
ax.text(3.8, 7.3, "BASE MODEL WORKFLOW (Duan et al., 2023)", 
        ha="center", va="center", fontsize=12, fontweight="bold", color="#C53030")
ax.text(3.8, 7.0, "Author Replication (Standard 80/20 Chronological Split)", 
        ha="center", va="center", fontsize=9.5, fontstyle="italic", color="#718096")

base_boxes = [
    ("1. Data Ingestion & Global Normalization\n• Raw hourly data aggregated to 24h daily mean\n• Min-Max normalization across ENTIRE 7-year sequence\n⚠️ Global bounds leak future test peaks into training", 
     (0.8, 5.7), 6.0, 1.05, "#FFF5F5", "#FEB2B2", "#9B2C2C"),
    
    ("2. Linear ARIMA Modeling\n• 80% Train / 20% Test split (2,022 train / 506 test)\n• Order (p, d, q) via BIC: Beijing ARIMA(3, 1, 1)\n• Generates linear projection L̂_t", 
     (0.8, 4.4), 6.0, 1.05, "#EDF2F7", "#CBD5E0", "#2D3748"),
    
    ("3. Residual Extraction & DBO Swarm Tuning\n• Non-linear residuals e_t = x_t - L̂_t with lag window w=10\n• 30 Dung beetles optimize 7 hyperparameters\n⚠️ Fitness evaluated directly on test-like evaluation set", 
     (0.8, 3.1), 6.0, 1.05, "#FFF5F5", "#FEB2B2", "#9B2C2C"),
    
    ("4. CNN-LSTM Residual Modeling & Recombination\n• 1D-CNN + MaxPool + Multi-layer LSTM\n• Final additive forecast: x̂_t = L̂_t + ê_t", 
     (0.8, 1.8), 6.0, 1.05, "#EDF2F7", "#CBD5E0", "#2D3748"),
    
    ("5. Author Evaluation & Flattering R² Claim\n• Evaluated on nominal test split without persistence benchmark\n• Claimed R² = 0.962 - 0.989 (Artifact of seasonal variance)\n⚠️ Severe degradation on hazardous days (AQI ≥ 200)", 
     (0.8, 0.5), 6.0, 1.05, "#FFF5F5", "#FEB2B2", "#9B2C2C"),
]

for text, (x, y), w, h, bg, border, tc in base_boxes:
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                                        facecolor=bg, edgecolor=border, linewidth=1.2))
    ax.text(x + 0.2, y + h/2, text, ha="left", va="center", fontsize=8.2, color=tc, linespacing=1.2)

# Downward arrows in base
for y in [5.7, 4.4, 3.1, 1.8]:
    ax.annotate("", xy=(3.8, y - 0.05), xytext=(3.8, y + 0.05),
                arrowprops=dict(arrowstyle="->", color="#A0AEC0", lw=1.5))


# Right Column: Enhanced Model
ax.add_patch(patches.FancyBboxPatch((7.8, 0.4), 6.8, 7.3, boxstyle="round,pad=0.2", 
                                    facecolor="#F0F4F8", edgecolor="#BEE3F8", linewidth=1.5))
ax.text(11.2, 7.3, "ENHANCED AUDITED WORKFLOW (Phase 2 & 3)", 
        ha="center", va="center", fontsize=12, fontweight="bold", color="#2B6CB0")
ax.text(11.2, 7.0, "Student Forensic Audit & Methodological Enhancements", 
        ha="center", va="center", fontsize=9.5, fontstyle="italic", color="#718096")

enh_boxes = [
    ("1. Leak-Free Ingestion & Train-Only Scaling\n• 3-Way Chronological Split (70% Train, 15% Val, 15% Test)\n• Normalizer fitted STRICTLY on training partition [1, T_train]\n✓ Complete isolation of test & validation bounds", 
     (8.2, 5.7), 6.0, 1.05, "#EBF8FF", "#90CDF4", "#2B6CB0"),
    
    ("2. Linear ARIMA Modeling + Seasonal Audit\n• ARIMA(3, 1, 1) rolling one-step-ahead test forecast\n• Seasonal SARIMA evaluation for annual cycle (s = 365)\n✓ Full stationarity & normality diagnostic audit", 
     (8.2, 4.4), 6.0, 1.05, "#FFFFFF", "#BEE3F8", "#2D3748"),
    
    ("3. Quarantined DBO Swarm Optimization\n• 30 Dung beetles search 7 hyperparameter dimensions\n• Fitness scored strictly on RESERVED 15% Validation partition\n✓ Zero test data leakage during hyperparameter search", 
     (8.2, 3.1), 6.0, 1.05, "#EBF8FF", "#90CDF4", "#2B6CB0"),
    
    ("4. Multi-Model Benchmarking & Alternatives\n• Additive Hybrid (ARIMA + DBO-CNN-LSTM)\n• Proposed Alternative: Gradient Boosted Trees (GBDT, 0.42s)\n• Mandatory Baseline: Naive Persistence (ŷ_t = x_{t-1})", 
     (8.2, 1.8), 6.0, 1.05, "#FFFFFF", "#BEE3F8", "#2D3748"),
    
    ("5. Rigorous Multi-Dimensional Evaluation\n• Diebold-Mariano tests (Hybrid vs. Persistence: p = 0.5316)\n• Scale-free MASE metric (0.596 vs. 0.608 for persistence)\n• Peak-day error stratification (AQI ≥ 200 RMSE = 145.67)\n• Gaussian noise jitter & multi-pollutant stress-testing", 
     (8.2, 0.5), 6.0, 1.05, "#EBF8FF", "#90CDF4", "#2B6CB0"),
]

for text, (x, y), w, h, bg, border, tc in enh_boxes:
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                                        facecolor=bg, edgecolor=border, linewidth=1.2))
    ax.text(x + 0.2, y + h/2, text, ha="left", va="center", fontsize=8.2, color=tc, linespacing=1.2)

# Downward arrows in enhanced
for y in [5.7, 4.4, 3.1, 1.8]:
    ax.annotate("", xy=(11.2, y - 0.05), xytext=(11.2, y + 0.05),
                arrowprops=dict(arrowstyle="->", color="#63B3ED", lw=1.5))

# Connecting comparison arrows across columns
ax.annotate("Audit: Leakage Elimination", xy=(8.2, 6.22), xytext=(6.8, 6.22),
            arrowprops=dict(arrowstyle="->", color="#3182CE", lw=1.2, linestyle="--"),
            fontsize=7.5, fontweight="bold", color="#2B6CB0", va="bottom", ha="center")

ax.annotate("Audit: Validation Quarantine", xy=(8.2, 3.62), xytext=(6.8, 3.62),
            arrowprops=dict(arrowstyle="->", color="#3182CE", lw=1.2, linestyle="--"),
            fontsize=7.5, fontweight="bold", color="#2B6CB0", va="bottom", ha="center")

ax.annotate("Audit: DM Significance & Benchmarks", xy=(8.2, 1.02), xytext=(6.8, 1.02),
            arrowprops=dict(arrowstyle="->", color="#3182CE", lw=1.2, linestyle="--"),
            fontsize=7.5, fontweight="bold", color="#2B6CB0", va="bottom", ha="center")

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Flowchart successfully generated at: {output_path}")
