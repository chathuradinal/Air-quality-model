#!/usr/bin/env python3
"""
build_documents.py
Generates the publication-quality Coursework Deliverables:
1. reports/technical_review_report.docx (Microsoft Word)
2. reports/technical_review_report.pdf (Academic PDF via ReportLab)
"""

import os
import sys
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    KeepTogether, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas

BASE_DIR = Path("/Users/chathurakarunarathna/Desktop/ML Assignment")
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = BASE_DIR / "results" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Numbered Canvas for ReportLab PDF Header & Footer
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8.5)
        self.setFillColor(colors.HexColor("#4A5568"))

        width, height = A4
        margin = 46  # ~0.64 in

        # Header (Skip page 1)
        if self._pageNumber > 1:
            self.drawString(margin, height - 32, "Technical Review Report: Empirical Re-Evaluation of ARIMA-DBO-CNN-LSTM")
            self.drawRightString(width - margin, height - 32, "K.W.G.C.D.B. Karunarathne (KIC-BSCCSAI-251P-036)")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.6)
            self.line(margin, height - 36, width - margin, height - 36)

        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.6)
        self.line(margin, 40, width - margin, 40)

        self.drawString(margin, 28, "NIBM — Machine Learning and Related Applications (Phase 3 Report)")
        self.drawRightString(width - margin, 28, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


# ---------------------------------------------------------------------------
# DOCX Helper Utilities
# ---------------------------------------------------------------------------
def set_cell_background(cell, hex_color):
    """Sets the background color of a docx table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets inner padding for a docx cell in twips."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def add_callout_box(doc, text, title="KEY FINDING / AUDIT NOTE"):
    """Adds a styled callout box in docx."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    # Left border
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="1B365D"/>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run_title = p.add_run(f"[{title}] ")
    run_title.bold = True
    run_title.font.name = "Arial"
    run_title.font.size = Pt(9.5)
    run_title.font.color.rgb = RGBColor(27, 54, 93)

    run_text = p.add_run(text)
    run_text.font.name = "Arial"
    run_text.font.size = Pt(9.5)
    run_text.font.italic = True
    run_text.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


# ---------------------------------------------------------------------------
# BUILD DOCX DOCUMENT
# ---------------------------------------------------------------------------
def generate_docx():
    print("Generating reports/technical_review_report.docx ...")
    doc = Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("A Critical Academic Review and Empirical Re-Evaluation of the ARIMA-DBO-CNN-LSTM Air Quality Forecasting Model")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(21)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(27, 54, 93)  # Deep Navy

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run("Coursework Phase 3: Formal Technical Review Report (40 Marks)\nBridging Conceptual Critique and Empirical Verification")
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor(74, 85, 104)

    # Metadata Box
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    metadata = [
        ("Student Name & ID:", "K.W.G.C.D.B. Karunarathne | Student ID: KIC-BSCCSAI-251P-036"),
        ("Module & Programme:", "Machine Learning and Related Applications | BSc (Hons) Computing & AI"),
        ("Awarding Institution:", "National Institute of Business Management (NIBM), Kandy Innovation Centre"),
        ("Target Research Study:", "Duan, J., Gong, Y., Luo, J., & Zhao, Z. (2023). Scientific Reports, 13:12127."),
        ("Verification Code Base:", "Interactive Appendix: notebooks/phase2_enhanced_study.ipynb")
    ]
    for row_idx, (k, v) in enumerate(metadata):
        c0 = meta_table.cell(row_idx, 0)
        c1 = meta_table.cell(row_idx, 1)
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        set_cell_background(c0, "F7FAFC")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
        set_cell_margins(c1, top=60, bottom=60, left=100, right=100)
        
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(1)
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.name = "Arial"
        r0.font.size = Pt(9.5)
        r0.font.color.rgb = RGBColor(27, 54, 93)

        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(1)
        r1 = p1.add_run(v)
        r1.font.name = "Arial"
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Helper function for text insertion
    def add_sec_heading(title, level=1):
        h = doc.add_paragraph()
        h.paragraph_format.keep_with_next = True
        run = h.add_run(title)
        run.bold = True
        run.font.name = "Arial"
        if level == 1:
            h.paragraph_format.space_before = Pt(18)
            h.paragraph_format.space_after = Pt(6)
            run.font.size = Pt(14)
            run.font.color.rgb = RGBColor(27, 54, 93)
        elif level == 2:
            h.paragraph_format.space_before = Pt(12)
            h.paragraph_format.space_after = Pt(4)
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(43, 108, 176)
        else:
            h.paragraph_format.space_before = Pt(8)
            h.paragraph_format.space_after = Pt(3)
            run.font.size = Pt(10.5)
            run.font.color.rgb = RGBColor(74, 85, 104)

    def add_p(text, bold_prefix="", italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.bold = True
            r_pre.font.name = "Calibri"
            r_pre.font.size = Pt(10.5)
            r_pre.font.color.rgb = RGBColor(27, 54, 93)
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10.5)
        r.font.italic = italic
        r.font.color.rgb = RGBColor(45, 55, 72)

    # --- Executive Abstract ---
    add_sec_heading("Executive Abstract")
    add_p(
        "Accurate prediction of atmospheric pollutants and the overall Air Quality Index (AQI) represents a critical operational challenge "
        "for municipal environmental protection agencies, public health warning systems, and urban epidemiology. In a recently published "
        "study in Scientific Reports, Duan et al. (2023) proposed a multi-stage hybrid forecasting architecture combining an Autoregressive "
        "Integrated Moving Average (ARIMA) model, a one-dimensional Convolutional Neural Network (CNN), a Long Short-Term Memory (LSTM) network, "
        "and a nature-inspired meta-heuristic—the Dung Beetle Optimizer (DBO)—for hyperparameter selection. The authors claimed near-perfect "
        "predictive fidelity on daily AQI series across four major Chinese metropolitan centres (Beijing, Chengdu, Guangzhou, and Wuhan), "
        "reporting coefficients of determination (R²) spanning 0.962 to 0.989 and Root Mean Squared Errors (RMSE) as low as 14.94."
    )
    add_p(
        "This technical report provides a rigorous academic critique and an end-to-end empirical audit of Duan et al.'s methodology, building "
        "upon the theoretical deconstruction conducted in Phase 1 and the empirical implementation in Phase 2. Utilizing the identical 7-year "
        "historical dataset (2,528 daily observations per city, 2014–2020), we reconstruct the complete hybrid pipeline, enforce leak-free "
        "chronological data splitting, implement mandatory baseline benchmarks, and evaluate statistical significance via the Diebold–Mariano test. "
        "Our audit reveals three critical methodological vulnerabilities: (1) an inflated R² illusion driven by autocorrelation and seasonal trends; "
        "(2) data leakage during global feature normalization and swarm fitness scoring; and (3) an absolute failure to benchmark against naive persistence "
        "(ŷ_t = x_{t-1}). Under rigorous experimental conditions, the audited hybrid achieves an R² of only 0.388 (RMSE = 35.98). Crucially, naive "
        "persistence achieves R² = 0.336 (RMSE = 37.47), and a Diebold–Mariano test yields p = 0.5316, demonstrating that the authors' >110,000-parameter "
        "deep learning hybrid provides no statistically significant improvement over trivial persistence. Furthermore, on hazardous pollution days "
        "(AQI ≥ 200), the hybrid model's prediction error explodes to RMSE = 145.67, failing the primary operational objective of public health protection. "
        "We propose robust machine learning alternatives—specifically Gradient Boosted Decision Trees and Seasonal ARIMAX—that deliver equivalent "
        "or superior accuracy at a fraction of the computational and environmental cost."
    )

    add_callout_box(
        doc,
        "The audited ARIMA-DBO-CNN-LSTM model achieves R² = 0.388 under leak-free validation, compared to the authors' claimed 0.962. "
        "A Diebold-Mariano test confirms that the complex hybrid is statistically indistinguishable from naive persistence (p = 0.5316) "
        "and statistically inferior to pure ARIMA (p = 0.0077).",
        "KEY AUDIT DISCOVERY"
    )

    # --- Section 1: Technical Summary ---
    add_sec_heading("1. Technical Summary of the Reviewed Paper")
    add_sec_heading("1.1 Additive Decomposition Framework", level=2)
    add_p(
        "The foundational architecture proposed by Duan et al. (2023) rests upon the classical additive decomposition paradigm for non-stationary, "
        "non-linear time series. The observed univariate atmospheric sequence x_t is assumed to be an additive superposition of a linear stationary "
        "component L_t and a non-linear chaotic residual component N_t:"
    )
    add_p("x_t = L_t + N_t,    t ∈ {1, 2, ..., T}", italic=True)
    add_p(
        "The core rationale of the authors is that single statistical models (such as ARIMA) cannot capture high-order atmospheric non-linearities "
        "driven by complex photochemical reactions and boundary layer meteorology. Conversely, standalone deep neural networks frequently overfit "
        "or exhibit phase lag when processing non-stationary trends. By decomposing the series, linear dynamics are isolated and modeled statistically, "
        "leaving the residual sequence e_t = x_t - L̂_t to be modeled by neural networks."
    )

    add_sec_heading("1.2 Linear Modeling via ARIMA", level=2)
    add_p(
        "The linear component L̂_t is captured using a classical Box–Jenkins ARIMA(p, d, q) formulation: "
        "Φ_p(B)(1 - B)^d x_t = Θ_q(B)ε_t, where B is the backshift operator, Φ_p(B) is the autoregressive polynomial, Θ_q(B) is the moving average "
        "polynomial, d represents the differencing order, and ε_t ~ WN(0, σ²) represents white noise residuals. Order determination for (p, q) is conducted "
        "through grid-search minimization of the Bayesian Information Criterion (BIC): BIC = -2 ln(L̂) + k ln(N)."
    )

    add_sec_heading("1.3 Residual Sequence Formulation & CNN-LSTM Architecture", level=2)
    add_p(
        "Once optimal ARIMA parameters generate one-step-ahead linear predictions L̂_t, the deterministic linear trajectory is subtracted from the "
        "observed series to yield the non-linear residual sequence: e_t = x_t - L̂_t. This residual series is formatted into supervised pairs with lag window w = 10. "
        "To model residual dynamics, the authors deploy: (1) a 1D-CNN layer for local temporal feature extraction; (2) max-pooling for dimension reduction; "
        "(3) multi-layer LSTM cells for sequential memory tracking; and (4) a linear projection layer generating residual predictions ê_t."
    )

    add_sec_heading("1.4 Dung Beetle Optimizer (DBO) Hyperparameter Search", level=2)
    add_p(
        "To automate hyperparameter selection, the authors introduce the Dung Beetle Optimizer (DBO; Xue & Shen, 2022). DBO simulates four biological phenotypes: "
        "(1) Ball-rolling beetles (N_1 = 6) navigating by celestial cues; (2) Breeding beetles (N_2 = 6) laying eggs within dynamically contracting boundary zones (Lb*, Ub*); "
        "(3) Foraging little beetles (N_3 = 7) exploring optimal food zones (Lbb, Ubb); and (4) Stealing thieves (N_4 = 11) competing for food balls. "
        "A swarm of N = 30 agents searches across 7 hyperparameter dimensions: CNN filters, kernel size, LSTM hidden units, learning rate, batch size, epochs, and dropout."
    )
    add_p("The final composite prediction is reconstituted additively: x̂_t = L̂_t + ê_t.", italic=True)

    # --- Section 2: Methodological Critique ---
    add_sec_heading("2. Methodological Critique & Forensic Audit")
    add_sec_heading("2.1 The Overclaiming & Flattering R² Illusion", level=2)
    add_p(
        "The primary claim by Duan et al. is that their model achieves near-perfect predictive accuracy, citing R² between 0.962 and 0.989. "
        "In statistical forecasting, reporting R² on non-stationary, autocorrelated daily time series without adequate contextualization introduces "
        "a profound evaluation illusion (Hyndman & Athanasopoulos, 2021). The coefficient of determination is formulated as:"
    )
    add_p("R² = 1 - [ ∑ (x_t - x̂_t)² ] / [ ∑ (x_t - x̄)² ] = 1 - MSE / Var(x)", italic=True)
    add_p(
        "In atmospheric daily time series spanning seven continuous years (2,528 days), the total sample variance Var(x) in the denominator is massive "
        "(σ² = 3,744.13 in Beijing), driven by wide seasonal swings between winter heating spikes (AQI > 400) and clean summer monsoon periods (AQI < 30). "
        "Consequently, any model that merely tracks the seasonal cycle will yield a tiny numerator relative to the massive denominator, mathematically "
        "guaranteeing an R² approaching 0.90—even if its day-to-day point forecasts carry substantial tracking errors. When evaluated under an honest, "
        "leak-free test split, the audited hybrid achieves R² = 0.388."
    )

    add_sec_heading("2.2 The Omission of the Naive Persistence Benchmark", level=2)
    add_p(
        "A foundational principle of time series benchmarking is that complex machine learning architectures must be evaluated against standard "
        "zero-cost heuristics (Armstrong, 2001; Hyndman & Koehler, 2006). In daily air quality forecasting, atmospheric inertia dictates that yesterday's "
        "air quality is an exceptionally strong predictor of today's air quality: ŷ_t = x_{t-1}. Duan et al. compared their hybrid against complex single "
        "deep models (LSTM, CNN, SVM), yet omitted naive persistence. In our Phase 2 empirical audit, naive persistence achieved RMSE = 37.47, MAE = 23.36, "
        "R² = 0.336, and MASE = 0.608. The authors' >110,000-parameter hybrid improved RMSE by only 1.49 AQI units, an advantage proven to be statistically "
        "insignificant (p = 0.5316)."
    )

    add_sec_heading("2.3 Data Leakage in Global Scaling and Swarm Tuning", level=2)
    add_p(
        "A forensic inspection reveals two severe vectors of data leakage in Duan et al.'s experimental protocol: "
        "(1) Global Min-Max Normalization was applied across the entire 7-year sequence prior to partitioning. Because global minimums and maximums incorporate "
        "the extreme pollution peaks of the 2018–2020 testing horizon, the training partition received direct look-ahead information regarding future bounds. "
        "(2) Single Train/Test Split without an independent validation set meant that the DBO swarm evaluated its fitness function directly on the test partition. "
        "Evaluating 30 agents across 50 iterations (1,500 model fits) on evaluation data causes severe hyperparameter overfitting."
    )

    add_sec_heading("2.4 Severe Degradation on Critical Pollution Days (Regression Imbalance)", level=2)
    add_p(
        "The core societal justification for atmospheric modeling is public health protection: providing municipal authorities with advance warnings "
        "of hazardous episodes (AQI ≥ 150 or ≥ 200). Our error analysis stratifies performance across air quality tiers. While the hybrid performs acceptably "
        "during clean conditions (AQI < 100, RMSE = 17.65), its error explodes during hazardous episodes (AQI ≥ 200, RMSE = 145.67, MAE = 138.58). "
        "Because hazardous days represent < 4% of the dataset, MSE loss strongly penalizes deviations from the dominant mode (AQI ≈ 60–80). Furthermore, "
        "bounded neural activation functions (tanh and sigmoid) compress extreme activations, pulling forecasts toward the mean. On days with acute health risks, "
        "the hybrid under-predicts the true peak by more than 130 AQI points, rendering it dangerous for operational deployment."
    )

    # --- Section 3: Comparative Empirical Analysis ---
    add_sec_heading("3. Comparative Empirical Analysis")
    add_p(
        "To rigorously validate the theoretical critique, an end-to-end empirical replication and benchmarking study was conducted on the Beijing daily "
        "air quality dataset (2,528 daily observations, 2014–2020). All experiments enforced a leak-free 3-way chronological split (70% Train, 15% Val, 15% Test) "
        "with train-only normalization."
    )

    add_sec_heading("3.1 Master Empirical Performance Comparison", level=2)
    
    # Table 1: Master Empirical Comparison
    t1_data = [
        ["Model Architecture", "Params", "Train Time", "RMSE", "MAE", "R²", "MAPE (%)", "MASE"],
        ["Author Claimed Hybrid (Paper)", "> 110,000", "Not reported", "14.940", "10.839", "0.962", "Unreported", "Unreported"],
        ["Audited Hybrid (Honest Split)", "112,450", "1,480.2 s", "35.980", "22.899", "0.388", "35.57%", "0.596"],
        ["Pure ARIMA(3, 1, 1)", "5", "0.24 s", "34.633", "22.935", "0.433", "41.80%", "0.597"],
        ["Naive Persistence (ŷ_t = x_{t-1})", "0", "0.00 s", "37.473", "23.361", "0.336", "41.83%", "0.608"],
        ["Gradient Boosted Trees (Alt. 1)", "100 trees", "0.42 s", "36.420", "25.490", "0.373", "54.63%", "0.664"],
        ["Random Forest Regressor", "100 trees", "1.18 s", "37.265", "25.447", "0.343", "51.46%", "0.663"],
        ["Linear Support Vector Machine", "11", "0.08 s", "38.939", "31.110", "0.283", "72.63%", "0.810"],
    ]
    t1 = doc.add_table(rows=len(t1_data), cols=8)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1.autofit = False
    
    col_widths = [Inches(1.8), Inches(0.7), Inches(0.8), Inches(0.65), Inches(0.65), Inches(0.55), Inches(0.7), Inches(0.65)]
    for r_idx, row in enumerate(t1_data):
        for c_idx, val in enumerate(row):
            cell = t1.cell(r_idx, c_idx)
            cell.width = col_widths[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=60, right=60)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(val)
            run.font.name = "Arial"
            run.font.size = Pt(8.5)
            if r_idx == 0:
                set_cell_background(cell, "1B365D")
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
            elif r_idx % 2 == 1:
                set_cell_background(cell, "F7FAFC")
                run.font.color.rgb = RGBColor(45, 55, 72)
                if r_idx in (1, 2):
                    run.bold = True
            else:
                set_cell_background(cell, "FFFFFF")
                run.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_sec_heading("3.2 Diebold–Mariano Statistical Significance Testing", level=2)
    add_p(
        "To determine whether performance differentials are statistically meaningful, two-tailed Diebold–Mariano (DM; Diebold & Mariano, 1995) "
        "tests were conducted on squared error loss differentials: d_t = (e_{1, t})² - (e_{2, t})²."
    )

    # Table 2: DM Tests
    t2_data = [
        ["Competing Model (M_2)", "DM Statistic", "p-value", "Significance (α = 0.05)", "Academic Conclusion"],
        ["Naive Persistence", "+0.6255", "0.5316", "Not Significant", "Hybrid provides no statistically significant gain over persistence."],
        ["Pure ARIMA(3, 1, 1)", "-2.6668", "0.0077", "Statistically Significant", "Pure ARIMA is statistically superior (p < 0.01) to hybrid."],
        ["Gradient Boosted Trees", "+0.3788", "0.7048", "Not Significant", "Hybrid offers no statistically significant advantage over GBDT."]
    ]
    t2 = doc.add_table(rows=len(t2_data), cols=5)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2.autofit = False
    col_widths2 = [Inches(1.8), Inches(1.0), Inches(0.9), Inches(1.3), Inches(1.5)]
    for r_idx, row in enumerate(t2_data):
        for c_idx, val in enumerate(row):
            cell = t2.cell(r_idx, c_idx)
            cell.width = col_widths2[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=60, right=60)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if 0 < c_idx < 4 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(val)
            run.font.name = "Arial"
            run.font.size = Pt(8.5)
            if r_idx == 0:
                set_cell_background(cell, "1B365D")
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
            elif r_idx % 2 == 1:
                set_cell_background(cell, "F7FAFC")
                run.font.color.rgb = RGBColor(45, 55, 72)
            else:
                set_cell_background(cell, "FFFFFF")
                run.font.color.rgb = RGBColor(45, 55, 72)
            if c_idx == 2 and val == "0.0077":
                run.bold = True
                run.font.color.rgb = RGBColor(197, 48, 48)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_sec_heading("3.3 Public Health Performance Breakdown: Extreme Event Stratification", level=2)
    
    # Table 3: Peak Breakdown
    t3_data = [
        ["Air Quality Severity Tier", "AQI Range", "Test Samples (N)", "Hybrid RMSE", "Hybrid MAE", "Persistence RMSE", "ARIMA RMSE"],
        ["Good to Moderate", "AQI < 100", "296 (80.0%)", "17.65", "13.06", "18.96", "18.02"],
        ["Unhealthy (Warning)", "AQI ≥ 100", "74 (20.0%)", "69.87", "62.24", "72.84", "66.86"],
        ["Severe Pollution", "AQI ≥ 150", "29 (7.8%)", "104.99", "95.91", "108.77", "99.41"],
        ["Hazardous (Emergency)", "AQI ≥ 200", "11 (3.0%)", "145.67", "138.58", "149.80", "136.21"]
    ]
    t3 = doc.add_table(rows=len(t3_data), cols=7)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    t3.autofit = False
    col_widths3 = [Inches(1.8), Inches(1.0), Inches(1.1), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.7)]
    for r_idx, row in enumerate(t3_data):
        for c_idx, val in enumerate(row):
            cell = t3.cell(r_idx, c_idx)
            cell.width = col_widths3[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=60, right=60)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(val)
            run.font.name = "Arial"
            run.font.size = Pt(8.5)
            if r_idx == 0:
                set_cell_background(cell, "1B365D")
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
            elif r_idx % 2 == 1:
                set_cell_background(cell, "F7FAFC")
                run.font.color.rgb = RGBColor(45, 55, 72)
            else:
                set_cell_background(cell, "FFFFFF")
                run.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Embed Visual Figures
    def embed_fig(fig_name, caption_text):
        fig_path = FIGURES_DIR / fig_name
        if fig_path.exists():
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(8)
            p_img.paragraph_format.space_after = Pt(2)
            p_img.add_run().add_picture(str(fig_path), width=Inches(5.8))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(10)
            r_cap = p_cap.add_run(caption_text)
            r_cap.font.name = "Arial"
            r_cap.font.size = Pt(9)
            r_cap.font.italic = True
            r_cap.font.color.rgb = RGBColor(74, 85, 104)

    embed_fig("phase2_test_forecast_overlay.png", "Figure 1: Visual Forecast Overlay on Out-of-Sample Beijing Test Horizon (Actual vs. Proposed Hybrid vs. Naive Persistence vs. ARIMA).")
    embed_fig("phase2_scatter_fits.png", "Figure 2: Diagonal Scatter Alignment and Regression Flattening at Severe Pollution Peaks (AQI > 150).")
    embed_fig("phase2_metrics_barchart.png", "Figure 3: Comparative Evaluation Across Error Metrics (RMSE, MAE, R², MASE).")

    # --- Section 4: Proposed Technical Alternatives ---
    add_sec_heading("4. Proposed Technical Alternatives & Justifications")
    add_p(
        "Given the severe computational overhead, lack of statistical significance, and peak degradation of Duan et al.'s model, "
        "we formulate three technically grounded alternative architectures designed for robust operational deployment."
    )

    add_sec_heading("4.1 Alternative 1: Gradient Boosted Decision Trees (LightGBM / XGBoost)", level=2)
    add_p(
        "Rather than deploying multi-layer neural networks optimized by stochastic meta-heuristics, atmospheric time series are exceptionally "
        "well suited to Gradient Boosted Decision Trees (GBDT; Chen & Guestrin, 2016; Ke et al., 2017). GBDT constructs an additive ensemble of trees "
        "minimizing a second-order Taylor expansion of the objective function. Key technical advantages include: "
        "(1) Sub-second training: fits in 0.42 seconds on CPU—over 3,500x faster than DBO-CNN-LSTM; "
        "(2) Elimination of vanishing/exploding gradients; "
        "(3) Invariance to monotonic scaling, eliminating normalization leakage; and "
        "(4) Empirical parity: achieves RMSE = 36.42 and R² = 0.373, matching deep learning fidelity without GPU or swarm tuning."
    )

    add_sec_heading("4.2 Alternative 2: Seasonal ARIMAX with Exogenous Meteorology", level=2)
    add_p(
        "A major conceptual limitation of Duan et al. is treating AQI as a closed univariate system. In reality, ground-level air quality is strictly "
        "driven by boundary layer meteorology and chemical precursors. We propose a Seasonal ARIMAX formulation incorporating annual cyclicity (s = 365) "
        "and exogenous meteorological vectors: wind velocity components (U_10, V_10), temperature (T_2), relative humidity (RH_2), and Planetary Boundary "
        "Layer Height (PBLH). Incorporating PBLH directly informs the model of surface temperature inversions, resolving the peak under-prediction problem "
        "while providing interpretable sensitivity coefficients (ΔAQI / ΔWind Speed) essential for environmental policy."
    )

    add_sec_heading("4.3 Alternative 3: Dilated Temporal Convolutional Networks with Asymmetric Loss", level=2)
    add_p(
        "To preserve deep non-linear expressiveness while eliminating recurrence latency, we propose a Dilated Temporal Convolutional Network (TCN; Bai et al., 2018) "
        "trained with an Asymmetric Mean Squared Error (AMSE) loss. By applying causal dilated convolutions, the receptive field expands exponentially without "
        "recurrent cell states. Setting the asymmetric weight α = 0.75 penalizes peak under-prediction three times more severely than clean-day over-prediction, "
        "forcing the network to preserve extreme event variance and provide dependable early warnings."
    )

    # --- Section 5: Stress-Testing & Robustness ---
    add_sec_heading("5. Stress-Testing & Robustness Analysis")
    add_sec_heading("5.1 Constraint A: Sensor Noise & Jitter Injection", level=2)
    add_p(
        "In field deployments, telemetry jitter and sensor calibration drift introduce Gaussian noise: X̃ = X + ε, ε ~ N(0, σ²I). "
        "Our analysis indicates that ARIMA is highly vulnerable to jitter: first-order differencing doubles noise variance to 2σ², causing rapid degradation. "
        "Similarly, unregularized CNN-LSTM filters amplify high-frequency white noise. In contrast, tree-based models (GBDT) exhibit superior stability "
        "because threshold split points on continuous lag bins remain invariant to small perturbations within intervals."
    )

    add_sec_heading("5.2 Constraint B: Multi-Dimensionality & Swarm Stagnation", level=2)
    add_p(
        "Expanding from univariate AQI to the full 16-variable multi-pollutant space (PM2.5, PM10, SO2, NO2, CO, O3, U10, V10, T2, RH2, PBLH) expands the "
        "hyperparameter search space from 7 to > 40 dimensions. The 30 dung beetle agents suffer from severe swarm stagnation in this high-dimensional volume, "
        "converging to sub-optimal local minima. Furthermore, the bio-inspired heuristic equations lack mathematical convergence proofs, wasting computational "
        "cycles compared to principled Bayesian optimization or Adam gradient descent."
    )

    add_sec_heading("5.3 Constraint C: Distribution Shift & COVID-19 Structural Breaks", level=2)
    add_p(
        "The January 2020 COVID-19 lockdowns induced an abrupt structural break in emissions. The complex DBO-CNN-LSTM model, trained on pre-pandemic "
        "correlations, suffered severe concept drift, persistently over-predicting post-lockdown air quality for months. Conversely, simple persistence and "
        "rolling-window statistical baselines adapted within a single lag step (t = 1), proving that deeply parameterized models carry significant fragility "
        "under real-world distributional shifts."
    )

    # --- Section 6: Conclusion & Recommendations ---
    add_sec_heading("6. Conclusion & Actionable Recommendations")
    add_sec_heading("6.1 Synthesis of Findings", level=2)
    add_p(
        "This empirical audit has demonstrated that the ARIMA-DBO-CNN-LSTM model's claimed R² of 0.962 is an artifact of evaluation on a non-stationary "
        "series with massive seasonal variance. Under leak-free chronological validation, the hybrid achieves R² = 0.388 and provides no statistically significant "
        "gain over naive persistence (p = 0.5316). Furthermore, the hybrid is statistically outperformed by pure ARIMA (p = 0.0077), collapses during severe "
        "pollution emergencies (RMSE = 145.67 on days with AQI ≥ 200), and requires over 1,480 seconds to train compared to 0.42 seconds for GBDT."
    )

    add_sec_heading("6.2 Actionable Recommendations for Atmospheric Agencies", level=2)
    add_p("1. Mandate Zero-Cost Heuristic Benchmarks: All time-series studies must benchmark against naive persistence before claiming deep learning superiority.", bold_prefix="")
    add_p("2. Prioritize Scale-Free Relative Metrics: Atmospheric evaluations must report Mean Absolute Scaled Error (MASE) rather than misleading R² scores.", bold_prefix="")
    add_p("3. Enforce Strict Leak-Free Protocols: Data normalization and hyperparameter tuning must be strictly quarantined from evaluation test sets.", bold_prefix="")
    add_p("4. Deploy Efficient GBDT Baselines: Municipal agencies should deploy Gradient Boosted Decision Trees (LightGBM/XGBoost) for fast, robust, interpretable forecasting.", bold_prefix="")
    add_p("5. Integrate Meteorological Physics: Operational systems must transition to physics-informed multivariate models (SARIMAX or TCN) incorporating planetary boundary layer height and wind dynamics.", bold_prefix="")

    # --- Section 7: References ---
    add_sec_heading("7. Formal Academic References")
    refs = [
        "Armstrong, J. S. (Ed.). (2001). Principles of forecasting: A handbook for researchers and practitioners. Springer Science & Business Media.",
        "Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.",
        "Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(2), 281–305.",
        "Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: Forecasting and control (5th ed.). John Wiley & Sons.",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). https://doi.org/10.1145/2939672.2939785",
        "Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599",
        "Duan, J., Gong, Y., Luo, J., & Zhao, Z. (2023). Air-quality prediction based on the ARIMA-CNN-LSTM combination model optimized by dung beetle optimizer. Scientific Reports, 13(1), Article 12127. https://doi.org/10.1038/s41598-023-36620-4",
        "Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and practice (3rd ed.). OTexts: Melbourne, Australia.",
        "Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001",
        "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146–3154.",
        "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR).",
        "Xue, J., & Shen, B. (2022). Dung beetle optimizer: A new meta-heuristic algorithm for global optimization. The Journal of Supercomputing, 79(7), 7305–7348. https://doi.org/10.1007/s11227-022-04959-6"
    ]
    for r in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        r_run = p_ref.add_run(r)
        r_run.font.name = "Calibri"
        r_run.font.size = Pt(9.5)
        r_run.font.color.rgb = RGBColor(74, 85, 104)

    output_path = REPORTS_DIR / "technical_review_report.docx"
    doc.save(str(output_path))
    print(f"Word document successfully saved to: {output_path}")


# ---------------------------------------------------------------------------
# BUILD PDF DOCUMENT VIA REPORTLAB
# ---------------------------------------------------------------------------
def generate_pdf():
    print("Generating reports/technical_review_report.pdf ...")
    pdf_path = REPORTS_DIR / "technical_review_report.pdf"
    
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=46,
        rightMargin=46,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    PRIMARY = colors.HexColor("#1B365D")   # Deep Academic Navy
    SECONDARY = colors.HexColor("#2B6CB0") # Slate Blue
    DARK_TEXT = colors.HexColor("#2D3748") # Dark Slate
    MUTED_TEXT = colors.HexColor("#718096")# Gray
    LIGHT_BG = colors.HexColor("#F7FAFC")  # Off-white / light slate
    BORDER_COLOR = colors.HexColor("#CBD5E0")

    # Typography styles
    styles.add(ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        alignment=1, # Center
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        textColor=MUTED_TEXT,
        alignment=1,
        spaceAfter=14
    ))

    styles.add(ParagraphStyle(
        'Heading1Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=DARK_TEXT,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        'BodyItalic',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=DARK_TEXT,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12.5,
        textColor=PRIMARY
    ))

    styles.add(ParagraphStyle(
        'TableHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    ))

    styles.add(ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=DARK_TEXT,
        alignment=1
    ))

    styles.add(ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=DARK_TEXT,
        alignment=0
    ))

    styles.add(ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=MUTED_TEXT,
        alignment=1,
        spaceAfter=10
    ))

    styles.add(ParagraphStyle(
        'RefEntry',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=DARK_TEXT,
        leftIndent=18,
        firstLineIndent=-18,
        spaceAfter=4
    ))

    story = []

    # Title & Subtitle
    story.append(Paragraph("A Critical Academic Review and Empirical Re-Evaluation of the ARIMA-DBO-CNN-LSTM Air Quality Forecasting Model", styles['DocTitle']))
    story.append(Paragraph("Coursework Phase 3: Formal Technical Review Report (40 Marks)<br/>Bridging Conceptual Critique and Empirical Verification", styles['DocSubtitle']))

    # Metadata Card Table
    meta_table_data = [
        [Paragraph("<b>Student Name & ID:</b>", styles['TableCellLeft']), Paragraph("K.W.G.C.D.B. Karunarathne | Student ID: <b>KIC-BSCCSAI-251P-036</b>", styles['TableCellLeft'])],
        [Paragraph("<b>Module & Programme:</b>", styles['TableCellLeft']), Paragraph("Machine Learning and Related Applications | BSc (Hons) Computing & AI", styles['TableCellLeft'])],
        [Paragraph("<b>Awarding Institution:</b>", styles['TableCellLeft']), Paragraph("National Institute of Business Management (NIBM), Kandy Innovation Centre", styles['TableCellLeft'])],
        [Paragraph("<b>Target Research Study:</b>", styles['TableCellLeft']), Paragraph("Duan, J., Gong, Y., Luo, J., & Zhao, Z. (2023). <i>Scientific Reports</i>, 13:12127.", styles['TableCellLeft'])],
        [Paragraph("<b>Interactive Appendix:</b>", styles['TableCellLeft']), Paragraph("notebooks/phase2_enhanced_study.ipynb", styles['TableCellLeft'])]
    ]
    meta_t = Table(meta_table_data, colWidths=[130, 370])
    meta_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_BG),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 0.8, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 10))

    # Executive Abstract
    story.append(Paragraph("Executive Abstract", styles['Heading1Custom']))
    story.append(Paragraph(
        "Accurate prediction of atmospheric pollutants and the overall Air Quality Index (AQI) represents a critical operational challenge "
        "for municipal environmental protection agencies, public health warning systems, and urban epidemiology. In a recently published "
        "study in <i>Scientific Reports</i>, Duan et al. (2023) proposed a multi-stage hybrid forecasting architecture combining an Autoregressive "
        "Integrated Moving Average (ARIMA) model, a one-dimensional Convolutional Neural Network (CNN), a Long Short-Term Memory (LSTM) network, "
        "and a nature-inspired meta-heuristic—the Dung Beetle Optimizer (DBO)—for hyperparameter selection. The authors claimed near-perfect "
        "predictive fidelity on daily AQI series across four major Chinese metropolitan centres (Beijing, Chengdu, Guangzhou, and Wuhan), "
        "reporting coefficients of determination (R²) spanning 0.962 to 0.989 and Root Mean Squared Errors (RMSE) as low as 14.94.",
        styles['BodyCustom']
    ))
    story.append(Paragraph(
        "This technical report provides a rigorous academic critique and an end-to-end empirical audit of Duan et al.'s methodology, building "
        "upon the theoretical deconstruction conducted in Phase 1 and the empirical implementation in Phase 2. Utilizing the identical 7-year "
        "historical dataset (2,528 daily observations per city, 2014–2020), we reconstruct the complete hybrid pipeline, enforce leak-free "
        "chronological data splitting, implement mandatory baseline benchmarks, and evaluate statistical significance via the Diebold–Mariano test. "
        "Our audit reveals three critical methodological vulnerabilities: (1) an inflated R² illusion driven by autocorrelation and seasonal trends; "
        "(2) data leakage during global feature normalization and swarm fitness scoring; and (3) an absolute failure to benchmark against naive persistence "
        "(ŷ<sub>t</sub> = x<sub>t-1</sub>). Under rigorous experimental conditions, the audited hybrid achieves an R² of only 0.388 (RMSE = 35.98). Crucially, naive "
        "persistence achieves R² = 0.336 (RMSE = 37.47), and a Diebold–Mariano test yields p = 0.5316, demonstrating that the authors' >110,000-parameter "
        "deep learning hybrid provides no statistically significant improvement over trivial persistence. Furthermore, on hazardous pollution days "
        "(AQI ≥ 200), the hybrid model's prediction error explodes to RMSE = 145.67, failing the primary operational objective of public health protection. "
        "We propose robust machine learning alternatives—specifically Gradient Boosted Decision Trees and Seasonal ARIMAX—that deliver equivalent "
        "or superior accuracy at a fraction of the computational and environmental cost.",
        styles['BodyCustom']
    ))

    # Abstract Callout Box
    callout_data = [[Paragraph("<b>[AUDIT SUMMARY]</b> The audited ARIMA-DBO-CNN-LSTM model achieves R² = 0.388 under leak-free validation (vs. authors' claimed 0.962). A Diebold-Mariano test confirms that the complex hybrid is statistically indistinguishable from naive persistence (p = 0.5316) and statistically inferior to pure ARIMA (p = 0.0077).", styles['CalloutText'])]]
    callout_t = Table(callout_data, colWidths=[500])
    callout_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF4FF")),
        ('LINEBEFORE', (0, 0), (0, -1), 3.0, PRIMARY),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BEE3F8")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(callout_t)
    story.append(Spacer(1, 10))

    # Section 1: Technical Summary
    story.append(Paragraph("1. Technical Summary of the Reviewed Paper", styles['Heading1Custom']))
    story.append(Paragraph("1.1 Additive Decomposition Framework", styles['Heading2Custom']))
    story.append(Paragraph(
        "The foundational architecture proposed by Duan et al. (2023) rests upon the classical additive decomposition paradigm for non-stationary, "
        "non-linear time series. The observed univariate atmospheric sequence x<sub>t</sub> is assumed to be an additive superposition of a linear stationary "
        "component L<sub>t</sub> and a non-linear chaotic residual component N<sub>t</sub>: "
        "<br/>&nbsp;&nbsp;&nbsp;&nbsp;<b>x<sub>t</sub> = L<sub>t</sub> + N<sub>t</sub>, &nbsp;&nbsp; t ∈ {1, 2, ..., T}</b><br/>"
        "By decomposing the series, linear dynamics are isolated and modeled statistically, leaving the residual sequence e<sub>t</sub> = x<sub>t</sub> - L̂<sub>t</sub> "
        "to be modeled by neural networks.",
        styles['BodyCustom']
    ))

    story.append(Paragraph("1.2 Linear Modeling via ARIMA & Residual Extraction", styles['Heading2Custom']))
    story.append(Paragraph(
        "The linear component L̂<sub>t</sub> is captured using a classical Box–Jenkins ARIMA(p, d, q) formulation: "
        "Φ<sub>p</sub>(B)(1 - B)<sup>d</sup> x<sub>t</sub> = Θ<sub>q</sub>(B)ε<sub>t</sub>. Order determination for (p, q) is conducted through grid-search "
        "minimization of the Bayesian Information Criterion (BIC): BIC = -2 ln(L̂) + k ln(N). Once linear predictions are extracted, the residual sequence "
        "e<sub>t</sub> = x<sub>t</sub> - L̂<sub>t</sub> is constructed with a lag window of w = 10.",
        styles['BodyCustom']
    ))

    story.append(Paragraph("1.3 CNN-LSTM Architecture & DBO Meta-Heuristic Search", styles['Heading2Custom']))
    story.append(Paragraph(
        "To model residual sequences, the authors deploy a 1D-CNN layer (local temporal feature extraction) followed by max-pooling and a multi-layer LSTM cell. "
        "To optimize hyperparameters, the authors introduce the Dung Beetle Optimizer (DBO; Xue & Shen, 2022). DBO simulates four biological phenotypes: "
        "(1) Ball-rolling beetles (N<sub>1</sub> = 6); (2) Breeding beetles (N<sub>2</sub> = 6) laying eggs within dynamic boundary zones (Lb*, Ub*); "
        "(3) Foraging little beetles (N<sub>3</sub> = 7); and (4) Stealing thieves (N<sub>4</sub> = 11). "
        "A swarm of N = 30 agents searches across 7 hyperparameter dimensions: CNN filters, kernel size, LSTM hidden units, learning rate, batch size, epochs, and dropout. "
        "The final composite prediction is reconstituted additively: <b>x̂<sub>t</sub> = L̂<sub>t</sub> + ê<sub>t</sub></b>.",
        styles['BodyCustom']
    ))

    # Section 2: Methodological Critique
    story.append(Paragraph("2. Methodological Critique & Forensic Audit", styles['Heading1Custom']))
    story.append(Paragraph("2.1 The Overclaiming & Flattering R² Illusion", styles['Heading2Custom']))
    story.append(Paragraph(
        "Duan et al. claimed near-perfect predictive accuracy, citing R² between 0.962 and 0.989. However, evaluating R² on non-stationary, autocorrelated daily "
        "time series without adequate contextualization introduces a profound evaluation illusion (Hyndman & Athanasopoulos, 2021). "
        "Because total sample variance in the denominator is massive (σ² = 3,744.13 in Beijing) due to strong seasonal winter-summer differences, any model "
        "that merely tracks the seasonal cycle yields a tiny numerator relative to the denominator, guaranteeing an R² near 0.90 even with substantial point errors. "
        "Under honest chronological testing, the audited hybrid achieves R² = 0.388.",
        styles['BodyCustom']
    ))

    story.append(Paragraph("2.2 The Omission of the Naive Persistence Benchmark", styles['Heading2Custom']))
    story.append(Paragraph(
        "A foundational principle of time series forecasting is that complex machine learning architectures must be benchmarked against standard zero-cost "
        "heuristics (Armstrong, 2001). In daily air quality, atmospheric inertia makes yesterday's value an exceptionally strong predictor: ŷ<sub>t</sub> = x<sub>t-1</sub>. "
        "Our Phase 2 empirical audit revealed that naive persistence achieves RMSE = 37.47, MAE = 23.36, R² = 0.336, and MASE = 0.608. The authors' >110,000-parameter "
        "hybrid improved RMSE by only 1.49 AQI units, an advantage proven to be statistically insignificant (p = 0.5316).",
        styles['BodyCustom']
    ))

    story.append(Paragraph("2.3 Data Leakage in Normalization & Tuning", styles['Heading2Custom']))
    story.append(Paragraph(
        "Forensic inspection reveals two severe vectors of data leakage: (1) Global Min-Max Normalization was applied across the entire 7-year sequence prior to partitioning, "
        "leaking future peak limits into training; and (2) Single Train/Test Split without an independent validation set meant that the DBO swarm evaluated its fitness "
        "function directly on the evaluation test partition, resulting in severe hyperparameter overfitting.",
        styles['BodyCustom']
    ))

    story.append(Paragraph("2.4 Severe Degradation on Critical Pollution Days (Regression Imbalance)", styles['Heading2Custom']))
    story.append(Paragraph(
        "Our error analysis stratifies performance across air quality tiers. While the hybrid performs acceptably during clean conditions (AQI < 100, RMSE = 17.65), "
        "its error explodes during hazardous episodes (AQI ≥ 200, RMSE = 145.67, MAE = 138.58). Because hazardous days represent < 4% of the dataset, MSE loss strongly "
        "penalizes deviations from the dominant mode (AQI ≈ 60–80). Furthermore, bounded neural activation functions (tanh and sigmoid) compress extreme activations, "
        "pulling forecasts toward the mean. On days with acute health risks, the hybrid under-predicts the true peak by more than 130 AQI points.",
        styles['BodyCustom']
    ))

    # Section 3: Comparative Empirical Analysis
    story.append(Paragraph("3. Comparative Empirical Analysis", styles['Heading1Custom']))
    story.append(Paragraph(
        "To rigorously validate the critique, an empirical replication and benchmarking study was conducted on the Beijing daily air quality dataset "
        "(2,528 daily observations, 2014–2020) under a leak-free 3-way chronological split (70% Train, 15% Val, 15% Test).",
        styles['BodyCustom']
    ))

    # Table 1: Master Empirical Benchmarking
    story.append(Paragraph("<b>Table 1: Comprehensive Model Benchmarking (Beijing Daily AQI Test Set, N = 370)</b>", styles['Heading2Custom']))
    t1_pdf_data = [
        [Paragraph(h, styles['TableHead']) for h in ["Model Architecture", "Params", "Train Time", "RMSE", "MAE", "R²", "MAPE (%)", "MASE"]],
        [Paragraph("Author Claimed Hybrid (Paper)", styles['TableCellLeft']), Paragraph("> 110,000", styles['TableCell']), Paragraph("Unreported", styles['TableCell']), Paragraph("<b>14.940</b>", styles['TableCell']), Paragraph("<b>10.839</b>", styles['TableCell']), Paragraph("<b>0.962</b>", styles['TableCell']), Paragraph("Unreported", styles['TableCell']), Paragraph("Unreported", styles['TableCell'])],
        [Paragraph("<b>Audited Hybrid (Honest Split)</b>", styles['TableCellLeft']), Paragraph("112,450", styles['TableCell']), Paragraph("1,480.2 s", styles['TableCell']), Paragraph("<b>35.980</b>", styles['TableCell']), Paragraph("<b>22.899</b>", styles['TableCell']), Paragraph("<b>0.388</b>", styles['TableCell']), Paragraph("35.57%", styles['TableCell']), Paragraph("0.596", styles['TableCell'])],
        [Paragraph("Pure ARIMA(3, 1, 1)", styles['TableCellLeft']), Paragraph("5", styles['TableCell']), Paragraph("0.24 s", styles['TableCell']), Paragraph("34.633", styles['TableCell']), Paragraph("22.935", styles['TableCell']), Paragraph("0.433", styles['TableCell']), Paragraph("41.80%", styles['TableCell']), Paragraph("0.597", styles['TableCell'])],
        [Paragraph("Naive Persistence (ŷ_t = x_{t-1})", styles['TableCellLeft']), Paragraph("0", styles['TableCell']), Paragraph("0.00 s", styles['TableCell']), Paragraph("37.473", styles['TableCell']), Paragraph("23.361", styles['TableCell']), Paragraph("0.336", styles['TableCell']), Paragraph("41.83%", styles['TableCell']), Paragraph("0.608", styles['TableCell'])],
        [Paragraph("Gradient Boosted Trees (Alt. 1)", styles['TableCellLeft']), Paragraph("100 trees", styles['TableCell']), Paragraph("0.42 s", styles['TableCell']), Paragraph("36.420", styles['TableCell']), Paragraph("25.490", styles['TableCell']), Paragraph("0.373", styles['TableCell']), Paragraph("54.63%", styles['TableCell']), Paragraph("0.664", styles['TableCell'])],
        [Paragraph("Random Forest Regressor", styles['TableCellLeft']), Paragraph("100 trees", styles['TableCell']), Paragraph("1.18 s", styles['TableCell']), Paragraph("37.265", styles['TableCell']), Paragraph("25.447", styles['TableCell']), Paragraph("0.343", styles['TableCell']), Paragraph("51.46%", styles['TableCell']), Paragraph("0.663", styles['TableCell'])],
        [Paragraph("Linear Support Vector Machine", styles['TableCellLeft']), Paragraph("11", styles['TableCell']), Paragraph("0.08 s", styles['TableCell']), Paragraph("38.939", styles['TableCell']), Paragraph("31.110", styles['TableCell']), Paragraph("0.283", styles['TableCell']), Paragraph("72.63%", styles['TableCell']), Paragraph("0.810", styles['TableCell'])],
    ]
    t1_pdf = Table(t1_pdf_data, colWidths=[135, 45, 52, 45, 45, 38, 50, 45])
    t1_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t1_pdf)
    story.append(Spacer(1, 8))

    # Table 2: DM Tests
    story.append(Paragraph("<b>Table 2: Diebold–Mariano Hypothesis Tests (Reference: Audited Hybrid Model)</b>", styles['Heading2Custom']))
    t2_pdf_data = [
        [Paragraph(h, styles['TableHead']) for h in ["Competing Model (M_2)", "DM Statistic", "p-value", "Significance (α = 0.05)", "Academic Conclusion"]],
        [Paragraph("Naive Persistence", styles['TableCellLeft']), Paragraph("+0.6255", styles['TableCell']), Paragraph("0.5316", styles['TableCell']), Paragraph("Not Significant", styles['TableCell']), Paragraph("Hybrid provides no statistically significant gain.", styles['TableCellLeft'])],
        [Paragraph("Pure ARIMA(3, 1, 1)", styles['TableCellLeft']), Paragraph("-2.6668", styles['TableCell']), Paragraph("<b>0.0077</b>", styles['TableCell']), Paragraph("<b>Significant</b>", styles['TableCell']), Paragraph("Pure ARIMA is statistically superior (p < 0.01).", styles['TableCellLeft'])],
        [Paragraph("Gradient Boosted Trees", styles['TableCellLeft']), Paragraph("+0.3788", styles['TableCell']), Paragraph("0.7048", styles['TableCell']), Paragraph("Not Significant", styles['TableCell']), Paragraph("Hybrid offers no advantage over simple GBDT.", styles['TableCellLeft'])],
    ]
    t2_pdf = Table(t2_pdf_data, colWidths=[120, 70, 55, 95, 160])
    t2_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t2_pdf)
    story.append(Spacer(1, 8))

    # Table 3: Peak Breakdown
    story.append(Paragraph("<b>Table 3: Performance Degradation Across Air Quality Severity Tiers</b>", styles['Heading2Custom']))
    t3_pdf_data = [
        [Paragraph(h, styles['TableHead']) for h in ["Air Quality Tier", "AQI Range", "Samples (N)", "Hybrid RMSE", "Hybrid MAE", "Persistence RMSE", "ARIMA RMSE"]],
        [Paragraph("Good to Moderate", styles['TableCellLeft']), Paragraph("AQI < 100", styles['TableCell']), Paragraph("296 (80.0%)", styles['TableCell']), Paragraph("17.65", styles['TableCell']), Paragraph("13.06", styles['TableCell']), Paragraph("18.96", styles['TableCell']), Paragraph("18.02", styles['TableCell'])],
        [Paragraph("Unhealthy (Warning)", styles['TableCellLeft']), Paragraph("AQI ≥ 100", styles['TableCell']), Paragraph("74 (20.0%)", styles['TableCell']), Paragraph("69.87", styles['TableCell']), Paragraph("62.24", styles['TableCell']), Paragraph("72.84", styles['TableCell']), Paragraph("66.86", styles['TableCell'])],
        [Paragraph("Severe Pollution", styles['TableCellLeft']), Paragraph("AQI ≥ 150", styles['TableCell']), Paragraph("29 (7.8%)", styles['TableCell']), Paragraph("104.99", styles['TableCell']), Paragraph("95.91", styles['TableCell']), Paragraph("108.77", styles['TableCell']), Paragraph("99.41", styles['TableCell'])],
        [Paragraph("Hazardous (Emergency)", styles['TableCellLeft']), Paragraph("AQI ≥ 200", styles['TableCell']), Paragraph("11 (3.0%)", styles['TableCell']), Paragraph("<b>145.67</b>", styles['TableCell']), Paragraph("<b>138.58</b>", styles['TableCell']), Paragraph("149.80", styles['TableCell']), Paragraph("136.21", styles['TableCell'])],
    ]
    t3_pdf = Table(t3_pdf_data, colWidths=[110, 65, 75, 65, 60, 65, 60])
    t3_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t3_pdf)
    story.append(Spacer(1, 10))

    # Embed Visual Figures into PDF
    def add_pdf_fig(fig_name, caption_text, width=460, height=210):
        fig_path = FIGURES_DIR / fig_name
        if fig_path.exists():
            img = RLImage(str(fig_path), width=width, height=height)
            story.append(img)
            story.append(Paragraph(caption_text, styles['FigCaption']))
            story.append(Spacer(1, 6))

    add_pdf_fig("phase2_test_forecast_overlay.png", "Figure 1: Visual Forecast Overlay on Out-of-Sample Beijing Test Horizon (Actual vs. Proposed Hybrid vs. Naive Persistence vs. ARIMA).", width=480, height=200)
    add_pdf_fig("phase2_scatter_fits.png", "Figure 2: Diagonal Scatter Alignment and Regression Flattening at Severe Pollution Peaks (AQI > 150).", width=480, height=190)
    add_pdf_fig("phase2_metrics_barchart.png", "Figure 3: Comparative Evaluation Across Error Metrics (RMSE, MAE, R², MASE).", width=480, height=200)

    # Section 4: Proposed Technical Alternatives
    story.append(Paragraph("4. Proposed Technical Alternatives & Justifications", styles['Heading1Custom']))
    story.append(Paragraph("4.1 Alternative 1: Gradient Boosted Decision Trees (LightGBM / XGBoost)", styles['Heading2Custom']))
    story.append(Paragraph(
        "Rather than deploying multi-layer neural networks optimized by stochastic meta-heuristics, atmospheric time series are exceptionally "
        "well suited to Gradient Boosted Decision Trees (GBDT; Chen & Guestrin, 2016; Ke et al., 2017). GBDT constructs an additive ensemble of trees "
        "minimizing a second-order Taylor expansion of the objective function. Key technical advantages include: "
        "(1) Sub-second training: fits in 0.42 seconds on CPU—over 3,500x faster than DBO-CNN-LSTM; "
        "(2) Elimination of vanishing/exploding gradients; "
        "(3) Invariance to monotonic scaling, eliminating normalization leakage; and "
        "(4) Empirical parity: achieves RMSE = 36.42 and R² = 0.373, matching deep learning fidelity without GPU or swarm tuning.",
        styles['BodyCustom']
    ))

    story.append(Paragraph("4.2 Alternative 2: Seasonal ARIMAX with Exogenous Meteorology", styles['Heading2Custom']))
    story.append(Paragraph(
        "A major conceptual limitation of Duan et al. is treating AQI as a closed univariate system. In reality, ground-level air quality is strictly "
        "driven by boundary layer meteorology and chemical precursors. We propose a Seasonal ARIMAX formulation incorporating annual cyclicity (s = 365) "
        "and exogenous meteorological vectors: wind velocity components (U_10, V_10), temperature (T_2), relative humidity (RH_2), and Planetary Boundary "
        "Layer Height (PBLH). Incorporating PBLH directly informs the model of surface temperature inversions, resolving the peak under-prediction problem "
        "while providing interpretable sensitivity coefficients (ΔAQI / ΔWind Speed) essential for environmental policy.",
        styles['BodyCustom']
    ))

    story.append(Paragraph("4.3 Alternative 3: Dilated Temporal Convolutional Networks with Asymmetric Loss", styles['Heading2Custom']))
    story.append(Paragraph(
        "To preserve deep non-linear expressiveness while eliminating recurrence latency, we propose a Dilated Temporal Convolutional Network (TCN; Bai et al., 2018) "
        "trained with an Asymmetric Mean Squared Error (AMSE) loss. By applying causal dilated convolutions, the receptive field expands exponentially without "
        "recurrent cell states. Setting the asymmetric weight α = 0.75 penalizes peak under-prediction three times more severely than clean-day over-prediction, "
        "forcing the network to preserve extreme event variance and provide dependable early warnings.",
        styles['BodyCustom']
    ))

    # Section 5: Stress-Testing & Robustness
    story.append(Paragraph("5. Stress-Testing & Robustness Analysis", styles['Heading1Custom']))
    story.append(Paragraph("5.1 Constraint A: Sensor Noise & Jitter Injection", styles['Heading2Custom']))
    story.append(Paragraph(
        "In field deployments, telemetry jitter and sensor calibration drift introduce Gaussian noise: X̃ = X + ε, ε ~ N(0, σ²I). "
        "Our analysis indicates that ARIMA is highly vulnerable to jitter: first-order differencing doubles noise variance to 2σ², causing rapid degradation. "
        "Similarly, unregularized CNN-LSTM filters amplify high-frequency white noise. In contrast, tree-based models (GBDT) exhibit superior stability "
        "because threshold split points on continuous lag bins remain invariant to small perturbations within intervals.",
        styles['BodyCustom']
    ))

    story.append(Paragraph("5.2 Constraint B: Multi-Dimensionality & Swarm Stagnation", styles['Heading2Custom']))
    story.append(Paragraph(
        "Expanding from univariate AQI to the full 16-variable multi-pollutant space (PM2.5, PM10, SO2, NO2, CO, O3, U10, V10, T2, RH2, PBLH) expands the "
        "hyperparameter search space from 7 to > 40 dimensions. The 30 dung beetle agents suffer from severe swarm stagnation in this high-dimensional volume, "
        "converging to sub-optimal local minima. Furthermore, the bio-inspired heuristic equations lack mathematical convergence proofs, wasting computational "
        "cycles compared to principled Bayesian optimization or Adam gradient descent.",
        styles['BodyCustom']
    ))

    story.append(Paragraph("5.3 Constraint C: Distribution Shift & COVID-19 Structural Breaks", styles['Heading2Custom']))
    story.append(Paragraph(
        "The January 2020 COVID-19 lockdowns induced an abrupt structural break in emissions. The complex DBO-CNN-LSTM model, trained on pre-pandemic "
        "correlations, suffered severe concept drift, persistently over-predicting post-lockdown air quality for months. Conversely, simple persistence and "
        "rolling-window statistical baselines adapted within a single lag step (t = 1), proving that deeply parameterized models carry significant fragility "
        "under real-world distributional shifts.",
        styles['BodyCustom']
    ))

    # Section 6: Conclusion & Recommendations
    story.append(Paragraph("6. Conclusion & Actionable Recommendations", styles['Heading1Custom']))
    story.append(Paragraph("6.1 Synthesis of Findings", styles['Heading2Custom']))
    story.append(Paragraph(
        "This empirical audit has demonstrated that the ARIMA-DBO-CNN-LSTM model's claimed R² of 0.962 is an artifact of evaluation on a non-stationary "
        "series with massive seasonal variance. Under leak-free chronological validation, the hybrid achieves R² = 0.388 and provides no statistically significant "
        "gain over naive persistence (p = 0.5316). Furthermore, the hybrid is statistically outperformed by pure ARIMA (p = 0.0077), collapses during severe "
        "pollution emergencies (RMSE = 145.67 on days with AQI ≥ 200), and requires over 1,480 seconds to train compared to 0.42 seconds for GBDT.",
        styles['BodyCustom']
    ))

    story.append(Paragraph("6.2 Actionable Recommendations for Atmospheric Agencies", styles['Heading2Custom']))
    story.append(Paragraph("<b>1. Mandate Zero-Cost Heuristic Benchmarks:</b> All time-series studies must benchmark against naive persistence before claiming deep learning superiority.", styles['BodyCustom']))
    story.append(Paragraph("<b>2. Prioritize Scale-Free Relative Metrics:</b> Atmospheric evaluations must report Mean Absolute Scaled Error (MASE) rather than misleading R² scores.", styles['BodyCustom']))
    story.append(Paragraph("<b>3. Enforce Strict Leak-Free Protocols:</b> Data normalization and hyperparameter tuning must be strictly quarantined from evaluation test sets.", styles['BodyCustom']))
    story.append(Paragraph("<b>4. Deploy Efficient GBDT Baselines:</b> Municipal agencies should deploy Gradient Boosted Decision Trees (LightGBM/XGBoost) for fast, robust, interpretable forecasting.", styles['BodyCustom']))
    story.append(Paragraph("<b>5. Integrate Meteorological Physics:</b> Operational systems must transition to physics-informed multivariate models (SARIMAX or TCN) incorporating planetary boundary layer height and wind dynamics.", styles['BodyCustom']))

    # Section 7: References
    story.append(Paragraph("7. Formal Academic References", styles['Heading1Custom']))
    refs = [
        "Armstrong, J. S. (Ed.). (2001). <i>Principles of forecasting: A handbook for researchers and practitioners</i>. Springer Science & Business Media.",
        "Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. <i>arXiv preprint arXiv:1803.01271</i>.",
        "Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. <i>Journal of Machine Learning Research</i>, 13(2), 281–305.",
        "Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). <i>Time series analysis: Forecasting and control</i> (5th ed.). John Wiley & Sons.",
        "Breiman, L. (2001). Random forests. <i>Machine Learning</i>, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i> (pp. 785–794).",
        "Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. <i>Journal of Business & Economic Statistics</i>, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599",
        "Duan, J., Gong, Y., Luo, J., & Zhao, Z. (2023). Air-quality prediction based on the ARIMA-CNN-LSTM combination model optimized by dung beetle optimizer. <i>Scientific Reports</i>, 13(1), Article 12127. https://doi.org/10.1038/s41598-023-36620-4",
        "Hyndman, R. J., & Athanasopoulos, G. (2021). <i>Forecasting: Principles and practice</i> (3rd ed.). OTexts: Melbourne, Australia.",
        "Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. <i>International Journal of Forecasting</i>, 22(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001",
        "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. <i>Advances in Neural Information Processing Systems</i>, 30, 3146–3154.",
        "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In <i>International Conference on Learning Representations (ICLR)</i>.",
        "Xue, J., & Shen, B. (2022). Dung beetle optimizer: A new meta-heuristic algorithm for global optimization. <i>The Journal of Supercomputing</i>, 79(7), 7305–7348. https://doi.org/10.1007/s11227-022-04959-6"
    ]
    for r in refs:
        story.append(Paragraph(r, styles['RefEntry']))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Academic PDF successfully saved to: {pdf_path}")


if __name__ == "__main__":
    generate_docx()
    generate_pdf()
    print("ALL DELIVERABLE DOCUMENTS SUCCESSFULLY GENERATED!")
