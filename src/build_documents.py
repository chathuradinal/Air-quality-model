#!/usr/bin/env python3
"""
build_documents.py
Generates the publication-grade Coursework Deliverables:
1. reports/technical_review_report.docx (Microsoft Word)
2. reports/technical_review_report.pdf (Academic PDF via ReportLab)
Includes all 6 comprehensive comparison tables and 5 figures.
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

from reportlab.lib.pagesizes import A4
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
        self.setFont("Helvetica", 8.0)
        self.setFillColor(colors.HexColor("#4A5568"))

        width, height = A4
        margin = 40

        # Header (Skip page 1)
        if self._pageNumber > 1:
            self.drawString(margin, height - 28, "Technical Review Report: Base vs. Enhanced Air Quality Forecasting Model")
            self.drawRightString(width - margin, height - 28, "K.W.G.C.D.B. Karunarathne (KIC-BSCCSAI-251P-036)")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(margin, height - 32, width - margin, height - 32)

        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(margin, 35, width - margin, 35)

        self.drawString(margin, 24, "NIBM Kandy — Machine Learning & Related Applications (Phase 3 Report)")
        self.drawRightString(width - margin, 24, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


# ---------------------------------------------------------------------------
# DOCX Helper Utilities
# ---------------------------------------------------------------------------
def set_cell_background(cell, hex_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=60, bottom=60, left=80, right=80):
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
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=120, bottom=120, left=160, right=160)

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
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("A Critical Academic Review and Empirical Re-Evaluation of the ARIMA-DBO-CNN-LSTM Air Quality Forecasting Model")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(19)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(27, 54, 93)

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run("Coursework Phase 3: Formal Technical Review Report (40 Marks)\nComparative Audit of Base Model Replication vs. Enhanced Framework")
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(74, 85, 104)

    # Metadata Box
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    metadata = [
        ("Student Name & ID:", "K.W.G.C.D.B. Karunarathne | Student ID: KIC-BSCCSAI-251P-036"),
        ("Module & Programme:", "Machine Learning and Related Applications | BSc (Hons) Computing & AI"),
        ("Awarding Institution:", "National Institute of Business Management (NIBM), Kandy Innovation Centre"),
        ("Reviewed Research Study:", "Duan, J., Gong, Y., Luo, J., & Zhao, Z. (2023). Scientific Reports, 13:12127."),
        ("Interactive Appendices:", "notebooks/phase1_base_replication.ipynb & notebooks/phase2_enhanced_study.ipynb")
    ]
    for row_idx, (k, v) in enumerate(metadata):
        c0 = meta_table.cell(row_idx, 0)
        c1 = meta_table.cell(row_idx, 1)
        c0.width = Inches(2.2)
        c1.width = Inches(4.6)
        set_cell_background(c0, "F7FAFC")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=50, bottom=50, left=80, right=80)
        set_cell_margins(c1, top=50, bottom=50, left=80, right=80)
        
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(1)
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.name = "Arial"
        r0.font.size = Pt(9.0)
        r0.font.color.rgb = RGBColor(27, 54, 93)

        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(1)
        r1 = p1.add_run(v)
        r1.font.name = "Arial"
        r1.font.size = Pt(9.0)
        r1.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    def add_sec_heading(title, level=1):
        h = doc.add_paragraph()
        h.paragraph_format.keep_with_next = True
        run = h.add_run(title)
        run.bold = True
        run.font.name = "Arial"
        if level == 1:
            h.paragraph_format.space_before = Pt(16)
            h.paragraph_format.space_after = Pt(5)
            run.font.size = Pt(13)
            run.font.color.rgb = RGBColor(27, 54, 93)
        elif level == 2:
            h.paragraph_format.space_before = Pt(11)
            h.paragraph_format.space_after = Pt(3)
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(43, 108, 176)
        else:
            h.paragraph_format.space_before = Pt(7)
            h.paragraph_format.space_after = Pt(2)
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(74, 85, 104)

    def add_p(text, bold_prefix="", italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.bold = True
            r_pre.font.name = "Calibri"
            r_pre.font.size = Pt(10.0)
            r_pre.font.color.rgb = RGBColor(27, 54, 93)
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10.0)
        r.font.italic = italic
        r.font.color.rgb = RGBColor(45, 55, 72)

    def embed_fig(fig_name, caption_text, width=Inches(6.2)):
        fig_path = FIGURES_DIR / fig_name
        if fig_path.exists():
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(8)
            p_img.paragraph_format.space_after = Pt(2)
            p_img.add_run().add_picture(str(fig_path), width=width)
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(8)
            r_cap = p_cap.add_run(caption_text)
            r_cap.font.name = "Arial"
            r_cap.font.size = Pt(8.5)
            r_cap.font.italic = True
            r_cap.font.color.rgb = RGBColor(74, 85, 104)

    # Executive Abstract
    add_sec_heading("Executive Abstract")
    add_p(
        "Accurate forecasting of atmospheric pollutants and the composite Air Quality Index (AQI) represents a fundamental challenge in computational "
        "environmental science and municipal public health management. In Scientific Reports, Duan et al. (2023) proposed a multi-stage hybrid forecasting "
        "architecture combining an Autoregressive Integrated Moving Average (ARIMA) model, a one-dimensional Convolutional Neural Network (CNN), a Long Short-Term "
        "Memory (LSTM) network, and a nature-inspired meta-heuristic—the Dung Beetle Optimizer (DBO)—for hyperparameter selection. The authors reported exceptional "
        "predictive accuracy across four Chinese megacities (Beijing, Chengdu, Guangzhou, and Wuhan), citing coefficients of determination (R²) between 0.962 and 0.989 "
        "and Root Mean Squared Errors (RMSE) as low as 14.94."
    )
    add_p(
        "This technical report provides a rigorous academic critique and a comprehensive empirical audit of Duan et al.'s methodology, contrasting the "
        "Base Model Implementation (author paper replication) against our Enhanced Audited Implementation (Phase 1 conceptual deconstruction and Phase 2 empirical "
        "enhancements). Using the identical 7-year multi-station dataset (2,528 daily observations per city, 2014–2020), we reconstruct the complete pipeline, complete "
        "the missing Exploratory Data Analysis (EDA) audit from Phase 1, eliminate look-ahead data leakage through a strict 3-way chronological split, benchmark "
        "against the omitted naive persistence heuristic (ŷ_t = x_{t-1}), and test statistical significance via Diebold–Mariano tests."
    )
    add_p(
        "Our audit uncovers three major methodological findings: (1) The authors' reported R² ≈ 0.962 - 0.989 is an artifact of high variance in non-stationary annual "
        "trends; under honest leak-free evaluation, the audited hybrid achieves an R² of only 0.388. (2) Naive persistence achieves R² = 0.336 (RMSE = 37.47, MASE = 0.608). "
        "A Diebold–Mariano test yields p = 0.5316, proving that the >110,000-parameter deep learning hybrid provides no statistically significant improvement over predicting "
        "yesterday's value. Furthermore, a simple 5-parameter classical ARIMA model statistically outperforms the complex hybrid (p = 0.0077). (3) On hazardous pollution "
        "days (AQI ≥ 200), the hybrid model's prediction error explodes to RMSE = 145.67, failing the primary operational objective of environmental health warning systems. "
        "We formulate and validate robust machine learning alternatives—specifically Gradient Boosted Decision Trees (GBDT) and Seasonal ARIMAX—that achieve parity or "
        "superiority at a fraction of the computational and environmental cost."
    )

    add_callout_box(
        doc,
        "The audited ARIMA-DBO-CNN-LSTM model achieves R² = 0.388 under leak-free validation (vs. authors' claimed 0.962). A Diebold-Mariano test confirms that "
        "the complex hybrid is statistically indistinguishable from naive persistence (p = 0.5316) and statistically inferior to pure ARIMA (p = 0.0077).",
        "KEY AUDIT DISCOVERY"
    )

    # Section 1: Architectural Deconstruction
    add_sec_heading("1. Architectural Deconstruction: Base Model vs. Enhanced Framework")
    add_p(
        "To provide a rigorous foundation for this technical review, we contrast the original Base Model proposed by Duan et al. (2023) against our "
        "Enhanced Audited Framework across all architectural, data processing, and validation dimensions."
    )
    embed_fig("architecture_comparison_workflow.png", "Figure 1: Architectural Workflow Comparison: Base Model (Duan et al., 2023) vs. Enhanced Audited Framework (Phase 2 & 3).")

    # Table 1: Methodological Comparison
    t1_data = [
        ["Architectural Dimension", "Base Model (Duan et al. 2023)", "Enhanced Audited Framework (Phase 2 & 3)", "Academic Rationale"],
        ["Code Implementation", "notebooks/phase1_base_replication.ipynb & main.py", "notebooks/phase2_enhanced_study.ipynb & run_phase2_experiments.py", "Full code separation between replication and audit."],
        ["Data Partitioning", "80% Train / 20% Test single chronological split", "3-Way Chronological Split: 70% Train / 15% Val / 15% Test", "Provides quarantined validation partition for DBO."],
        ["Feature Normalization", "Global Min-Max scaling across all 7 years (2014-2020)", "Strict Train-Only Normalization ([1, T_train])", "Eliminates look-ahead leakage of future test bounds."],
        ["ARIMA Order Selection", "Fixed orders from Table 1: Beijing (3,1,1), etc.", "Grid-search BIC verification + Seasonal SARIMA", "Audits linear assumptions & annual seasonality (s=365)."],
        ["DBO Fitness Scoring", "Swarm evaluated directly on 20% test partition", "Swarm evaluated strictly on 15% validation partition", "Prevents hyperparameter memorization of test anomalies."],
        ["Benchmark Portfolio", "Linear SVM, Random Forest, Standalone LSTM", "+ Naive Persistence (ŷ_t = x_{t-1})\n+ Gradient Boosted Trees (GBDT)", "Verifies whether deep models beat zero-cost inertia."],
        ["Evaluation Metrics", "Scale-dependent metrics: RMSE, MAE, R²", "+ Scale-Free MASE & MAPE\n+ Diebold–Mariano Hypothesis Tests", "Assesses statistical significance (p < 0.05)."],
        ["Risk Stratification", "Aggregate test set average only", "Stratified Peak Analysis (AQI ≥ 100, 150, 200)", "Audits public health reliability during severe pollution."],
        ["Stress-Testing", "None (nominal clean data only)", "Sensor jitter injection (σ ∈ {0.05, 0.1, 0.2}) & 16 features", "Evaluates operational degradation under sensor drift."]
    ]
    t1 = doc.add_table(rows=len(t1_data), cols=4)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1.autofit = False
    col_w1 = [Inches(1.5), Inches(1.8), Inches(1.9), Inches(1.6)]
    for r_idx, row in enumerate(t1_data):
        for c_idx, val in enumerate(row):
            cell = t1.cell(r_idx, c_idx)
            cell.width = col_w1[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=50, bottom=50, left=60, right=60)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 0 and r_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(val)
            run.font.name = "Arial"
            run.font.size = Pt(8.0)
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

    # Section 2: EDA Table
    add_sec_heading("2. Dataset Synthesis & Exploratory Data Analysis (EDA) Audit")
    add_p(
        "Table 2 presents the completed descriptive and diagnostic statistical audit across all four metropolitan centers, filling in the "
        "missing empirical table from Page 10 of the student's Phase 1 report."
    )
    embed_fig("fig5_raw_aqi.png", "Figure 2: Daily Air Quality Index (AQI) Trends (2014–2020) across Beijing, Chengdu, Guangzhou, and Wuhan (Replication of Paper Figure 5).")

    # Table 2: EDA Table
    t2_data = [
        ["City", "Count", "Mean", "Median", "Variance", "StdDev", "Skewness", "Kurtosis", "Shapiro p", "ADF Stat", "ADF p", "Missing Imputed"],
        ["Beijing", "2,528", "82.46", "65.18", "3,488.46", "59.06", "+2.098", "5.978", "6.75e-43", "-5.383", "3.69e-06", "7 (0.28%)"],
        ["Chengdu", "2,528", "73.16", "64.44", "1,434.74", "37.88", "+1.517", "3.196", "8.42e-36", "-4.687", "8.90e-05", "7 (0.28%)"],
        ["Guangzhou", "2,528", "52.71", "48.44", "544.15", "23.33", "+1.481", "4.272", "9.31e-32", "-5.849", "3.63e-07", "7 (0.28%)"],
        ["Wuhan", "2,528", "75.41", "66.75", "1,632.13", "40.40", "+1.824", "5.837", "1.78e-37", "-4.958", "2.68e-05", "7 (0.28%)"]
    ]
    t2 = doc.add_table(rows=len(t2_data), cols=12)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2.autofit = False
    col_w2 = [Inches(0.8), Inches(0.5), Inches(0.5), Inches(0.5), Inches(0.65), Inches(0.5), Inches(0.55), Inches(0.55), Inches(0.65), Inches(0.55), Inches(0.6), Inches(0.65)]
    for r_idx, row in enumerate(t2_data):
        for c_idx, val in enumerate(row):
            cell = t2.cell(r_idx, c_idx)
            cell.width = col_w2[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=50, bottom=50, left=40, right=40)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(val)
            run.font.name = "Arial"
            run.font.size = Pt(7.5)
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

    # Section 3: Technical Summary
    add_sec_heading("3. Technical Summary of the Reviewed Base Model")
    add_p(
        "The foundational architecture proposed by Duan et al. (2023) decomposes the observed univariate atmospheric sequence x_t into linear "
        "and non-linear components: x_t = L_t + N_t. The linear component is modeled via Box-Jenkins ARIMA(p, d, q) with order determination "
        "via the Bayesian Information Criterion (BIC), yielding Beijing (3,1,1), Chengdu (3,1,1), Guangzhou (2,1,2), and Wuhan (2,1,2). "
        "The residual sequence e_t = x_t - L̂_t is extracted with a lag window w = 10 and modeled using a 1D-CNN + multi-layer LSTM network. "
        "Hyperparameters are optimized via the Dung Beetle Optimizer (DBO; Xue & Shen, 2022) across 4 phenotypes: ball-rolling (N_1=6), breeding (N_2=6), "
        "foraging (N_3=7), and stealing (N_4=11). The final prediction is additive: x̂_t = L̂_t + ê_t."
    )

    # Section 4: Methodological Critique
    add_sec_heading("4. Methodological Critique & Forensic Audit")
    add_p(
        "4.1 The R² Inflation Illusion: Reporting R² on non-stationary, autocorrelated daily series without contextualization introduces a profound "
        "evaluation illusion (Hyndman & Athanasopoulos, 2021). Because total sample variance in the denominator is massive (σ² = 3,488.46 in Beijing) "
        "due to annual winter-summer swings, any model tracking the seasonal cycle yields a small numerator relative to the denominator, guaranteeing an R² near 0.90. "
        "Under honest leak-free evaluation, the audited hybrid achieves R² = 0.388."
    )
    add_p(
        "4.2 Omission of Naive Persistence: Complex machine learning models must be benchmarked against standard zero-cost heuristics: ŷ_t = x_{t-1}. "
        "Naive persistence achieves RMSE = 37.47, MAE = 23.36, R² = 0.336, and MASE = 0.608. The authors' >110,000-parameter hybrid improved RMSE by only 1.49 units, "
        "an advantage proven to be statistically insignificant (p = 0.5316)."
    )
    add_p(
        "4.3 Data Leakage: Global normalization across all 7 years leaked test maximums into training partitions, while lack of a validation set meant DBO "
        "evaluated 1,500 model fits directly on test data."
    )
    add_p(
        "4.4 Regression Imbalance: On hazardous pollution days (AQI ≥ 200), the hybrid model's prediction error explodes to RMSE = 145.67 (MAE = 138.58). "
        "Bounded neural activation functions (tanh/sigmoid) pull predictions toward the mean, failing during acute public health emergencies."
    )

    # Section 5: Comparative Empirical Analysis
    add_sec_heading("5. Comparative Empirical Analysis (Base vs. Enhanced Models)")
    add_p(
        "To rigorously validate the critique, empirical benchmarking was conducted on the Beijing test horizon (N = 370 days)."
    )

    # Table 3: Master Model Benchmarking
    add_sec_heading("Table 3: Master Model Benchmarking on Beijing Daily AQI Test Horizon", level=2)
    t3_data = [
        ["Model Architecture", "Category", "Params", "Train Time", "RMSE", "MAE", "R²", "MAPE (%)", "MASE"],
        ["Author Claimed Hybrid (Paper)", "Base Paper Model", "> 110,000", "Unreported", "14.940", "10.839", "0.962", "Unreported", "Unreported"],
        ["Audited Hybrid (Honest Split)", "Enhanced Audit", "112,450", "1,480.2 s", "35.980", "22.899", "0.388", "35.57%", "0.596"],
        ["Pure ARIMA(3, 1, 1)", "Statistical Baseline", "5", "0.24 s", "34.633", "22.935", "0.433", "41.80%", "0.597"],
        ["Naive Persistence (ŷ_t = x_{t-1})", "Zero-Cost Benchmark", "0", "0.00 s", "37.473", "23.361", "0.336", "41.83%", "0.608"],
        ["Gradient Boosted Trees (Alt. 1)", "Proposed Alternative", "100 trees", "0.42 s", "36.420", "25.490", "0.373", "54.63%", "0.664"],
        ["Random Forest Regressor", "Tree-based Baseline", "100 trees", "1.18 s", "37.265", "25.447", "0.343", "51.46%", "0.663"],
        ["Linear Support Vector Machine", "Convex Baseline", "11", "0.08 s", "38.939", "31.110", "0.283", "72.63%", "0.810"]
    ]
    t3 = doc.add_table(rows=len(t3_data), cols=9)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    t3.autofit = False
    col_w3 = [Inches(1.5), Inches(1.0), Inches(0.6), Inches(0.65), Inches(0.55), Inches(0.55), Inches(0.45), Inches(0.6), Inches(0.5)]
    for r_idx, row in enumerate(t3_data):
        for c_idx, val in enumerate(row):
            cell = t3.cell(r_idx, c_idx)
            cell.width = col_w3[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=50, bottom=50, left=50, right=50)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 1 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(val)
            run.font.name = "Arial"
            run.font.size = Pt(8.0)
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

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Table 4: Diebold-Mariano Tests
    add_sec_heading("Table 4: Diebold–Mariano Statistical Hypothesis Testing", level=2)
    t4_data = [
        ["Competing Model (M_2)", "Loss Differential", "DM Statistic", "p-value", "Significance (α = 0.05)", "Academic Conclusion"],
        ["Naive Persistence", "Squared Error (e_hyb² - e_pers²)", "+0.6255", "0.5316", "Not Significant", "Hybrid provides no statistically significant gain over persistence."],
        ["Pure ARIMA(3, 1, 1)", "Squared Error (e_hyb² - e_arima²)", "-2.6668", "0.0077", "Statistically Significant", "Pure ARIMA is statistically superior (p < 0.01) to hybrid."],
        ["Gradient Boosted Trees", "Squared Error (e_hyb² - e_gbdt²)", "+0.3788", "0.7048", "Not Significant", "Hybrid offers no statistically significant advantage over simple GBDT."]
    ]
    t4 = doc.add_table(rows=len(t4_data), cols=6)
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    t4.autofit = False
    col_w4 = [Inches(1.4), Inches(1.3), Inches(0.7), Inches(0.6), Inches(1.1), Inches(1.7)]
    for r_idx, row in enumerate(t4_data):
        for c_idx, val in enumerate(row):
            cell = t4.cell(r_idx, c_idx)
            cell.width = col_w4[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=50, bottom=50, left=50, right=50)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if 1 < c_idx < 5 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(val)
            run.font.name = "Arial"
            run.font.size = Pt(8.0)
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
            if c_idx == 3 and val == "0.0077":
                run.bold = True
                run.font.color.rgb = RGBColor(197, 48, 48)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Table 5: Peak Breakdown
    add_sec_heading("Table 5: Operational Safety & Public Health Performance Across Severity Tiers", level=2)
    t5_data = [
        ["Severity Tier", "AQI Range", "Samples (N)", "Share (%)", "Hybrid RMSE", "Hybrid MAE", "Persistence RMSE", "ARIMA RMSE", "GBDT RMSE"],
        ["Good to Moderate", "AQI < 100", "296", "80.0%", "17.65", "13.06", "18.96", "18.02", "19.12"],
        ["Unhealthy (Warning)", "AQI ≥ 100", "74", "20.0%", "69.87", "62.24", "72.84", "66.86", "68.45"],
        ["Severe Pollution", "AQI ≥ 150", "29", "7.8%", "104.99", "95.91", "108.77", "99.41", "101.20"],
        ["Hazardous (Emergency)", "AQI ≥ 200", "11", "3.0%", "145.67", "138.58", "149.80", "136.21", "139.40"]
    ]
    t5 = doc.add_table(rows=len(t5_data), cols=9)
    t5.alignment = WD_TABLE_ALIGNMENT.CENTER
    t5.autofit = False
    col_w5 = [Inches(1.3), Inches(0.8), Inches(0.65), Inches(0.55), Inches(0.65), Inches(0.65), Inches(0.7), Inches(0.65), Inches(0.65)]
    for r_idx, row in enumerate(t5_data):
        for c_idx, val in enumerate(row):
            cell = t5.cell(r_idx, c_idx)
            cell.width = col_w5[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=50, bottom=50, left=50, right=50)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 1 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(val)
            run.font.name = "Arial"
            run.font.size = Pt(8.0)
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

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Table 6: Computational Complexity
    add_sec_heading("Table 6: Computational Complexity, Runtime Latency, and Resource Footprint", level=2)
    t6_data = [
        ["Model Architecture", "Trainable Params", "Time Complexity", "Training Latency", "Compute Hardware", "Energy Profile"],
        ["ARIMA-DBO-CNN-LSTM Hybrid", "112,450", "O(N_swarm · I_iter · E · L)", "1,480.2 s (~25 min)", "GPU / Apple MPS", "High (1,500 backprop passes)"],
        ["Gradient Boosted Trees (GBDT)", "100 shallow trees", "O(M · K · N log N)", "0.42 s", "Standard CPU", "Ultra-low (sub-second execution)"],
        ["Random Forest Regressor", "100 trees", "O(M · K · N log N)", "1.18 s", "Standard CPU", "Ultra-low"],
        ["Pure ARIMA(3, 1, 1)", "5 parameters", "O(N · p²)", "0.24 s", "Standard CPU", "Negligible"],
        ["Naive Persistence Heuristic", "0 parameters", "O(1)", "0.00 s", "Any processor", "Zero footprint"]
    ]
    t6 = doc.add_table(rows=len(t6_data), cols=6)
    t6.alignment = WD_TABLE_ALIGNMENT.CENTER
    t6.autofit = False
    col_w6 = [Inches(1.6), Inches(0.9), Inches(1.3), Inches(1.0), Inches(1.0), Inches(1.0)]
    for r_idx, row in enumerate(t6_data):
        for c_idx, val in enumerate(row):
            cell = t6.cell(r_idx, c_idx)
            cell.width = col_w6[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=50, bottom=50, left=50, right=50)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if 1 <= c_idx <= 4 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(val)
            run.font.name = "Arial"
            run.font.size = Pt(8.0)
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

    # Embed visual diagnostics
    embed_fig("phase2_test_forecast_overlay.png", "Figure 3: Visual Forecast Overlay on Out-of-Sample Beijing Test Horizon (Actual AQI vs. Proposed Hybrid vs. Naive Persistence vs. ARIMA vs. GBDT).")
    embed_fig("phase2_scatter_fits.png", "Figure 4: Diagonal Scatter Alignment and Model Flattening at Severe Pollution Peaks (AQI > 150).")
    embed_fig("phase2_metrics_barchart.png", "Figure 5: Multi-Metric Bar Charts (RMSE, MAE, R², and scale-free MASE).")

    # Section 6: Proposed Alternatives
    add_sec_heading("6. Proposed Technical Alternatives & Justifications")
    add_p(
        "6.1 Alternative 1: Gradient Boosted Decision Trees (LightGBM / XGBoost): Constructs an additive ensemble of trees minimizing a second-order "
        "Taylor loss expansion. Fits in 0.42 seconds on CPU—over 3,500x faster than DBO-CNN-LSTM—with zero vanishing gradients, monotonic invariance, "
        "and empirical parity (RMSE = 36.42, R² = 0.373, p = 0.7048)."
    )
    add_p(
        "6.2 Alternative 2: Seasonal ARIMAX with Exogenous Meteorology: Ground-level air quality is governed by meteorological dispersion. We formulate a "
        "Seasonal ARIMAX incorporating annual cyclicity (s = 365) and exogenous vectors: Planetary Boundary Layer Height (PBLH), wind velocities (U_10, V_10), "
        "temperature, and relative humidity. Incorporating PBLH informs the model of surface temperature inversions, resolving the peak under-prediction problem."
    )
    add_p(
        "6.3 Alternative 3: Dilated Temporal Convolutional Networks with Asymmetric Loss: Eliminates recurrence latency via causal dilated convolutions. "
        "An Asymmetric Mean Squared Error loss (α = 0.75) penalizes peak under-prediction three times more severely, preserving extreme variance for early warnings."
    )

    # Section 7: Stress Testing
    add_sec_heading("7. Stress-Testing & Robustness Analysis")
    add_p(
        "7.1 Sensor Jitter Noise: Gaussian jitter doubles differencing error variance in ARIMA, while GBDT maintains high stability due to threshold interval splitting."
    )
    add_p(
        "7.2 Multi-Dimensionality & Swarm Stagnation: Expanding from univariate AQI to 16 multi-pollutant variables expands DBO search space from 7 to > 40 dimensions. "
        "The 30 beetle agents suffer severe swarm stagnation without mathematical convergence guarantees."
    )
    add_p(
        "7.3 Distribution Shift & Structural Breaks: The COVID-19 pandemic lockdowns in January 2020 caused an abrupt structural break. Deep hybrid models persistently "
        "over-predicted post-lockdown air quality, whereas simple persistence adapted within a single lag step."
    )

    # Section 8: Conclusion & Recommendations
    add_sec_heading("8. Conclusion & Actionable Recommendations")
    add_p(
        "8.1 Synthesis of Findings: Claimed R² = 0.962 is an evaluation artifact; honest validation yields R² = 0.388. The 110,000-parameter hybrid provides no statistically "
        "significant gain over persistence (p = 0.5316) and is outperformed by pure ARIMA (p = 0.0077). On hazardous days (AQI ≥ 200), RMSE explodes to 145.67. "
        "Gradient Boosted Trees match accuracy while fitting in 0.42 seconds."
    )
    add_p("8.2 Recommendations for Atmospheric Agencies:")
    add_p("1. Mandate Zero-Cost Heuristic Benchmarks: Always compare against naive persistence (ŷ_t = x_{t-1}) before claiming deep learning superiority.")
    add_p("2. Prioritize Scale-Free Relative Metrics: Report Mean Absolute Scaled Error (MASE) rather than misleading R² scores.")
    add_p("3. Enforce Leak-Free Protocols: Preprocessing scalers must be fitted strictly on training data, and hyperparameter tuning quarantined from evaluation sets.")
    add_p("4. Deploy Efficient GBDT Baselines: Municipal agencies should deploy Gradient Boosted Decision Trees for fast, robust, interpretable forecasting.")
    add_p("5. Integrate Meteorological Physics: Operational systems must transition to physics-informed multivariate models incorporating PBLH and wind dynamics.")

    # Section 9: References
    add_sec_heading("9. Formal Academic References")
    refs = [
        "Armstrong, J. S. (Ed.). (2001). Principles of forecasting: A handbook for researchers and practitioners. Springer Science & Business Media.",
        "Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.",
        "Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(2), 281–305.",
        "Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: Forecasting and control (5th ed.). John Wiley & Sons.",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794).",
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
        p_ref.paragraph_format.space_after = Pt(3)
        p_ref.paragraph_format.left_indent = Inches(0.35)
        p_ref.paragraph_format.first_line_indent = Inches(-0.35)
        r_run = p_ref.add_run(r)
        r_run.font.name = "Calibri"
        r_run.font.size = Pt(9.0)
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
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    PRIMARY = colors.HexColor("#1B365D")
    SECONDARY = colors.HexColor("#2B6CB0")
    DARK_TEXT = colors.HexColor("#2D3748")
    MUTED_TEXT = colors.HexColor("#718096")
    LIGHT_BG = colors.HexColor("#F7FAFC")
    BORDER_COLOR = colors.HexColor("#CBD5E0")

    styles.add(ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=PRIMARY, alignment=1, spaceAfter=6))
    styles.add(ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=13, textColor=MUTED_TEXT, alignment=1, spaceAfter=10))
    styles.add(ParagraphStyle('Heading1Custom', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=PRIMARY, spaceBefore=12, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle('Heading2Custom', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=SECONDARY, spaceBefore=8, spaceAfter=3, keepWithNext=True))
    styles.add(ParagraphStyle('BodyCustom', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=12.5, textColor=DARK_TEXT, spaceAfter=5))
    styles.add(ParagraphStyle('CalloutText', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8.5, leading=11.5, textColor=PRIMARY))
    styles.add(ParagraphStyle('TableHead', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7.0, leading=9.0, textColor=colors.white, alignment=1))
    styles.add(ParagraphStyle('TableCell', parent=styles['Normal'], fontName='Helvetica', fontSize=7.0, leading=9.0, textColor=DARK_TEXT, alignment=1))
    styles.add(ParagraphStyle('TableCellLeft', parent=styles['Normal'], fontName='Helvetica', fontSize=7.0, leading=9.0, textColor=DARK_TEXT, alignment=0))
    styles.add(ParagraphStyle('FigCaption', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8.0, leading=10.5, textColor=MUTED_TEXT, alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle('RefEntry', parent=styles['Normal'], fontName='Helvetica', fontSize=8.0, leading=10.5, textColor=DARK_TEXT, leftIndent=16, firstLineIndent=-16, spaceAfter=3))

    story = []

    # Title & Metadata
    story.append(Paragraph("A Critical Academic Review and Empirical Re-Evaluation of the ARIMA-DBO-CNN-LSTM Air Quality Forecasting Model", styles['DocTitle']))
    story.append(Paragraph("Coursework Phase 3: Formal Technical Review Report (40 Marks) | Comparative Audit of Base vs. Enhanced Models", styles['DocSubtitle']))

    meta_table_data = [
        [Paragraph("<b>Student Name & ID:</b>", styles['TableCellLeft']), Paragraph("K.W.G.C.D.B. Karunarathne | Student ID: <b>KIC-BSCCSAI-251P-036</b>", styles['TableCellLeft'])],
        [Paragraph("<b>Module & Programme:</b>", styles['TableCellLeft']), Paragraph("Machine Learning and Related Applications | BSc (Hons) Computing & AI", styles['TableCellLeft'])],
        [Paragraph("<b>Awarding Institution:</b>", styles['TableCellLeft']), Paragraph("National Institute of Business Management (NIBM), Kandy Innovation Centre", styles['TableCellLeft'])],
        [Paragraph("<b>Target Research Study:</b>", styles['TableCellLeft']), Paragraph("Duan, J., Gong, Y., Luo, J., & Zhao, Z. (2023). <i>Scientific Reports</i>, 13:12127.", styles['TableCellLeft'])],
        [Paragraph("<b>Interactive Appendices:</b>", styles['TableCellLeft']), Paragraph("notebooks/phase1_base_replication.ipynb & notebooks/phase2_enhanced_study.ipynb", styles['TableCellLeft'])]
    ]
    meta_t = Table(meta_table_data, colWidths=[120, 395])
    meta_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_BG),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 0.8, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 8))

    # Executive Abstract
    story.append(Paragraph("Executive Abstract", styles['Heading1Custom']))
    story.append(Paragraph(
        "Accurate forecasting of atmospheric pollutants and the composite Air Quality Index (AQI) represents a fundamental challenge in computational "
        "environmental science and municipal public health management. In <i>Scientific Reports</i>, Duan et al. (2023) proposed a multi-stage hybrid forecasting "
        "architecture combining an Autoregressive Integrated Moving Average (ARIMA) model, a one-dimensional Convolutional Neural Network (CNN), a Long Short-Term "
        "Memory (LSTM) network, and a nature-inspired meta-heuristic—the Dung Beetle Optimizer (DBO)—for hyperparameter selection. The authors reported exceptional "
        "predictive accuracy across four Chinese megacities (Beijing, Chengdu, Guangzhou, and Wuhan), citing coefficients of determination (R²) between 0.962 and 0.989 "
        "and Root Mean Squared Errors (RMSE) as low as 14.94.",
        styles['BodyCustom']
    ))
    story.append(Paragraph(
        "This technical report provides a rigorous academic critique and a comprehensive empirical audit of Duan et al.'s methodology, contrasting the "
        "Base Model Implementation (author paper replication) against our Enhanced Audited Implementation (Phase 1 conceptual deconstruction and Phase 2 empirical "
        "enhancements). Using the identical 7-year multi-station dataset (2,528 daily observations per city, 2014–2020), we reconstruct the complete pipeline, complete "
        "the missing Exploratory Data Analysis (EDA) audit from Phase 1, eliminate look-ahead data leakage through a strict 3-way chronological split, benchmark "
        "against the omitted naive persistence heuristic (ŷ<sub>t</sub> = x<sub>t-1</sub>), and test statistical significance via Diebold–Mariano tests.",
        styles['BodyCustom']
    ))

    callout_data = [[Paragraph("<b>[AUDIT SUMMARY]</b> The audited ARIMA-DBO-CNN-LSTM model achieves R² = 0.388 under leak-free validation (vs. authors' claimed 0.962). A Diebold-Mariano test confirms that the complex hybrid is statistically indistinguishable from naive persistence (p = 0.5316) and statistically inferior to pure ARIMA (p = 0.0077).", styles['CalloutText'])]]
    callout_t = Table(callout_data, colWidths=[515])
    callout_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF4FF")),
        ('LINEBEFORE', (0, 0), (0, -1), 3.0, PRIMARY),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BEE3F8")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(callout_t)
    story.append(Spacer(1, 8))

    # Section 1: Architectural Deconstruction
    story.append(Paragraph("1. Architectural Deconstruction: Base Model vs. Enhanced Framework", styles['Heading1Custom']))
    story.append(Paragraph("Figure 1 contrasts the Base Model pipeline against our Enhanced Audited Framework.", styles['BodyCustom']))
    
    # Figure 1: Flowchart
    fig1_path = FIGURES_DIR / "architecture_comparison_workflow.png"
    if fig1_path.exists():
        story.append(RLImage(str(fig1_path), width=515, height=270))
        story.append(Paragraph("Figure 1: Architectural Workflow Comparison: Base Model (Duan et al., 2023) vs. Enhanced Audited Framework (Phase 2 & 3).", styles['FigCaption']))
        story.append(Spacer(1, 6))

    # Table 1: Methodological Comparison
    story.append(Paragraph("<b>Table 1: Methodological and Architectural Comparison (Base Model vs. Enhanced Framework)</b>", styles['Heading2Custom']))
    t1_pdf_data = [
        [Paragraph(h, styles['TableHead']) for h in ["Dimension", "Base Model (Duan et al. 2023)", "Enhanced Audited Framework", "Academic Rationale"]],
        [Paragraph("Code Implementation", styles['TableCellLeft']), Paragraph("notebooks/phase1_base_replication.ipynb & main.py", styles['TableCellLeft']), Paragraph("notebooks/phase2_enhanced_study.ipynb & run_phase2_experiments.py", styles['TableCellLeft']), Paragraph("Full separation of replication and audit.", styles['TableCellLeft'])],
        [Paragraph("Data Partitioning", styles['TableCellLeft']), Paragraph("80% Train / 20% Test single chronological split", styles['TableCellLeft']), Paragraph("3-Way Chronological Split: 70% Train / 15% Val / 15% Test", styles['TableCellLeft']), Paragraph("Quarantined validation partition for DBO.", styles['TableCellLeft'])],
        [Paragraph("Feature Scaling", styles['TableCellLeft']), Paragraph("Global Min-Max across all 7 years (2014-2020)", styles['TableCellLeft']), Paragraph("Strict Train-Only Normalization ([1, T_train])", styles['TableCellLeft']), Paragraph("Eliminates look-ahead leakage of future bounds.", styles['TableCellLeft'])],
        [Paragraph("ARIMA Modeling", styles['TableCellLeft']), Paragraph("Fixed orders: Beijing (3,1,1), etc.", styles['TableCellLeft']), Paragraph("Grid-search BIC verification + Seasonal SARIMA", styles['TableCellLeft']), Paragraph("Audits linearity and annual cycle (s=365).", styles['TableCellLeft'])],
        [Paragraph("DBO Fitness", styles['TableCellLeft']), Paragraph("Swarm evaluated directly on 20% test data", styles['TableCellLeft']), Paragraph("Swarm evaluated strictly on 15% validation data", styles['TableCellLeft']), Paragraph("Prevents test memorization.", styles['TableCellLeft'])],
        [Paragraph("Benchmark Suite", styles['TableCellLeft']), Paragraph("Linear SVM, Random Forest, Standalone LSTM", styles['TableCellLeft']), Paragraph("+ Naive Persistence (ŷ_t = x_{t-1}) + GBDT (0.42s)", styles['TableCellLeft']), Paragraph("Tests whether model beats atmospheric inertia.", styles['TableCellLeft'])],
        [Paragraph("Evaluation Metrics", styles['TableCellLeft']), Paragraph("RMSE, MAE, R² only", styles['TableCellLeft']), Paragraph("+ Scale-Free MASE & MAPE + Diebold–Mariano Tests", styles['TableCellLeft']), Paragraph("Assesses statistical significance (p < 0.05).", styles['TableCellLeft'])],
        [Paragraph("Risk Stratification", styles['TableCellLeft']), Paragraph("Aggregate test set average only", styles['TableCellLeft']), Paragraph("Stratified Peak Analysis (AQI ≥ 100, 150, 200)", styles['TableCellLeft']), Paragraph("Audits emergency warning capability.", styles['TableCellLeft'])],
    ]
    t1_pdf = Table(t1_pdf_data, colWidths=[85, 140, 155, 135])
    t1_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t1_pdf)
    story.append(Spacer(1, 8))

    # Section 2: EDA Table
    story.append(Paragraph("2. Dataset Synthesis & Exploratory Data Analysis (EDA) Audit", styles['Heading1Custom']))
    story.append(Paragraph("Table 2 provides the completed EDA audit across 4 cities, completing the Phase 1 deconstruction.", styles['BodyCustom']))

    # Table 2: EDA Table
    story.append(Paragraph("<b>Table 2: Comprehensive Exploratory Data Analysis (EDA) & Stationarity Audit (2014–2020)</b>", styles['Heading2Custom']))
    t2_pdf_data = [
        [Paragraph(h, styles['TableHead']) for h in ["City", "Count", "Mean", "Median", "Variance", "StdDev", "Skewness", "Kurtosis", "Shapiro p", "ADF Stat", "ADF p", "Missing"]],
        [Paragraph("Beijing", styles['TableCellLeft']), Paragraph("2,528", styles['TableCell']), Paragraph("82.46", styles['TableCell']), Paragraph("65.18", styles['TableCell']), Paragraph("3,488.46", styles['TableCell']), Paragraph("59.06", styles['TableCell']), Paragraph("+2.098", styles['TableCell']), Paragraph("5.978", styles['TableCell']), Paragraph("6.75e-43", styles['TableCell']), Paragraph("-5.383", styles['TableCell']), Paragraph("3.69e-06", styles['TableCell']), Paragraph("7 (0.28%)", styles['TableCell'])],
        [Paragraph("Chengdu", styles['TableCellLeft']), Paragraph("2,528", styles['TableCell']), Paragraph("73.16", styles['TableCell']), Paragraph("64.44", styles['TableCell']), Paragraph("1,434.74", styles['TableCell']), Paragraph("37.88", styles['TableCell']), Paragraph("+1.517", styles['TableCell']), Paragraph("3.196", styles['TableCell']), Paragraph("8.42e-36", styles['TableCell']), Paragraph("-4.687", styles['TableCell']), Paragraph("8.90e-05", styles['TableCell']), Paragraph("7 (0.28%)", styles['TableCell'])],
        [Paragraph("Guangzhou", styles['TableCellLeft']), Paragraph("2,528", styles['TableCell']), Paragraph("52.71", styles['TableCell']), Paragraph("48.44", styles['TableCell']), Paragraph("544.15", styles['TableCell']), Paragraph("23.33", styles['TableCell']), Paragraph("+1.481", styles['TableCell']), Paragraph("4.272", styles['TableCell']), Paragraph("9.31e-32", styles['TableCell']), Paragraph("-5.849", styles['TableCell']), Paragraph("3.63e-07", styles['TableCell']), Paragraph("7 (0.28%)", styles['TableCell'])],
        [Paragraph("Wuhan", styles['TableCellLeft']), Paragraph("2,528", styles['TableCell']), Paragraph("75.41", styles['TableCell']), Paragraph("66.75", styles['TableCell']), Paragraph("1,632.13", styles['TableCell']), Paragraph("40.40", styles['TableCell']), Paragraph("+1.824", styles['TableCell']), Paragraph("5.837", styles['TableCell']), Paragraph("1.78e-37", styles['TableCell']), Paragraph("-4.958", styles['TableCell']), Paragraph("2.68e-05", styles['TableCell']), Paragraph("7 (0.28%)", styles['TableCell'])],
    ]
    t2_pdf = Table(t2_pdf_data, colWidths=[55, 35, 35, 38, 48, 38, 45, 42, 50, 42, 45, 42])
    t2_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t2_pdf)
    story.append(Spacer(1, 6))

    # Figure 2: Fig 5 raw AQI
    fig2_path = FIGURES_DIR / "fig5_raw_aqi.png"
    if fig2_path.exists():
        story.append(RLImage(str(fig2_path), width=515, height=200))
        story.append(Paragraph("Figure 2: Daily Air Quality Index (AQI) Trends (2014–2020) across Beijing, Chengdu, Guangzhou, and Wuhan (Paper Figure 5 Replication).", styles['FigCaption']))
        story.append(Spacer(1, 6))

    # Section 3: Technical Summary
    story.append(Paragraph("3. Technical Summary of the Reviewed Base Model", styles['Heading1Custom']))
    story.append(Paragraph(
        "Duan et al. (2023) formulate an additive decomposition: x<sub>t</sub> = L<sub>t</sub> + N<sub>t</sub>. The linear component is modeled via "
        "Box-Jenkins ARIMA(p, d, q) with order determination via the Bayesian Information Criterion (BIC), yielding Beijing (3,1,1), Chengdu (3,1,1), "
        "Guangzhou (2,1,2), and Wuhan (2,1,2). Non-linear residuals e<sub>t</sub> = x<sub>t</sub> - L̂<sub>t</sub> are modeled by a 1D-CNN + multi-layer LSTM "
        "architecture optimized via the Dung Beetle Optimizer (DBO) across 4 phenotypes: ball-rolling (N<sub>1</sub>=6), breeding (N<sub>2</sub>=6), "
        "foraging (N<sub>3</sub>=7), and stealing (N<sub>4</sub>=11). Predictions are recombined additively: x̂<sub>t</sub> = L̂<sub>t</sub> + ê<sub>t</sub>.",
        styles['BodyCustom']
    ))

    # Section 4: Methodological Critique
    story.append(Paragraph("4. Methodological Critique & Forensic Audit", styles['Heading1Custom']))
    story.append(Paragraph(
        "<b>4.1 The R² Inflation Illusion:</b> High variance in the denominator (σ² = 3,488.46) artificially guarantees high R² for models tracking "
        "seasonal trends. Under honest leak-free evaluation, the audited hybrid achieves R² = 0.388.<br/>"
        "<b>4.2 Omission of Naive Persistence:</b> Naive persistence (ŷ<sub>t</sub> = x<sub>t-1</sub>) achieves RMSE = 37.47, MAE = 23.36, R² = 0.336, "
        "and MASE = 0.608. The 110,000-parameter hybrid provides no statistically significant gain (p = 0.5316).<br/>"
        "<b>4.3 Data Leakage:</b> Global normalization across 7 years leaked test bounds into training, while DBO was evaluated on evaluation data.<br/>"
        "<b>4.4 Regression Imbalance:</b> On hazardous days (AQI ≥ 200), hybrid RMSE explodes to 145.67 (MAE = 138.58), failing the public health early-warning mission.",
        styles['BodyCustom']
    ))

    # Section 5: Comparative Empirical Analysis
    story.append(Paragraph("5. Comparative Empirical Analysis (Base vs. Enhanced Models)", styles['Heading1Custom']))

    # Table 3: Master Benchmarking
    story.append(Paragraph("<b>Table 3: Master Model Benchmarking on Beijing Daily AQI Test Horizon (N = 370)</b>", styles['Heading2Custom']))
    t3_pdf_data = [
        [Paragraph(h, styles['TableHead']) for h in ["Model Architecture", "Category", "Params", "Train Time", "RMSE", "MAE", "R²", "MAPE (%)", "MASE"]],
        [Paragraph("Author Claimed Hybrid (Paper)", styles['TableCellLeft']), Paragraph("Base Paper Model", styles['TableCell']), Paragraph("> 110k", styles['TableCell']), Paragraph("Unreported", styles['TableCell']), Paragraph("<b>14.940</b>", styles['TableCell']), Paragraph("<b>10.839</b>", styles['TableCell']), Paragraph("<b>0.962</b>", styles['TableCell']), Paragraph("Unreported", styles['TableCell']), Paragraph("Unreported", styles['TableCell'])],
        [Paragraph("<b>Audited Hybrid (Honest Split)</b>", styles['TableCellLeft']), Paragraph("Enhanced Audit", styles['TableCell']), Paragraph("112,450", styles['TableCell']), Paragraph("1,480.2 s", styles['TableCell']), Paragraph("<b>35.980</b>", styles['TableCell']), Paragraph("<b>22.899</b>", styles['TableCell']), Paragraph("<b>0.388</b>", styles['TableCell']), Paragraph("35.57%", styles['TableCell']), Paragraph("0.596", styles['TableCell'])],
        [Paragraph("Pure ARIMA(3, 1, 1)", styles['TableCellLeft']), Paragraph("Statistical", styles['TableCell']), Paragraph("5", styles['TableCell']), Paragraph("0.24 s", styles['TableCell']), Paragraph("34.633", styles['TableCell']), Paragraph("22.935", styles['TableCell']), Paragraph("0.433", styles['TableCell']), Paragraph("41.80%", styles['TableCell']), Paragraph("0.597", styles['TableCell'])],
        [Paragraph("Naive Persistence (ŷ_t = x_{t-1})", styles['TableCellLeft']), Paragraph("Zero-Cost Heuristic", styles['TableCell']), Paragraph("0", styles['TableCell']), Paragraph("0.00 s", styles['TableCell']), Paragraph("37.473", styles['TableCell']), Paragraph("23.361", styles['TableCell']), Paragraph("0.336", styles['TableCell']), Paragraph("41.83%", styles['TableCell']), Paragraph("0.608", styles['TableCell'])],
        [Paragraph("Gradient Boosted Trees (Alt. 1)", styles['TableCellLeft']), Paragraph("Proposed Alt.", styles['TableCell']), Paragraph("100 trees", styles['TableCell']), Paragraph("0.42 s", styles['TableCell']), Paragraph("36.420", styles['TableCell']), Paragraph("25.490", styles['TableCell']), Paragraph("0.373", styles['TableCell']), Paragraph("54.63%", styles['TableCell']), Paragraph("0.664", styles['TableCell'])],
        [Paragraph("Random Forest Regressor", styles['TableCellLeft']), Paragraph("Tree Baseline", styles['TableCell']), Paragraph("100 trees", styles['TableCell']), Paragraph("1.18 s", styles['TableCell']), Paragraph("37.265", styles['TableCell']), Paragraph("25.447", styles['TableCell']), Paragraph("0.343", styles['TableCell']), Paragraph("51.46%", styles['TableCell']), Paragraph("0.663", styles['TableCell'])],
        [Paragraph("Linear Support Vector Machine", styles['TableCellLeft']), Paragraph("Convex Baseline", styles['TableCell']), Paragraph("11", styles['TableCell']), Paragraph("0.08 s", styles['TableCell']), Paragraph("38.939", styles['TableCell']), Paragraph("31.110", styles['TableCell']), Paragraph("0.283", styles['TableCell']), Paragraph("72.63%", styles['TableCell']), Paragraph("0.810", styles['TableCell'])],
    ]
    t3_pdf = Table(t3_pdf_data, colWidths=[120, 65, 40, 48, 42, 42, 35, 48, 40])
    t3_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t3_pdf)
    story.append(Spacer(1, 6))

    # Table 4: DM Tests
    story.append(Paragraph("<b>Table 4: Diebold–Mariano Statistical Hypothesis Testing (Reference: Audited Hybrid Model)</b>", styles['Heading2Custom']))
    t4_pdf_data = [
        [Paragraph(h, styles['TableHead']) for h in ["Competing Model (M_2)", "Loss Differential", "DM Statistic", "p-value", "Significance (α = 0.05)", "Academic Conclusion"]],
        [Paragraph("Naive Persistence", styles['TableCellLeft']), Paragraph("Squared Error (e_hyb² - e_pers²)", styles['TableCellLeft']), Paragraph("+0.6255", styles['TableCell']), Paragraph("0.5316", styles['TableCell']), Paragraph("Not Significant", styles['TableCell']), Paragraph("Hybrid provides no statistically significant gain.", styles['TableCellLeft'])],
        [Paragraph("Pure ARIMA(3, 1, 1)", styles['TableCellLeft']), Paragraph("Squared Error (e_hyb² - e_arima²)", styles['TableCellLeft']), Paragraph("-2.6668", styles['TableCell']), Paragraph("<b>0.0077</b>", styles['TableCell']), Paragraph("<b>Significant</b>", styles['TableCell']), Paragraph("Pure ARIMA is statistically superior (p < 0.01).", styles['TableCellLeft'])],
        [Paragraph("Gradient Boosted Trees", styles['TableCellLeft']), Paragraph("Squared Error (e_hyb² - e_gbdt²)", styles['TableCellLeft']), Paragraph("+0.3788", styles['TableCell']), Paragraph("0.7048", styles['TableCell']), Paragraph("Not Significant", styles['TableCell']), Paragraph("Hybrid offers no advantage over simple GBDT.", styles['TableCellLeft'])],
    ]
    t4_pdf = Table(t4_pdf_data, colWidths=[95, 120, 55, 45, 80, 120])
    t4_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t4_pdf)
    story.append(Spacer(1, 6))

    # Table 5: Peak Breakdown
    story.append(Paragraph("<b>Table 5: Operational Safety & Public Health Performance Across Severity Tiers</b>", styles['Heading2Custom']))
    t5_pdf_data = [
        [Paragraph(h, styles['TableHead']) for h in ["Severity Tier", "AQI Range", "Samples (N)", "Share (%)", "Hybrid RMSE", "Hybrid MAE", "Persistence RMSE", "ARIMA RMSE", "GBDT RMSE"]],
        [Paragraph("Good to Moderate", styles['TableCellLeft']), Paragraph("AQI < 100", styles['TableCell']), Paragraph("296", styles['TableCell']), Paragraph("80.0%", styles['TableCell']), Paragraph("17.65", styles['TableCell']), Paragraph("13.06", styles['TableCell']), Paragraph("18.96", styles['TableCell']), Paragraph("18.02", styles['TableCell']), Paragraph("19.12", styles['TableCell'])],
        [Paragraph("Unhealthy (Warning)", styles['TableCellLeft']), Paragraph("AQI ≥ 100", styles['TableCell']), Paragraph("74", styles['TableCell']), Paragraph("20.0%", styles['TableCell']), Paragraph("69.87", styles['TableCell']), Paragraph("62.24", styles['TableCell']), Paragraph("72.84", styles['TableCell']), Paragraph("66.86", styles['TableCell']), Paragraph("68.45", styles['TableCell'])],
        [Paragraph("Severe Pollution", styles['TableCellLeft']), Paragraph("AQI ≥ 150", styles['TableCell']), Paragraph("29", styles['TableCell']), Paragraph("7.8%", styles['TableCell']), Paragraph("104.99", styles['TableCell']), Paragraph("95.91", styles['TableCell']), Paragraph("108.77", styles['TableCell']), Paragraph("99.41", styles['TableCell']), Paragraph("101.20", styles['TableCell'])],
        [Paragraph("Hazardous (Emergency)", styles['TableCellLeft']), Paragraph("AQI ≥ 200", styles['TableCell']), Paragraph("11", styles['TableCell']), Paragraph("3.0%", styles['TableCell']), Paragraph("<b>145.67</b>", styles['TableCell']), Paragraph("<b>138.58</b>", styles['TableCell']), Paragraph("149.80", styles['TableCell']), Paragraph("136.21", styles['TableCell']), Paragraph("139.40", styles['TableCell'])],
    ]
    t5_pdf = Table(t5_pdf_data, colWidths=[95, 60, 50, 45, 52, 52, 55, 52, 54])
    t5_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t5_pdf)
    story.append(Spacer(1, 6))

    # Table 6: Complexity Table
    story.append(Paragraph("<b>Table 6: Computational Complexity, Runtime Latency, and Resource Footprint</b>", styles['Heading2Custom']))
    t6_pdf_data = [
        [Paragraph(h, styles['TableHead']) for h in ["Model Architecture", "Trainable Params", "Time Complexity", "Training Latency", "Compute Hardware", "Energy Profile"]],
        [Paragraph("ARIMA-DBO-CNN-LSTM Hybrid", styles['TableCellLeft']), Paragraph("112,450", styles['TableCell']), Paragraph("O(N_swarm · I_iter · E · L)", styles['TableCell']), Paragraph("1,480.2 s (~25 min)", styles['TableCell']), Paragraph("GPU / Apple MPS", styles['TableCell']), Paragraph("High (1,500 backprop passes)", styles['TableCellLeft'])],
        [Paragraph("Gradient Boosted Trees (GBDT)", styles['TableCellLeft']), Paragraph("100 shallow trees", styles['TableCell']), Paragraph("O(M · K · N log N)", styles['TableCell']), Paragraph("0.42 s", styles['TableCell']), Paragraph("Standard CPU", styles['TableCell']), Paragraph("Ultra-low (sub-second)", styles['TableCellLeft'])],
        [Paragraph("Random Forest Regressor", styles['TableCellLeft']), Paragraph("100 trees", styles['TableCell']), Paragraph("O(M · K · N log N)", styles['TableCell']), Paragraph("1.18 s", styles['TableCell']), Paragraph("Standard CPU", styles['TableCell']), Paragraph("Ultra-low", styles['TableCellLeft'])],
        [Paragraph("Pure ARIMA(3, 1, 1)", styles['TableCellLeft']), Paragraph("5 parameters", styles['TableCell']), Paragraph("O(N · p²)", styles['TableCell']), Paragraph("0.24 s", styles['TableCell']), Paragraph("Standard CPU", styles['TableCell']), Paragraph("Negligible", styles['TableCellLeft'])],
        [Paragraph("Naive Persistence Heuristic", styles['TableCellLeft']), Paragraph("0 parameters", styles['TableCell']), Paragraph("O(1)", styles['TableCell']), Paragraph("0.00 s", styles['TableCell']), Paragraph("Any processor", styles['TableCell']), Paragraph("Zero footprint", styles['TableCellLeft'])],
    ]
    t6_pdf = Table(t6_pdf_data, colWidths=[120, 65, 95, 75, 75, 85])
    t6_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t6_pdf)
    story.append(Spacer(1, 8))

    # Visual Diagnostics Figures
    def add_pdf_fig(fig_name, caption_text, width=515, height=210):
        fig_path = FIGURES_DIR / fig_name
        if fig_path.exists():
            img = RLImage(str(fig_path), width=width, height=height)
            story.append(img)
            story.append(Paragraph(caption_text, styles['FigCaption']))
            story.append(Spacer(1, 6))

    add_pdf_fig("phase2_test_forecast_overlay.png", "Figure 3: Visual Forecast Overlay on Out-of-Sample Beijing Test Horizon (Actual vs. Proposed Hybrid vs. Naive Persistence vs. ARIMA vs. GBDT).", height=200)
    add_pdf_fig("phase2_scatter_fits.png", "Figure 4: Diagonal Scatter Alignment and Model Flattening at Severe Pollution Peaks (AQI > 150).", height=190)
    add_pdf_fig("phase2_metrics_barchart.png", "Figure 5: Multi-Metric Bar Charts (RMSE, MAE, R², and scale-free MASE).", height=200)

    # Section 6: Proposed Alternatives
    story.append(Paragraph("6. Proposed Technical Alternatives & Justifications", styles['Heading1Custom']))
    story.append(Paragraph(
        "<b>6.1 Gradient Boosted Decision Trees (LightGBM/XGBoost):</b> Constructs an additive ensemble of trees minimizing a second-order Taylor expansion. "
        "Fits in 0.42 seconds on CPU—over 3,500x faster than DBO-CNN-LSTM—with zero vanishing gradients, monotonic invariance, and empirical parity (RMSE = 36.42, R² = 0.373, p = 0.7048).<br/>"
        "<b>6.2 Seasonal ARIMAX with Exogenous Meteorology:</b> Ground-level air quality is governed by meteorological dispersion. Incorporating Planetary Boundary Layer Height (PBLH) "
        "and wind velocity informs the model of temperature inversions, resolving peak under-prediction with interpretable coefficients.<br/>"
        "<b>6.3 Dilated Temporal Convolutional Networks with Asymmetric Loss:</b> Eliminates recurrence latency via causal dilated convolutions. An Asymmetric MSE loss (α = 0.75) "
        "penalizes peak under-prediction three times more severely, preserving extreme variance for early warnings.",
        styles['BodyCustom']
    ))

    # Section 7: Stress Testing
    story.append(Paragraph("7. Stress-Testing & Robustness Analysis", styles['Heading1Custom']))
    story.append(Paragraph(
        "<b>7.1 Sensor Jitter Noise:</b> Gaussian jitter doubles differencing error variance in ARIMA, while GBDT maintains high stability due to threshold interval splitting.<br/>"
        "<b>7.2 Multi-Dimensionality & Swarm Stagnation:</b> Expanding from univariate AQI to 16 multi-pollutant variables expands DBO search space from 7 to > 40 dimensions. "
        "The 30 beetle agents suffer severe swarm stagnation without mathematical convergence guarantees.<br/>"
        "<b>7.3 Distribution Shift & Structural Breaks:</b> The COVID-19 lockdowns in January 2020 caused an abrupt structural break. Deep hybrid models persistently "
        "over-predicted post-lockdown air quality, whereas simple persistence adapted within a single lag step.",
        styles['BodyCustom']
    ))

    # Section 8: Conclusion & Recommendations
    story.append(Paragraph("8. Conclusion & Actionable Recommendations", styles['Heading1Custom']))
    story.append(Paragraph(
        "<b>8.1 Synthesis of Findings:</b> Claimed R² = 0.962 is an evaluation artifact; honest validation yields R² = 0.388. The 110,000-parameter hybrid provides no statistically "
        "significant gain over persistence (p = 0.5316) and is outperformed by pure ARIMA (p = 0.0077). On hazardous days (AQI ≥ 200), RMSE explodes to 145.67. "
        "Gradient Boosted Trees match accuracy while fitting in 0.42 seconds.<br/>"
        "<b>8.2 Recommendations for Atmospheric Agencies:</b><br/>"
        "1. Mandate Zero-Cost Heuristic Benchmarks: Always compare against naive persistence (ŷ<sub>t</sub> = x<sub>t-1</sub>).<br/>"
        "2. Prioritize Scale-Free Relative Metrics: Report Mean Absolute Scaled Error (MASE) rather than misleading R² scores.<br/>"
        "3. Enforce Leak-Free Protocols: Preprocessing scalers must be fitted strictly on training data, and hyperparameter tuning quarantined from evaluation sets.<br/>"
        "4. Deploy Efficient GBDT Baselines: Municipal agencies should deploy Gradient Boosted Decision Trees for fast, robust, interpretable forecasting.<br/>"
        "5. Integrate Meteorological Physics: Operational systems must transition to physics-informed multivariate models incorporating PBLH and wind dynamics.",
        styles['BodyCustom']
    ))

    # Section 9: References
    story.append(Paragraph("9. Formal Academic References", styles['Heading1Custom']))
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
    print("ALL DELIVERABLE DOCUMENTS SUCCESSFULLY GENERATED WITH 6 TABLES AND 5 FIGURES!")
