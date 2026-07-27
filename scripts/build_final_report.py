"""Build the compliant three-page audited FMRG final report."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results/final_submission/metrics.json"
FIGURES = ROOT / "results/final_submission/figures"
OUTPUT = ROOT / "deliverables/report/FMRG_Final_Report_Audited.pdf"

NAVY = colors.HexColor("#061E2B")
BLUE = colors.HexColor("#1769FF")
ORANGE = colors.HexColor("#E85D24")
INK = colors.HexColor("#1D2730")
MUTED = colors.HexColor("#5E6A73")
PALE = colors.HexColor("#EEF2F5")
WHITE = colors.white


def register_fonts():
    pdfmetrics.registerFont(
        TTFont("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("Arial-Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    )


def styles():
    return {
        "kicker": ParagraphStyle(
            "Kicker",
            fontName="Arial-Bold",
            fontSize=10,
            leading=12,
            textColor=BLUE,
            spaceAfter=8,
        ),
        "title": ParagraphStyle(
            "Title",
            fontName="Arial-Bold",
            fontSize=24,
            leading=27,
            textColor=NAVY,
            spaceAfter=7,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName="Arial",
            fontSize=10,
            leading=13,
            textColor=MUTED,
            spaceAfter=9,
        ),
        "h1": ParagraphStyle(
            "H1",
            fontName="Arial-Bold",
            fontSize=16,
            leading=19,
            textColor=NAVY,
            spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "H2",
            fontName="Arial-Bold",
            fontSize=11,
            leading=13,
            textColor=BLUE,
            spaceBefore=4,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName="Arial",
            fontSize=10,
            leading=12.2,
            textColor=INK,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "Small",
            fontName="Arial",
            fontSize=10,
            leading=11.5,
            textColor=MUTED,
        ),
        "callout": ParagraphStyle(
            "Callout",
            fontName="Arial-Bold",
            fontSize=10,
            leading=12.5,
            textColor=NAVY,
        ),
        "metric": ParagraphStyle(
            "Metric",
            fontName="Arial-Bold",
            fontSize=16,
            leading=18,
            textColor=BLUE,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            fontName="Arial",
            fontSize=10,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "table": ParagraphStyle(
            "Table",
            fontName="Arial",
            fontSize=10,
            leading=11.5,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            fontName="Arial-Bold",
            fontSize=10,
            leading=11.5,
            textColor=WHITE,
        ),
    }


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#C8D0D6"))
    canvas.line(inch, 0.63 * inch, 7.5 * inch, 0.63 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Arial", 8)
    canvas.drawString(inch, 0.44 * inch, "FMRG Data Challenge - Audited Final Submission")
    canvas.drawRightString(7.5 * inch, 0.44 * inch, str(doc.page))
    canvas.restoreState()


def section(title, style):
    return KeepTogether(
        [
            Paragraph(title, style["h1"]),
            HRFlowable(width="100%", thickness=1.4, color=BLUE, spaceAfter=6),
        ]
    )


def table(rows, style, widths):
    cells = []
    for row_index, row in enumerate(rows):
        cell_style = style["table_head"] if row_index == 0 else style["table"]
        cells.append([Paragraph(str(value), cell_style) for value in row])
    result = Table(cells, colWidths=widths, repeatRows=1)
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F7F9FA")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CDD4DA")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def metric_card(value, label, style):
    result = Table(
        [
            [Paragraph(value, style["metric"])],
            [Paragraph(label, style["metric_label"])],
        ],
        colWidths=[1.55 * inch],
        rowHeights=[0.28 * inch, 0.32 * inch],
    )
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD7DD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return result


def bullet(text, style):
    return Paragraph(f"- {text}", style["body"])


def build_report():
    register_fonts()
    style = styles()
    metrics = json.loads(METRICS_PATH.read_text())
    baseline = metrics["baseline"]
    corrected = metrics["corrected"]
    boundary = metrics["boundary_metrics"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title="FMRG Final Report - Audited Local Geometry Prediction",
        author="Team Submission",
    )
    story = [
        # Page 1 - result first.
        Paragraph("NSF FUTURE MANUFACTURING DATA CHALLENGE | FINAL SUBMISSION", style["kicker"]),
        Paragraph("Predicting local DED track width from thermal history", style["title"]),
        Paragraph(
            "Audited physical alignment, local boundaries, uncertainty, and held-out "
            "cross-condition evaluation.",
            style["subtitle"],
        ),
        HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=7),
        Paragraph("Executive result", style["h1"]),
        Paragraph(
            "On untouched Track 21, the audited thermal-history Ridge model reduces local-width "
            "MAE by <b>12.3%</b> relative to the original notebook Gradient Boosting baseline "
            "retrained under the same split. It is a better benchmark, not a control-ready model: "
            "held-out R^2 remains negative and the nominal 90% interval under-covers.",
            style["body"],
        ),
        Table(
            [[
                metric_card("0.139 mm", "Held-out MAE", style),
                metric_card("12.3%", "MAE reduction", style),
                metric_card("0.088 mm", "Grouped CV MAE", style),
                metric_card("76.5%", "Interval coverage", style),
            ]],
            colWidths=[1.62 * inch] * 4,
        ),
        Spacer(1, 7),
        table(
            [
                ["Model", "MAE", "RMSE", "R^2", "Coverage"],
                [
                    "Notebook Gradient Boosting",
                    f"{baseline['test_metrics']['mae_mm']:.3f} mm",
                    f"{baseline['test_metrics']['rmse_mm']:.3f} mm",
                    f"{baseline['test_metrics']['r2']:.2f}",
                    f"{100 * baseline['test_interval_metrics']['coverage']:.1f}%",
                ],
                [
                    "Audited Ridge alpha=10",
                    f"{corrected['test_metrics']['mae_mm']:.3f} mm",
                    f"{corrected['test_metrics']['rmse_mm']:.3f} mm",
                    f"{corrected['test_metrics']['r2']:.2f}",
                    f"{100 * corrected['test_interval_metrics']['coverage']:.1f}%",
                ],
            ],
            style,
            [2.25 * inch, 1.05 * inch, 1.05 * inch, 0.75 * inch, 1.4 * inch],
        ),
        Spacer(1, 7),
        Image(
            str(FIGURES / "track21_held_out_comparison.png"),
            width=6.45 * inch,
            height=2.55 * inch,
        ),
        Paragraph(
            "Figure 1. Measured local width, reproduced notebook baseline, audited prediction, "
            "and development-calibrated interval on held-out Track 21.",
            style["small"],
        ),
        PageBreak(),

        # Page 2 - method and audit.
        section("Method: local geometry without label leakage", style),
        Paragraph(
            "<b>Target and alignment.</b> Each profilometer cross-section is robustly detrended. "
            "Connected 30%-of-peak crossings around the central bead maximum define left and right "
            "boundaries; width is their difference. Invalid acquisition gaps remain excluded. "
            "Thermal frames map to physical x using 10 mm/s at 50 fps (0.2 mm/frame).",
            style["body"],
        ),
        Paragraph(
            "<b>Thermal history.</b> Features include hot-region area, bounding-box geometry, "
            "equivalent diameter, elongation, temperature percentiles, hot-region mean, thermal "
            "mass, gradients, asymmetry, deltas, one-frame lags, three-frame rolling means, and "
            "low-order physical-x harmonics.",
            style["body"],
        ),
        table(
            [
                ["Audit item", "Prior artifact", "Audited protocol"],
                ["Target", "Height/average language", "Spatial width plus both boundaries"],
                ["Alignment", "Normalized row position", "Physical x in millimeters"],
                ["Selection", "Random split / test reused", "Grouped CV on Tracks 8, 10, 14"],
                ["Final test", "No untouched condition", "Track 21 opened once"],
                ["SEM", "Assumed dominant", "Masked ablation; rejected when CV worsened"],
                ["Uncertainty", "Implied reliability", "Conformal interval plus observed coverage"],
            ],
            style,
            [1.15 * inch, 2.25 * inch, 3.1 * inch],
        ),
        Spacer(1, 6),
        Paragraph("Selection and interpretation", style["h2"]),
        Paragraph(
            "Ridge alpha=10 with thermal-only features minimizes leave-one-track-out development "
            "MAE (0.0876 mm). The best thermal-plus-masked-SEM candidate reaches 0.1135 mm, so SEM "
            "is not selected. With only four conditions, process and substrate effects are not "
            "identifiable; the report makes no causal substrate claim.",
            style["body"],
        ),
        Image(
            str(FIGURES / "feature_importance.png"),
            width=5.95 * inch,
            height=2.65 * inch,
        ),
        Paragraph(
            "Figure 2. Development-validation permutation importance. Hot-region temperature and "
            "thermal mass lead; importance indicates association, not causality.",
            style["small"],
        ),
        PageBreak(),

        # Page 3 - boundaries, limitations, reproducibility.
        section("Boundary results, uncertainty, and release gates", style),
        table(
            [
                ["Held-out signal", "MAE", "RMSE", "R^2"],
                [
                    "Left boundary",
                    f"{boundary['left']['mae_mm']:.3f} mm",
                    f"{boundary['left']['rmse_mm']:.3f} mm",
                    f"{boundary['left']['r2']:.2f}",
                ],
                [
                    "Right boundary",
                    f"{boundary['right']['mae_mm']:.3f} mm",
                    f"{boundary['right']['rmse_mm']:.3f} mm",
                    f"{boundary['right']['r2']:.2f}",
                ],
                ["Mean boundary MAE", f"{boundary['mean_boundary_mae_mm']:.3f} mm", "-", "-"],
            ],
            style,
            [2.55 * inch, 1.35 * inch, 1.35 * inch, 1.25 * inch],
        ),
        Spacer(1, 7),
        Paragraph("What is supported", style["h2"]),
        bullet("Held-out width MAE and RMSE improve under a strict, same-sample comparison.", style),
        bullet("Short thermal history generalizes better than masked SEM on this split.", style),
        bullet("Connected boundary extraction produces explicit local left/right predictions.", style),
        Paragraph("What remains open", style["h2"]),
        bullet("Held-out R^2 is -0.58; the unseen-condition shift still dominates local fit.", style),
        bullet("Nominal 90% intervals cover 76.5% of Track 21, below the target.", style),
        bullet("Track 21 profilometry is incomplete; only valid aligned regions are scored.", style),
        bullet("Four conditions cannot separate substrate-driven from process-driven variation.", style),
        Spacer(1, 4),
        Table(
            [[Paragraph(
                "Release gate: do not use this model for closed-loop control until interval coverage "
                "and positive held-out R^2 are demonstrated across additional powers, plates, and repeats.",
                style["callout"],
            )]],
            colWidths=[6.5 * inch],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE),
                    ("BOX", (0, 0), (-1, -1), 0.7, ORANGE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            ),
        ),
        Paragraph("Reproduce", style["h2"]),
        Paragraph(
            "<font name='Courier'>python scripts/run_final_analysis.py --raw-dir /path/to/data "
            "--output-dir results/final_submission</font><br/>"
            "<font name='Courier'>python -m pytest</font><br/>"
            "<font name='Courier'>python scripts/build_final_report.py</font>",
            style["small"],
        ),
        Paragraph("Generative AI disclosure", style["h2"]),
        Paragraph(
            "OpenAI Codex assisted with code review, test generation, debugging, and artifact "
            "layout. Reported metrics were produced by the tracked analysis code and verified "
            "against saved predictions; AI did not supply or alter experimental measurements.",
            style["small"],
        ),
        Paragraph("Sources and submission record", style["h2"]),
        Paragraph(
            "Dataset: https://doi.org/10.5281/zenodo.21285367<br/>"
            "Paper: https://arxiv.org/abs/2607.07965<br/>"
            "Locked metrics: results/final_submission/metrics.json<br/>"
            "Executed notebook: notebooks/03_final_submission_audited.ipynb",
            style["small"],
        ),
        Spacer(1, 6),
        Paragraph(
            "<b>Bottom line:</b> the audited model is measurably better than the reproduced notebook "
            "baseline while the negative R^2 and interval under-coverage keep the claim honest.",
            style["body"],
        ),
    ]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
