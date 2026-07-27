"""Build the compliant three-page improved FMRG final report."""

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
METRICS_PATH = ROOT / "results/improved_submission/metrics.json"
FIGURES = ROOT / "results/improved_submission/figures"
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
            spaceAfter=7,
        ),
        "title": ParagraphStyle(
            "Title",
            fontName="Arial-Bold",
            fontSize=23,
            leading=26,
            textColor=NAVY,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName="Arial",
            fontSize=10,
            leading=12,
            textColor=MUTED,
            spaceAfter=7,
        ),
        "h1": ParagraphStyle(
            "H1",
            fontName="Arial-Bold",
            fontSize=15,
            leading=17,
            textColor=NAVY,
            spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "H2",
            fontName="Arial-Bold",
            fontSize=11,
            leading=13,
            textColor=BLUE,
            spaceBefore=3,
            spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName="Arial",
            fontSize=10,
            leading=11.8,
            textColor=INK,
            spaceAfter=3,
        ),
        "small": ParagraphStyle(
            "Small",
            fontName="Arial",
            fontSize=10,
            leading=11.3,
            textColor=MUTED,
        ),
        "callout": ParagraphStyle(
            "Callout",
            fontName="Arial-Bold",
            fontSize=10,
            leading=12,
            textColor=NAVY,
        ),
        "metric": ParagraphStyle(
            "Metric",
            fontName="Arial-Bold",
            fontSize=16,
            leading=17,
            textColor=BLUE,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            fontName="Arial",
            fontSize=10,
            leading=10.8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "table": ParagraphStyle(
            "Table",
            fontName="Arial",
            fontSize=10,
            leading=11,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            fontName="Arial-Bold",
            fontSize=10,
            leading=11,
            textColor=WHITE,
        ),
    }


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#C8D0D6"))
    canvas.line(inch, 0.63 * inch, 7.5 * inch, 0.63 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Arial", 8)
    canvas.drawString(inch, 0.44 * inch, "FMRG Data Challenge - Final Submission")
    canvas.drawRightString(7.5 * inch, 0.44 * inch, str(doc.page))
    canvas.restoreState()


def section(title, style):
    return KeepTogether(
        [
            Paragraph(title, style["h1"]),
            HRFlowable(width="100%", thickness=1.3, color=BLUE, spaceAfter=5),
        ]
    )


def report_table(rows, style, widths):
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
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD3D9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def metric_strip(items, style):
    cells = [
        [
            Paragraph(value, style["metric"]),
            Paragraph(label, style["metric_label"]),
        ]
        for value, label in items
    ]
    result = Table([cells], colWidths=[6.5 * inch / len(items)] * len(items))
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#CAD3DA")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CAD3DA")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return result


def bullet(text, style):
    return Paragraph(f"&#8226;&nbsp; {text}", style["body"])


def build_report():
    register_fonts()
    style = styles()
    metrics = json.loads(METRICS_PATH.read_text())
    incumbent = metrics["incumbent"]["metrics"]
    promoted = metrics["candidates"]["nested_metrics"]
    uncertainty = metrics["uncertainty"]

    improvement = 100 * (
        incumbent["track_balanced_width_mae_mm"]
        - promoted["track_balanced_width_mae_mm"]
    ) / incumbent["track_balanced_width_mae_mm"]
    worst_improvement = 100 * (
        incumbent["worst_track_width_mae_mm"]
        - promoted["worst_track_width_mae_mm"]
    ) / incumbent["worst_track_width_mae_mm"]
    boundary_improvement = 100 * (
        incumbent["mean_boundary_mae_mm"] - promoted["mean_boundary_mae_mm"]
    ) / incumbent["mean_boundary_mae_mm"]

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=0.8 * inch,
        title="FMRG Final Submission - Hierarchical Local Geometry Prediction",
        author="Team Submission",
    )
    story = []

    story.extend(
        [
            Paragraph("NSF FUTURE MANUFACTURING DATA CHALLENGE", style["kicker"]),
            Paragraph(
                "Predicting local DED track geometry from thermal history",
                style["title"],
            ),
            Paragraph(
                "Hierarchical condition baselines, causal multiscale descriptors, physically "
                "consistent boundaries, and nested four-track validation",
                style["subtitle"],
            ),
            metric_strip(
                [
                    (f"{promoted['track_balanced_width_mae_mm']:.3f} mm", "four-track width MAE"),
                    (f"{improvement:.1f}%", "lower than direct Ridge"),
                    (f"{promoted['mean_boundary_mae_mm']:.3f} mm", "mean boundary MAE"),
                    (f"{uncertainty['conditional']['coverage'] * 100:.1f}%", "conditional coverage"),
                ],
                style,
            ),
            Spacer(1, 7),
            section("Executive summary", style),
            Paragraph(
                "A laser track is a spatial signal, not one average width. We align each active "
                "thermal frame to physical x, extract the local track center and connected "
                "left/right profilometry boundaries, and evaluate every track as an untouched "
                "condition. The promoted nested selector reduces track-balanced width MAE from "
                f"<b>{incumbent['track_balanced_width_mae_mm']:.3f} mm</b> to "
                f"<b>{promoted['track_balanced_width_mae_mm']:.3f} mm</b> ({improvement:.1f}%), "
                f"worst-track MAE by {worst_improvement:.1f}%, and boundary MAE by "
                f"{boundary_improvement:.1f}%.",
                style["body"],
            ),
            Paragraph(
                "The key modeling change is to predict condition-level baseline center and "
                "log-width from track summaries, then predict local residuals from frame history. "
                "This directly addresses cross-power mean shifts and over-smoothed local signals.",
                style["body"],
            ),
            section("Problem formulation and data alignment", style),
            Paragraph(
                "For each valid x location, targets are width, center, left boundary, and right "
                "boundary. Profilometry cross-sections are robustly detrended; connected "
                "30%-of-peak crossings around the bead define the boundaries. Thermal frames map "
                "to x using 10 mm/s at 50 fps (0.2 mm/frame). Frames farther than 0.10 mm from "
                "valid geometry are excluded, and only the 24-96 mm steady-state region is scored.",
                style["body"],
            ),
            Paragraph("Leakage controls", style["h2"]),
            bullet(
                "Outer leave-one-track-out tests hold out Tracks 8, 10, 14, and 21 in turn.",
                style,
            ),
            bullet(
                "Feature family, estimator, preprocessing, and interval calibration are chosen "
                "inside the other three tracks by inner leave-one-track-out validation.",
                style,
            ),
            bullet(
                "Thermal history is causal: current and earlier frames only. Headline scores are "
                "unweighted means of per-track outer metrics.",
                style,
            ),
            Paragraph("Generative AI use", style["h2"]),
            Paragraph(
                "OpenAI Codex assisted with code review, tests, debugging, and document layout. "
                "All numbers come from tracked analysis code and saved outer-fold predictions; "
                "AI did not supply or alter experimental measurements.",
                style["body"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            section("Method: preserve condition shifts and local variation", style),
            Paragraph(
                "<b>Thermal descriptors.</b> Instantaneous features cover hot-region area and "
                "axes, temperature percentiles, thermal mass, gradients, asymmetry, cooling-tail "
                "area/decay, centroid velocity, and pool-shape change. Causal 5-, 10-, and "
                "20-frame windows add means, slopes, changes, persistence above fixed temperature "
                "thresholds, and history-availability flags. Robust within-track normalization "
                "isolates local deviations without exposing held-out geometry.",
                style["body"],
            ),
            Paragraph(
                "<b>Constrained multi-output prediction.</b> Track thermal summaries predict "
                "baseline center and baseline log-width. Local features jointly predict center "
                "and log-width residuals. Exponentiating log-width guarantees positive width; "
                "left/right are reconstructed from one shared center and width, so boundaries "
                "remain ordered.",
                style["body"],
            ),
            Paragraph(
                "<b>Low-capacity model ladder.</b> Inner folds compare Ridge, elastic net, partial "
                "least squares, spline-Ridge, and a Gaussian process. Within 0.02 mm of the best "
                "inner MAE, selection favors lower variation-scale error and higher residual "
                "correlation. Spline-Ridge with normalized compact thermal features wins three "
                "outer folds; Track 14 independently chooses normalized multiscale Ridge.",
                style["body"],
            ),
            Spacer(1, 4),
            Image(str(FIGURES / "nested_outer_predictions.png"), width=6.5 * inch, height=3.28 * inch),
            Paragraph(
                "Figure 1. Measured and predicted local width for four untouched outer tracks. "
                "Each panel is produced by a model selected without that track's labels.",
                style["small"],
            ),
            Spacer(1, 5),
            report_table(
                [
                    ["Metric", "Direct Ridge", "Hierarchical selector", "Change"],
                    [
                        "Track-balanced width MAE",
                        f"{incumbent['track_balanced_width_mae_mm']:.3f} mm",
                        f"{promoted['track_balanced_width_mae_mm']:.3f} mm",
                        f"-{improvement:.1f}%",
                    ],
                    [
                        "Worst-track width MAE",
                        f"{incumbent['worst_track_width_mae_mm']:.3f} mm",
                        f"{promoted['worst_track_width_mae_mm']:.3f} mm",
                        f"-{worst_improvement:.1f}%",
                    ],
                    [
                        "Mean left/right boundary MAE",
                        f"{incumbent['mean_boundary_mae_mm']:.3f} mm",
                        f"{promoted['mean_boundary_mae_mm']:.3f} mm",
                        f"-{boundary_improvement:.1f}%",
                    ],
                    [
                        "Residual correlation",
                        f"{incumbent['residual_correlation']:.3f}",
                        f"{promoted['residual_correlation']:.3f}",
                        f"{promoted['residual_correlation'] / incumbent['residual_correlation']:.2f}x",
                    ],
                    [
                        "Predicted/measured variation std.",
                        f"{incumbent['variation_std_ratio']:.3f}",
                        f"{promoted['variation_std_ratio']:.3f}",
                        "+0.142",
                    ],
                ],
                style,
                [2.15 * inch, 1.35 * inch, 1.7 * inch, 1.3 * inch],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            section("Uncertainty, interpretation, and limits", style),
            Image(str(FIGURES / "before_after_scorecard.png"), width=6.5 * inch, height=2.48 * inch),
            Paragraph(
                "Figure 2. The promoted model improves accuracy, boundary position, and local "
                "variation fidelity under the same four-track outer protocol.",
                style["small"],
            ),
            Paragraph("Conditional uncertainty", style["h2"]),
            Paragraph(
                "Normalized conformal intervals scale by predicted local difficulty. Conditional "
                f"coverage is <b>{uncertainty['conditional']['coverage'] * 100:.1f}%</b> with "
                f"<b>{uncertainty['conditional']['mean_width_mm']:.3f} mm</b> mean width, versus "
                f"{uncertainty['global']['coverage'] * 100:.1f}% and "
                f"{uncertainty['global']['mean_width_mm']:.3f} mm for a fixed global interval. "
                "Mean width expands from "
                f"{uncertainty['conditional_by_difficulty']['low']['mean_width_mm']:.3f} mm in "
                f"easy regions to {uncertainty['conditional_by_difficulty']['high']['mean_width_mm']:.3f} mm "
                "in difficult regions.",
                style["body"],
            ),
            Paragraph("Interpretable links and substrate evidence", style["h2"]),
            Paragraph(
                "Track-level hot area, maximum temperature, thermal mass, and cooling-tail "
                "summaries anchor the condition baseline. Normalized pool shape, temperature, "
                "asymmetry, and recent history contribute local residual information. These are "
                "predictive associations. The available SEM is post-process; after masking the "
                "processed center, SEM is selected in <b>zero</b> outer folds. We therefore make "
                "no causal pre-process substrate claim.",
                style["body"],
            ),
            Paragraph("Limitations", style["h2"]),
            bullet(
                f"Track-balanced R-squared remains {promoted['track_balanced_width_r2']:.2f}; "
                "the result is not ready for closed-loop control.",
                style,
            ),
            bullet(
                "Only four independent tracks are available, so model ranking and interval "
                "coverage remain uncertain across new plates and recipes.",
                style,
            ),
            bullet(
                "The earlier 0.139 mm Track 21 result is a historical tuned split. Under the "
                f"stronger nested protocol, Track 21 MAE is "
                f"{promoted['per_track']['21']['width_mae_mm']:.3f} mm.",
                style,
            ),
            bullet(
                "Genuine pre-process SEM or surface measurements registered to physical x are "
                "needed to distinguish substrate-driven from process-driven variation.",
                style,
            ),
            Paragraph("Conclusion", style["h2"]),
            Paragraph(
                "Separating condition-level geometry from local residual variation produces a "
                "more accurate, less over-smoothed, physically consistent predictor. The next "
                "decisive study should add powers, plates, repeats, and registered pre-process "
                "surface measurements, then test the locked pipeline on blind tracks.",
                style["body"],
            ),
            Paragraph("Reproduction and sources", style["h2"]),
            Paragraph(
                "<font name='Courier'>PYTHONPATH=src LOKY_MAX_CPU_COUNT=1 MPLBACKEND=Agg "
                ".venv/bin/python scripts/run_improvement_experiments.py "
                "--raw-dir /path/to/data --cache-dir /path/to/cache "
                "--output-dir results/improved_submission</font><br/>"
                "Dataset: doi:10.5281/zenodo.21285367; paper: arXiv:2607.07965; "
                "locked metrics: results/improved_submission/metrics.json.",
                style["small"],
            ),
        ]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
