"""Create and execute the audited final-submission notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/03_final_submission_audited.ipynb"


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbformat.v4.new_code_cell(dedent(text).strip())


def build_notebook():
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        markdown(
            """
            # FMRG Final Submission: Audited Local Geometry Prediction

            This executed notebook is the compact submission record for predicting spatially
            varying DED track width and left/right boundaries from thermal history.

            **Locked protocol:** model selection and uncertainty calibration use Tracks 8, 10,
            and 14 only. Track 21 is opened once for final scoring.
            """
        ),
        markdown(
            """
            ## Reproducibility

            The full raw-data pipeline is implemented in `scripts/run_final_analysis.py`.
            Recreate the tracked outputs with:

            ```bash
            python scripts/run_final_analysis.py \
              --raw-dir /path/to/extracted/zenodo/data \
              --output-dir results/final_submission
            ```

            The official Zenodo archives were verified against the record's MD5 checksums before
            analysis. This notebook reads the tracked, locked results so it executes without
            redistributing the large raw archives.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import pandas as pd
            from IPython.display import Image, display

            ROOT = Path.cwd()
            if not (ROOT / "results/final_submission/metrics.json").exists():
                ROOT = ROOT.parent

            RESULTS = ROOT / "results/final_submission"
            metrics = json.loads((RESULTS / "metrics.json").read_text())
            metrics["data_split"]
            """
        ),
        markdown(
            """
            ## Audit corrections

            The prior report/deck asserted a 21.18 micrometer MAE, positive R-squared, a Random
            Forest result, and dominant SEM importance. Those claims were not produced by the
            supplied notebook. This submission replaces them with a reproduced notebook baseline
            and an audited pipeline evaluated on the same untouched Track 21 samples.
            """
        ),
        code(
            """
            baseline = metrics["baseline"]
            corrected = metrics["corrected"]

            comparison = pd.DataFrame(
                [
                    {
                        "model": "Notebook Gradient Boosting",
                        "development_CV_MAE_mm": baseline["development_oof_metrics"]["mae_mm"],
                        "Track21_MAE_mm": baseline["test_metrics"]["mae_mm"],
                        "Track21_RMSE_mm": baseline["test_metrics"]["rmse_mm"],
                        "Track21_R2": baseline["test_metrics"]["r2"],
                        "interval_coverage": baseline["test_interval_metrics"]["coverage"],
                        "interval_width_mm": baseline["test_interval_metrics"]["mean_width_mm"],
                    },
                    {
                        "model": "Audited Ridge alpha=10",
                        "development_CV_MAE_mm": corrected["development_oof_metrics"]["mae_mm"],
                        "Track21_MAE_mm": corrected["test_metrics"]["mae_mm"],
                        "Track21_RMSE_mm": corrected["test_metrics"]["rmse_mm"],
                        "Track21_R2": corrected["test_metrics"]["r2"],
                        "interval_coverage": corrected["test_interval_metrics"]["coverage"],
                        "interval_width_mm": corrected["test_interval_metrics"]["mean_width_mm"],
                    },
                ]
            )
            comparison.round(4)
            """
        ),
        code(
            """
            improvement = metrics["held_out_mae_improvement_percent"]
            print(f"Held-out MAE improvement: {improvement:.1f}%")
            print(
                "Interpretation: the audited model is better on the untouched condition, "
                "but negative R^2 and interval under-coverage prevent a deployment claim."
            )
            """
        ),
        markdown(
            """
            ## Local geometry target

            Each profilometer cross-section is robustly detrended. Connected 30%-of-peak crossings
            around the central bead maximum define the local left and right boundaries; width is
            their difference. Invalid acquisition gaps remain excluded. Thermal frames map to the
            physical x-axis using 10 mm/s at 50 fps, or 0.2 mm per frame.
            """
        ),
        code(
            """
            boundary = metrics["boundary_metrics"]
            pd.DataFrame(
                [
                    {"signal": "left boundary", **boundary["left"]},
                    {"signal": "right boundary", **boundary["right"]},
                    {
                        "signal": "mean boundary MAE",
                        "mae_mm": boundary["mean_boundary_mae_mm"],
                        "rmse_mm": float("nan"),
                        "r2": float("nan"),
                    },
                ]
            ).round(4)
            """
        ),
        markdown(
            """
            ## Model selection and SEM ablation

            Candidate models and feature families are compared by leave-one-track-out development
            cross-validation. Masked SEM is tested as a substrate-only feature family, not assumed
            beneficial.
            """
        ),
        code(
            """
            candidates = corrected["candidate_cv_mae_mm"]
            (
                pd.Series(candidates, name="grouped_CV_MAE_mm")
                .sort_values()
                .rename_axis("candidate")
                .to_frame()
                .round(4)
            )
            """
        ),
        code(
            """
            print("Selected model:", corrected["model_name"])
            print("Selected feature family:", corrected["feature_set"])
            print("Masked SEM selected:", metrics["masked_sem_selected"])
            """
        ),
        markdown(
            """
            ## Held-out spatial predictions and uncertainty

            The interval half-width is calibrated from grouped development residuals only. Observed
            Track 21 coverage is reported directly, even though it falls below the nominal 90%.
            """
        ),
        code(
            """
            display(
                Image(
                    filename=str(RESULTS / "figures/track21_held_out_comparison.png"),
                    width=1050,
                )
            )
            """
        ),
        markdown(
            """
            ## Interpretable thermal links

            Validation permutation importance identifies hot-region mean temperature and thermal
            mass as the strongest predictive associations. These features reflect melt-pool thermal
            state and short process memory; importance is not a causal substrate/process separation.
            """
        ),
        code(
            """
            display(
                Image(
                    filename=str(RESULTS / "figures/feature_importance.png"),
                    width=900,
                )
            )
            """
        ),
        markdown(
            """
            ## Honest conclusion

            - Track 21 local-width MAE improves from **0.159 mm to 0.139 mm** (12.3%).
            - Development grouped-CV MAE improves from **0.137 mm to 0.088 mm**.
            - Held-out R-squared remains **-0.58**.
            - Nominal 90% intervals cover **76.5%** of held-out samples.
            - Mean left/right boundary MAE is **0.174 mm**.
            - Masked SEM worsens grouped CV and is not selected.

            The contribution is a reproducible, leakage-controlled benchmark and spatial pipeline,
            not a closed-loop-ready controller.
            """
        ),
        markdown(
            """
            ## Generative AI disclosure

            OpenAI Codex assisted with code review, test generation, debugging, and artifact
            layout. Reported metrics were produced by the tracked analysis code and verified
            against saved predictions; AI did not supply or alter experimental measurements.
            """
        ),
        markdown(
            """
            ## Sources

            - Official dataset: https://doi.org/10.5281/zenodo.21285367
            - Dataset paper: https://arxiv.org/abs/2607.07965
            - Locked metrics: `results/final_submission/metrics.json`
            """
        ),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_notebook()
