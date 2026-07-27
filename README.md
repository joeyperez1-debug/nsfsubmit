# FMRG Improved Final Submission

This repository contains the final model and submission artifacts for
predicting spatially varying DED track width, center, and left/right boundaries
from thermal history.

The official multimodal dataset is hosted on Zenodo:
[10.5281/zenodo.21285367](https://doi.org/10.5281/zenodo.21285367).

## Nested four-track result

Tracks 8, 10, 14, and 21 are each held out once. Feature family, estimator,
preprocessing, and uncertainty calibration are selected by leave-one-track-out
validation using only the other three tracks.

| Metric | Direct Ridge | Hierarchical selector | Change |
|---|---:|---:|---:|
| Track-balanced width MAE | 0.187 mm | 0.163 mm | -13.1% |
| Worst-track width MAE | 0.308 mm | 0.219 mm | -28.9% |
| Mean boundary MAE | 0.180 mm | 0.148 mm | -17.6% |
| Residual correlation | 0.055 | 0.124 | 2.26× |
| Predicted/measured variation std. | 0.214 | 0.356 | +0.142 |

Conditional conformal intervals cover **91.4%** of outer samples with
**0.738 mm** mean width, versus **94.2%** and **0.824 mm** for a fixed global
interval. Track-balanced R² remains **-0.34**, so this is a stronger benchmark,
not a closed-loop-ready controller.

The earlier 0.139 mm Track 21 result is retained only as a historical tuned
split. Under the stronger nested protocol, Track 21 receives no special tuning
and scores 0.219 mm.

## Method

- Track-level thermal summaries predict baseline center and log-width.
- Causal local descriptors predict center and log-width residuals.
- Thermal history includes pool shape, cooling tail, motion, asymmetry,
  persistence, and 5-, 10-, and 20-frame statistics.
- Positive width and ordered boundaries are enforced by reconstruction from
  one shared center and exponentiated log-width.
- Ridge, elastic net, partial least squares, spline-Ridge, and Gaussian process
  candidates are selected inside nested track-level folds.
- Available SEM is post-process, the processed center is masked, and SEM is
  selected in zero outer folds. No causal substrate claim is made.

## Final deliverables

- `notebooks/03_final_submission_audited.ipynb` - executed final notebook.
- `deliverables/report/FMRG_Final_Report_Audited.pdf` - compliant three-page report.
- `deliverables/presentation/FMRG_Final_Submission_Audited.pptx` - editable deck.
- `deliverables/presentation/FMRG_Final_Submission_Audited.pdf` - exported deck.
- `results/improved_submission/` - nested metrics, predictions, and figures.
- `src/fmrg_submission/` - geometry, thermal, targets, evaluation, uncertainty,
  SEM ablation, and experiment code.
- `tests/` - regression tests for the critical pipeline logic.

The prior `notebooks/finalnotebook.ipynb` is retained as a legacy artifact for
comparison.

## Reproduce

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src LOKY_MAX_CPU_COUNT=1 MPLBACKEND=Agg \
  .venv/bin/python scripts/run_improvement_experiments.py \
  --raw-dir /path/to/extracted/zenodo/data \
  --cache-dir /path/to/cache \
  --output-dir results/improved_submission
python -m pytest
```

Build the final artifacts:

```bash
python scripts/build_final_notebook.py
python scripts/build_final_report.py
python scripts/build_submission_package.py
```

Raw Zenodo archives, extracted raw data, temporary renders, virtual
environments, and credentials are intentionally excluded.
