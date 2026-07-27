# FMRG Audited Final Submission

This repository contains the final model and submission artifacts for
predicting spatially varying DED track width and left/right boundaries from
thermal history.

The official multimodal dataset is hosted on Zenodo:
[10.5281/zenodo.21285367](https://doi.org/10.5281/zenodo.21285367).

## Audited result

Tracks 8, 10, and 14 are used for grouped model selection and uncertainty
calibration. Track 21 is held out until final scoring.

| Model | Track 21 MAE | Track 21 RMSE | Track 21 R² |
|---|---:|---:|---:|
| Reproduced notebook Gradient Boosting | 0.159 mm | 0.187 mm | -0.99 |
| Audited Ridge alpha 10 | 0.139 mm | 0.167 mm | -0.58 |

The audited model reduces held-out MAE by **12.3%**. Nominal 90% interval
coverage is **76.5%**, so this is a better benchmark rather than a
closed-loop-ready controller.

## Final deliverables

- `notebooks/03_final_submission_audited.ipynb` - executed final notebook.
- `deliverables/report/FMRG_Final_Report_Audited.pdf` - compliant three-page report.
- `deliverables/presentation/FMRG_Final_Submission_Audited.pptx` - editable deck.
- `deliverables/presentation/FMRG_Final_Submission_Audited.pdf` - exported deck.
- `results/final_submission/` - locked metrics, predictions, and figures.
- `src/fmrg_submission/` - audited geometry, thermal, SEM, and modeling code.
- `tests/` - regression tests for the critical pipeline logic.

The prior `notebooks/finalnotebook.ipynb` is retained as a legacy artifact for
comparison.

## Reproduce

```bash
python -m pip install -r requirements.txt
python scripts/run_final_analysis.py \
  --raw-dir /path/to/extracted/zenodo/data \
  --output-dir results/final_submission
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
