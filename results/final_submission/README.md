# Audited final results

This directory is the locked evidence base for the final notebook, report, and
presentation.

## Protocol

- Development and grouped model selection: Tracks 8, 10, and 14.
- Untouched final evaluation: Track 21.
- Primary target: local track width in millimeters.
- Additional targets: left and right boundary positions.
- Selected model: Ridge regression with alpha 10 and thermal-history features.
- Uncertainty: group-robust conformal residual radius calibrated without Track 21.

## Headline comparison

| Model | Track 21 MAE | Track 21 RMSE | Track 21 R² |
|---|---:|---:|---:|
| Reproduced notebook Gradient Boosting | 0.159 mm | 0.187 mm | -0.99 |
| Audited Ridge alpha 10 | 0.139 mm | 0.167 mm | -0.58 |

The held-out MAE improvement is 12.3%. Nominal 90% interval coverage is 76.5%,
so the uncertainty result is not deployment-ready.

`metrics.json` is the source of truth. CSV files contain aligned observations,
out-of-fold predictions, held-out predictions, and permutation importance.
