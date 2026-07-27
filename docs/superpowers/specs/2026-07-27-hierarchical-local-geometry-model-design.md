# Hierarchical Local Geometry Model Design

**Date:** 2026-07-27  
**Status:** Approved for specification  
**Scope:** Improve the FMRG laser-track submission without contaminating evaluation

## Objective

Improve prediction of the final laser track as a spatially varying signal, with particular emphasis on the two weaknesses exposed by the audited submission:

- the held-out track has a condition-level mean-width shift; and
- the predicted local signal is substantially smoother than the measured geometry.

The improved submission must remain executable, interpretable, physically consistent, and honest about the four-track sample size. It must not select features or models by repeatedly optimizing Track 21 after its labels have been inspected.

## Evaluation Authority

All promotion decisions use nested leave-one-track-out validation over Tracks 8, 10, 14, and 21.

For each outer fold:

1. Hold out one complete track.
2. Use only the remaining three tracks for feature selection, model selection, hyperparameter selection, preprocessing, and conformal calibration.
3. Fit the chosen configuration on those three tracks.
4. Predict the untouched outer track.

The primary score is the mean of the four per-track MAEs, so a track with more aligned samples cannot dominate the result. Secondary scores are:

- worst-track MAE;
- track-balanced RMSE and R²;
- local-variation correlation after removing each track's mean;
- local-variation standard-deviation ratio;
- left- and right-boundary MAE;
- 90% interval coverage and mean interval width; and
- roughness and waviness descriptor error.

The existing Track 21 result of MAE 0.1393538 is retained as a historical benchmark, not used as the new selection target.

A candidate is promoted only if it lowers track-balanced outer MAE relative to the current audited pipeline, does not materially worsen worst-track MAE, and improves at least one spatial-fidelity metric. If no candidate clears that gate, the submission keeps the current predictive model and reports the negative experiment honestly.

## Data and Leakage Controls

The four laser tracks are the independent experimental units. Individual aligned frames from the same track are correlated and must never be split across training and validation folds.

Every learned transformation is fit inside its training fold:

- missing-value imputation;
- scaling;
- spline bases;
- dimensionality reduction;
- model hyperparameters;
- baseline-width models;
- residual models; and
- uncertainty normalization.

Thermal features may use only the current and earlier frames. Centered or future-looking rolling windows are prohibited.

The repository does not currently contain authoritative pre-process SEM. The SEM images are post-process measurements. SEM-derived substrate features therefore use only masked regions outside the observed track boundaries and are described as surrounding post-process substrate morphology, not causal pre-process evidence.

## Hierarchical Target Representation

The model separates condition-level geometry from local variation.

For each track, define:

- baseline center as the robust median center position;
- baseline log-width as the log of the robust median positive width;
- local center residual as center minus baseline center; and
- local log-width residual as log-width minus baseline log-width.

At prediction time, no ground-truth summary from the held-out track is available. A baseline model predicts center and log-width baselines from track-level thermal summaries computed from the full thermal sequence. These summaries include robust quantiles and dispersion of melt-pool size, intensity, gradients, asymmetry, cooling-tail behavior, and motion.

A local model predicts center and log-width residuals from causal frame-level thermal history. Final outputs are reconstructed as:

```text
center = predicted baseline center + predicted local center residual
width  = exp(predicted baseline log-width + predicted local log-width residual)
left   = center - width / 2
right  = center + width / 2
```

This guarantees positive widths and left/right ordering while allowing center and width to share the same thermal evidence.

## Multiscale Thermal History

The existing instantaneous, lag-1, and three-frame features are retained as the compact baseline. New causal descriptors are added at 5-, 10-, and 20-frame horizons:

- rolling mean, standard deviation, minimum, maximum, and range;
- change from the oldest value in the window;
- least-squares temporal slope;
- exponentially weighted recent value;
- fraction of frames above thresholds defined from training-fold intensity quantiles;
- time since the most recent threshold crossing;
- persistence of hot-pixel area and high-temperature area;
- melt-pool centroid velocity and acceleration;
- area, major-axis, minor-axis, eccentricity, orientation, and their velocities;
- left/right and leading/trailing intensity imbalance;
- cooling-tail length, integrated intensity, decay slope, and asymmetry; and
- frame-to-frame mask overlap and shape-change magnitude.

Thresholds and normalization constants are learned from training tracks only. Missing early-history values receive deterministic causal defaults plus missingness indicators.

## Model Candidates

The dataset is too small for unrestricted image networks. The candidate ladder therefore emphasizes low-capacity models for correlated smooth signals:

1. Ridge regression as the reference.
2. Elastic net for sparse multiscale feature selection.
3. Partial least squares with a small component grid.
4. A generalized-additive approximation using spline transforms plus ridge regression.
5. Gaussian-process regression on a training-fold dimensionality-reduced feature space, only when runtime and matrix size remain practical.

Each candidate predicts two outputs: center and log-width. The hierarchical variant has separate baseline and residual stages but shares preprocessing and the same evaluation contract.

A low-capacity temporal convolution is excluded from the initial promotion ladder because PyTorch is not installed and adding a deep-learning dependency is not justified by four tracks. It may be attempted only after the scikit-learn candidates are exhausted and only if it can be implemented with an already available dependency, deterministic training, and the same nested evaluation.

Hyperparameter grids are intentionally small. Inner leave-one-track-out validation on the three training tracks selects the model and configuration by track-balanced MAE, with spatial-fidelity metrics used as tie-breakers.

## Conditional Uncertainty

The improved model uses normalized conformal prediction rather than one global residual width.

An auxiliary difficulty model predicts absolute residual scale from training-fold evidence such as:

- distance from the training feature distribution;
- local thermal volatility;
- disagreement among candidate models or inner-fold predictions;
- melt-pool instability; and
- missing-history indicators.

Calibration scores are absolute residuals divided by predicted local scale. The conformal quantile is applied to the held-out track's predicted scale, causing intervals to widen in difficult regions and contract in stable regions.

Calibration remains track-grouped. The report includes empirical coverage, mean width, and coverage by predicted-difficulty tercile. If normalized conformal undercovers or produces wider intervals without better conditional behavior, the existing global conformal method remains the submission default.

## SEM Registration and Substrate Evidence

SEM work is an ablation, not an assumed improvement.

The processed track is registered to physical x using the extracted track centerline and scan-direction endpoints. Morphology is sampled from fixed-width flanking bands outside the measured boundaries. Candidate features include texture energy, local contrast, edge density, orientation entropy, and left/right flank asymmetry.

The SEM ablation must pass the same nested outer validation as thermal-only models. The report will state:

- whether registered flank features improve prediction;
- uncertainty in the alignment;
- that the source imagery is post-process; and
- that causal separation of original substrate effects requires genuine registered pre-process measurements in future experiments.

If registration is unreliable or validation worsens, SEM features are excluded from the final predictive model and retained only as a documented negative result.

## Components

The implementation will use focused modules:

- `thermal.py` for causal instantaneous and multiscale descriptors;
- `targets.py` for center/log-width targets, baseline/residual decomposition, and boundary reconstruction;
- `modeling.py` for candidate pipelines and grouped inner selection;
- `evaluation.py` for nested outer validation, track-balanced metrics, and promotion rules;
- `uncertainty.py` for normalized conformal calibration;
- `sem.py` for physical-x registration and flank-only morphology;
- `run_improvement_experiments.py` for the reproducible experiment matrix and machine-readable outputs; and
- existing artifact builders for the final notebook, report, slide deck, and ZIP.

## Testing and Reproducibility

Implementation follows test-driven development.

Unit tests must prove:

- rolling descriptors are causal;
- windows do not cross track boundaries;
- fold-learned thresholds use training data only;
- width reconstruction is always positive and boundaries are ordered;
- outer-held-out labels cannot affect preprocessing or model choice;
- track-balanced metrics weight tracks equally;
- conditional conformal intervals expand with predicted difficulty;
- SEM features are sampled outside the measured track; and
- deterministic seeds reproduce metrics.

The experiment runner writes configuration, package versions, per-fold choices, predictions, metrics, and calibration results to versioned JSON/CSV files. One clean execution must recreate every metric and plot cited in the report and presentation.

## Deliverables and Pull Requests

After model selection:

1. Regenerate the executable final notebook.
2. Regenerate the maximum-three-page report PDF with minimum 10 pt Arial and one-inch margins.
3. Regenerate the self-contained presentation deck and PDF.
4. Rebuild the single submission ZIP.
5. Visually and programmatically verify all artifacts.
6. Commit and push the verified changes to the existing feature branches.
7. Update the bodies of both existing pull requests with the new evaluation protocol, verified before/after metrics, limitations, and reproduction commands.

The PRs must distinguish historical Track 21 performance from the new nested four-track evidence and must not claim that an unverified experiment improved the model.
