# Hierarchical Local Geometry Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and honestly evaluate a hierarchical, multiscale, physically constrained local-geometry model, then regenerate the verified competition submission and amend both existing pull requests.

**Architecture:** Thermal frames become causal multiscale descriptors. A nested leave-one-track-out evaluator selects low-capacity pipelines that predict track-level baselines plus local center/log-width residuals; positive widths reconstruct ordered boundaries. Conditional conformal intervals and registered post-process SEM flank features are evaluated as separate promotion-gated ablations.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy, scikit-learn 1.9, matplotlib, pytest, Jupyter, ReportLab, PptxGenJS, LibreOffice, Git, GitHub CLI/browser.

## Global Constraints

- Tracks 8, 10, 14, and 21 are the independent experimental units.
- Feature selection, preprocessing, model selection, hyperparameter selection, and calibration occur inside the training side of every outer fold.
- Thermal history is causal: current and earlier frames only.
- Width is modeled in log space and reconstructed with `exp`, guaranteeing positive width and ordered boundaries.
- Primary promotion metric is mean per-track outer MAE; worst-track MAE must not materially worsen.
- Track 21 MAE 0.1393538 remains a historical benchmark, not a tuning target.
- SEM imagery is post-process; only registered flanking substrate regions may be used, and no pre-process causal claim is permitted.
- PyTorch and pyGAM are not dependencies. The initial ladder uses scikit-learn models only.
- Every cited metric and plot must be reproducible from one clean experiment command.
- Final report remains at most 3 pages, Arial at least 10 pt, with one-inch margins.
- Final deliverable remains one ZIP containing the report PDF, executable notebook, and presentation deck.

---

### Task 1: Causal Multiscale Thermal History

**Files:**
- Modify: `src/fmrg_submission/thermal.py:18-105`
- Modify: `tests/test_thermal.py:1-30`

**Interfaces:**
- Consumes: `frames: np.ndarray`, `x_mm: np.ndarray`, fixed absolute melt threshold.
- Produces: `extract_thermal_descriptors(frames, x_mm, *, threshold=1500.0, windows=(5, 10, 20)) -> pd.DataFrame`.
- Produces columns for component shape, cooling tail, velocity, causal rolling statistics, slopes, persistence, and missing-history flags.

- [ ] **Step 1: Write failing causality and multiscale tests**

```python
def test_multiscale_history_is_causal_and_contains_required_windows():
    frames = np.full((24, 30, 30), 800.0)
    for i in range(24):
        frames[i, 10:20, 8 : 12 + i // 4] = 1600.0 + 10.0 * i
    original = extract_thermal_descriptors(frames, np.arange(24.0))
    changed = frames.copy()
    changed[20:] = 4000.0
    perturbed = extract_thermal_descriptors(changed, np.arange(24.0))

    required = {
        "roll5_hot_area_px_mean",
        "roll10_thermal_mass_slope",
        "roll20_max_temperature_persistence",
        "centroid_velocity_px",
        "shape_change",
        "cooling_tail_decay",
    }
    assert required.issubset(original.columns)
    assert np.allclose(
        original.loc[:19, sorted(required)],
        perturbed.loc[:19, sorted(required)],
    )


def test_early_history_has_missingness_flags_and_finite_defaults():
    frames = np.full((3, 20, 20), 900.0)
    result = extract_thermal_descriptors(frames, [1.0, 2.0, 3.0])
    assert result.loc[0, "roll20_history_fraction"] == 0.05
    assert np.isfinite(result.select_dtypes("number")).all().all()
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_thermal.py -q`  
Expected: FAIL because the new multiscale columns and `windows` argument do not exist.

- [ ] **Step 3: Extend instantaneous descriptors**

Add component covariance outputs (`major_axis_px`, `minor_axis_px`, `orientation_rad`, `eccentricity`), mask overlap, leading/trailing cooling-tail area and integrated intensity, and decay slope. Keep empty-component outputs finite.

```python
def _linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, np.asarray(values, dtype=float), 1)[0])
```

- [ ] **Step 4: Add causal rolling features**

For each dynamic descriptor and each window in `(5, 10, 20)`, compute backward-looking mean, standard deviation, range, oldest-to-current change, and slope with `rolling(window, min_periods=1)`. Add `roll{window}_history_fraction = min(frame_index + 1, window) / window`.

For `max_temperature`, add persistence above deterministic absolute thresholds 1500, 1750, and 2000 K. Training-fold quantile thresholds remain an evaluator-side transformation so held-out values cannot define thresholds.

- [ ] **Step 5: Run thermal tests**

Run: `pytest tests/test_thermal.py -q`  
Expected: all thermal tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fmrg_submission/thermal.py tests/test_thermal.py
git commit -m "feat: add causal multiscale thermal history"
```

---

### Task 2: Physically Constrained Hierarchical Targets

**Files:**
- Create: `src/fmrg_submission/targets.py`
- Create: `tests/test_targets.py`
- Modify: `src/fmrg_submission/__init__.py:1-6`

**Interfaces:**
- Consumes: aligned frame table with `track_id`, `center_mm`, and `width_mm`.
- Produces: `add_hierarchical_targets(data: pd.DataFrame) -> pd.DataFrame`.
- Produces: `track_thermal_summaries(data, feature_columns) -> pd.DataFrame`.
- Produces: `reconstruct_geometry(baseline_center, baseline_log_width, center_residual, log_width_residual) -> pd.DataFrame`.

- [ ] **Step 1: Write failing decomposition and reconstruction tests**

```python
def test_hierarchical_targets_separate_track_baseline_and_local_residual():
    data = pd.DataFrame(
        {
            "track_id": [8, 8, 10, 10],
            "center_mm": [1.0, 1.2, 2.0, 2.2],
            "width_mm": [0.4, 0.6, 0.8, 1.2],
        }
    )
    result = add_hierarchical_targets(data)
    assert np.allclose(result.groupby("track_id")["center_residual_mm"].median(), 0)
    assert np.allclose(result.groupby("track_id")["log_width_residual"].median(), 0)


def test_reconstruction_is_positive_and_boundaries_are_ordered():
    result = reconstruct_geometry(
        baseline_center=np.array([1.0, 1.0]),
        baseline_log_width=np.log(np.array([0.5, 0.5])),
        center_residual=np.array([0.1, -0.1]),
        log_width_residual=np.array([100.0, -100.0]),
    )
    assert (result["width_mm"] > 0).all()
    assert (result["left_mm"] < result["right_mm"]).all()
    assert np.allclose(result["center_mm"], [1.1, 0.9])
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_targets.py -q`  
Expected: collection FAIL because `fmrg_submission.targets` does not exist.

- [ ] **Step 3: Implement target decomposition**

Use per-track medians for `baseline_center_mm` and `baseline_log_width`. Reject nonfinite or nonpositive widths with `ValueError`. Add residual columns without modifying input order.

- [ ] **Step 4: Implement track-level thermal summaries**

For each requested thermal feature, emit median, interquartile range, 10th percentile, and 90th percentile by track. Preserve one row per `track_id`.

```python
def track_thermal_summaries(
    data: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    grouped = data.groupby("track_id", sort=True)
    records = []
    for track_id, frame in grouped:
        record = {"track_id": track_id}
        for column in feature_columns:
            values = frame[column].to_numpy(dtype=float)
            record[f"{column}__median"] = float(np.nanmedian(values))
            record[f"{column}__iqr"] = float(
                np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
            )
            record[f"{column}__p10"] = float(np.nanpercentile(values, 10))
            record[f"{column}__p90"] = float(np.nanpercentile(values, 90))
        records.append(record)
    return pd.DataFrame.from_records(records)
```

- [ ] **Step 5: Implement stable reconstruction**

Clip combined log-width to `[-20, 20]` before exponentiation to prevent numerical overflow, then derive left and right from center and width.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_targets.py tests/test_geometry.py -q`  
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/fmrg_submission/targets.py src/fmrg_submission/__init__.py tests/test_targets.py
git commit -m "feat: add constrained hierarchical geometry targets"
```

---

### Task 3: Nested Track-Level Model Selection

**Files:**
- Modify: `src/fmrg_submission/modeling.py:9-370`
- Create: `src/fmrg_submission/evaluation.py`
- Extend: `tests/test_modeling.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `candidate_estimators(random_state=42) -> dict[str, BaseEstimator]`.
- Produces: `fit_hierarchical_candidate(train, test, local_features, summary_features, estimator) -> pd.DataFrame`.
- Produces: `nested_leave_one_track_out(data, feature_sets, estimator_factories, *, coverage=0.90) -> dict`.
- Produces: `track_balanced_metrics(predictions: pd.DataFrame) -> dict[str, float]`.

- [ ] **Step 1: Write failing candidate-model tests**

```python
def test_candidate_ladder_contains_requested_low_capacity_models():
    names = candidate_estimators().keys()
    assert {"ridge", "elastic_net", "pls", "spline_ridge", "gaussian_process"} <= set(
        names
    )


def test_hierarchical_prediction_ignores_held_out_geometry_summaries():
    train, test = synthetic_four_track_data()
    first = fit_hierarchical_candidate(
        train, test, ["signal"], ["signal__median"], candidate_estimators()["ridge"]
    )
    changed = test.copy()
    changed[["width_mm", "center_mm"]] = 99.0
    second = fit_hierarchical_candidate(
        train, changed, ["signal"], ["signal__median"], candidate_estimators()["ridge"]
    )
    assert np.allclose(first["width_prediction_mm"], second["width_prediction_mm"])
    assert np.allclose(first["center_prediction_mm"], second["center_prediction_mm"])
```

- [ ] **Step 2: Run model tests and verify they fail**

Run: `pytest tests/test_modeling.py -q`  
Expected: FAIL because the public candidate ladder and hierarchical fitter do not exist.

- [ ] **Step 3: Implement deterministic estimator factories**

Use:

- Ridge with alpha grid `(1, 10, 100)`;
- `MultiTaskElasticNet` with alpha `(0.001, 0.01, 0.1)` and l1 ratio `(0.1, 0.5)`;
- `PLSRegression` with components `(2, 4, 6)`, capped below available features/training rows;
- `SplineTransformer(n_knots=4, degree=2)` plus Ridge `(10, 100)`; and
- `PCA(n_components <= 8)` plus independent `GaussianProcessRegressor` targets with fixed `ConstantKernel * RBF + WhiteKernel`.

All linear pipelines include median imputation and standard scaling. Set every available random seed to 42.

- [ ] **Step 4: Fit baseline and residual stages**

Fit the baseline model on one summary row per training track. Predict held-out baselines from held-out thermal summaries only. Fit the local multi-output model on `[center_residual_mm, log_width_residual]`. Reconstruct center, width, left, and right using `reconstruct_geometry`.

- [ ] **Step 5: Write failing nested-validation tests**

```python
def test_nested_outer_fold_never_uses_held_track_for_selection(monkeypatch):
    seen = []

    def recording_selector(inner_train, inner_valid, *args, **kwargs):
        seen.append(
            (
                set(inner_train["track_id"]),
                set(inner_valid["track_id"]),
            )
        )
        return "ridge"

    monkeypatch.setattr(evaluation, "_score_inner_candidate", recording_selector)
    nested_leave_one_track_out(
        synthetic_four_track_data(),
        {"thermal": ["signal"]},
        {"ridge": lambda: Ridge(alpha=10)},
    )
    assert all(train_ids.isdisjoint(valid_ids) for train_ids, valid_ids in seen)


def test_track_balanced_mae_weights_tracks_equally():
    predictions = pd.DataFrame(
        {
            "track_id": [8] * 100 + [10],
            "width_mm": [0.0] * 101,
            "width_prediction_mm": [0.1] * 100 + [1.0],
            "center_mm": [0.0] * 101,
            "center_prediction_mm": [0.0] * 101,
            "left_mm": [0.0] * 101,
            "left_prediction_mm": [0.0] * 101,
            "right_mm": [0.0] * 101,
            "right_prediction_mm": [0.0] * 101,
        }
    )
    metrics = track_balanced_metrics(predictions)
    assert np.isclose(metrics["track_balanced_width_mae_mm"], 0.55)
```

- [ ] **Step 6: Run evaluation tests and verify they fail**

Run: `pytest tests/test_evaluation.py -q`  
Expected: collection FAIL because `fmrg_submission.evaluation` does not exist.

- [ ] **Step 7: Implement nested outer/inner evaluation**

For every outer held track, run inner leave-one-track-out on the remaining three tracks. Select by mean inner per-track width MAE; break ties by residual correlation, boundary MAE, then model name for determinism. Save inner scores, selected feature set/model, outer predictions, and per-track metrics.

- [ ] **Step 8: Implement spatial-fidelity and promotion metrics**

For each track, subtract the true and predicted track means before correlation and standard-deviation ratio. Compute boundary MAE, roughness as RMS first difference per millimetre, and waviness as RMS deviation from a Savitzky–Golay low-pass curve. Aggregate by unweighted mean across tracks.

- [ ] **Step 9: Run modeling and evaluation tests**

Run: `pytest tests/test_modeling.py tests/test_evaluation.py -q`  
Expected: all tests PASS.

- [ ] **Step 10: Commit**

```bash
git add src/fmrg_submission/modeling.py src/fmrg_submission/evaluation.py tests/test_modeling.py tests/test_evaluation.py
git commit -m "feat: add nested hierarchical model selection"
```

---

### Task 4: Locally Scaled Conformal Uncertainty

**Files:**
- Create: `src/fmrg_submission/uncertainty.py`
- Create: `tests/test_uncertainty.py`
- Modify: `src/fmrg_submission/evaluation.py`

**Interfaces:**
- Produces: `fit_local_scale(calibration: pd.DataFrame, feature_columns: list[str]) -> BaseEstimator`.
- Produces: `normalized_conformal_interval(y_cal, pred_cal, scale_cal, pred_test, scale_test, coverage=0.90) -> tuple[np.ndarray, np.ndarray, float]`.
- Produces difficulty-tercile coverage metrics.

- [ ] **Step 1: Write failing normalized-conformal tests**

```python
def test_normalized_conformal_expands_for_difficult_regions():
    lower, upper, quantile = normalized_conformal_interval(
        y_cal=np.array([0.0, 0.2, 0.0, 0.4]),
        pred_cal=np.zeros(4),
        scale_cal=np.array([1.0, 2.0, 1.0, 4.0]),
        pred_test=np.array([0.0, 0.0]),
        scale_test=np.array([1.0, 3.0]),
        coverage=0.75,
    )
    widths = upper - lower
    assert quantile > 0
    assert np.isclose(widths[1] / widths[0], 3.0)


def test_normalized_conformal_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="positive"):
        normalized_conformal_interval(
            np.array([0.0]), np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([1.0])
        )
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_uncertainty.py -q`  
Expected: collection FAIL because `fmrg_submission.uncertainty` does not exist.

- [ ] **Step 3: Implement normalized conformal**

Fit a positive residual-scale model by predicting `log(abs(residual) + 1e-6)` with Ridge, then exponentiating and flooring scale at `1e-6`. Calibrate `abs(y - pred) / scale` using the finite-sample conformal rank already used in `modeling.py`.

- [ ] **Step 4: Integrate grouped calibration**

Within each outer fold, generate inner out-of-fold residuals on the three training tracks. Fit and calibrate the scale model from those residuals only. Compare conditional and global intervals on the outer track. Record coverage and width overall and by predicted-scale tercile.

- [ ] **Step 5: Run uncertainty and evaluation tests**

Run: `pytest tests/test_uncertainty.py tests/test_evaluation.py -q`  
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fmrg_submission/uncertainty.py src/fmrg_submission/evaluation.py tests/test_uncertainty.py tests/test_evaluation.py
git commit -m "feat: add locally scaled conformal intervals"
```

---

### Task 5: Registered SEM Flank Ablation

**Files:**
- Modify: `src/fmrg_submission/sem.py:11-76`
- Modify: `tests/test_sem.py:1-21`

**Interfaces:**
- Produces: `flank_sem_descriptors(image, left_row, right_row, *, flank_width_px) -> dict[str, float]`.
- Extends: `extract_sem_descriptors_at_positions(..., boundary_rows=None, registration_uncertainty_mm=0.0) -> pd.DataFrame`.
- Keeps `masked_sem_descriptors` for historical reproducibility.

- [ ] **Step 1: Write failing flank-isolation and registration tests**

```python
def test_flank_descriptors_ignore_track_and_use_left_right_separately():
    image = np.zeros((100, 80), dtype=float)
    image[:35] = 10.0
    image[65:] = 30.0
    first = flank_sem_descriptors(
        image, left_row=40, right_row=60, flank_width_px=20
    )
    altered = image.copy()
    altered[40:61] = 255.0
    second = flank_sem_descriptors(
        altered, left_row=40, right_row=60, flank_width_px=20
    )
    assert first == second
    assert first["sem_right_mean"] > first["sem_left_mean"]


def test_registration_uncertainty_is_emitted_in_physical_units():
    result = extract_sem_descriptors_at_positions(
        tile_paths, np.array([90.0]), registration_uncertainty_mm=0.25
    )
    assert result.loc[0, "sem_registration_uncertainty_mm"] == 0.25
```

- [ ] **Step 2: Run SEM tests and verify they fail**

Run: `pytest tests/test_sem.py -q`  
Expected: FAIL because flank descriptors and registration uncertainty do not exist.

- [ ] **Step 3: Implement flank descriptors**

Sample fixed-width bands immediately outside `left_row` and `right_row`, clipped to image bounds. Compute mean, standard deviation, gradient mean/P95, Laplacian standard deviation, edge density above the region's 90th-percentile gradient, orientation entropy, and left/right differences.

- [ ] **Step 4: Extend physical-x interpolation**

Preserve the current tile-center mapping, emit the assumed registration uncertainty, and accept optional registered boundary rows. When boundary rows are unavailable, use the documented central exclusion as a fallback and mark `sem_registration_mode = "center_mask_fallback"`.

- [ ] **Step 5: Run SEM tests**

Run: `pytest tests/test_sem.py -q`  
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fmrg_submission/sem.py tests/test_sem.py
git commit -m "feat: add registered SEM flank descriptors"
```

---

### Task 6: Reproducible Improvement Experiment and Promotion Gate

**Files:**
- Create: `scripts/run_improvement_experiments.py`
- Modify: `scripts/run_final_analysis.py:37-420`
- Create: `tests/test_experiment_runner.py`
- Generate: `analysis/improved/metrics.json`
- Generate: `analysis/improved/outer_fold_predictions.csv`
- Generate: `analysis/improved/candidate_scores.csv`
- Generate: `analysis/improved/figures/*.png`

**Interfaces:**
- CLI: `python scripts/run_improvement_experiments.py --raw-dir PATH --output-dir analysis/improved`.
- Produces machine-readable nested-fold results, chosen model, uncertainty comparison, SEM ablation, promotion decision, package versions, and deterministic seed.
- `run_final_analysis.py` consumes the promoted configuration rather than independently choosing a model.

- [ ] **Step 1: Write failing runner contract test**

```python
def test_experiment_output_contract(tmp_path, synthetic_raw_dir):
    result = run_experiments(synthetic_raw_dir, tmp_path)
    assert {
        "protocol",
        "historical_benchmark",
        "candidates",
        "selected",
        "promotion",
        "uncertainty",
        "sem_ablation",
        "software_versions",
    } <= result.keys()
    assert set(pd.read_csv(tmp_path / "outer_fold_predictions.csv")["track_id"]) == {
        8, 10, 14, 21
    }
```

- [ ] **Step 2: Run contract test and verify it fails**

Run: `pytest tests/test_experiment_runner.py -q`  
Expected: collection FAIL because the improvement runner does not exist.

- [ ] **Step 3: Implement cached data loading**

Reuse `_load_track`, `align_geometry_to_frames`, and the existing raw-data package. Write aligned per-track feature tables once, then reload them for model candidates so model comparison does not repeatedly decode thermal videos.

- [ ] **Step 4: Run the complete experiment matrix**

Evaluate:

- compact current thermal features;
- multiscale thermal features;
- each of Ridge, elastic net, PLS, spline-ridge, and reduced GPR;
- direct versus hierarchical center/log-width targets;
- thermal-only versus thermal plus registered SEM flank features; and
- global versus normalized conformal intervals.

Write one candidate row per outer fold and aggregate configuration.

- [ ] **Step 5: Apply the promotion rule**

Promote only when:

```python
promote = (
    candidate["track_balanced_width_mae_mm"]
    < incumbent["track_balanced_width_mae_mm"]
    and candidate["worst_track_width_mae_mm"]
    <= incumbent["worst_track_width_mae_mm"] * 1.05
    and (
        candidate["residual_correlation"]
        > incumbent["residual_correlation"]
        or candidate["mean_boundary_mae_mm"]
        < incumbent["mean_boundary_mae_mm"]
        or candidate["variation_std_ratio_error"]
        < incumbent["variation_std_ratio_error"]
    )
)
```

If no candidate passes, preserve the incumbent and save the full negative result.

- [ ] **Step 6: Produce competition plots**

Generate:

- four-panel outer-fold measured versus predicted width;
- before/after track-balanced scorecard;
- residual variation correlation and standard-deviation ratio;
- center/left/right boundary reconstruction;
- interval coverage versus local difficulty;
- thermal feature importance using outer-fold permutation importance; and
- SEM thermal-only ablation comparison.

- [ ] **Step 7: Run the unit contract**

Run: `pytest tests/test_experiment_runner.py -q`  
Expected: PASS on synthetic fixtures.

- [ ] **Step 8: Run the real experiment once**

Run:

```bash
python scripts/run_improvement_experiments.py \
  --raw-dir /Users/goon/nsf-data/zenodo-21285367/extracted \
  --output-dir analysis/improved
```

Expected: exit 0; `metrics.json` contains all four outer tracks, a deterministic selected configuration, and an explicit promotion boolean.

- [ ] **Step 9: Review metrics before artifact generation**

Check:

```bash
python -m json.tool analysis/improved/metrics.json
```

Confirm every headline number comes from nested outer folds and the historical Track 21 benchmark is labeled separately.

- [ ] **Step 10: Commit**

```bash
git add scripts/run_improvement_experiments.py scripts/run_final_analysis.py \
  tests/test_experiment_runner.py analysis/improved
git commit -m "feat: evaluate and select improved local geometry model"
```

---

### Task 7: Notebook, Report, Presentation, ZIP, and PR Updates

**Files:**
- Modify: `scripts/build_final_notebook.py`
- Modify: `scripts/build_final_report.py`
- Modify: `scripts/build_final_deck.mjs`
- Modify: `scripts/build_submission_package.py`
- Modify: `deliverables/README.md`
- Regenerate: `notebooks/03_final_submission_audited.ipynb`
- Regenerate: `deliverables/report/FMRG_Final_Report_Audited.pdf`
- Regenerate: `deliverables/presentation/FMRG_Final_Submission_Audited.pptx`
- Regenerate: `deliverables/presentation/FMRG_Final_Submission_Audited.pdf`
- Regenerate: `deliverables/submission/FMRG_Final_Submission_Audited.zip`

**Interfaces:**
- Artifact builders consume `analysis/improved/metrics.json`, tables, and plots.
- Notebook executes the promoted configuration and reproduces cited outer-fold metrics.
- Both existing PR bodies receive identical verified result summaries and reproduction commands.

- [ ] **Step 1: Update the notebook builder**

Explain the nested protocol, hierarchical target, physical constraints, multiscale descriptors, conditional uncertainty, SEM limitation, Generative AI use, and all finalist-facing caveats. Include executable cells that load or recompute the promoted result.

- [ ] **Step 2: Execute and verify the notebook**

Run:

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/03_final_submission_audited.ipynb \
  --output /private/tmp/fmrg_verified.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Expected: exit 0 with no cell error outputs.

- [ ] **Step 3: Update the three-page report**

Replace single Track 21 headline claims with track-balanced nested outer results. Include executive summary, methodology and AI disclosure, local width/boundary outcomes, spatial descriptors, uncertainty, interpretability, SEM limitation, and conclusion.

- [ ] **Step 4: Update the ten-minute presentation**

Lead with the condition-plus-residual insight, show the four-track scorecard, demonstrate physical boundary reconstruction, explain difficult-region uncertainty, and close with limitations plus future registered pre-process SEM.

- [ ] **Step 5: Build all artifacts**

Run:

```bash
python scripts/build_final_notebook.py
python scripts/build_final_report.py
node scripts/build_final_deck.mjs
python scripts/build_submission_package.py
```

Expected: all commands exit 0 and regenerate the four named deliverables.

- [ ] **Step 6: Verify report and deck visually and programmatically**

Render every report/deck page to images. Check:

- report page count is at most 3;
- margins are one inch and body text is at least 10 pt Arial;
- no text is clipped or overlaps;
- plots are readable at normal zoom;
- every metric matches `analysis/improved/metrics.json`;
- no unsupported pre-process SEM or causal substrate claim appears; and
- the deck fits a ten-minute presentation.

- [ ] **Step 7: Verify ZIP contents**

Run: `unzip -l deliverables/submission/FMRG_Final_Submission_Audited.zip`  
Expected: report PDF, executable notebook, PPTX, repository links, and manifest are present exactly once; no caches or raw data are included.

- [ ] **Step 8: Run the full test suite once**

Run: `pytest -q`  
Expected: all tests PASS.

- [ ] **Step 9: Commit regenerated submission**

```bash
git add scripts notebooks deliverables analysis/improved
git commit -m "docs: regenerate improved FMRG final submission"
```

- [ ] **Step 10: Push the main PR branch**

Run: `git push origin codex/fmrg-final-rebuild`  
Expected: `alphons3t/nsf-chuds` PR 1 advances to the final commit.

- [ ] **Step 11: Synchronize and push the mirror branch**

Copy the verified commit state into `/Users/goon/nsf-worktrees/joeyperez1-nsfsubmit`, commit only if histories require a mirror commit, and push `codex/fmrg-final-rebuild` to the authorized fork remote used by `joeyperez1-debug/nsfsubmit` PR 1.

- [ ] **Step 12: Amend both existing PR bodies**

Update:

- `https://github.com/alphons3t/nsf-chuds/pull/1`
- `https://github.com/joeyperez1-debug/nsfsubmit/pull/1`

Each body must include:

- nested four-track protocol;
- before/after track-balanced MAE, worst-track MAE, boundary MAE, residual correlation, and interval coverage/width;
- promoted model and feature set;
- SEM ablation outcome and post-process limitation;
- exact reproduction commands;
- artifact list; and
- test and notebook execution evidence.

- [ ] **Step 13: Verify remote state**

Confirm both PRs show the final commit, updated body, correct files, and no failing checks. Do not merge unless the user separately requests merging.

