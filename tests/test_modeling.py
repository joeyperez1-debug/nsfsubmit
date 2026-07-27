import numpy as np
import pandas as pd

from fmrg_submission.modeling import (
    align_geometry_to_frames,
    conformal_half_width,
    fit_grouped_final_comparison,
    fit_leakage_safe_comparison,
    group_robust_conformal_half_width,
    interval_metrics,
)


def test_alignment_uses_physical_coordinates_not_normalized_index():
    features = pd.DataFrame({"x_mm": [20.2, 20.8], "signal": [1.0, 2.0]})
    geometry = pd.DataFrame(
        {
            "x_mm": [20.0, 20.5, 21.0],
            "width_mm": [0.4, 0.6, 1.0],
            "left_mm": [0.7, 0.6, 0.4],
            "right_mm": [1.1, 1.2, 1.4],
            "valid": [True, True, True],
        }
    )

    aligned = align_geometry_to_frames(features, geometry, max_gap_mm=0.30)

    assert np.allclose(aligned["width_mm"], [0.48, 0.84])
    assert np.allclose(aligned["left_mm"], [0.66, 0.48])


def test_alignment_does_not_bridge_large_measurement_gaps():
    features = pd.DataFrame({"x_mm": [20.1, 21.0, 21.9]})
    geometry = pd.DataFrame(
        {
            "x_mm": [20.0, 20.2, 21.8, 22.0],
            "width_mm": [0.5, 0.5, 0.8, 0.8],
            "left_mm": [0.7, 0.7, 0.5, 0.5],
            "right_mm": [1.2, 1.2, 1.3, 1.3],
            "valid": [True, True, True, True],
        }
    )

    aligned = align_geometry_to_frames(features, geometry, max_gap_mm=0.15)

    assert list(aligned["x_mm"]) == [20.1, 21.9]


def test_split_conformal_interval_has_finite_calibrated_width():
    residuals = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    half_width = conformal_half_width(residuals, coverage=0.80)

    assert half_width == 0.05
    metrics = interval_metrics(
        y_true=np.array([0.50, 0.60]),
        lower=np.array([0.45, 0.50]),
        upper=np.array([0.55, 0.70]),
    )
    assert metrics["coverage"] == 1.0
    assert np.isclose(metrics["mean_width_mm"], 0.15)


def test_group_robust_conformal_uses_worst_development_condition():
    residuals = np.array([0.01, 0.02, 0.20, 0.30])
    groups = np.array([8, 8, 10, 10])

    half_width = group_robust_conformal_half_width(
        residuals, groups, coverage=0.50
    )

    assert half_width == 0.30


def test_model_selection_and_predictions_do_not_use_validation_or_test_labels():
    rows = []
    for track_id, offset in [(8, 0.0), (10, 0.1), (14, 0.2), (21, 0.3)]:
        for i in range(14):
            signal = i / 13
            rows.append(
                {
                    "track_id": track_id,
                    "signal": signal,
                    "signal2": signal**2,
                    "baseline_signal": signal,
                    "width_mm": 0.5 + 0.2 * signal + offset,
                }
            )
    data = pd.DataFrame(rows)
    train = data[data["track_id"].isin([8, 10])].copy()
    validation = data[data["track_id"] == 14].copy()
    test = data[data["track_id"] == 21].copy()

    first = fit_leakage_safe_comparison(
        train,
        validation,
        test,
        baseline_features=["baseline_signal"],
        corrected_feature_sets={"thermal": ["signal", "signal2"]},
    )
    validation_scrambled = validation.copy()
    validation_scrambled["width_mm"] = validation_scrambled["width_mm"][::-1].to_numpy()
    test_scrambled = test.copy()
    test_scrambled["width_mm"] = 10.0
    second = fit_leakage_safe_comparison(
        train,
        validation_scrambled,
        test_scrambled,
        baseline_features=["baseline_signal"],
        corrected_feature_sets={"thermal": ["signal", "signal2"]},
    )

    assert first["corrected"]["model_name"] == second["corrected"]["model_name"]
    assert np.allclose(
        first["corrected"]["test_prediction"],
        second["corrected"]["test_prediction"],
    )


def test_grouped_final_comparison_uses_only_development_tracks_for_selection():
    rows = []
    for track_id, offset in [(8, 0.0), (10, 0.1), (14, 0.2), (21, 0.3)]:
        for i in range(12):
            signal = i / 11
            rows.append(
                {
                    "track_id": track_id,
                    "signal": signal,
                    "signal2": signal**2,
                    "baseline_signal": signal,
                    "width_mm": 0.5 + 0.2 * signal + offset,
                }
            )
    data = pd.DataFrame(rows)
    development = data[data["track_id"].isin([8, 10, 14])].copy()
    test = data[data["track_id"] == 21].copy()

    first = fit_grouped_final_comparison(
        development,
        test,
        baseline_features=["baseline_signal"],
        corrected_feature_sets={"thermal": ["signal", "signal2"]},
    )
    test_changed = test.copy()
    test_changed["width_mm"] = 100.0
    second = fit_grouped_final_comparison(
        development,
        test_changed,
        baseline_features=["baseline_signal"],
        corrected_feature_sets={"thermal": ["signal", "signal2"]},
    )

    assert first["corrected"]["model_name"] == second["corrected"]["model_name"]
    assert np.allclose(
        first["corrected"]["test_prediction"],
        second["corrected"]["test_prediction"],
    )
