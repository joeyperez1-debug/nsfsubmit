import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fmrg_submission.evaluation import (
    condition_summary_features,
    nested_leave_one_track_out,
    select_inner_candidate,
    track_balanced_metrics,
)


def _synthetic_four_track_data():
    rows = []
    for track_id, offset in [(8, 0.0), (10, 0.1), (14, 0.2), (21, 0.3)]:
        for index in range(12):
            x = index / 11
            thermal = offset + x
            width = 0.45 + offset + 0.04 * np.sin(2 * np.pi * x)
            center = 1.0 + 0.2 * offset + 0.02 * np.cos(2 * np.pi * x)
            rows.append(
                {
                    "track_id": track_id,
                    "x_mm": x,
                    "thermal": thermal,
                    "width_mm": width,
                    "center_mm": center,
                    "left_mm": center - width / 2,
                    "right_mm": center + width / 2,
                }
            )
    return pd.DataFrame(rows)


def _ridge_factory():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def test_nested_evaluation_predicts_every_outer_track():
    result = nested_leave_one_track_out(
        _synthetic_four_track_data(),
        feature_sets={"thermal": ["thermal"]},
        estimator_factories={"ridge": _ridge_factory},
    )

    assert set(result["predictions"]["track_id"]) == {8, 10, 14, 21}
    assert len(result["outer_folds"]) == 4


def test_outer_track_labels_do_not_affect_its_prediction_or_selection():
    data = _synthetic_four_track_data()
    first = nested_leave_one_track_out(
        data,
        feature_sets={"thermal": ["thermal"]},
        estimator_factories={"ridge": _ridge_factory},
    )
    changed = data.copy()
    changed.loc[
        changed["track_id"] == 21,
        ["width_mm", "center_mm", "left_mm", "right_mm"],
    ] = 99.0
    second = nested_leave_one_track_out(
        changed,
        feature_sets={"thermal": ["thermal"]},
        estimator_factories={"ridge": _ridge_factory},
    )

    first_fold = first["outer_folds"][21]
    second_fold = second["outer_folds"][21]
    assert first_fold["selected_model"] == second_fold["selected_model"]
    assert first_fold["selected_feature_set"] == second_fold["selected_feature_set"]
    first_prediction = first["predictions"].query("track_id == 21")
    second_prediction = second["predictions"].query("track_id == 21")
    assert np.allclose(
        first_prediction["width_prediction_mm"],
        second_prediction["width_prediction_mm"],
    )


def test_track_balanced_mae_weights_tracks_equally():
    predictions = pd.DataFrame(
        {
            "track_id": [8] * 100 + [10],
            "x_mm": np.arange(101, dtype=float),
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


def test_nested_evaluation_calibrates_global_and_local_intervals_from_inner_folds():
    result = nested_leave_one_track_out(
        _synthetic_four_track_data(),
        feature_sets={"thermal": ["thermal"]},
        estimator_factories={"ridge": _ridge_factory},
        coverage=0.90,
    )

    required = {
        "width_lower_90_mm",
        "width_upper_90_mm",
        "global_width_lower_90_mm",
        "global_width_upper_90_mm",
        "predicted_residual_scale_mm",
    }
    assert required.issubset(result["predictions"].columns)
    assert (
        result["predictions"]["width_upper_90_mm"]
        > result["predictions"]["width_lower_90_mm"]
    ).all()
    assert {"conditional", "global"} <= result["uncertainty"].keys()


def test_condition_baseline_uses_compact_summaries_not_local_history_expansion():
    features = [
        "hot_area_px",
        "max_temperature",
        "thermal_mass",
        "roll20_hot_area_px_slope",
        "roll10_thermal_mass_std",
        "x_sin_1",
    ]

    summaries = condition_summary_features(features)

    assert summaries == [
        "hot_area_px__median",
        "max_temperature__median",
        "thermal_mass__median",
    ]


def test_inner_selection_prefers_spatial_fidelity_within_accuracy_tolerance():
    flat = {
        "feature_set": "wide",
        "model": "gp",
        "error": None,
        "metrics": {
            "track_balanced_width_mae_mm": 0.150,
            "mean_boundary_mae_mm": 0.140,
            "residual_correlation": 0.00,
            "variation_std_ratio_error": 1.00,
        },
    }
    spatial = {
        "feature_set": "compact",
        "model": "spline_ridge",
        "error": None,
        "metrics": {
            "track_balanced_width_mae_mm": 0.165,
            "mean_boundary_mae_mm": 0.145,
            "residual_correlation": 0.12,
            "variation_std_ratio_error": 0.60,
        },
    }

    selected = select_inner_candidate([flat, spatial], accuracy_tolerance_mm=0.02)

    assert selected["model"] == "spline_ridge"
