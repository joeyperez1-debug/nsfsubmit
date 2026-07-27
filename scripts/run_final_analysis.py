#!/usr/bin/env python3
"""Run the audited baseline and leakage-safe final FMRG analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fmrg_submission.geometry import (  # noqa: E402
    extract_local_geometry,
    smooth_local_geometry,
)
from fmrg_submission.modeling import (  # noqa: E402
    fit_grouped_final_comparison,
    group_robust_conformal_half_width,
    point_metrics,
)
from fmrg_submission.sem import extract_sem_descriptors_at_positions  # noqa: E402
from fmrg_submission.thermal import extract_thermal_descriptors  # noqa: E402
from nsf_fmrg_data import (  # noqa: E402
    extract_final_thermal_frames,
    get_sem_tile_paths,
    load_wyko_asc,
)

TRACK_IDS = (8, 10, 14, 21)
TRAIN_TRACKS = (8, 10)
VALIDATION_TRACK = 14
TEST_TRACK = 21
STEADY_STATE_RANGE_MM = (24.0, 96.0)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _load_track(raw_dir: Path, track_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    thermal = extract_final_thermal_frames(raw_dir / "thermal", track_id)
    descriptors = extract_thermal_descriptors(
        thermal["frames"], thermal["x_mm_center"]
    )
    sem = extract_sem_descriptors_at_positions(
        get_sem_tile_paths(raw_dir / "sem", track_id),
        descriptors["x_mm"].to_numpy(),
        mask_fraction=0.30,
    )
    descriptors = descriptors.merge(sem, on="x_mm", validate="one_to_one")
    descriptors["track_id"] = track_id
    descriptors["x_scaled"] = (
        descriptors["x_mm"] - STEADY_STATE_RANGE_MM[0]
    ) / (STEADY_STATE_RANGE_MM[1] - STEADY_STATE_RANGE_MM[0])
    descriptors["x_sin_1"] = np.sin(2.0 * np.pi * descriptors["x_scaled"])
    descriptors["x_cos_1"] = np.cos(2.0 * np.pi * descriptors["x_scaled"])
    descriptors["x_sin_2"] = np.sin(4.0 * np.pi * descriptors["x_scaled"])
    descriptors["x_cos_2"] = np.cos(4.0 * np.pi * descriptors["x_scaled"])

    height = load_wyko_asc(raw_dir / "height_maps", track_id)
    geometry = smooth_local_geometry(
        extract_local_geometry(
            height["Z_mm"], height["x_actual_mm"], height["y_mm"]
        ),
        window_mm=0.40,
    )
    geometry_frame = pd.DataFrame(geometry)
    return descriptors, geometry_frame


def _add_notebook_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["track_id", "x_mm"]).copy()
    grouped = frame.groupby("track_id", sort=False)
    frame["size_norm"] = frame["hot_area_px"] - grouped["hot_area_px"].transform(
        "mean"
    )
    frame["temp_norm"] = frame["max_temperature"] - grouped[
        "max_temperature"
    ].transform("mean")
    frame["delta_size"] = grouped["size_norm"].diff().fillna(0.0)
    frame["lag1_size"] = grouped["size_norm"].shift(1).fillna(0.0)
    frame["lag2_size"] = grouped["size_norm"].shift(2).fillna(0.0)
    return frame


def _plot_predictions(
    path: Path,
    data: pd.DataFrame,
    baseline: dict,
    corrected: dict,
    *,
    title: str,
) -> None:
    order = np.argsort(data["x_mm"].to_numpy())
    x_mm = data["x_mm"].to_numpy()[order]
    actual = data["width_mm"].to_numpy()[order]
    baseline_prediction = np.asarray(baseline["test_prediction"])[order]
    corrected_prediction = np.asarray(corrected["test_prediction"])[order]
    lower = np.asarray(corrected["test_lower"])[order]
    upper = np.asarray(corrected["test_upper"])[order]
    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(x_mm, actual, color="#17324d", linewidth=2.0, label="Measured width")
    axis.plot(
        x_mm,
        baseline_prediction,
        color="#999999",
        linewidth=1.4,
        label="Original notebook baseline",
    )
    axis.plot(
        x_mm,
        corrected_prediction,
        color="#d65f2e",
        linewidth=1.8,
        label="Audited model",
    )
    axis.fill_between(
        x_mm,
        lower,
        upper,
        color="#f2b38d",
        alpha=0.35,
        label="90% split-conformal interval",
    )
    axis.set(
        title=title,
        xlabel="Physical position along scan, x (mm)",
        ylabel="Local track width (mm)",
    )
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(raw_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    aligned_tracks = []
    geometry_tracks = []
    from fmrg_submission.modeling import align_geometry_to_frames

    for track_id in TRACK_IDS:
        print(f"Processing Track {track_id}...", flush=True)
        descriptors, geometry = _load_track(raw_dir, track_id)
        aligned = align_geometry_to_frames(
            descriptors, geometry, max_gap_mm=0.10
        )
        aligned = aligned[
            aligned["x_mm"].between(*STEADY_STATE_RANGE_MM)
        ].copy()
        aligned_tracks.append(aligned)
        geometry_tracks.append(geometry.assign(track_id=track_id))
        aligned.to_csv(tables_dir / f"aligned_track_{track_id}.csv", index=False)

    data = _add_notebook_features(pd.concat(aligned_tracks, ignore_index=True))
    train = data[data["track_id"].isin(TRAIN_TRACKS)].copy()
    validation = data[data["track_id"] == VALIDATION_TRACK].copy()
    test = data[data["track_id"] == TEST_TRACK].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("One or more train/validation/test splits are empty")

    baseline_features = [
        "size_norm",
        "temp_norm",
        "delta_size",
        "lag1_size",
        "lag2_size",
    ]
    excluded = {
        "x_mm",
        "track_id",
        "width_mm",
        "left_mm",
        "right_mm",
        "center_mm",
        *baseline_features,
    }
    thermal_features = [
        column
        for column in data.columns
        if column not in excluded and not column.startswith("sem_")
    ]
    sem_features = [column for column in data.columns if column.startswith("sem_")]
    development = pd.concat([train, validation], ignore_index=True)
    comparison = fit_grouped_final_comparison(
        development,
        test,
        baseline_features=baseline_features,
        corrected_feature_sets={
            "thermal": thermal_features,
            "thermal_plus_masked_sem": thermal_features + sem_features,
        },
        coverage=0.90,
    )
    baseline = comparison["baseline"]
    corrected = comparison["corrected"]

    center_oof_prediction = np.full(len(development), np.nan)
    for held_track in sorted(development["track_id"].unique()):
        train_mask = development["track_id"] != held_track
        held_mask = ~train_mask
        center_fold_model = clone(corrected["model"]).fit(
            development.loc[train_mask, corrected["features"]],
            development.loc[train_mask, "center_mm"],
        )
        center_oof_prediction[held_mask.to_numpy()] = center_fold_model.predict(
            development.loc[held_mask, corrected["features"]]
        )
    center_half_width = group_robust_conformal_half_width(
        np.abs(
            development["center_mm"].to_numpy() - center_oof_prediction
        ),
        development["track_id"].to_numpy(),
        coverage=0.90,
    )
    center_model = clone(corrected["model"]).fit(
        development[corrected["features"]], development["center_mm"]
    )
    test_center_prediction = center_model.predict(test[corrected["features"]])
    test_left_prediction = (
        test_center_prediction - 0.5 * corrected["test_prediction"]
    )
    test_right_prediction = (
        test_center_prediction + 0.5 * corrected["test_prediction"]
    )
    boundary_metrics = {
        "left": point_metrics(test["left_mm"], test_left_prediction),
        "right": point_metrics(test["right_mm"], test_right_prediction),
        "mean_boundary_mae_mm": float(
            0.5
            * (
                np.mean(np.abs(test["left_mm"] - test_left_prediction))
                + np.mean(np.abs(test["right_mm"] - test_right_prediction))
            )
        ),
        "center_calibration_half_width_mm": center_half_width,
    }

    importance_train = development[
        development["track_id"] != VALIDATION_TRACK
    ]
    importance_validation = development[
        development["track_id"] == VALIDATION_TRACK
    ]
    importance_model = clone(corrected["model"]).fit(
        importance_train[corrected["features"]],
        importance_train["width_mm"],
    )
    importance = permutation_importance(
        importance_model,
        importance_validation[corrected["features"]],
        importance_validation["width_mm"],
        scoring="neg_mean_absolute_error",
        n_repeats=20,
        random_state=42,
    )
    importance_table = pd.DataFrame(
        {
            "feature": corrected["features"],
            "importance_mae_increase_mm": importance.importances_mean,
            "importance_std_mm": importance.importances_std,
        }
    ).sort_values("importance_mae_increase_mm", ascending=False)
    importance_table.to_csv(tables_dir / "feature_importance.csv", index=False)

    test_predictions = test[
        ["track_id", "x_mm", "width_mm", "left_mm", "right_mm", "center_mm"]
    ].copy()
    test_predictions["baseline_width_prediction_mm"] = baseline["test_prediction"]
    test_predictions["corrected_width_prediction_mm"] = corrected[
        "test_prediction"
    ]
    test_predictions["corrected_width_lower_90_mm"] = corrected["test_lower"]
    test_predictions["corrected_width_upper_90_mm"] = corrected["test_upper"]
    test_predictions["corrected_center_prediction_mm"] = test_center_prediction
    test_predictions["corrected_left_prediction_mm"] = test_left_prediction
    test_predictions["corrected_right_prediction_mm"] = test_right_prediction
    test_predictions.to_csv(tables_dir / "track21_predictions.csv", index=False)

    development_predictions = development[
        ["track_id", "x_mm", "width_mm", "left_mm", "right_mm", "center_mm"]
    ].copy()
    development_predictions["baseline_width_oof_prediction_mm"] = baseline[
        "development_oof_prediction"
    ]
    development_predictions["corrected_width_oof_prediction_mm"] = corrected[
        "development_oof_prediction"
    ]
    development_predictions["corrected_width_lower_90_mm"] = (
        corrected["development_oof_prediction"]
        - corrected["calibration_half_width_mm"]
    )
    development_predictions["corrected_width_upper_90_mm"] = (
        corrected["development_oof_prediction"]
        + corrected["calibration_half_width_mm"]
    )
    development_predictions.to_csv(
        tables_dir / "development_oof_predictions.csv", index=False
    )

    _plot_predictions(
        figures_dir / "track21_held_out_comparison.png",
        test,
        baseline,
        corrected,
        title="Untouched held-out test: Track 21",
    )
    validation_mask = development["track_id"] == VALIDATION_TRACK
    validation = development.loc[validation_mask].reset_index(drop=True)
    validation_baseline = {
        "test_prediction": baseline["development_oof_prediction"][
            validation_mask.to_numpy()
        ]
    }
    validation_corrected = {
        "test_prediction": corrected["development_oof_prediction"][
            validation_mask.to_numpy()
        ]
    }
    validation_corrected["test_lower"] = (
        validation_corrected["test_prediction"]
        - corrected["calibration_half_width_mm"]
    )
    validation_corrected["test_upper"] = (
        validation_corrected["test_prediction"]
        + corrected["calibration_half_width_mm"]
    )
    _plot_predictions(
        figures_dir / "track14_validation_comparison.png",
        validation,
        validation_baseline,
        validation_corrected,
        title="Calibration track: Track 14",
    )

    top_importance = importance_table.head(12).sort_values(
        "importance_mae_increase_mm"
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.barh(
        top_importance["feature"],
        top_importance["importance_mae_increase_mm"] * 1000.0,
        color="#d65f2e",
    )
    axis.set(
        xlabel="Validation MAE increase after permutation (µm)",
        title="Most informative thermal / substrate descriptors",
    )
    axis.grid(axis="x", alpha=0.22)
    figure.tight_layout()
    figure.savefig(figures_dir / "feature_importance.png", dpi=220)
    plt.close(figure)

    baseline_mae = baseline["test_metrics"]["mae_mm"]
    corrected_mae = corrected["test_metrics"]["mae_mm"]
    improvement_percent = 100.0 * (baseline_mae - corrected_mae) / baseline_mae
    metrics = {
        "data_split": {
            "development_tracks": [*TRAIN_TRACKS, VALIDATION_TRACK],
            "selection_and_calibration": "leave-one-track-out cross-validation",
            "untouched_test_track": TEST_TRACK,
            "steady_state_x_mm": list(STEADY_STATE_RANGE_MM),
            "samples": {
                "development": len(development),
                "test": len(test),
            },
        },
        "baseline": {
            key: value for key, value in baseline.items() if key != "model"
        },
        "corrected": {
            key: value for key, value in corrected.items() if key != "model"
        },
        "boundary_metrics": boundary_metrics,
        "held_out_mae_improvement_percent": improvement_percent,
        "masked_sem_selected": corrected["feature_set"]
        == "thermal_plus_masked_sem",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_ready(metrics), indent=2) + "\n"
    )
    metric_rows = []
    for model_name, result in [("original", baseline), ("audited", corrected)]:
        metric_rows.append(
            {
                "model": model_name,
                "split": "development_oof",
                **result["development_oof_metrics"],
            }
        )
        metric_rows.append(
            {
                "model": model_name,
                "split": "test",
                **result["test_metrics"],
                **result["test_interval_metrics"],
            }
        )
    pd.DataFrame(metric_rows).to_csv(tables_dir / "metric_summary.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing thermal/, sem/, and height_maps/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = run(args.raw_dir.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "baseline_test": metrics["baseline"]["test_metrics"],
                "corrected_test": metrics["corrected"]["test_metrics"],
                "improvement_percent": metrics[
                    "held_out_mae_improvement_percent"
                ],
                "selected_model": metrics["corrected"]["model_name"],
                "selected_feature_set": metrics["corrected"]["feature_set"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
