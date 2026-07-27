"""Reproducible four-track improvement experiments and promotion gate."""

from __future__ import annotations

import importlib.metadata
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import nested_leave_one_track_out, track_balanced_metrics
from .modeling import (
    align_geometry_to_frames,
    candidate_estimators,
)
from .sem import extract_sem_descriptors_at_positions
from .thermal import (
    add_within_track_normalized_features,
    extract_thermal_descriptors,
)


TRACK_IDS = (8, 10, 14, 21)
STEADY_STATE_RANGE_MM = (24.0, 96.0)
HISTORICAL_TRACK21 = {
    "original_notebook_mae_mm": 0.1589159,
    "audited_ridge_mae_mm": 0.1393538,
    "mae_improvement_percent": 12.3097,
    "role": "historical benchmark only; not used for model selection",
}
LEGACY_THERMAL_FEATURES = [
    "hot_area_px",
    "bbox_width_px",
    "bbox_height_px",
    "equivalent_diameter_px",
    "elongation",
    "centroid_row_px",
    "centroid_col_px",
    "max_temperature",
    "p95_temperature",
    "p99_temperature",
    "hot_mean_temperature",
    "thermal_mass",
    "gradient_mean",
    "gradient_p95",
    "right_left_asymmetry",
    "rear_front_asymmetry",
    "delta_hot_area_px",
    "lag1_hot_area_px",
    "roll3_hot_area_px",
    "delta_bbox_width_px",
    "lag1_bbox_width_px",
    "roll3_bbox_width_px",
    "delta_bbox_height_px",
    "lag1_bbox_height_px",
    "roll3_bbox_height_px",
    "delta_max_temperature",
    "lag1_max_temperature",
    "roll3_max_temperature",
    "delta_thermal_mass",
    "lag1_thermal_mass",
    "roll3_thermal_mass",
    "delta_rear_front_asymmetry",
    "lag1_rear_front_asymmetry",
    "roll3_rear_front_asymmetry",
    "x_scaled",
    "x_sin_1",
    "x_cos_1",
    "x_sin_2",
    "x_cos_2",
]


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def default_feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    """Build compact, multiscale, and optional SEM feature sets."""
    compact = [column for column in LEGACY_THERMAL_FEATURES if column in data]
    instantaneous = [
        column
        for column in (
            "major_axis_px",
            "minor_axis_px",
            "eccentricity",
            "orientation_rad",
            "cooling_tail_area_px",
            "cooling_tail_integral",
            "cooling_tail_decay",
            "centroid_velocity_px",
            "centroid_acceleration_px",
            "shape_change",
        )
        if column in data
    ]
    multiscale_bases = (
        "hot_area_px",
        "bbox_width_px",
        "bbox_height_px",
        "max_temperature",
        "thermal_mass",
        "right_left_asymmetry",
        "rear_front_asymmetry",
        "cooling_tail_integral",
        "cooling_tail_decay",
        "centroid_velocity_px",
        "shape_change",
    )
    multiscale = []
    for window in (5, 10, 20):
        history_fraction = f"roll{window}_history_fraction"
        if history_fraction in data:
            multiscale.append(history_fraction)
        for base in multiscale_bases:
            for statistic in ("mean", "std", "change", "slope"):
                column = f"roll{window}_{base}_{statistic}"
                if column in data:
                    multiscale.append(column)
        for threshold in (1500, 1750, 2000):
            column = (
                f"roll{window}_max_temperature_persistence_{threshold}"
            )
            if column in data:
                multiscale.append(column)
    thermal = list(dict.fromkeys(compact + instantaneous + multiscale))
    result = {"compact_thermal": compact, "multiscale_thermal": thermal}
    condition_core = [
        column
        for column in (
            "hot_area_px",
            "max_temperature",
            "thermal_mass",
            "cooling_tail_integral",
        )
        if column in data
    ]
    position = [
        column
        for column in ("x_scaled", "x_sin_1", "x_cos_1", "x_sin_2", "x_cos_2")
        if column in data
    ]
    normalized_compact = [
        f"local_{column}"
        for column in compact
        if f"local_{column}" in data and not column.startswith("x_")
    ]
    normalized_multiscale = [
        f"local_{column}"
        for column in thermal
        if f"local_{column}" in data and not column.startswith("x_")
    ]
    if normalized_compact:
        result["normalized_compact_thermal"] = list(
            dict.fromkeys(condition_core + normalized_compact + position)
        )
    if normalized_multiscale:
        result["normalized_multiscale_thermal"] = list(
            dict.fromkeys(condition_core + normalized_multiscale + position)
        )
    sem = [
        column
        for column in data.select_dtypes(include="number").columns
        if column.startswith("sem_")
    ]
    if sem:
        result["multiscale_thermal_plus_postprocess_sem"] = thermal + sem
    return {name: columns for name, columns in result.items() if columns}


def _direct_outer_predictions(
    data: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    predictions = []
    estimator = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    for outer_track in sorted(data["track_id"].unique()):
        train = data[data["track_id"] != outer_track]
        test = data[data["track_id"] == outer_track]
        model = clone(estimator).fit(
            train[features], train[["center_mm", "width_mm"]]
        )
        output = np.asarray(model.predict(test[features]), dtype=float)
        center = output[:, 0]
        width = np.maximum(output[:, 1], 1e-6)
        fold = test[
            [
                "track_id",
                "x_mm",
                "width_mm",
                "center_mm",
                "left_mm",
                "right_mm",
            ]
        ].reset_index(drop=True)
        fold["center_prediction_mm"] = center
        fold["width_prediction_mm"] = width
        fold["left_prediction_mm"] = center - 0.5 * width
        fold["right_prediction_mm"] = center + 0.5 * width
        predictions.append(fold)
    return pd.concat(predictions, ignore_index=True)


def _promotion_decision(
    incumbent: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    lower_mae = (
        candidate["track_balanced_width_mae_mm"]
        < incumbent["track_balanced_width_mae_mm"]
    )
    robust = (
        candidate["worst_track_width_mae_mm"]
        <= incumbent["worst_track_width_mae_mm"] * 1.05
    )
    spatial_checks = {
        "higher_residual_correlation": (
            candidate["residual_correlation"]
            > incumbent["residual_correlation"]
        ),
        "lower_boundary_mae": (
            candidate["mean_boundary_mae_mm"]
            < incumbent["mean_boundary_mae_mm"]
        ),
        "better_variation_scale": (
            candidate["variation_std_ratio_error"]
            < incumbent["variation_std_ratio_error"]
        ),
    }
    promoted = bool(lower_mae and robust and any(spatial_checks.values()))
    return {
        "promoted": promoted,
        "lower_track_balanced_mae": bool(lower_mae),
        "worst_track_within_five_percent": bool(robust),
        "spatial_checks": spatial_checks,
        "default_for_artifacts": "hierarchical" if promoted else "incumbent",
    }


def _software_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
    ):
        versions[package] = importlib.metadata.version(package)
    return versions


def _score_rows(inner_scores: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for score in inner_scores:
        row = {
            "outer_track": score["outer_track"],
            "feature_set": score["feature_set"],
            "model": score["model"],
            "error": score["error"],
        }
        metrics = score["metrics"]
        for key, value in metrics.items():
            if key != "per_track":
                row[key] = value
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _write_figures(
    output_dir: Path,
    incumbent_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    incumbent_metrics: dict[str, object],
    candidate_metrics: dict[str, object],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
    for axis, track_id in zip(axes.flat, TRACK_IDS, strict=True):
        candidate = candidate_predictions[
            candidate_predictions["track_id"] == track_id
        ].sort_values("x_mm")
        incumbent = incumbent_predictions[
            incumbent_predictions["track_id"] == track_id
        ].sort_values("x_mm")
        axis.plot(
            candidate["x_mm"],
            candidate["width_mm"],
            color="#17324d",
            linewidth=1.8,
            label="Measured",
        )
        axis.plot(
            incumbent["x_mm"],
            incumbent["width_prediction_mm"],
            color="#9b9b9b",
            linewidth=1.2,
            label="Direct Ridge",
        )
        axis.plot(
            candidate["x_mm"],
            candidate["width_prediction_mm"],
            color="#d65f2e",
            linewidth=1.4,
            label="Nested hierarchical",
        )
        axis.fill_between(
            candidate["x_mm"],
            candidate["width_lower_90_mm"],
            candidate["width_upper_90_mm"],
            color="#f2b38d",
            alpha=0.25,
        )
        axis.set_title(f"Outer-held Track {track_id}")
        axis.set_xlabel("x (mm)")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Local width (mm)")
    axes[1, 0].set_ylabel("Local width (mm)")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "nested_outer_predictions.png", dpi=220)
    plt.close(figure)

    labels = ["Track-balanced MAE", "Worst-track MAE", "Boundary MAE"]
    incumbent_values = [
        incumbent_metrics["track_balanced_width_mae_mm"],
        incumbent_metrics["worst_track_width_mae_mm"],
        incumbent_metrics["mean_boundary_mae_mm"],
    ]
    candidate_values = [
        candidate_metrics["track_balanced_width_mae_mm"],
        candidate_metrics["worst_track_width_mae_mm"],
        candidate_metrics["mean_boundary_mae_mm"],
    ]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(
        positions - 0.18,
        incumbent_values,
        width=0.36,
        color="#9b9b9b",
        label="Direct Ridge",
    )
    axis.bar(
        positions + 0.18,
        candidate_values,
        width=0.36,
        color="#d65f2e",
        label="Nested hierarchical",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Error (mm; lower is better)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figures / "before_after_scorecard.png", dpi=220)
    plt.close(figure)


def run_experiments(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    feature_sets: dict[str, list[str]] | None = None,
    estimator_factories: (
        dict[str, Callable[[], object] | object] | None
    ) = None,
    incumbent_features: list[str] | None = None,
    coverage: float = 0.90,
    accuracy_tolerance_mm: float = 0.02,
) -> dict[str, object]:
    """Run the fixed incumbent and nested candidate selector, then gate promotion."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = feature_sets or default_feature_sets(data)
    estimator_factories = estimator_factories or candidate_estimators()
    if incumbent_features is None:
        incumbent_features = [
            feature for feature in LEGACY_THERMAL_FEATURES if feature in data
        ]
    if not incumbent_features:
        raise ValueError("No incumbent features are available")

    incumbent_predictions = _direct_outer_predictions(data, incumbent_features)
    incumbent_metrics = track_balanced_metrics(incumbent_predictions)
    candidate = nested_leave_one_track_out(
        data,
        feature_sets,
        estimator_factories,
        coverage=coverage,
        accuracy_tolerance_mm=accuracy_tolerance_mm,
    )
    candidate_predictions = candidate["predictions"]
    candidate_metrics = candidate["metrics"]
    selections = Counter(
        (
            fold["selected_feature_set"],
            fold["selected_model"],
        )
        for fold in candidate["outer_folds"].values()
    )
    selected_pair, selected_count = min(
        selections.items(), key=lambda item: (-item[1], item[0])
    )
    sem_selections = sum(
        count
        for (feature_set, _), count in selections.items()
        if "sem" in feature_set
    )
    conditional_interval = candidate["uncertainty"]["conditional"]
    global_interval = candidate["uncertainty"]["global"]
    conditional_selected = (
        abs(conditional_interval["coverage"] - coverage)
        <= abs(global_interval["coverage"] - coverage)
        and conditional_interval["mean_width_mm"]
        <= global_interval["mean_width_mm"]
    )
    uncertainty = {
        **candidate["uncertainty"],
        "selected": "conditional" if conditional_selected else "global",
        "selection_rule": (
            "closest coverage to target, then no wider than global"
        ),
    }
    result = {
        "protocol": {
            "name": "nested leave-one-track-out",
            "outer_tracks": list(TRACK_IDS),
            "selection": "inner leave-one-track-out on the other three tracks",
            "primary_metric": "unweighted mean of per-track width MAE",
            "coverage_target": coverage,
            "accuracy_tolerance_mm": accuracy_tolerance_mm,
            "steady_state_x_mm": list(STEADY_STATE_RANGE_MM),
            "random_seed": 42,
        },
        "historical_benchmark": HISTORICAL_TRACK21,
        "incumbent": {
            "name": "direct Ridge alpha=10 under four-track outer validation",
            "features": incumbent_features,
            "metrics": incumbent_metrics,
        },
        "candidates": {
            "feature_sets": feature_sets,
            "models": sorted(estimator_factories),
            "nested_metrics": candidate_metrics,
        },
        "selected": {
            "feature_set": selected_pair[0],
            "model": selected_pair[1],
            "outer_fold_selection_count": selected_count,
            "per_outer_fold": candidate["outer_folds"],
        },
        "promotion": _promotion_decision(incumbent_metrics, candidate_metrics),
        "uncertainty": uncertainty,
        "sem_ablation": {
            "source": "post-process SEM with the processed center masked",
            "preprocess_sem_available": False,
            "outer_folds_selecting_sem": sem_selections,
            "selected_by_any_outer_fold": sem_selections > 0,
            "causal_substrate_claim": False,
        },
        "software_versions": _software_versions(),
    }
    candidate_predictions.to_csv(
        output_dir / "outer_fold_predictions.csv", index=False
    )
    incumbent_predictions.to_csv(
        output_dir / "incumbent_outer_fold_predictions.csv", index=False
    )
    _score_rows(candidate["inner_scores"]).to_csv(
        output_dir / "candidate_scores.csv", index=False
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_ready(result), indent=2) + "\n"
    )
    _write_figures(
        output_dir,
        incumbent_predictions,
        candidate_predictions,
        incumbent_metrics,
        candidate_metrics,
    )
    return result


def load_cached_aligned_data(
    cache_dir: Path,
    raw_dir: Path,
    *,
    aligned_output_dir: Path | None = None,
) -> pd.DataFrame:
    """Extract current descriptors from cached thermal frames and geometry."""
    from nsf_fmrg_data import get_sem_tile_paths

    cache_dir = Path(cache_dir)
    raw_dir = Path(raw_dir)
    if aligned_output_dir is not None:
        aligned_output_dir = Path(aligned_output_dir)
        aligned_output_dir.mkdir(parents=True, exist_ok=True)
    aligned_tracks = []
    for track_id in TRACK_IDS:
        thermal = np.load(cache_dir / f"thermal_{track_id}.npz")
        descriptors = extract_thermal_descriptors(
            thermal["frames"], thermal["x_mm_center"]
        )
        descriptors["track_id"] = track_id
        descriptors["x_scaled"] = (
            descriptors["x_mm"] - STEADY_STATE_RANGE_MM[0]
        ) / (STEADY_STATE_RANGE_MM[1] - STEADY_STATE_RANGE_MM[0])
        descriptors["x_sin_1"] = np.sin(
            2.0 * np.pi * descriptors["x_scaled"]
        )
        descriptors["x_cos_1"] = np.cos(
            2.0 * np.pi * descriptors["x_scaled"]
        )
        descriptors["x_sin_2"] = np.sin(
            4.0 * np.pi * descriptors["x_scaled"]
        )
        descriptors["x_cos_2"] = np.cos(
            4.0 * np.pi * descriptors["x_scaled"]
        )
        sem = extract_sem_descriptors_at_positions(
            get_sem_tile_paths(raw_dir / "sem", track_id),
            descriptors["x_mm"].to_numpy(),
            mask_fraction=0.30,
            registration_uncertainty_mm=0.25,
        )
        descriptors = descriptors.merge(sem, on="x_mm", validate="one_to_one")
        geometry = pd.read_csv(cache_dir / f"geometry_{track_id}.csv")
        aligned = align_geometry_to_frames(
            descriptors, geometry, max_gap_mm=0.10
        )
        aligned = aligned[
            aligned["x_mm"].between(*STEADY_STATE_RANGE_MM)
        ].reset_index(drop=True)
        if aligned_output_dir is not None:
            aligned.to_csv(
                aligned_output_dir / f"aligned_track_{track_id}.csv",
                index=False,
            )
        aligned_tracks.append(aligned)
    combined = pd.concat(aligned_tracks, ignore_index=True)
    excluded = {
        "track_id",
        "x_mm",
        "x_scaled",
        "x_sin_1",
        "x_cos_1",
        "x_sin_2",
        "x_cos_2",
        "width_mm",
        "center_mm",
        "left_mm",
        "right_mm",
    }
    thermal_columns = [
        column
        for column in combined.select_dtypes(include="number").columns
        if column not in excluded
        and not column.startswith("sem_")
        and not column.startswith("local_")
    ]
    return add_within_track_normalized_features(combined, thermal_columns)
