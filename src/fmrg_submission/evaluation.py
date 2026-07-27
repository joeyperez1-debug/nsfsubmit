"""Nested track-grouped evaluation for local laser-track geometry."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import clone

from .modeling import (
    fit_hierarchical_candidate,
    group_robust_conformal_half_width,
    interval_metrics,
)
from .uncertainty import (
    fit_local_scale,
    normalized_conformal_interval,
    predict_local_scale,
)


def _finite_mean(values: list[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _signal_descriptors(
    x_mm: np.ndarray, values: np.ndarray
) -> tuple[float, float]:
    order = np.argsort(x_mm)
    x_mm = np.asarray(x_mm, dtype=float)[order]
    values = np.asarray(values, dtype=float)[order]
    if len(values) < 3:
        return 0.0, 0.0
    spacing = np.diff(x_mm)
    valid = spacing > 1e-12
    gradient = np.diff(values)[valid] / spacing[valid]
    roughness = float(np.sqrt(np.mean(gradient**2))) if len(gradient) else 0.0
    window = min(21, len(values) if len(values) % 2 else len(values) - 1)
    if window < 5:
        smooth = np.full_like(values, np.mean(values))
    else:
        smooth = savgol_filter(values, window_length=window, polyorder=2)
    waviness = float(np.sqrt(np.mean((values - smooth) ** 2)))
    return roughness, waviness


def track_balanced_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    """Score each track independently, then average tracks equally."""
    required = {
        "track_id",
        "x_mm",
        "width_mm",
        "width_prediction_mm",
        "center_mm",
        "center_prediction_mm",
        "left_mm",
        "left_prediction_mm",
        "right_mm",
        "right_prediction_mm",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    per_track: dict[int, dict[str, float]] = {}
    for track_id, frame in predictions.groupby("track_id", sort=True):
        actual_width = frame["width_mm"].to_numpy(dtype=float)
        predicted_width = frame["width_prediction_mm"].to_numpy(dtype=float)
        actual_center = frame["center_mm"].to_numpy(dtype=float)
        predicted_center = frame["center_prediction_mm"].to_numpy(dtype=float)
        actual_residual = actual_width - np.mean(actual_width)
        predicted_residual = predicted_width - np.mean(predicted_width)
        actual_std = float(np.std(actual_residual))
        predicted_std = float(np.std(predicted_residual))
        if actual_std > 1e-12 and predicted_std > 1e-12:
            residual_correlation = float(
                np.corrcoef(actual_residual, predicted_residual)[0, 1]
            )
        else:
            residual_correlation = float("nan")
        variation_ratio = predicted_std / max(actual_std, 1e-12)
        actual_roughness, actual_waviness = _signal_descriptors(
            frame["x_mm"].to_numpy(dtype=float), actual_width
        )
        predicted_roughness, predicted_waviness = _signal_descriptors(
            frame["x_mm"].to_numpy(dtype=float), predicted_width
        )
        width_errors = actual_width - predicted_width
        center_errors = actual_center - predicted_center
        left_error = np.abs(
            frame["left_mm"].to_numpy(dtype=float)
            - frame["left_prediction_mm"].to_numpy(dtype=float)
        )
        right_error = np.abs(
            frame["right_mm"].to_numpy(dtype=float)
            - frame["right_prediction_mm"].to_numpy(dtype=float)
        )
        denominator = np.sum(
            (actual_width - np.mean(actual_width)) ** 2
        )
        r2 = (
            1.0 - np.sum(width_errors**2) / denominator
            if denominator > 1e-12
            else float("nan")
        )
        per_track[int(track_id)] = {
            "width_mae_mm": float(np.mean(np.abs(width_errors))),
            "width_rmse_mm": float(np.sqrt(np.mean(width_errors**2))),
            "width_r2": float(r2),
            "center_mae_mm": float(np.mean(np.abs(center_errors))),
            "mean_boundary_mae_mm": float(
                0.5 * (np.mean(left_error) + np.mean(right_error))
            ),
            "residual_correlation": residual_correlation,
            "variation_std_ratio": float(variation_ratio),
            "variation_std_ratio_error": float(abs(variation_ratio - 1.0)),
            "roughness_error": float(abs(predicted_roughness - actual_roughness)),
            "waviness_error_mm": float(
                abs(predicted_waviness - actual_waviness)
            ),
        }

    def values(name: str) -> list[float]:
        return [track[name] for track in per_track.values()]

    return {
        "track_balanced_width_mae_mm": _finite_mean(values("width_mae_mm")),
        "track_balanced_width_rmse_mm": _finite_mean(values("width_rmse_mm")),
        "track_balanced_width_r2": _finite_mean(values("width_r2")),
        "worst_track_width_mae_mm": float(max(values("width_mae_mm"))),
        "track_balanced_center_mae_mm": _finite_mean(values("center_mae_mm")),
        "mean_boundary_mae_mm": _finite_mean(values("mean_boundary_mae_mm")),
        "residual_correlation": _finite_mean(values("residual_correlation")),
        "variation_std_ratio": _finite_mean(values("variation_std_ratio")),
        "variation_std_ratio_error": _finite_mean(
            values("variation_std_ratio_error")
        ),
        "roughness_error": _finite_mean(values("roughness_error")),
        "waviness_error_mm": _finite_mean(values("waviness_error_mm")),
        "per_track": per_track,
    }


def _instantiate(factory: Callable[[], object] | object) -> object:
    return factory() if callable(factory) else clone(factory)


def nested_leave_one_track_out(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    estimator_factories: dict[str, Callable[[], object] | object],
    *,
    coverage: float = 0.90,
) -> dict[str, object]:
    """Select models in inner track folds and evaluate each untouched outer track."""
    tracks = sorted(int(track) for track in data["track_id"].unique())
    if len(tracks) < 4:
        raise ValueError("Nested evaluation requires at least four tracks")
    if not feature_sets or not estimator_factories:
        raise ValueError("At least one feature set and estimator are required")

    outer_predictions: list[pd.DataFrame] = []
    outer_folds: dict[int, dict[str, object]] = {}
    all_inner_scores: list[dict[str, object]] = []
    for outer_track in tracks:
        outer_train = data[data["track_id"] != outer_track].copy()
        outer_test = data[data["track_id"] == outer_track].copy()
        scored: list[tuple[tuple[float, float, float, str, str], dict[str, object]]] = []
        for feature_set_name, features in feature_sets.items():
            summary_features = [f"{feature}__median" for feature in features]
            for model_name, factory in estimator_factories.items():
                inner_predictions: list[pd.DataFrame] = []
                error: str | None = None
                try:
                    for inner_track in sorted(outer_train["track_id"].unique()):
                        inner_train = outer_train[
                            outer_train["track_id"] != inner_track
                        ]
                        inner_test = outer_train[
                            outer_train["track_id"] == inner_track
                        ]
                        prediction = fit_hierarchical_candidate(
                            inner_train,
                            inner_test,
                            local_features=features,
                            summary_features=summary_features,
                            estimator=_instantiate(factory),
                        )
                        inner_predictions.append(prediction)
                    combined = pd.concat(inner_predictions, ignore_index=True)
                    metrics = track_balanced_metrics(combined)
                    correlation = metrics["residual_correlation"]
                    if not np.isfinite(correlation):
                        correlation = -1.0
                    key = (
                        float(metrics["track_balanced_width_mae_mm"]),
                        float(metrics["mean_boundary_mae_mm"]),
                        -float(correlation),
                        feature_set_name,
                        model_name,
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    metrics = {}
                    key = (
                        float("inf"),
                        float("inf"),
                        float("inf"),
                        feature_set_name,
                        model_name,
                    )
                score = {
                    "outer_track": outer_track,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "metrics": metrics,
                    "error": error,
                }
                all_inner_scores.append(score)
                scored.append((key, score))

        selected_key, selected = min(scored, key=lambda item: item[0])
        if not np.isfinite(selected_key[0]):
            raise RuntimeError(f"Every candidate failed for outer Track {outer_track}")
        selected_features = feature_sets[str(selected["feature_set"])]
        selected_factory = estimator_factories[str(selected["model"])]
        calibration_predictions: list[pd.DataFrame] = []
        for calibration_track in sorted(outer_train["track_id"].unique()):
            calibration_train = outer_train[
                outer_train["track_id"] != calibration_track
            ]
            calibration_test = outer_train[
                outer_train["track_id"] == calibration_track
            ]
            calibration_prediction = fit_hierarchical_candidate(
                calibration_train,
                calibration_test,
                local_features=selected_features,
                summary_features=[
                    f"{feature}__median" for feature in selected_features
                ],
                estimator=_instantiate(selected_factory),
            )
            for feature in selected_features:
                calibration_prediction[feature] = calibration_test[
                    feature
                ].to_numpy()
            calibration_predictions.append(calibration_prediction)
        calibration = pd.concat(calibration_predictions, ignore_index=True)

        prediction = fit_hierarchical_candidate(
            outer_train,
            outer_test,
            local_features=selected_features,
            summary_features=[
                f"{feature}__median" for feature in selected_features
            ],
            estimator=_instantiate(selected_factory),
        )
        residuals = np.abs(
            calibration["width_mm"].to_numpy(dtype=float)
            - calibration["width_prediction_mm"].to_numpy(dtype=float)
        )
        scale_model = fit_local_scale(
            calibration,
            residuals,
            feature_columns=selected_features,
        )
        calibration_scale = predict_local_scale(
            scale_model,
            calibration,
            feature_columns=selected_features,
        )
        test_scale = predict_local_scale(
            scale_model,
            outer_test,
            feature_columns=selected_features,
        )
        local_lower, local_upper, local_quantile = (
            normalized_conformal_interval(
                calibration["width_mm"].to_numpy(dtype=float),
                calibration["width_prediction_mm"].to_numpy(dtype=float),
                calibration_scale,
                prediction["width_prediction_mm"].to_numpy(dtype=float),
                test_scale,
                coverage=coverage,
                groups_cal=calibration["track_id"].to_numpy(),
            )
        )
        global_half_width = group_robust_conformal_half_width(
            residuals,
            calibration["track_id"].to_numpy(),
            coverage=coverage,
        )
        prediction["predicted_residual_scale_mm"] = test_scale
        prediction["width_lower_90_mm"] = local_lower
        prediction["width_upper_90_mm"] = local_upper
        prediction["global_width_lower_90_mm"] = (
            prediction["width_prediction_mm"] - global_half_width
        )
        prediction["global_width_upper_90_mm"] = (
            prediction["width_prediction_mm"] + global_half_width
        )
        prediction["outer_track"] = outer_track
        prediction["selected_feature_set"] = selected["feature_set"]
        prediction["selected_model"] = selected["model"]
        outer_predictions.append(prediction)
        outer_folds[outer_track] = {
            "selected_feature_set": selected["feature_set"],
            "selected_model": selected["model"],
            "inner_metrics": selected["metrics"],
            "outer_metrics": track_balanced_metrics(prediction),
            "conditional_interval": {
                **interval_metrics(
                    prediction["width_mm"].to_numpy(),
                    local_lower,
                    local_upper,
                ),
                "normalized_quantile": local_quantile,
            },
            "global_interval": {
                **interval_metrics(
                    prediction["width_mm"].to_numpy(),
                    prediction["global_width_lower_90_mm"].to_numpy(),
                    prediction["global_width_upper_90_mm"].to_numpy(),
                ),
                "half_width_mm": global_half_width,
            },
        }

    predictions = pd.concat(outer_predictions, ignore_index=True)
    conditional = interval_metrics(
        predictions["width_mm"].to_numpy(),
        predictions["width_lower_90_mm"].to_numpy(),
        predictions["width_upper_90_mm"].to_numpy(),
    )
    global_interval = interval_metrics(
        predictions["width_mm"].to_numpy(),
        predictions["global_width_lower_90_mm"].to_numpy(),
        predictions["global_width_upper_90_mm"].to_numpy(),
    )
    difficulty_rank = predictions["predicted_residual_scale_mm"].rank(
        method="first"
    )
    terciles = pd.qcut(
        difficulty_rank, q=3, labels=["low", "medium", "high"]
    )
    by_difficulty: dict[str, dict[str, float]] = {}
    for label in ("low", "medium", "high"):
        mask = np.asarray(terciles == label)
        by_difficulty[label] = interval_metrics(
            predictions.loc[mask, "width_mm"].to_numpy(),
            predictions.loc[mask, "width_lower_90_mm"].to_numpy(),
            predictions.loc[mask, "width_upper_90_mm"].to_numpy(),
        )
    return {
        "predictions": predictions,
        "metrics": track_balanced_metrics(predictions),
        "outer_folds": outer_folds,
        "inner_scores": all_inner_scores,
        "uncertainty": {
            "conditional": conditional,
            "global": global_interval,
            "conditional_by_difficulty": by_difficulty,
        },
    }
