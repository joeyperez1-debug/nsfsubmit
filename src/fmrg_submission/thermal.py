"""Interpretable frame-level thermal descriptors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


def _linear_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def _frame_descriptors(frame: np.ndarray, threshold: float) -> dict[str, float]:
    frame = np.asarray(frame, dtype=float)
    component = _largest_component(frame > threshold)
    rows, cols = np.nonzero(component)
    hot_values = frame[component]
    image_center_row = 0.5 * (frame.shape[0] - 1)
    image_center_col = 0.5 * (frame.shape[1] - 1)

    if len(rows):
        weights = np.maximum(hot_values - threshold, 1.0)
        centroid_row = float(np.average(rows, weights=weights))
        centroid_col = float(np.average(cols, weights=weights))
        centered = np.column_stack([rows - centroid_row, cols - centroid_col])
        if len(rows) >= 2:
            covariance = np.atleast_2d(
                np.cov(centered, rowvar=False, aweights=weights)
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(np.maximum(eigenvalues, 0.0))
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            major_vector = eigenvectors[:, order[-1]]
            major_axis = float(4.0 * np.sqrt(eigenvalues[-1]))
            minor_axis = float(4.0 * np.sqrt(eigenvalues[0]))
            elongation = major_axis / max(minor_axis, 1e-12)
            eccentricity = float(
                np.sqrt(
                    max(
                        0.0,
                        1.0 - eigenvalues[0] / max(eigenvalues[-1], 1e-12),
                    )
                )
            )
            orientation = float(np.arctan2(major_vector[0], major_vector[1]))
        else:
            major_axis = 0.0
            minor_axis = 0.0
            elongation = 0.0
            eccentricity = 0.0
            orientation = 0.0
        bbox_height = int(rows.max() - rows.min() + 1)
        bbox_width = int(cols.max() - cols.min() + 1)
        hot_mean = float(np.mean(hot_values))
    else:
        centroid_row = image_center_row
        centroid_col = image_center_col
        major_axis = 0.0
        minor_axis = 0.0
        elongation = 0.0
        eccentricity = 0.0
        orientation = 0.0
        bbox_height = 0
        bbox_width = 0
        hot_mean = 0.0

    excess = np.maximum(frame - threshold, 0.0)
    total_excess = float(excess.sum())
    right = float(excess[:, int(np.ceil(image_center_col)) :].sum())
    left = float(excess[:, : int(np.ceil(image_center_col))].sum())
    rear = float(excess[int(np.ceil(image_center_row)) :, :].sum())
    front = float(excess[: int(np.ceil(image_center_row)), :].sum())
    rear_profile = excess[int(np.ceil(image_center_row)) :, :].sum(axis=1)
    gradient = np.hypot(*np.gradient(frame))
    return {
        "hot_area_px": float(component.sum()),
        "bbox_width_px": float(bbox_width),
        "bbox_height_px": float(bbox_height),
        "equivalent_diameter_px": float(
            2.0 * np.sqrt(component.sum() / np.pi) if component.any() else 0.0
        ),
        "major_axis_px": major_axis,
        "minor_axis_px": minor_axis,
        "elongation": elongation,
        "eccentricity": eccentricity,
        "orientation_rad": orientation,
        "centroid_row_px": centroid_row,
        "centroid_col_px": centroid_col,
        "max_temperature": float(np.nanmax(frame)),
        "p95_temperature": float(np.nanpercentile(frame, 95)),
        "p99_temperature": float(np.nanpercentile(frame, 99)),
        "hot_mean_temperature": hot_mean,
        "thermal_mass": total_excess,
        "gradient_mean": float(np.nanmean(gradient)),
        "gradient_p95": float(np.nanpercentile(gradient, 95)),
        "right_left_asymmetry": (right - left) / max(total_excess, 1.0),
        "rear_front_asymmetry": (rear - front) / max(total_excess, 1.0),
        "cooling_tail_area_px": float(np.count_nonzero(rear_profile)),
        "cooling_tail_integral": rear,
        "cooling_tail_decay": _linear_slope(rear_profile),
    }


def extract_thermal_descriptors(
    frames: np.ndarray,
    x_mm: np.ndarray | list[float],
    *,
    threshold: float = 1500.0,
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Return instantaneous descriptors and causal multiscale history."""
    frames = np.asarray(frames)
    x_mm = np.asarray(x_mm, dtype=float)
    if len(frames) != len(x_mm):
        raise ValueError("frames and x_mm must have the same length")
    records = [_frame_descriptors(frame, threshold) for frame in frames]
    result = pd.DataFrame.from_records(records)
    result.insert(0, "x_mm", x_mm)

    derived: dict[str, pd.Series | np.ndarray] = {}
    row_delta = result["centroid_row_px"].diff().fillna(0.0)
    col_delta = result["centroid_col_px"].diff().fillna(0.0)
    velocity = pd.Series(np.hypot(row_delta, col_delta), index=result.index)
    derived["centroid_velocity_px"] = velocity
    derived["centroid_acceleration_px"] = velocity.diff().fillna(0.0)
    area_scale = np.maximum(
        result["hot_area_px"].shift(1).fillna(result["hot_area_px"]),
        result["hot_area_px"],
    ).clip(lower=1.0)
    derived["shape_change"] = (
        result["hot_area_px"].diff().abs().fillna(0.0) / area_scale
    )
    result = pd.concat([result, pd.DataFrame(derived, index=result.index)], axis=1)

    dynamic_columns = [
        "hot_area_px",
        "bbox_width_px",
        "bbox_height_px",
        "major_axis_px",
        "minor_axis_px",
        "elongation",
        "eccentricity",
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
        "cooling_tail_area_px",
        "cooling_tail_integral",
        "cooling_tail_decay",
        "centroid_velocity_px",
        "centroid_acceleration_px",
        "shape_change",
    ]
    history: dict[str, pd.Series | np.ndarray] = {}
    for column in dynamic_columns:
        history[f"delta_{column}"] = result[column].diff().fillna(0.0)
        history[f"lag1_{column}"] = (
            result[column].shift(1).bfill().fillna(0.0)
        )
        history[f"roll3_{column}"] = result[column].rolling(
            3, min_periods=1
        ).mean()
    frame_count = np.arange(1, len(result) + 1, dtype=float)
    for window in windows:
        if window < 1:
            raise ValueError("history windows must be positive")
        history[f"roll{window}_history_fraction"] = np.minimum(
            frame_count, float(window)
        ) / float(window)
        for column in dynamic_columns:
            rolling = result[column].rolling(window, min_periods=1)
            history[f"roll{window}_{column}_mean"] = rolling.mean()
            history[f"roll{window}_{column}_std"] = rolling.std(ddof=0)
            history[f"roll{window}_{column}_range"] = (
                rolling.max() - rolling.min()
            )
            history[f"roll{window}_{column}_change"] = (
                result[column]
                - result[column]
                .shift(window - 1)
                .fillna(result[column].iloc[0])
            )
            history[f"roll{window}_{column}_slope"] = rolling.apply(
                _linear_slope, raw=True
            )
        for persistence_threshold in (1500.0, 1750.0, 2000.0):
            indicator = (
                result["max_temperature"] > persistence_threshold
            ).astype(float)
            threshold_name = int(persistence_threshold)
            history[
                f"roll{window}_max_temperature_persistence_{threshold_name}"
            ] = indicator.rolling(window, min_periods=1).mean()
    return pd.concat(
        [result, pd.DataFrame(history, index=result.index)], axis=1
    )


def add_within_track_normalized_features(
    data: pd.DataFrame,
    feature_columns: list[str],
    *,
    prefix: str = "local_",
) -> pd.DataFrame:
    """Add label-free robust deviations from each track's thermal condition."""
    missing = {"track_id", *feature_columns}.difference(data.columns)
    if missing:
        raise ValueError(f"Missing normalization columns: {sorted(missing)}")
    grouped = data.groupby("track_id", sort=False)
    normalized: dict[str, pd.Series] = {}
    for column in feature_columns:
        values = data[column].astype(float)
        median = grouped[column].transform("median")
        q25 = grouped[column].transform(lambda series: series.quantile(0.25))
        q75 = grouped[column].transform(lambda series: series.quantile(0.75))
        scale = (q75 - q25).where((q75 - q25).abs() > 1e-12, 1.0)
        normalized[f"{prefix}{column}"] = (values - median) / scale
    return pd.concat(
        [data.copy(), pd.DataFrame(normalized, index=data.index)], axis=1
    )
