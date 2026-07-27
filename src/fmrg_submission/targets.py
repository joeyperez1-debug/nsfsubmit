"""Hierarchical targets and physically constrained boundary reconstruction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_hierarchical_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Add robust per-track baselines and local center/log-width residuals."""
    required = {"track_id", "center_mm", "width_mm"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")
    width = data["width_mm"].to_numpy(dtype=float)
    center = data["center_mm"].to_numpy(dtype=float)
    if not np.isfinite(width).all() or np.any(width <= 0):
        raise ValueError("width_mm values must be finite and positive")
    if not np.isfinite(center).all():
        raise ValueError("center_mm values must be finite")

    result = data.copy()
    result["log_width"] = np.log(width)
    grouped = result.groupby("track_id", sort=False)
    result["baseline_center_mm"] = grouped["center_mm"].transform("median")
    result["baseline_log_width"] = grouped["log_width"].transform("median")
    result["center_residual_mm"] = (
        result["center_mm"] - result["baseline_center_mm"]
    )
    result["log_width_residual"] = (
        result["log_width"] - result["baseline_log_width"]
    )
    return result


def track_thermal_summaries(
    data: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Return robust track-level summaries computed from thermal evidence."""
    missing = {"track_id", *feature_columns}.difference(data.columns)
    if missing:
        raise ValueError(f"Missing summary columns: {sorted(missing)}")
    records: list[dict[str, float | int]] = []
    for track_id, frame in data.groupby("track_id", sort=True):
        record: dict[str, float | int] = {"track_id": track_id}
        for column in feature_columns:
            values = frame[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                record[f"{column}__median"] = 0.0
                record[f"{column}__iqr"] = 0.0
                record[f"{column}__p10"] = 0.0
                record[f"{column}__p90"] = 0.0
                continue
            p10, p25, p50, p75, p90 = np.percentile(
                finite, [10, 25, 50, 75, 90]
            )
            record[f"{column}__median"] = float(p50)
            record[f"{column}__iqr"] = float(p75 - p25)
            record[f"{column}__p10"] = float(p10)
            record[f"{column}__p90"] = float(p90)
        records.append(record)
    return pd.DataFrame.from_records(records)


def reconstruct_geometry(
    baseline_center: np.ndarray,
    baseline_log_width: np.ndarray,
    center_residual: np.ndarray,
    log_width_residual: np.ndarray,
) -> pd.DataFrame:
    """Reconstruct positive width and ordered boundaries from model outputs."""
    baseline_center = np.asarray(baseline_center, dtype=float)
    baseline_log_width = np.asarray(baseline_log_width, dtype=float)
    center_residual = np.asarray(center_residual, dtype=float)
    log_width_residual = np.asarray(log_width_residual, dtype=float)
    arrays = (
        baseline_center,
        baseline_log_width,
        center_residual,
        log_width_residual,
    )
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("All reconstruction inputs must have the same shape")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Reconstruction inputs must be finite")

    center = baseline_center + center_residual
    width = np.exp(
        np.clip(baseline_log_width + log_width_residual, -20.0, 20.0)
    )
    return pd.DataFrame(
        {
            "center_mm": center,
            "width_mm": width,
            "left_mm": center - 0.5 * width,
            "right_mm": center + 0.5 * width,
        }
    )
