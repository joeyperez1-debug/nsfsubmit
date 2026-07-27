"""Interpretable frame-level thermal descriptors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


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
        covariance = np.cov(centered, rowvar=False, aweights=weights)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
        elongation = float(
            np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-12))
        )
        bbox_height = int(rows.max() - rows.min() + 1)
        bbox_width = int(cols.max() - cols.min() + 1)
        hot_mean = float(np.mean(hot_values))
    else:
        centroid_row = image_center_row
        centroid_col = image_center_col
        elongation = 0.0
        bbox_height = 0
        bbox_width = 0
        hot_mean = 0.0

    excess = np.maximum(frame - threshold, 0.0)
    total_excess = float(excess.sum())
    right = float(excess[:, int(np.ceil(image_center_col)) :].sum())
    left = float(excess[:, : int(np.ceil(image_center_col))].sum())
    rear = float(excess[int(np.ceil(image_center_row)) :, :].sum())
    front = float(excess[: int(np.ceil(image_center_row)), :].sum())
    gradient = np.hypot(*np.gradient(frame))
    return {
        "hot_area_px": float(component.sum()),
        "bbox_width_px": float(bbox_width),
        "bbox_height_px": float(bbox_height),
        "equivalent_diameter_px": float(
            2.0 * np.sqrt(component.sum() / np.pi) if component.any() else 0.0
        ),
        "elongation": elongation,
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
    }


def extract_thermal_descriptors(
    frames: np.ndarray,
    x_mm: np.ndarray | list[float],
    *,
    threshold: float = 1500.0,
) -> pd.DataFrame:
    """Return frame descriptors and short thermal-history features."""
    frames = np.asarray(frames)
    x_mm = np.asarray(x_mm, dtype=float)
    if len(frames) != len(x_mm):
        raise ValueError("frames and x_mm must have the same length")
    records = [_frame_descriptors(frame, threshold) for frame in frames]
    result = pd.DataFrame.from_records(records)
    result.insert(0, "x_mm", x_mm)

    dynamic_columns = [
        "hot_area_px",
        "bbox_width_px",
        "bbox_height_px",
        "max_temperature",
        "thermal_mass",
        "rear_front_asymmetry",
    ]
    for column in dynamic_columns:
        result[f"delta_{column}"] = result[column].diff().fillna(0.0)
        result[f"lag1_{column}"] = result[column].shift(1).bfill().fillna(0.0)
        result[f"roll3_{column}"] = (
            result[column].rolling(3, min_periods=1).mean()
        )
    return result
