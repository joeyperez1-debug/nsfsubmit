"""Height-map processing and local track-geometry extraction."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


def _interpolate_profile(y_mm: np.ndarray, z_mm: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(z_mm)
    if valid.sum() < max(20, len(z_mm) // 8):
        return None
    return np.interp(y_mm, y_mm[valid], z_mm[valid])


def _robust_line(y_mm: np.ndarray, z_mm: np.ndarray) -> np.ndarray:
    """Fit a substrate line while down-weighting the raised bead."""
    design = np.column_stack([y_mm, np.ones_like(y_mm)])
    keep = np.ones(len(y_mm), dtype=bool)
    coef = np.linalg.lstsq(design, z_mm, rcond=None)[0]
    for _ in range(4):
        residual = z_mm - design @ coef
        cutoff = np.quantile(residual[keep], 0.58)
        keep = residual <= cutoff
        if keep.sum() < 20:
            break
        coef = np.linalg.lstsq(design[keep], z_mm[keep], rcond=None)[0]
    return design @ coef


def _crossing(
    y_mm: np.ndarray,
    residual_mm: np.ndarray,
    center_index: int,
    threshold_mm: float,
    direction: int,
) -> float:
    index = center_index
    while 0 <= index + direction < len(y_mm):
        next_index = index + direction
        if residual_mm[next_index] < threshold_mm:
            y0, y1 = y_mm[index], y_mm[next_index]
            z0, z1 = residual_mm[index], residual_mm[next_index]
            if np.isclose(z0, z1):
                return float(0.5 * (y0 + y1))
            fraction = (threshold_mm - z0) / (z1 - z0)
            return float(y0 + fraction * (y1 - y0))
        index = next_index
    return float("nan")


def extract_local_geometry(
    z_mm: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    *,
    threshold_fraction: float = 0.30,
    minimum_peak_um: float = 3.0,
    search_y_mm: tuple[float, float] = (0.30, 1.65),
) -> dict[str, np.ndarray]:
    """Extract local left/right boundaries and width from each cross-section.

    A robust line removes substrate slope. Boundaries are the connected
    threshold crossings around the bead peak, not a count of every elevated
    pixel in the column. This makes isolated dust and missing profilometer
    pixels much less influential.
    """
    z_mm = np.asarray(z_mm, dtype=float)
    x_mm = np.asarray(x_mm, dtype=float)
    y_mm = np.asarray(y_mm, dtype=float)
    if z_mm.shape != (len(y_mm), len(x_mm)):
        raise ValueError("z_mm must have shape (len(y_mm), len(x_mm))")

    n_x = len(x_mm)
    raw_center = np.full(n_x, np.nan)
    profiles: list[np.ndarray | None] = []
    search = (y_mm >= search_y_mm[0]) & (y_mm <= search_y_mm[1])
    search_indices = np.flatnonzero(search)

    for j in range(n_x):
        profile = _interpolate_profile(y_mm, z_mm[:, j])
        profiles.append(profile)
        if profile is None:
            continue
        residual = gaussian_filter1d(profile - _robust_line(y_mm, profile), 2.0)
        raw_center[j] = y_mm[search_indices[np.argmax(residual[search])]]

    center_filled = raw_center.copy()
    finite_center = np.isfinite(raw_center)
    if finite_center.any():
        center_filled[~finite_center] = np.interp(
            x_mm[~finite_center], x_mm[finite_center], raw_center[finite_center]
        )
        filter_size = min(101, n_x if n_x % 2 else n_x - 1)
        filter_size = max(3, filter_size)
        center_filled = median_filter(center_filled, size=filter_size, mode="nearest")
        sigma_x = max(1.0, 0.20 / max(np.median(np.diff(x_mm)), 1e-6))
        center_filled = gaussian_filter1d(center_filled, sigma=sigma_x)

    left = np.full(n_x, np.nan)
    right = np.full(n_x, np.nan)
    peak_um = np.full(n_x, np.nan)
    minimum_peak_mm = minimum_peak_um / 1000.0

    for j, profile in enumerate(profiles):
        if profile is None or not np.isfinite(center_filled[j]):
            continue
        # Keep the line fit clear of the bead shoulders. A 30%-height
        # boundary can sit roughly 0.6 mm from center for the widest tracks.
        outside = np.abs(y_mm - center_filled[j]) >= 0.65
        if outside.sum() >= 20:
            design = np.column_stack([y_mm[outside], np.ones(outside.sum())])
            coef = np.linalg.lstsq(design, profile[outside], rcond=None)[0]
            baseline = coef[0] * y_mm + coef[1]
        else:
            baseline = _robust_line(y_mm, profile)
        residual = gaussian_filter1d(profile - baseline, 2.0)
        local_search = np.abs(y_mm - center_filled[j]) <= 0.25
        if not local_search.any():
            continue
        local_indices = np.flatnonzero(local_search)
        center_index = local_indices[np.argmax(residual[local_search])]
        peak = residual[center_index]
        peak_um[j] = peak * 1000.0
        if not np.isfinite(peak) or peak < minimum_peak_mm:
            continue
        threshold = max(minimum_peak_mm * 0.60, threshold_fraction * peak)
        left[j] = _crossing(y_mm, residual, center_index, threshold, -1)
        right[j] = _crossing(y_mm, residual, center_index, threshold, 1)

    width = right - left
    valid = (
        np.isfinite(width)
        & (width >= 0.20)
        & (width <= 1.60)
        & np.isfinite(peak_um)
    )
    left[~valid] = np.nan
    right[~valid] = np.nan
    width[~valid] = np.nan
    center = 0.5 * (left + right)
    return {
        "x_mm": x_mm,
        "left_mm": left,
        "right_mm": right,
        "center_mm": center,
        "width_mm": width,
        "peak_height_um": peak_um,
        "valid": valid,
    }


def smooth_local_geometry(
    geometry: dict[str, np.ndarray], *, window_mm: float = 0.40
) -> dict[str, np.ndarray]:
    """Denoise scanner-scale boundary jitter over a stated physical window."""
    result = {key: np.asarray(value).copy() for key, value in geometry.items()}
    x_mm = result["x_mm"]
    valid = result["valid"].astype(bool)
    if valid.sum() < 2:
        return result
    spacing = float(np.median(np.diff(x_mm)))
    window = max(1, int(round(window_mm / max(spacing, 1e-12))))
    if window % 2 == 0:
        window += 1
    for column in ("left_mm", "right_mm"):
        values = result[column]
        filled = np.interp(x_mm, x_mm[valid], values[valid])
        filtered = median_filter(filled, size=window, mode="nearest")
        filtered = gaussian_filter1d(
            filtered, sigma=max(1.0, 0.25 * window), mode="nearest"
        )
        result[column][valid] = filtered[valid]
    result["center_mm"] = 0.5 * (result["left_mm"] + result["right_mm"])
    result["width_mm"] = result["right_mm"] - result["left_mm"]
    for column in ("left_mm", "right_mm", "center_mm", "width_mm"):
        result[column][~valid] = np.nan
    return result
