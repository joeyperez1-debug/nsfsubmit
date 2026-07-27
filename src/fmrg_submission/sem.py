"""Leakage-safe SEM descriptors from substrate-only image regions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage


def _texture_descriptors(
    region: np.ndarray, *, prefix: str
) -> dict[str, float]:
    region = np.asarray(region, dtype=float)
    if region.ndim != 2 or not region.size:
        raise ValueError("SEM flank region must be a nonempty grayscale image")
    gradient_y, gradient_x = np.gradient(region)
    magnitude = np.hypot(gradient_x, gradient_y)
    threshold = float(np.percentile(magnitude, 90))
    orientation = np.arctan2(gradient_y, gradient_x)
    weights, _ = np.histogram(
        orientation,
        bins=8,
        range=(-np.pi, np.pi),
        weights=magnitude,
    )
    probability = weights / max(float(weights.sum()), 1e-12)
    nonzero = probability > 0
    entropy = float(-np.sum(probability[nonzero] * np.log2(probability[nonzero])))
    return {
        f"{prefix}_mean": float(np.mean(region)),
        f"{prefix}_std": float(np.std(region)),
        f"{prefix}_gradient_mean": float(np.mean(magnitude)),
        f"{prefix}_gradient_p95": float(np.percentile(magnitude, 95)),
        f"{prefix}_laplacian_std": float(np.std(ndimage.laplace(region))),
        f"{prefix}_edge_density": float(np.mean(magnitude > threshold)),
        f"{prefix}_orientation_entropy": entropy,
    }


def flank_sem_descriptors(
    image: np.ndarray,
    left_row: float,
    right_row: float,
    *,
    flank_width_px: int,
) -> dict[str, float]:
    """Describe fixed substrate bands immediately outside registered boundaries."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("SEM image must be grayscale")
    if flank_width_px < 1:
        raise ValueError("flank_width_px must be positive")
    left = int(round(left_row))
    right = int(round(right_row))
    if not 0 < left <= right < image.shape[0] - 1:
        raise ValueError("registered boundary rows must lie inside the image")
    left_region = image[max(0, left - flank_width_px) : left, :]
    right_region = image[
        right + 1 : min(image.shape[0], right + 1 + flank_width_px), :
    ]
    left_features = _texture_descriptors(left_region, prefix="sem_left")
    right_features = _texture_descriptors(right_region, prefix="sem_right")
    result = {**left_features, **right_features}
    for statistic in (
        "mean",
        "std",
        "gradient_mean",
        "gradient_p95",
        "laplacian_std",
        "edge_density",
        "orientation_entropy",
    ):
        result[f"sem_flank_difference_{statistic}"] = (
            right_features[f"sem_right_{statistic}"]
            - left_features[f"sem_left_{statistic}"]
        )
    return result


def masked_sem_descriptors(
    image: np.ndarray, *, mask_fraction: float = 0.30
) -> dict[str, float]:
    """Describe texture after excluding the processed center band."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("SEM image must be grayscale")
    half_mask = 0.5 * mask_fraction
    low = int(round(image.shape[0] * (0.5 - half_mask)))
    high = int(round(image.shape[0] * (0.5 + half_mask)))
    regions = [image[:low], image[high:]]
    substrate = np.concatenate([region.ravel() for region in regions])
    # Differentiate each substrate region independently so the excluded
    # center cannot leak into the first unmasked pixel through a stencil.
    gradients = []
    laplacians = []
    for region in regions:
        gradient_y, gradient_x = np.gradient(region)
        gradients.append(np.hypot(gradient_x, gradient_y).ravel())
        laplacians.append(ndimage.laplace(region).ravel())
    substrate_gradient = np.concatenate(gradients)
    substrate_laplacian = np.concatenate(laplacians)
    return {
        "sem_mean": float(np.mean(substrate)),
        "sem_std": float(np.std(substrate)),
        "sem_gradient_mean": float(np.mean(substrate_gradient)),
        "sem_gradient_p95": float(np.percentile(substrate_gradient, 95)),
        "sem_laplacian_std": float(np.std(substrate_laplacian)),
    }


def sem_tile_centers_mm(
    tile_count: int, *, tile_width_mm: float = 6.41
) -> np.ndarray:
    """Return physical centers; tile 01 is nearest the 100 mm side."""
    indices = np.arange(tile_count, dtype=float)
    return 100.0 - (indices + 0.5) * tile_width_mm


def extract_sem_descriptors_at_positions(
    tile_paths: list,
    x_mm: np.ndarray,
    *,
    mask_fraction: float = 0.30,
    tile_width_mm: float = 6.41,
    boundary_rows: list[tuple[float, float]] | None = None,
    flank_width_px: int = 40,
    registration_uncertainty_mm: float = 0.0,
) -> pd.DataFrame:
    """Interpolate substrate-only tile descriptors onto frame coordinates."""
    if registration_uncertainty_mm < 0:
        raise ValueError("registration_uncertainty_mm must be nonnegative")
    if boundary_rows is not None and len(boundary_rows) != len(tile_paths):
        raise ValueError("boundary_rows must match the number of SEM tiles")
    records = []
    for index, path in enumerate(tile_paths):
        image = np.asarray(ImageOps.grayscale(Image.open(path)))
        if boundary_rows is None:
            records.append(
                masked_sem_descriptors(image, mask_fraction=mask_fraction)
            )
        else:
            left_row, right_row = boundary_rows[index]
            records.append(
                flank_sem_descriptors(
                    image,
                    left_row,
                    right_row,
                    flank_width_px=flank_width_px,
                )
            )
    if not records:
        raise ValueError("No SEM tiles were provided")
    tile_data = pd.DataFrame.from_records(records)
    tile_data["x_mm"] = sem_tile_centers_mm(
        len(records), tile_width_mm=tile_width_mm
    )
    tile_data = tile_data.sort_values("x_mm")
    result = pd.DataFrame({"x_mm": np.asarray(x_mm, dtype=float)})
    for column in tile_data.columns:
        if column == "x_mm":
            continue
        result[column] = np.interp(
            result["x_mm"], tile_data["x_mm"], tile_data[column]
        )
    result["sem_registration_uncertainty_mm"] = float(
        registration_uncertainty_mm
    )
    result.attrs["sem_registration_mode"] = (
        "center_mask_fallback" if boundary_rows is None else "registered_flanks"
    )
    return result
