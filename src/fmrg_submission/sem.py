"""Leakage-safe SEM descriptors from substrate-only image regions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage


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
) -> pd.DataFrame:
    """Interpolate substrate-only tile descriptors onto frame coordinates."""
    records = []
    for path in tile_paths:
        image = np.asarray(ImageOps.grayscale(Image.open(path)))
        records.append(masked_sem_descriptors(image, mask_fraction=mask_fraction))
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
    return result
