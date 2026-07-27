"""Locally scaled, leakage-safe conformal prediction intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import conformal_half_width


def fit_local_scale(
    calibration: pd.DataFrame,
    absolute_residuals: np.ndarray,
    *,
    feature_columns: list[str],
) -> object:
    """Fit a positive local residual-scale model in log space."""
    residuals = np.asarray(absolute_residuals, dtype=float)
    if len(calibration) != len(residuals):
        raise ValueError("calibration and residuals must have the same length")
    if np.any(~np.isfinite(residuals)) or np.any(residuals < 0):
        raise ValueError("absolute residuals must be finite and nonnegative")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    model.fit(
        calibration[feature_columns],
        np.log(np.maximum(residuals, 1e-6)),
    )
    return model


def predict_local_scale(
    model: object,
    data: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> np.ndarray:
    """Predict strictly positive local residual scales."""
    log_scale = np.asarray(model.predict(data[feature_columns]), dtype=float)
    return np.exp(np.clip(log_scale, -20.0, 20.0)).clip(min=1e-6)


def normalized_conformal_interval(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    scale_cal: np.ndarray,
    pred_test: np.ndarray,
    scale_test: np.ndarray,
    *,
    coverage: float = 0.90,
    groups_cal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calibrate residuals after dividing by predicted local difficulty."""
    y_cal = np.asarray(y_cal, dtype=float)
    pred_cal = np.asarray(pred_cal, dtype=float)
    scale_cal = np.asarray(scale_cal, dtype=float)
    pred_test = np.asarray(pred_test, dtype=float)
    scale_test = np.asarray(scale_test, dtype=float)
    if not (len(y_cal) == len(pred_cal) == len(scale_cal)):
        raise ValueError("calibration arrays must have the same length")
    if len(pred_test) != len(scale_test):
        raise ValueError("test prediction and scale must have the same length")
    if np.any(scale_cal <= 0) or np.any(scale_test <= 0):
        raise ValueError("conformal scales must be positive")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")

    scores = np.abs(y_cal - pred_cal) / scale_cal
    if groups_cal is None:
        quantile = conformal_half_width(scores, coverage=coverage)
    else:
        groups_cal = np.asarray(groups_cal)
        if len(groups_cal) != len(scores):
            raise ValueError("groups_cal must match calibration length")
        quantile = max(
            conformal_half_width(
                scores[groups_cal == group], coverage=coverage
            )
            for group in np.unique(groups_cal)
        )
    half_width = quantile * scale_test
    return pred_test - half_width, pred_test + half_width, float(quantile)
