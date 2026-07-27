"""Alignment, point metrics, and split-conformal uncertainty helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


GEOMETRY_COLUMNS = ("width_mm", "left_mm", "right_mm")


def align_geometry_to_frames(
    features: pd.DataFrame,
    geometry: pd.DataFrame,
    *,
    max_gap_mm: float = 0.10,
) -> pd.DataFrame:
    """Interpolate measured geometry at each physical thermal-frame coordinate."""
    valid = geometry["valid"].to_numpy(dtype=bool)
    source_x = geometry.loc[valid, "x_mm"].to_numpy(dtype=float)
    if len(source_x) < 2:
        raise ValueError("At least two valid geometry samples are required")
    frame_x = features["x_mm"].to_numpy(dtype=float)
    insertion = np.searchsorted(source_x, frame_x)
    left_index = np.clip(insertion - 1, 0, len(source_x) - 1)
    right_index = np.clip(insertion, 0, len(source_x) - 1)
    nearest_distance = np.minimum(
        np.abs(frame_x - source_x[left_index]),
        np.abs(frame_x - source_x[right_index]),
    )
    aligned = features.loc[nearest_distance <= max_gap_mm].copy()
    for column in GEOMETRY_COLUMNS:
        aligned[column] = np.interp(
            aligned["x_mm"].to_numpy(dtype=float),
            source_x,
            geometry.loc[valid, column].to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )
    aligned["center_mm"] = 0.5 * (aligned["left_mm"] + aligned["right_mm"])
    return aligned.dropna(subset=list(GEOMETRY_COLUMNS)).reset_index(drop=True)


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_mm": float(mean_absolute_error(y_true, y_pred)),
        "rmse_mm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def conformal_half_width(
    absolute_residuals: np.ndarray, *, coverage: float = 0.90
) -> float:
    residuals = np.sort(np.asarray(absolute_residuals, dtype=float))
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        raise ValueError("No finite calibration residuals")
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * coverage))
    return float(residuals[rank - 1])


def group_robust_conformal_half_width(
    absolute_residuals: np.ndarray,
    groups: np.ndarray,
    *,
    coverage: float = 0.90,
) -> float:
    """Use the largest per-condition conformal radius for power robustness."""
    residuals = np.asarray(absolute_residuals, dtype=float)
    groups = np.asarray(groups)
    if len(residuals) != len(groups):
        raise ValueError("residuals and groups must have the same length")
    radii = [
        conformal_half_width(residuals[groups == group], coverage=coverage)
        for group in np.unique(groups)
    ]
    return float(max(radii))


def interval_metrics(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return {
        "coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
        "mean_width_mm": float(np.mean(upper - lower)),
    }


def _candidate_estimators() -> dict[str, object]:
    return {
        "ridge_1": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
        ),
        "ridge_10": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=5,
                max_features=0.75,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=42,
            ),
        ),
    }


def _evaluate_fitted(
    model: object,
    features: list[str],
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    coverage: float,
) -> dict[str, object]:
    validation_prediction = model.predict(validation[features])
    calibration = conformal_half_width(
        np.abs(validation["width_mm"].to_numpy() - validation_prediction),
        coverage=coverage,
    )
    test_prediction = model.predict(test[features])
    lower = test_prediction - calibration
    upper = test_prediction + calibration
    return {
        "validation_prediction": validation_prediction,
        "test_prediction": test_prediction,
        "calibration_half_width_mm": calibration,
        "validation_metrics": point_metrics(
            validation["width_mm"], validation_prediction
        ),
        "test_metrics": point_metrics(test["width_mm"], test_prediction),
        "test_interval_metrics": interval_metrics(
            test["width_mm"].to_numpy(), lower, upper
        ),
        "test_lower": lower,
        "test_upper": upper,
    }


def fit_leakage_safe_comparison(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    baseline_features: list[str],
    corrected_feature_sets: dict[str, list[str]],
    coverage: float = 0.90,
) -> dict[str, dict[str, object]]:
    """Fit the fixed notebook baseline and select a corrected model by train-only CV.

    Candidate selection uses leave-one-training-track-out validation. Track 14
    is then used only to calibrate uncertainty, and Track 21 is evaluated once.
    """
    target = train["width_mm"].to_numpy()
    baseline_model = GradientBoostingRegressor(
        loss="squared_error", n_estimators=100, random_state=42
    )
    baseline_model.fit(train[baseline_features], target)
    baseline = _evaluate_fitted(
        baseline_model,
        baseline_features,
        validation,
        test,
        coverage=coverage,
    )
    baseline.update(
        {
            "model_name": "notebook_gradient_boosting",
            "feature_set": "notebook",
            "features": baseline_features,
            "model": baseline_model,
        }
    )

    groups = sorted(train["track_id"].unique())
    if len(groups) < 2:
        raise ValueError("At least two training tracks are required for model selection")
    candidates = _candidate_estimators()
    scores: list[tuple[float, str, str, list[str], object]] = []
    for feature_set, features in corrected_feature_sets.items():
        for model_name, estimator in candidates.items():
            fold_errors = []
            for held_track in groups:
                fold_train = train[train["track_id"] != held_track]
                fold_test = train[train["track_id"] == held_track]
                fitted = clone(estimator).fit(
                    fold_train[features], fold_train["width_mm"]
                )
                prediction = fitted.predict(fold_test[features])
                fold_errors.append(
                    mean_absolute_error(fold_test["width_mm"], prediction)
                )
            scores.append(
                (
                    float(np.mean(fold_errors)),
                    feature_set,
                    model_name,
                    features,
                    estimator,
                )
            )
    cv_mae, feature_set, model_name, features, estimator = min(
        scores, key=lambda item: item[0]
    )
    corrected_model = clone(estimator).fit(train[features], target)
    corrected = _evaluate_fitted(
        corrected_model,
        features,
        validation,
        test,
        coverage=coverage,
    )
    corrected.update(
        {
            "model_name": model_name,
            "feature_set": feature_set,
            "features": features,
            "train_group_cv_mae_mm": cv_mae,
            "candidate_cv_mae_mm": {
                f"{fs}:{name}": score for score, fs, name, _, _ in scores
            },
            "model": corrected_model,
        }
    )
    return {"baseline": baseline, "corrected": corrected}


def _group_oof_prediction(
    estimator: object, features: list[str], development: pd.DataFrame
) -> np.ndarray:
    prediction = np.full(len(development), np.nan)
    groups = sorted(development["track_id"].unique())
    for held_track in groups:
        train_mask = development["track_id"] != held_track
        held_mask = ~train_mask
        fitted = clone(estimator).fit(
            development.loc[train_mask, features],
            development.loc[train_mask, "width_mm"],
        )
        prediction[held_mask.to_numpy()] = fitted.predict(
            development.loc[held_mask, features]
        )
    return prediction


def _fit_full_with_oof_calibration(
    estimator: object,
    features: list[str],
    development: pd.DataFrame,
    test: pd.DataFrame,
    *,
    coverage: float,
) -> dict[str, object]:
    oof_prediction = _group_oof_prediction(estimator, features, development)
    half_width = group_robust_conformal_half_width(
        np.abs(development["width_mm"].to_numpy() - oof_prediction),
        development["track_id"].to_numpy(),
        coverage=coverage,
    )
    model = clone(estimator).fit(development[features], development["width_mm"])
    test_prediction = model.predict(test[features])
    lower = test_prediction - half_width
    upper = test_prediction + half_width
    return {
        "model": model,
        "features": features,
        "development_oof_prediction": oof_prediction,
        "development_oof_metrics": point_metrics(
            development["width_mm"], oof_prediction
        ),
        "calibration_half_width_mm": half_width,
        "test_prediction": test_prediction,
        "test_lower": lower,
        "test_upper": upper,
        "test_metrics": point_metrics(test["width_mm"], test_prediction),
        "test_interval_metrics": interval_metrics(
            test["width_mm"].to_numpy(), lower, upper
        ),
    }


def fit_grouped_final_comparison(
    development: pd.DataFrame,
    test: pd.DataFrame,
    *,
    baseline_features: list[str],
    corrected_feature_sets: dict[str, list[str]],
    coverage: float = 0.90,
) -> dict[str, dict[str, object]]:
    """Final protocol: grouped OOF selection/calibration, then fit all development data."""
    if development["track_id"].nunique() < 3:
        raise ValueError("At least three development tracks are required")
    baseline_estimator = GradientBoostingRegressor(
        loss="squared_error", n_estimators=100, random_state=42
    )
    baseline = _fit_full_with_oof_calibration(
        baseline_estimator,
        baseline_features,
        development,
        test,
        coverage=coverage,
    )
    baseline.update(
        {
            "model_name": "notebook_gradient_boosting",
            "feature_set": "notebook",
        }
    )

    scores: list[tuple[float, str, str, list[str], object]] = []
    for feature_set, features in corrected_feature_sets.items():
        for model_name, estimator in _candidate_estimators().items():
            oof_prediction = _group_oof_prediction(
                estimator, features, development
            )
            score = mean_absolute_error(
                development["width_mm"], oof_prediction
            )
            scores.append(
                (float(score), feature_set, model_name, features, estimator)
            )
    cv_mae, feature_set, model_name, features, estimator = min(
        scores, key=lambda item: item[0]
    )
    corrected = _fit_full_with_oof_calibration(
        estimator,
        features,
        development,
        test,
        coverage=coverage,
    )
    corrected.update(
        {
            "model_name": model_name,
            "feature_set": feature_set,
            "development_group_cv_mae_mm": cv_mae,
            "candidate_cv_mae_mm": {
                f"{fs}:{name}": score for score, fs, name, _, _ in scores
            },
        }
    )
    return {"baseline": baseline, "corrected": corrected}
