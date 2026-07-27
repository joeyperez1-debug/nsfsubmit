import numpy as np
import pandas as pd
import pytest

from fmrg_submission.uncertainty import (
    fit_local_scale,
    normalized_conformal_interval,
    predict_local_scale,
)


def test_normalized_conformal_expands_for_difficult_regions():
    lower, upper, quantile = normalized_conformal_interval(
        y_cal=np.array([0.0, 0.2, 0.0, 0.4]),
        pred_cal=np.zeros(4),
        scale_cal=np.array([1.0, 2.0, 1.0, 4.0]),
        pred_test=np.array([0.0, 0.0]),
        scale_test=np.array([1.0, 3.0]),
        coverage=0.75,
    )

    widths = upper - lower
    assert quantile > 0
    assert np.isclose(widths[1] / widths[0], 3.0)


def test_normalized_conformal_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="positive"):
        normalized_conformal_interval(
            y_cal=np.array([0.0]),
            pred_cal=np.array([0.0]),
            scale_cal=np.array([0.0]),
            pred_test=np.array([0.0]),
            scale_test=np.array([1.0]),
        )


def test_local_scale_increases_with_observed_difficulty():
    calibration = pd.DataFrame(
        {
            "volatility": [0.0, 0.2, 0.8, 1.0],
            "history_missing": [0.0, 0.0, 1.0, 1.0],
        }
    )
    residuals = np.array([0.01, 0.02, 0.20, 0.30])

    model = fit_local_scale(
        calibration,
        residuals,
        feature_columns=["volatility", "history_missing"],
    )
    scale = predict_local_scale(
        model,
        calibration,
        feature_columns=["volatility", "history_missing"],
    )

    assert (scale > 0).all()
    assert scale[-1] > scale[0]
