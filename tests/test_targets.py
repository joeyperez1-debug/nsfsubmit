import numpy as np
import pandas as pd
import pytest

from fmrg_submission.targets import (
    add_hierarchical_targets,
    reconstruct_geometry,
    track_thermal_summaries,
)


def test_hierarchical_targets_separate_track_baseline_and_local_residual():
    data = pd.DataFrame(
        {
            "track_id": [8, 8, 10, 10],
            "center_mm": [1.0, 1.2, 2.0, 2.2],
            "width_mm": [0.4, 0.6, 0.8, 1.2],
        }
    )

    result = add_hierarchical_targets(data)

    assert np.allclose(
        result.groupby("track_id")["center_residual_mm"].median(), 0.0
    )
    assert np.allclose(
        result.groupby("track_id")["log_width_residual"].median(), 0.0
    )


def test_hierarchical_targets_reject_nonpositive_width():
    data = pd.DataFrame(
        {
            "track_id": [8, 8],
            "center_mm": [1.0, 1.1],
            "width_mm": [0.5, 0.0],
        }
    )

    with pytest.raises(ValueError, match="positive"):
        add_hierarchical_targets(data)


def test_reconstruction_is_positive_and_boundaries_are_ordered():
    result = reconstruct_geometry(
        baseline_center=np.array([1.0, 1.0]),
        baseline_log_width=np.log(np.array([0.5, 0.5])),
        center_residual=np.array([0.1, -0.1]),
        log_width_residual=np.array([100.0, -100.0]),
    )

    assert (result["width_mm"] > 0).all()
    assert (result["left_mm"] < result["right_mm"]).all()
    assert np.allclose(result["center_mm"], [1.1, 0.9])


def test_track_summaries_emit_one_row_per_track_with_robust_statistics():
    data = pd.DataFrame(
        {
            "track_id": [8, 8, 8, 10, 10, 10],
            "thermal_mass": [1.0, 2.0, 100.0, 10.0, 20.0, 30.0],
        }
    )

    result = track_thermal_summaries(data, ["thermal_mass"])

    assert list(result["track_id"]) == [8, 10]
    assert result.loc[0, "thermal_mass__median"] == 2.0
    assert result.loc[1, "thermal_mass__iqr"] == 10.0
