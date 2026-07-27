import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fmrg_submission.experiments import default_feature_sets, run_experiments


def _synthetic_data():
    rows = []
    for track_id, offset in [(8, 0.0), (10, 0.1), (14, 0.2), (21, 0.3)]:
        for index in range(12):
            x = index / 11
            thermal = offset + x
            width = 0.45 + offset + 0.04 * np.sin(2 * np.pi * x)
            center = 1.0 + 0.2 * offset + 0.02 * np.cos(2 * np.pi * x)
            rows.append(
                {
                    "track_id": track_id,
                    "x_mm": x,
                    "thermal": thermal,
                    "width_mm": width,
                    "center_mm": center,
                    "left_mm": center - width / 2,
                    "right_mm": center + width / 2,
                }
            )
    return pd.DataFrame(rows)


def _ridge_factory():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def test_experiment_output_contract(tmp_path):
    result = run_experiments(
        _synthetic_data(),
        tmp_path,
        feature_sets={"thermal": ["thermal"]},
        estimator_factories={"ridge": _ridge_factory},
        incumbent_features=["thermal"],
    )

    assert {
        "protocol",
        "historical_benchmark",
        "incumbent",
        "candidates",
        "selected",
        "promotion",
        "uncertainty",
        "sem_ablation",
        "software_versions",
    } <= result.keys()
    predictions = pd.read_csv(tmp_path / "outer_fold_predictions.csv")
    assert set(predictions["track_id"]) == {8, 10, 14, 21}
    assert (tmp_path / "candidate_scores.csv").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["protocol"]["outer_tracks"] == [8, 10, 14, 21]
    assert saved["protocol"]["accuracy_tolerance_mm"] == 0.02
    assert saved["uncertainty"]["selected"] in {"conditional", "global"}


def test_default_feature_sets_separate_absolute_condition_and_local_deviation():
    data = _synthetic_data().rename(columns={"thermal": "hot_area_px"})
    data["max_temperature"] = 1500.0 + data["hot_area_px"]
    data["thermal_mass"] = 100.0 * data["hot_area_px"]
    data["local_hot_area_px"] = data.groupby("track_id")[
        "hot_area_px"
    ].transform(lambda values: values - values.median())
    data["local_max_temperature"] = data.groupby("track_id")[
        "max_temperature"
    ].transform(lambda values: values - values.median())
    data["local_thermal_mass"] = data.groupby("track_id")[
        "thermal_mass"
    ].transform(lambda values: values - values.median())

    feature_sets = default_feature_sets(data)

    assert "normalized_compact_thermal" in feature_sets
    normalized = feature_sets["normalized_compact_thermal"]
    assert "hot_area_px" in normalized
    assert "local_hot_area_px" in normalized
