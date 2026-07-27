import numpy as np

from fmrg_submission.thermal import extract_thermal_descriptors


def test_thermal_descriptors_capture_pool_geometry_and_asymmetry():
    frames = np.full((3, 40, 40), 900.0)
    frames[:, 15:25, 14:26] = 1800.0
    frames[1, 18:22, 26:32] = 1700.0
    frames[2, 15:25, 14:30] = 1900.0
    x_mm = np.array([20.1, 20.3, 20.5])

    features = extract_thermal_descriptors(frames, x_mm, threshold=1500.0)

    assert list(features["x_mm"]) == list(x_mm)
    assert features.loc[0, "hot_area_px"] == 120
    assert features.loc[1, "right_left_asymmetry"] > features.loc[
        0, "right_left_asymmetry"
    ]
    assert features.loc[2, "hot_area_px"] > features.loc[0, "hot_area_px"]
    assert features.loc[2, "delta_hot_area_px"] > 0
    assert np.isfinite(features.select_dtypes("number")).all().all()


def test_thermal_descriptors_handle_no_hot_component():
    frames = np.full((2, 12, 12), 800.0)
    features = extract_thermal_descriptors(frames, [20.1, 20.3], threshold=1500.0)

    assert (features["hot_area_px"] == 0).all()
    assert (features["bbox_width_px"] == 0).all()


def test_multiscale_history_is_causal_and_contains_required_windows():
    frames = np.full((24, 30, 30), 800.0)
    for index in range(24):
        frames[index, 10:20, 8 : 12 + index // 4] = 1600.0 + 10.0 * index
    x_mm = np.arange(24.0)

    original = extract_thermal_descriptors(frames, x_mm)
    changed = frames.copy()
    changed[20:] = 4000.0
    perturbed = extract_thermal_descriptors(changed, x_mm)

    required = {
        "roll5_hot_area_px_mean",
        "roll10_thermal_mass_slope",
        "roll20_max_temperature_persistence_1500",
        "centroid_velocity_px",
        "shape_change",
        "cooling_tail_decay",
    }
    assert required.issubset(original.columns)
    columns = sorted(required)
    assert np.allclose(original.loc[:19, columns], perturbed.loc[:19, columns])


def test_early_history_has_missingness_flags_and_finite_defaults():
    frames = np.full((3, 20, 20), 900.0)

    result = extract_thermal_descriptors(frames, [1.0, 2.0, 3.0])

    assert result.loc[0, "roll20_history_fraction"] == 0.05
    assert np.isfinite(result.select_dtypes("number")).all().all()
