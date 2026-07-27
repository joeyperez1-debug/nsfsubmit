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
