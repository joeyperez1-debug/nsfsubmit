import numpy as np

from fmrg_submission.geometry import extract_local_geometry


def test_extract_local_geometry_recovers_sloped_bead_with_missing_pixels():
    rng = np.random.default_rng(7)
    y_mm = np.linspace(0.0, 1.9, 480)
    x_mm = np.linspace(20.0, 100.0, 81)
    true_center = 0.96 + 0.05 * np.sin((x_mm - 20.0) / 9.0)
    true_width = 0.72 + 0.08 * np.sin((x_mm - 20.0) / 6.0)

    z_mm = np.empty((len(y_mm), len(x_mm)))
    for j, (center, width) in enumerate(zip(true_center, true_width)):
        sigma = width / 2.355
        bead = 0.026 * np.exp(-0.5 * ((y_mm - center) / sigma) ** 2)
        substrate = -0.010 + 0.006 * y_mm + 0.00008 * x_mm[j]
        z_mm[:, j] = substrate + bead + rng.normal(0.0, 0.00045, len(y_mm))

    z_mm[rng.random(z_mm.shape) < 0.35] = np.nan
    result = extract_local_geometry(
        z_mm,
        x_mm,
        y_mm,
        threshold_fraction=0.30,
        minimum_peak_um=5.0,
    )

    valid = result["valid"]
    assert valid.mean() > 0.95
    assert np.nanmean(np.abs(result["center_mm"][valid] - true_center[valid])) < 0.035
    # A 30%-of-peak Gaussian width is wider than FWHM by a known factor.
    expected_threshold_width = true_width * np.sqrt(
        np.log(1.0 / 0.30) / np.log(2.0)
    )
    assert np.nanmean(
        np.abs(result["width_mm"][valid] - expected_threshold_width[valid])
    ) < 0.06
    assert np.allclose(
        result["right_mm"][valid] - result["left_mm"][valid],
        result["width_mm"][valid],
    )


def test_extract_local_geometry_marks_flat_profiles_invalid():
    y_mm = np.linspace(0.0, 1.9, 100)
    x_mm = np.linspace(20.0, 21.0, 4)
    z_mm = np.tile(0.002 * y_mm[:, None], (1, len(x_mm)))

    result = extract_local_geometry(z_mm, x_mm, y_mm, minimum_peak_um=5.0)

    assert not result["valid"].any()
    assert np.isnan(result["width_mm"]).all()
