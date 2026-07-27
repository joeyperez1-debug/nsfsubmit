import numpy as np
from PIL import Image

from fmrg_submission.sem import (
    extract_sem_descriptors_at_positions,
    flank_sem_descriptors,
    masked_sem_descriptors,
    sem_tile_centers_mm,
)


def test_sem_descriptors_ignore_processed_center_band():
    base = np.tile(np.arange(60, dtype=float), (100, 1))
    altered = base.copy()
    altered[35:65, :] = 255.0

    a = masked_sem_descriptors(base, mask_fraction=0.30)
    b = masked_sem_descriptors(altered, mask_fraction=0.30)

    assert a.keys() == b.keys()
    assert all(np.isclose(a[key], b[key]) for key in a)


def test_sem_tile_coordinates_run_from_physical_100_mm_side():
    centers = sem_tile_centers_mm(3, tile_width_mm=6.0)

    assert np.allclose(centers, [97.0, 91.0, 85.0])


def test_flank_descriptors_ignore_track_and_keep_left_right_evidence_separate():
    image = np.zeros((100, 80), dtype=float)
    image[:40] = 10.0
    image[61:] = 30.0

    first = flank_sem_descriptors(
        image, left_row=40, right_row=60, flank_width_px=20
    )
    altered = image.copy()
    altered[40:61] = 255.0
    second = flank_sem_descriptors(
        altered, left_row=40, right_row=60, flank_width_px=20
    )

    assert first == second
    assert first["sem_right_mean"] > first["sem_left_mean"]


def test_registration_uncertainty_is_emitted_in_physical_units(tmp_path):
    tile_paths = []
    for index, value in enumerate((20, 40)):
        path = tmp_path / f"tile_{index}.png"
        Image.fromarray(np.full((40, 40), value, dtype=np.uint8)).save(path)
        tile_paths.append(path)

    result = extract_sem_descriptors_at_positions(
        tile_paths,
        np.array([90.0]),
        registration_uncertainty_mm=0.25,
    )

    assert result.loc[0, "sem_registration_uncertainty_mm"] == 0.25
