import numpy as np

from fmrg_submission.sem import masked_sem_descriptors, sem_tile_centers_mm


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
