import numpy as np

from omr_pipeline.models.source_image import SourceImage


def test_synthetic_staff_image_has_expected_dimensions(
    synthetic_staff_image: SourceImage,
):
    assert synthetic_staff_image.pixels.shape == (240, 400)


def test_synthetic_staff_image_has_five_staff_lines(
    synthetic_staff_image: SourceImage,
):
    projection = np.sum(
        synthetic_staff_image.pixels == 0,
        axis=1,
    )

    staff_rows = np.flatnonzero(projection > 0)

    assert len(staff_rows) == 10


def test_synthetic_staff_image_has_white_background(
    synthetic_staff_image: SourceImage,
):
    assert synthetic_staff_image.pixels[0, 0] == 255


def test_synthetic_staff_image_has_black_staff_lines(
    synthetic_staff_image: SourceImage,
):
    assert synthetic_staff_image.pixels[90, 100] == 0
