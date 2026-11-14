import numpy as np
import pytest

from omr_pipeline.models.source_image import SourceImage
from tests.helpers.image_generators import create_skewed_staff_image


def test_create_skewed_staff_image_zero_degrees(
    synthetic_staff_image: SourceImage,
):
    result = create_skewed_staff_image(synthetic_staff_image, 0.0)

    np.testing.assert_array_equal(
        result.pixels,
        synthetic_staff_image.pixels,
    )


@pytest.mark.parametrize("angle", [2.0, -2.0, 5.0, -5.0])
def test_create_skewed_staff_image_expands_dimensions(
    synthetic_staff_image: SourceImage,
    angle: float,
):
    result = create_skewed_staff_image(synthetic_staff_image, angle)

    assert result.height > synthetic_staff_image.height
    assert result.width > synthetic_staff_image.width


def test_create_skewed_staff_image_preserves_metadata(
    synthetic_staff_image: SourceImage,
):
    result = create_skewed_staff_image(synthetic_staff_image, 2.0)

    assert result.source_name == synthetic_staff_image.source_name
    assert result.source_format == synthetic_staff_image.source_format


def test_create_skewed_staff_image_returns_uint8(
    synthetic_staff_image: SourceImage,
):
    result = create_skewed_staff_image(synthetic_staff_image, 2.0)

    assert result.pixels.dtype == np.uint8


def test_create_skewed_staff_image_does_not_modify_original(
    synthetic_staff_image: SourceImage,
):
    original_pixels = synthetic_staff_image.pixels.copy()

    create_skewed_staff_image(synthetic_staff_image, 2.0)

    np.testing.assert_array_equal(
        synthetic_staff_image.pixels,
        original_pixels,
    )


def test_create_skewed_staff_image_changes_pixels(
    synthetic_staff_image: SourceImage,
):
    result = create_skewed_staff_image(synthetic_staff_image, 2.0)

    assert not np.array_equal(
        result.pixels,
        synthetic_staff_image.pixels,
    )


def test_create_skewed_staff_image_uses_white_background(
    synthetic_staff_image: SourceImage,
):
    result = create_skewed_staff_image(synthetic_staff_image, 5.0)

    # The corners should remain white after rotation.
    assert result.pixels[0, 0] == 255
    assert result.pixels[0, -1] == 255
    assert result.pixels[-1, 0] == 255
    assert result.pixels[-1, -1] == 255


@pytest.mark.parametrize(
    ("skew_angle", "expected_correction"),
    [
        (2.0, -2.0),
        (-2.0, 2.0),
        (4.0, -4.0),
        (-4.0, 4.0),
    ],
)
def test_skewed_image_has_expected_correction_angle(
    synthetic_staff_image: SourceImage,
    skew_angle: float,
    expected_correction: float,
):
    result = create_skewed_staff_image(
        synthetic_staff_image,
        skew_angle,
    )

    assert result.pixels.ndim == 2
    assert expected_correction == -skew_angle
