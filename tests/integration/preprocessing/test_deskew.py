import numpy as np
import pytest

from omr_pipeline.models.source_image import SourceImage
from omr_pipeline.preprocessing.deskew import deskew_image, estimate_skew_angle
from tests.helpers.image_generators import create_skewed_staff_image


@pytest.mark.parametrize(
    ("skew_angle", "expected_correction"),
    [
        (0.0, 0.0),
        (2.0, -2.0),
        (-2.0, 2.0),
        (4.0, -4.0),
        (-4.0, 4.0),
    ],
)
def test_deskew_image(
    synthetic_staff_image: SourceImage,
    skew_angle: float,
    expected_correction: float,
):
    skewed_image = create_skewed_staff_image(
        synthetic_staff_image,
        skew_angle,
    )

    result = deskew_image(skewed_image)

    assert isinstance(result, SourceImage)

    estimated_angle = estimate_skew_angle(result)

    assert estimated_angle == pytest.approx(0.0, abs=0.2)


def test_deskew_image_does_not_rotate_upright_image(synthetic_staff_image: SourceImage):
    result = deskew_image(synthetic_staff_image)

    assert result is synthetic_staff_image

    np.testing.assert_array_equal(
        result.pixels,
        synthetic_staff_image.pixels,
    )


def test_deskew_image_preserves_metadata(
    synthetic_staff_image: SourceImage,
):
    skewed_image = create_skewed_staff_image(
        synthetic_staff_image,
        2.0,
    )

    result = deskew_image(skewed_image)

    assert result.source_name == skewed_image.source_name
    assert result.source_format == skewed_image.source_format


def test_deskew_image_does_not_modify_input(
    synthetic_staff_image: SourceImage,
):
    skewed_image = create_skewed_staff_image(
        synthetic_staff_image,
        2.0,
    )

    original_pixels = skewed_image.pixels.copy()

    deskew_image(skewed_image)

    np.testing.assert_array_equal(
        skewed_image.pixels,
        original_pixels,
    )


@pytest.mark.parametrize(
    ("max_skew", "angle_step"),
    [
        (0, 0.1),
        (-1, 0.1),
        (5, 0),
        (5, -0.1),
        (5, 6),
        (float("inf"), 0.1),
        (5, float("nan")),
    ],
)
def test_deskew_image_rejects_invalid_parameters(
    synthetic_staff_image: SourceImage,
    max_skew: float,
    angle_step: float,
):
    with pytest.raises((TypeError, ValueError)):
        deskew_image(
            synthetic_staff_image,
            max_skew=max_skew,
            angle_step=angle_step,
        )
