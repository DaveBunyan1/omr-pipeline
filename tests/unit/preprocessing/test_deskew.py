"""Tests for deskewing preprocessing functions."""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from omr_pipeline.models.source_image import SourceImage
from omr_pipeline.preprocessing.deskew import (
    calculate_horizontal_projection,
    calculate_projection_score,
    create_foreground_mask,
    estimate_skew_angle,
    generate_candidate_angles,
    rotate_image,
    rotate_source_image,
)
from tests.helpers.image_generators import create_skewed_staff_image


def make_source_image(pixels: np.ndarray) -> SourceImage:
    """Create a SourceImage for testing."""
    return SourceImage(
        pixels=pixels.astype(np.uint8),
        source_name="test.png",
        source_format="PNG",
    )


def test_calculate_horizontal_projection():
    binary_image = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )

    result = calculate_horizontal_projection(binary_image)

    expected = np.array([0, 4, 0, 5])

    np.testing.assert_array_equal(result, expected)


def test_calculate_horizontal_projection_returns_correct_shape():
    binary_image = np.ones((10, 20), dtype=np.uint8)

    result = calculate_horizontal_projection(binary_image)

    assert result.shape == (10,)


def test_calculate_horizontal_projection_returns_integer_values():
    binary_image = np.ones((10, 20), dtype=np.uint8)

    result = calculate_horizontal_projection(binary_image)

    assert np.issubdtype(result.dtype, np.integer)


def test_calculate_horizontal_projection_handles_empty_rows():
    binary_image = np.zeros((5, 10), dtype=np.uint8)

    result = calculate_horizontal_projection(binary_image)

    np.testing.assert_array_equal(result, np.zeros(5, dtype=result.dtype))


def test_calculate_horizontal_projection_handles_full_rows():
    binary_image = np.ones((5, 10), dtype=np.uint8)

    result = calculate_horizontal_projection(binary_image)

    np.testing.assert_array_equal(result, np.full(5, 10))


@pytest.mark.parametrize("ndim", [1, 3])
def test_calculate_horizontal_projection_rejects_non_2d_input(ndim: int):
    binary_image = np.ones((10,) * ndim, dtype=np.uint8)

    with pytest.raises(ValueError):
        calculate_horizontal_projection(binary_image)


def test_create_foreground_mask():
    pixels = np.array(
        [
            [0, 0, 0, 255, 255],
            [0, 255, 255, 255, 255],
            [0, 0, 0, 0, 255],
        ],
        dtype=np.uint8,
    )

    source = make_source_image(pixels)

    result = create_foreground_mask(source)

    expected = np.array(
        [
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(result, expected)


def test_create_foreground_mask_returns_uint8():
    pixels = np.array(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )

    source = make_source_image(pixels)

    result = create_foreground_mask(source)

    assert result.dtype == np.uint8


def test_create_foreground_mask_contains_only_zero_and_one():
    pixels = np.random.default_rng(42).integers(0, 256, size=(50, 50), dtype=np.uint8)

    source = make_source_image(pixels)

    result = create_foreground_mask(source)

    assert set(np.unique(result)).issubset({0, 1})


def test_create_foreground_mask_handles_uniform_black_image():
    pixels = np.zeros((10, 10), dtype=np.uint8)
    source = make_source_image(pixels)

    result = create_foreground_mask(source)

    assert np.all(result == 1)


def test_create_foreground_mask_handles_uniform_white_image():
    pixels = np.full((10, 10), 255, dtype=np.uint8)
    source = make_source_image(pixels)

    result = create_foreground_mask(source)

    assert np.all(result == 0)


@pytest.mark.parametrize("channels", [3, 4])
def test_create_foreground_mask_rejects_colour_images(channels: int):
    pixels = np.full((10, 10, channels), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    with pytest.raises(ValueError):
        create_foreground_mask(source)


def test_create_foreground_mask_does_not_modify_original_image():
    pixels = np.array(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )

    source = make_source_image(pixels)
    original_pixels = source.pixels.copy()

    create_foreground_mask(source)

    np.testing.assert_array_equal(source.pixels, original_pixels)


def test_create_foreground_mask_preserves_image_metadata():
    pixels = np.full((10, 10), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    create_foreground_mask(source)

    assert source.source_name == "test.png"
    assert source.source_format == "PNG"


def test_calculate_projection_score():
    projection = np.array([0, 4, 0, 5])

    result = calculate_projection_score(projection)

    assert result == pytest.approx(np.var(projection))


def test_calculate_projection_score_uniform_profile():
    projection = np.array([5, 5, 5, 5])

    result = calculate_projection_score(projection)

    assert result == 0.0


def test_calculate_projection_score_returns_python_float():
    projection = np.array([0, 4, 0, 5])

    result = calculate_projection_score(projection)

    assert isinstance(result, float)


def test_rotate_image_by_zero_degrees():
    image = np.zeros((10, 20), dtype=np.uint8)
    image[2:5, 5:15] = 1

    result = rotate_image(image, 0.0)

    np.testing.assert_array_equal(result, image)


def test_rotate_image_by_90_degrees():
    image = np.zeros((10, 20), dtype=np.uint8)
    image[2:5, 5:15] = 1

    result = rotate_image(image, 90.0)

    assert result.shape == (20, 10)


def test_rotate_image_by_180_degrees():
    image = np.zeros((10, 20), dtype=np.uint8)
    image[2:5, 5:15] = 1

    result = rotate_image(image, 180.0)

    assert result.shape == image.shape
    np.testing.assert_array_equal(result, np.rot90(image, 2))


def test_rotate_image_by_small_angle_expands_dimensions():
    image = np.zeros((100, 200), dtype=np.uint8)
    image[20:25, 20:180] = 1

    result = rotate_image(image, 5.0)

    assert result.shape[0] > image.shape[0]
    assert result.shape[1] > image.shape[1]


def test_rotate_image_contains_only_binary_values():
    image = np.zeros((50, 50), dtype=np.uint8)
    image[10:20, 10:40] = 1

    result = rotate_image(image, 3.0)

    assert set(np.unique(result)).issubset({0, 1})


def test_rotate_image_returns_uint8():
    image = np.zeros((50, 50), dtype=np.uint8)

    result = rotate_image(image, 3.0)

    assert result.dtype == np.uint8


def test_rotate_image_fills_exposed_areas_with_zero():
    image = np.ones((50, 50), dtype=np.uint8)

    result = rotate_image(image, 5.0)

    assert np.any(result == 0)


def test_rotate_image_does_not_modify_original_image():
    image = np.zeros((50, 50), dtype=np.uint8)
    image[10:20, 10:40] = 1

    original = image.copy()

    rotate_image(image, 5.0)

    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("ndim", [1, 3])
def test_rotate_image_rejects_non_2d_input(ndim: int):
    image = np.ones((10,) * ndim, dtype=np.uint8)

    with pytest.raises(ValueError):
        rotate_image(image, 5.0)


@pytest.mark.parametrize(
    "angle",
    [float("inf"), float("-inf"), float("nan")],
)
def test_rotate_image_rejects_non_finite_angles(angle: float):
    image = np.zeros((50, 50), dtype=np.uint8)

    with pytest.raises(ValueError):
        rotate_image(image, angle)


@pytest.mark.parametrize(
    "shape",
    [
        (100, 100, 3),
        (100, 100, 4),
    ],
)
def test_estimate_skew_angle_rejects_non_grayscale_images(shape: int):
    import numpy as np

    from omr_pipeline.models import SourceImage

    source = SourceImage(
        pixels=np.zeros(shape, dtype=np.uint8),
        source_name="test.png",
        source_format="PNG",
    )

    with pytest.raises(ValueError, match="2D grayscale"):
        estimate_skew_angle(source)


@pytest.mark.parametrize(
    "max_skew",
    [
        "5",
        None,
        True,
        False,
        [],
    ],
)
def test_estimate_skew_angle_rejects_invalid_max_skew_types(
    synthetic_staff_image: SourceImage,
    max_skew: float,
):
    with pytest.raises(TypeError, match="max_skew must be a number"):
        estimate_skew_angle(
            synthetic_staff_image,
            max_skew=max_skew,
        )


@pytest.mark.parametrize(
    "max_skew",
    [
        0,
        -1,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_estimate_skew_angle_rejects_invalid_max_skew_values(
    synthetic_staff_image: SourceImage,
    max_skew: float,
):
    with pytest.raises(ValueError, match="max_skew"):
        estimate_skew_angle(
            synthetic_staff_image,
            max_skew=max_skew,
        )


@pytest.mark.parametrize(
    "angle_step",
    [
        "0.1",
        None,
        True,
        False,
        [],
    ],
)
def test_estimate_skew_angle_rejects_invalid_angle_step_types(
    synthetic_staff_image: SourceImage,
    angle_step: Any,
):
    with pytest.raises(TypeError, match="angle_step must be a number"):
        estimate_skew_angle(
            synthetic_staff_image,
            angle_step=angle_step,
        )


@pytest.mark.parametrize(
    "angle_step",
    [
        0,
        -0.1,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_estimate_skew_angle_rejects_invalid_angle_step_values(
    synthetic_staff_image: SourceImage,
    angle_step: float,
):
    with pytest.raises(ValueError, match="angle_step"):
        estimate_skew_angle(
            synthetic_staff_image,
            angle_step=angle_step,
        )


def test_estimate_skew_angle_rejects_angle_step_greater_than_max_skew(
    synthetic_staff_image: SourceImage,
):
    with pytest.raises(ValueError, match="must not exceed max_skew"):
        estimate_skew_angle(
            synthetic_staff_image,
            max_skew=2.0,
            angle_step=3.0,
        )


# TODO: Update after implementation
def test_estimate_skew_angle_allows_angle_step_equal_to_max_skew(
    synthetic_staff_image: SourceImage,
):
    # This will exercise the actual estimator once the search is implemented.
    # For now, validation should not reject these arguments.
    try:
        estimate_skew_angle(
            synthetic_staff_image,
            max_skew=1.0,
            angle_step=1.0,
        )
    except ValueError as exc:
        if "angle_step must not exceed max_skew" in str(exc):
            pytest.fail("Equal max_skew and angle_step should be allowed.")


def test_generate_candidate_angles_evenly_divisible():
    result = generate_candidate_angles(
        max_skew=1.0,
        angle_step=0.5,
    )

    expected = np.array(
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(result, expected)


def test_generate_candidate_angles_not_evenly_divisible():
    result = generate_candidate_angles(
        max_skew=1.0,
        angle_step=0.6,
    )

    expected = np.array(
        [-1.0, -0.4, 0.0, 0.2, 0.8, 1.0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("max_skew", "angle_step"),
    [
        (1.0, 0.3),
        (2.0, 0.7),
        (5.0, 0.1),
        (3.0, 1.0),
    ],
)
def test_generate_candidate_angles_includes_zero(max_skew: float, angle_step: float):
    result = generate_candidate_angles(max_skew, angle_step)

    assert 0.0 in result


def test_generate_candidate_angles_includes_endpoints():
    max_skew = 1.0

    result = generate_candidate_angles(
        max_skew=max_skew,
        angle_step=0.3,
    )

    assert result[0] == -max_skew
    assert result[-1] == max_skew


def test_generate_candidate_angles_are_ascending():
    result = generate_candidate_angles(
        max_skew=2.0,
        angle_step=0.3,
    )

    assert np.all(np.diff(result) > 0)


def test_generate_candidate_angles_returns_numpy_array():
    result = generate_candidate_angles(1.0, 0.5)

    assert isinstance(result, np.ndarray)


def test_generate_candidate_angles_returns_float64():
    result = generate_candidate_angles(1.0, 0.5)

    assert result.dtype == np.float64


def test_generate_candidate_angles_step_equals_max_skew():
    result = generate_candidate_angles(
        max_skew=1.0,
        angle_step=1.0,
    )

    expected = np.array(
        [-1.0, 0.0, 1.0],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(result, expected)


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
def test_estimate_skew_angle(
    synthetic_staff_image: SourceImage,
    skew_angle: float,
    expected_correction: float,
):
    skewed_image = create_skewed_staff_image(
        synthetic_staff_image,
        skew_angle,
    )

    result = estimate_skew_angle(
        skewed_image,
        max_skew=5.0,
        angle_step=0.1,
    )

    assert result == pytest.approx(expected_correction, abs=0.2)


def test_estimate_skew_angle_blank_image():
    pixels = np.full((240, 400), 255, dtype=np.uint8)

    source = SourceImage(
        pixels=pixels,
        source_name="blank.png",
        source_format="PNG",
    )

    with pytest.raises(ValueError, match="Unable to estimate skew"):
        estimate_skew_angle(source)


def test_estimate_skew_angle_rejects_rgb_image():
    pixels = np.zeros((240, 400, 3), dtype=np.uint8)

    source = SourceImage(
        pixels=pixels,
        source_name="rgb.png",
        source_format="PNG",
    )

    with pytest.raises(ValueError, match="grayscale"):
        estimate_skew_angle(source)


def test_rotate_source_image_zero_degrees(synthetic_staff_image: SourceImage):
    result = rotate_source_image(synthetic_staff_image, 0.0)

    np.testing.assert_array_equal(
        result.pixels,
        synthetic_staff_image.pixels,
    )


def test_rotate_source_image_90_degrees(synthetic_staff_image: SourceImage):
    result = rotate_source_image(synthetic_staff_image, 90.0)

    assert result.width == synthetic_staff_image.height
    assert result.height == synthetic_staff_image.width


def test_rotate_source_image_180_degrees(synthetic_staff_image: SourceImage):
    result = rotate_source_image(synthetic_staff_image, 180.0)

    expected = np.rot90(synthetic_staff_image.pixels, 2)

    np.testing.assert_array_equal(result.pixels, expected)


def test_rotate_source_image_small_angle_expands_dimensions(
    synthetic_staff_image: SourceImage,
):
    result = rotate_source_image(synthetic_staff_image, 2.0)

    assert result.width > synthetic_staff_image.width
    assert result.height > synthetic_staff_image.height


@pytest.mark.parametrize(
    "channels",
    [1, 3, 4],
)
def test_rotate_source_image_preserves_image_channels(channels: int):
    height = 100
    width = 150

    if channels == 1:
        pixels = np.full((height, width), 128, dtype=np.uint8)
    else:
        pixels = np.full(
            (height, width, channels),
            128,
            dtype=np.uint8,
        )

    source = SourceImage(
        pixels=pixels,
        source_name="test.png",
        source_format="PNG",
    )

    result = rotate_source_image(source, 2.0)

    if channels == 1:
        assert result.pixels.ndim == 2
    else:
        assert result.pixels.shape[2] == channels

    assert result.pixels.dtype == np.uint8


@pytest.mark.parametrize(
    ("channels", "expected"),
    [
        (1, 255),
        (3, [255, 255, 255]),
        (4, [255, 255, 255, 255]),
    ],
)
def test_rotate_source_image_fills_exposed_areas_white(
    channels: int,
    expected: NDArray[np.uint8],
):
    height = 100
    width = 150

    if channels == 1:
        pixels = np.zeros((height, width), dtype=np.uint8)
    else:
        pixels = np.zeros(
            (height, width, channels),
            dtype=np.uint8,
        )

    source = SourceImage(
        pixels=pixels,
        source_name="test.png",
        source_format="PNG",
    )

    result = rotate_source_image(source, 5.0)

    # The expanded image should contain white pixels.
    assert (
        np.any(np.all(result.pixels == expected, axis=-1))
        if channels > 1
        else np.any(result.pixels == expected)
    )


def test_rotate_source_image_preserves_metadata(synthetic_staff_image: SourceImage):
    result = rotate_source_image(synthetic_staff_image, 2.0)

    assert result.source_name == synthetic_staff_image.source_name
    assert result.source_format == synthetic_staff_image.source_format


def test_rotate_source_image_does_not_modify_input(synthetic_staff_image: SourceImage):
    original = synthetic_staff_image.pixels.copy()

    rotate_source_image(synthetic_staff_image, 2.0)

    np.testing.assert_array_equal(
        synthetic_staff_image.pixels,
        original,
    )


def test_rotate_source_image_returns_new_array(synthetic_staff_image: SourceImage):
    result = rotate_source_image(synthetic_staff_image, 2.0)

    assert result.pixels is not synthetic_staff_image.pixels


@pytest.mark.parametrize(
    "angle",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rotate_source_image_rejects_non_finite_angles(
    synthetic_staff_image: SourceImage,
    angle: float,
):
    with pytest.raises(ValueError, match="finite"):
        rotate_source_image(synthetic_staff_image, angle)


@pytest.mark.parametrize(
    "angle",
    [
        "2",
        None,
        [],
        True,
    ],
)
def test_rotate_source_image_rejects_invalid_angle_types(
    synthetic_staff_image: SourceImage,
    angle: float,
):
    with pytest.raises(TypeError, match="number"):
        rotate_source_image(synthetic_staff_image, angle)
