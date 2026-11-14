"""Tests for deskewing preprocessing functions."""

import numpy as np
import pytest

from omr_pipeline.models.source_image import SourceImage
from omr_pipeline.preprocessing.deskew import (
    calculate_horizontal_projection,
    calculate_projection_score,
    create_foreground_mask,
)


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
