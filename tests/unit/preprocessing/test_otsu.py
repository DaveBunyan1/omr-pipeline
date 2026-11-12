"""Tests for Otsu's automatic image thresholding."""

import numpy as np
import pytest

from omr_pipeline.models import SourceImage
from omr_pipeline.preprocessing import apply_otsu_threshold


def make_source_image(pixels: np.ndarray) -> SourceImage:
    """Create a grayscale SourceImage for testing."""
    return SourceImage(
        pixels=pixels.astype(np.uint8),
        source_name="score.png",
        source_format="PNG",
    )


def test_otsu_returns_grayscale_binary_image():
    pixels = np.full((100, 200), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_otsu_threshold(source)

    assert result.pixels.shape == (100, 200)
    assert result.pixels.dtype == np.uint8
    assert result.channels == 1


def test_otsu_output_contains_only_black_and_white_pixels():
    pixels = np.array(
        [[0, 32, 64, 96, 128, 160, 192, 224, 255]],
        dtype=np.uint8,
    )
    source = make_source_image(pixels)

    result = apply_otsu_threshold(source)

    assert set(np.unique(result.pixels)).issubset({0, 255})


def test_otsu_separates_two_distinct_intensity_groups():
    pixels = np.array(
        [
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [240, 240, 240, 240, 240],
            [240, 240, 240, 240, 240],
        ],
        dtype=np.uint8,
    )
    source = make_source_image(pixels)

    result = apply_otsu_threshold(source)

    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(result.pixels, expected)


def test_otsu_separates_black_and_white_pixels():
    pixels = np.array(
        [
            [0, 0, 0, 255, 255],
            [0, 0, 255, 255, 255],
        ],
        dtype=np.uint8,
    )
    source = make_source_image(pixels)

    result = apply_otsu_threshold(source)

    assert np.array_equal(result.pixels, pixels)


def test_otsu_handles_uniform_black_image():
    pixels = np.zeros((10, 10), dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_otsu_threshold(source)

    assert np.all(result.pixels == 0)


def test_otsu_handles_uniform_white_image():
    pixels = np.full((10, 10), 255, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_otsu_threshold(source)

    assert np.all(result.pixels == 255)


def test_otsu_handles_uniform_grey_image():
    pixels = np.full((10, 10), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_otsu_threshold(source)

    assert set(np.unique(result.pixels)).issubset({0, 255})


def test_otsu_preserves_image_dimensions():
    pixels = np.full((75, 150), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_otsu_threshold(source)

    assert result.width == source.width
    assert result.height == source.height


def test_otsu_preserves_source_metadata():
    pixels = np.full((10, 20), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_otsu_threshold(source)

    assert result.source_name == source.source_name
    assert result.source_format == source.source_format


def test_otsu_does_not_modify_original_pixels():
    pixels = np.array(
        [[10, 10, 10, 240, 240, 240]],
        dtype=np.uint8,
    )
    source = make_source_image(pixels)
    original_pixels = source.pixels.copy()

    apply_otsu_threshold(source)

    assert np.array_equal(source.pixels, original_pixels)


def test_otsu_returns_new_source_image():
    pixels = np.full((10, 10), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_otsu_threshold(source)

    assert isinstance(result, SourceImage)
    assert result is not source


def test_otsu_rejects_rgb_input():
    pixels = np.full((10, 10, 3), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    with pytest.raises(ValueError):
        apply_otsu_threshold(source)


def test_otsu_rejects_rgba_input():
    pixels = np.full((10, 10, 4), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    with pytest.raises(ValueError):
        apply_otsu_threshold(source)


def test_otsu_separates_adjacent_intensity_groups():
    pixels = np.array(
        [
            [10, 10, 10, 10],
            [11, 11, 11, 11],
        ],
        dtype=np.uint8,
    )
    source = make_source_image(pixels)

    result = apply_otsu_threshold(source)

    expected = np.array(
        [
            [0, 0, 0, 0],
            [255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(result.pixels, expected)
