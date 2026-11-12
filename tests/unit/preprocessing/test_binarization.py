"""Tests for image binarization."""

import numpy as np
import pytest

from omr_pipeline.models import SourceImage
from omr_pipeline.preprocessing import apply_threshold


def make_source_image(pixels: np.ndarray) -> SourceImage:
    """Create a grayscale SourceImage for testing."""
    return SourceImage(
        pixels=pixels.astype(np.uint8),
        source_name="score.png",
        source_format="PNG",
    )


def test_apply_threshold_returns_grayscale_image():
    pixels = np.full((100, 200), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_threshold(source)

    assert result.pixels.shape == (100, 200)
    assert result.pixels.dtype == np.uint8
    assert result.channels == 1


def test_pixels_above_threshold_become_white():
    pixels = np.array([[128, 150, 200, 255]], dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_threshold(source)

    assert np.all(result.pixels == 255)


def test_pixels_below_threshold_become_black():
    pixels = np.array([[0, 25, 100, 126]], dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_threshold(source)

    assert np.all(result.pixels == 0)


def test_pixels_equal_to_threshold():
    pixels = np.array([[127]], dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_threshold(source)

    assert result.pixels[0, 0] == 0


def test_custom_threshold_is_respected():
    pixels = np.array([[99, 100, 101]], dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_threshold(source, threshold=100)

    assert np.array_equal(
        result.pixels,
        np.array([[0, 0, 255]], dtype=np.uint8),
    )


def test_threshold_zero():
    pixels = np.array([[0, 1, 127, 255]], dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_threshold(source, threshold=0)

    assert np.array_equal(
        result.pixels,
        np.array([[0, 255, 255, 255]], dtype=np.uint8),
    )


def test_threshold_maximum():
    pixels = np.array([[0, 127, 254, 255]], dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_threshold(source, threshold=255)

    assert np.all(result.pixels == 0)


@pytest.mark.parametrize("threshold", [-1, 256])
def test_invalid_threshold_raises_error(threshold: int):
    pixels = np.full((10, 10), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    with pytest.raises(ValueError):
        apply_threshold(source, threshold=threshold)


def test_threshold_preserves_image_dimensions():
    pixels = np.full((75, 150), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_threshold(source)

    assert result.width == source.width
    assert result.height == source.height


def test_threshold_preserves_source_metadata():
    pixels = np.full((10, 20), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_threshold(source)

    assert result.source_name == source.source_name
    assert result.source_format == source.source_format


def test_threshold_does_not_modify_original_pixels():
    pixels = np.full((10, 10), 128, dtype=np.uint8)
    source = make_source_image(pixels)
    original_pixels = source.pixels.copy()

    apply_threshold(source)

    assert np.array_equal(source.pixels, original_pixels)


def test_threshold_returns_new_source_image():
    pixels = np.full((10, 10), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_threshold(source)

    assert isinstance(result, SourceImage)
    assert result is not source


def test_output_contains_only_black_and_white_pixels():
    pixels = np.array(
        [[0, 32, 64, 96, 128, 160, 192, 224, 255]],
        dtype=np.uint8,
    )
    source = make_source_image(pixels)

    result = apply_threshold(source)

    assert set(np.unique(result.pixels)).issubset({0, 255})
