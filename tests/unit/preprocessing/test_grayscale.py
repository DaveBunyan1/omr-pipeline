"""Tests for grayscale image conversion."""

import numpy as np

from omr_pipeline.models import SourceImage
from omr_pipeline.preprocessing import convert_to_grayscale


def make_source_image(pixels: np.ndarray) -> SourceImage:
    """Create a SourceImage for testing."""
    return SourceImage(
        pixels=pixels.astype(np.uint8),
        source_name="score.png",
        source_format="PNG",
    )


def test_convert_rgb_to_grayscale():
    pixels = np.full((100, 200, 3), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = convert_to_grayscale(source)

    assert result.pixels.shape == (100, 200)
    assert result.pixels.dtype == np.uint8
    assert result.channels == 1


def test_convert_rgba_to_grayscale():
    pixels = np.full((100, 200, 4), 128, dtype=np.uint8)
    pixels[:, :, 3] = 255
    source = make_source_image(pixels)

    result = convert_to_grayscale(source)

    assert result.pixels.shape == (100, 200)
    assert result.pixels.dtype == np.uint8
    assert result.channels == 1


def test_grayscale_input_remains_visually_unchanged():
    pixels = np.full((100, 200), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = convert_to_grayscale(source)

    assert np.array_equal(result.pixels, pixels)


def test_white_pixels_remain_white():
    pixels = np.full((10, 10, 3), 255, dtype=np.uint8)
    source = make_source_image(pixels)

    result = convert_to_grayscale(source)

    assert np.all(result.pixels == 255)


def test_black_pixels_remain_black():
    pixels = np.zeros((10, 10, 3), dtype=np.uint8)
    source = make_source_image(pixels)

    result = convert_to_grayscale(source)

    assert np.all(result.pixels == 0)


def test_mid_grey_pixels_remain_approximately_128():
    pixels = np.full((10, 10, 3), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = convert_to_grayscale(source)

    assert np.allclose(result.pixels, 128, atol=1)


def test_transparent_rgba_pixels_are_composited_against_white():
    pixels = np.zeros((10, 10, 4), dtype=np.uint8)
    pixels[:, :, :3] = 0
    pixels[:, :, 3] = 0
    source = make_source_image(pixels)

    result = convert_to_grayscale(source)

    # Fully transparent black should appear white over a white background.
    assert np.all(result.pixels == 255)


def test_semitransparent_rgba_pixels_are_composited_against_white():
    pixels = np.zeros((10, 10, 4), dtype=np.uint8)
    pixels[:, :, :3] = 0
    pixels[:, :, 3] = 128
    source = make_source_image(pixels)

    result = convert_to_grayscale(source)

    # Half-transparent black over white should produce approximately 127–128.
    assert np.allclose(result.pixels, 127, atol=1)


def test_conversion_preserves_image_dimensions():
    pixels = np.full((75, 150, 3), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = convert_to_grayscale(source)

    assert result.width == source.width
    assert result.height == source.height


def test_conversion_preserves_source_metadata():
    pixels = np.full((10, 20, 3), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = convert_to_grayscale(source)

    assert result.source_name == source.source_name
    assert result.source_format == source.source_format


def test_conversion_does_not_modify_original_pixels():
    pixels = np.full((10, 10, 3), 128, dtype=np.uint8)
    source = make_source_image(pixels)
    original_pixels = source.pixels.copy()

    convert_to_grayscale(source)

    assert np.array_equal(source.pixels, original_pixels)


def test_conversion_returns_new_source_image():
    pixels = np.full((10, 10, 3), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = convert_to_grayscale(source)

    assert isinstance(result, SourceImage)
    assert result is not source
