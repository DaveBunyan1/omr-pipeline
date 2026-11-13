"""Tests for adaptive image thresholding."""

import numpy as np
import pytest

from omr_pipeline.models import SourceImage
from omr_pipeline.preprocessing import apply_adaptive_threshold


def make_source_image(pixels: np.ndarray) -> SourceImage:
    """Create a grayscale SourceImage for testing."""
    return SourceImage(
        pixels=pixels.astype(np.uint8),
        source_name="score.png",
        source_format="PNG",
    )


def test_adaptive_threshold_returns_grayscale_binary_image():
    pixels = np.full((100, 200), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_adaptive_threshold(source)

    assert result.pixels.shape == (100, 200)
    assert result.pixels.dtype == np.uint8
    assert result.channels == 1


def test_adaptive_threshold_output_contains_only_black_and_white():
    pixels = np.random.default_rng(42).integers(0, 256, size=(100, 100), dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_adaptive_threshold(source)

    assert set(np.unique(result.pixels)).issubset({0, 255})


def test_adaptive_threshold_handles_uniform_black_image():
    pixels = np.zeros((20, 20), dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_adaptive_threshold(source)

    assert np.all(result.pixels == 0)


def test_adaptive_threshold_handles_uniform_white_image():
    pixels = np.full((20, 20), 255, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_adaptive_threshold(source)

    assert np.all(result.pixels == 255)


def test_adaptive_threshold_handles_uniform_grey_image():
    pixels = np.full((20, 20), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_adaptive_threshold(source)

    assert set(np.unique(result.pixels)).issubset({0, 255})


def test_adaptive_threshold_detects_dark_pixels():
    pixels = np.full((20, 20), 200, dtype=np.uint8)
    pixels[8:12, 8:12] = 50
    source = make_source_image(pixels)

    result = apply_adaptive_threshold(source, block_size=11, constant=2.0)

    assert np.all(result.pixels[8:12, 8:12] == 0)
    assert np.all(result.pixels[:5, :5] == 255)


def test_adaptive_threshold_handles_different_local_brightness():
    pixels = np.full((20, 40), 200, dtype=np.uint8)

    # Dark foreground on a bright background.
    pixels[8:12, 8:12] = 50

    # The same foreground/background contrast under darker illumination.
    pixels[:, 20:] = 100
    pixels[8:12, 28:32] = 0

    source = make_source_image(pixels)

    result = apply_adaptive_threshold(source, block_size=11, constant=2.0)

    assert np.all(result.pixels[8:12, 8:12] == 0)
    assert np.all(result.pixels[8:12, 28:32] == 0)


def test_adaptive_threshold_respects_block_size():
    pixels = np.full((30, 30), 200, dtype=np.uint8)
    pixels[13:17, 13:17] = 50
    source = make_source_image(pixels)

    result_small = apply_adaptive_threshold(
        source,
        block_size=3,
    )
    result_large = apply_adaptive_threshold(
        source,
        block_size=21,
    )

    assert result_small.pixels.shape == source.pixels.shape
    assert result_large.pixels.shape == source.pixels.shape


@pytest.mark.parametrize("block_size", [0, 1, -3, 2, 4, 10])
def test_invalid_block_size_raises_value_error(block_size: int):
    pixels = np.full((20, 20), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    with pytest.raises(ValueError):
        apply_adaptive_threshold(source, block_size=block_size)


@pytest.mark.parametrize("block_size", [3.5, "11", True])
def test_non_integer_block_size_raises_type_error(block_size: int):
    pixels = np.full((20, 20), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    with pytest.raises(TypeError):
        apply_adaptive_threshold(source, block_size=block_size)


@pytest.mark.parametrize("constant", [-1.0, 0.0, 2.0, 10.5])
def test_adaptive_threshold_accepts_valid_constant(constant: float):
    pixels = np.full((20, 20), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_adaptive_threshold(source, constant=constant)

    assert result.pixels.shape == source.pixels.shape


@pytest.mark.parametrize("constant", [float("inf"), float("-inf"), float("nan")])
def test_adaptive_threshold_rejects_non_finite_constant(constant: float):
    pixels = np.full((20, 20), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    with pytest.raises(ValueError):
        apply_adaptive_threshold(source, constant=constant)


def test_adaptive_threshold_rejects_non_numeric_constant():
    pixels = np.full((20, 20), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    with pytest.raises(TypeError):
        apply_adaptive_threshold(source, constant="-3.0")  # type: ignore


def test_adaptive_threshold_preserves_image_dimensions():
    pixels = np.full((75, 150), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_adaptive_threshold(source)

    assert result.width == source.width
    assert result.height == source.height


def test_adaptive_threshold_preserves_source_metadata():
    pixels = np.full((10, 20), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_adaptive_threshold(source)

    assert result.source_name == source.source_name
    assert result.source_format == source.source_format


def test_adaptive_threshold_does_not_modify_original_pixels():
    pixels = np.full((20, 20), 128, dtype=np.uint8)
    pixels[8:12, 8:12] = 50
    source = make_source_image(pixels)
    original_pixels = source.pixels.copy()

    apply_adaptive_threshold(source)

    assert np.array_equal(source.pixels, original_pixels)


def test_adaptive_threshold_returns_new_source_image():
    pixels = np.full((20, 20), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    result = apply_adaptive_threshold(source)

    assert isinstance(result, SourceImage)
    assert result is not source


@pytest.mark.parametrize("channels", [3, 4])
def test_adaptive_threshold_rejects_colour_images(channels: int):
    pixels = np.full((20, 20, channels), 128, dtype=np.uint8)
    source = make_source_image(pixels)

    with pytest.raises(ValueError):
        apply_adaptive_threshold(source)
