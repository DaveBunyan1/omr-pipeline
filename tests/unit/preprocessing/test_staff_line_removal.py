import numpy as np
import pytest

from omr_pipeline.models import SourceImage
from omr_pipeline.preprocessing.staff_line_removal import (
    extract_horizontal_lines,
    remove_staff_lines,
)


def create_source_image(pixels: np.ndarray) -> SourceImage:
    return SourceImage(
        pixels=pixels.astype(np.uint8),
        source_name="test.png",
        source_format="PNG",
    )


def test_extract_horizontal_lines_preserves_long_lines() -> None:
    pixels = np.full((3, 10), 255, dtype=np.uint8)
    pixels[1, 1:9] = 0

    source = create_source_image(pixels)

    result = extract_horizontal_lines(source, kernel_length=5)

    expected = np.zeros((3, 10), dtype=np.bool_)
    expected[1, 1:9] = True

    np.testing.assert_array_equal(result, expected)


def test_extract_horizontal_lines_removes_short_features() -> None:
    pixels = np.full((3, 10), 255, dtype=np.uint8)
    pixels[1, 2:5] = 0

    source = create_source_image(pixels)

    result = extract_horizontal_lines(source, kernel_length=5)

    expected = np.zeros((3, 10), dtype=np.bool_)

    np.testing.assert_array_equal(result, expected)


def test_extract_horizontal_lines_preserves_features_at_kernel_length() -> None:
    pixels = np.full((3, 10), 255, dtype=np.uint8)
    pixels[1, 2:7] = 0

    source = create_source_image(pixels)

    result = extract_horizontal_lines(source, kernel_length=5)

    expected = np.zeros((3, 10), dtype=np.bool_)
    expected[1, 2:7] = True

    np.testing.assert_array_equal(result, expected)


def test_extract_horizontal_lines_supports_multiple_rows() -> None:
    pixels = np.full((3, 10), 255, dtype=np.uint8)

    pixels[0, 1:9] = 0
    pixels[1, 2:5] = 0
    pixels[2, 1:9] = 0

    source = create_source_image(pixels)

    result = extract_horizontal_lines(source, kernel_length=5)

    expected = np.zeros((3, 10), dtype=np.bool_)
    expected[0, 1:9] = True
    expected[2, 1:9] = True

    np.testing.assert_array_equal(result, expected)


def test_extract_horizontal_lines_returns_boolean_array() -> None:
    pixels = np.full((3, 10), 255, dtype=np.uint8)
    pixels[1, 1:9] = 0

    source = create_source_image(pixels)

    result = extract_horizontal_lines(source, kernel_length=5)

    assert result.dtype == np.bool_


def test_extract_horizontal_lines_preserves_original_dimensions() -> None:
    pixels = np.full((5, 20), 255, dtype=np.uint8)
    source = create_source_image(pixels)

    result = extract_horizontal_lines(source, kernel_length=5)

    assert result.shape == pixels.shape


def test_extract_horizontal_lines_does_not_modify_source() -> None:
    pixels = np.full((3, 10), 255, dtype=np.uint8)
    pixels[1, 1:9] = 0

    source = create_source_image(pixels)
    original = source.pixels.copy()

    extract_horizontal_lines(source, kernel_length=5)

    np.testing.assert_array_equal(source.pixels, original)


@pytest.mark.parametrize("pixel_value", [100, 127, 200])
def test_extract_horizontal_lines_rejects_non_binary_images(
    pixel_value: int,
) -> None:
    pixels = np.full((3, 10), 255, dtype=np.uint8)
    pixels[1, 2] = pixel_value

    source = create_source_image(pixels)

    with pytest.raises(ValueError, match="binary image"):
        extract_horizontal_lines(source, kernel_length=5)


def test_extract_horizontal_lines_rejects_rgb_images() -> None:
    pixels = np.full((3, 10, 3), 255, dtype=np.uint8)
    source = create_source_image(pixels)

    with pytest.raises(ValueError, match="grayscale"):
        extract_horizontal_lines(source, kernel_length=5)


def test_extract_horizontal_lines_rejects_rgba_images() -> None:
    pixels = np.full((3, 10, 4), 255, dtype=np.uint8)
    source = create_source_image(pixels)

    with pytest.raises(ValueError, match="grayscale"):
        extract_horizontal_lines(source, kernel_length=5)


def test_remove_staff_lines_removes_horizontal_lines() -> None:
    pixels = np.full((5, 20), 255, dtype=np.uint8)
    pixels[2, :] = 0

    source = create_source_image(pixels)

    result = remove_staff_lines(source, kernel_length=10)

    expected = np.full((5, 20), 255, dtype=np.uint8)

    np.testing.assert_array_equal(result.pixels, expected)


def test_remove_staff_lines_preserves_short_features() -> None:
    pixels = np.full((5, 20), 255, dtype=np.uint8)

    # Short horizontal feature.
    pixels[1, 2:5] = 0

    # Vertical feature.
    pixels[1:4, 10] = 0

    source = create_source_image(pixels)

    result = remove_staff_lines(source, kernel_length=10)

    np.testing.assert_array_equal(result.pixels, pixels)


def test_remove_staff_lines_preserves_source_metadata() -> None:
    pixels = np.full((5, 20), 255, dtype=np.uint8)
    source = create_source_image(pixels)

    result = remove_staff_lines(source, kernel_length=10)

    assert result.source_name == source.source_name
    assert result.source_format == source.source_format


def test_remove_staff_lines_preserves_original_dimensions() -> None:
    pixels = np.full((5, 20), 255, dtype=np.uint8)
    source = create_source_image(pixels)

    result = remove_staff_lines(source, kernel_length=10)

    assert result.pixels.shape == source.pixels.shape


def test_remove_staff_lines_does_not_modify_source() -> None:
    pixels = np.full((5, 20), 255, dtype=np.uint8)
    pixels[2, :] = 0

    source = create_source_image(pixels)
    original = source.pixels.copy()

    remove_staff_lines(source, kernel_length=10)

    np.testing.assert_array_equal(source.pixels, original)


def test_remove_staff_lines_returns_uint8_image() -> None:
    pixels = np.full((5, 20), 255, dtype=np.uint8)
    source = create_source_image(pixels)

    result = remove_staff_lines(source, kernel_length=10)

    assert result.pixels.dtype == np.uint8
