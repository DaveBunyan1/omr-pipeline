"""Tests for source image loading."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from omr_pipeline.models import SourceImage
from omr_pipeline.preprocessing import ImageLoadError, load_image


def test_load_png_returns_source_image(tmp_path: Path):
    """A PNG should load into a valid SourceImage."""
    image_path = tmp_path / "score.png"
    Image.new("RGB", (200, 100), color="white").save(image_path)

    result = load_image(image_path)

    assert isinstance(result, SourceImage)
    assert result.pixels.shape == (100, 200, 3)
    assert result.pixels.dtype == np.uint8
    assert result.source_name == "score.png"
    assert result.source_format == "PNG"


def test_load_jpeg_returns_rgb_image(tmp_path: Path):
    """A JPEG should be normalized to RGB."""
    image_path = tmp_path / "score.jpg"
    Image.new("RGB", (200, 100), color="white").save(image_path)

    result = load_image(image_path)

    assert result.pixels.shape == (100, 200, 3)
    assert result.channels == 3
    assert result.source_format == "JPEG"


def test_load_tiff_returns_source_image(tmp_path: Path):
    """A TIFF should load into a valid SourceImage."""
    image_path = tmp_path / "score.tiff"
    Image.new("RGB", (200, 100), color="white").save(image_path)

    result = load_image(image_path)

    assert isinstance(result, SourceImage)
    assert result.pixels.shape == (100, 200, 3)
    assert result.source_format == "TIFF"


def test_load_grayscale_image_preserves_grayscale(tmp_path: Path):
    """An 8-bit grayscale image should remain grayscale."""
    image_path = tmp_path / "score.png"
    Image.new("L", (200, 100), color=128).save(image_path)

    result = load_image(image_path)

    assert result.pixels.shape == (100, 200)
    assert result.channels == 1


def test_load_transparent_image_preserves_alpha(tmp_path: Path):
    """An image with transparency should be normalized to RGBA."""
    image_path = tmp_path / "score.png"
    Image.new("RGBA", (200, 100), color=(255, 255, 255, 128)).save(image_path)

    result = load_image(image_path)

    assert result.pixels.shape == (100, 200, 4)
    assert result.channels == 4
    assert result.pixels[0, 0, 3] == 128


def test_load_nonexistent_file_raises_error(tmp_path: Path):
    """A missing image file should raise ImageLoadError."""
    image_path = tmp_path / "missing.png"

    with pytest.raises(ImageLoadError, match="does not exist"):
        load_image(image_path)


def test_load_directory_raises_error(tmp_path: Path):
    """A directory should not be accepted as an image file."""
    with pytest.raises(ImageLoadError, match="not a file"):
        load_image(tmp_path)


def test_load_unsupported_format_raises_error(tmp_path: Path):
    """An unsupported image format should raise ImageLoadError."""
    image_path = tmp_path / "score.bmp"
    Image.new("RGB", (200, 100)).save(image_path)

    with pytest.raises(ImageLoadError, match="Unsupported image format"):
        load_image(image_path)


def test_load_corrupt_image_raises_error(tmp_path: Path):
    """An invalid image file should raise ImageLoadError."""
    image_path = tmp_path / "corrupt.png"
    image_path.write_bytes(b"This is not a valid image.")

    with pytest.raises(ImageLoadError, match="Failed to load image"):
        load_image(image_path)


def test_load_multiframe_tiff_raises_error(tmp_path: Path):
    """A multi-page TIFF should be rejected."""
    image_path = tmp_path / "multipage.tiff"

    first_frame = Image.new("RGB", (200, 100), color="white")
    second_frame = Image.new("RGB", (200, 100), color="black")

    first_frame.save(
        image_path,
        save_all=True,
        append_images=[second_frame],
    )

    with pytest.raises(ImageLoadError, match="Multi-frame"):
        load_image(image_path)
