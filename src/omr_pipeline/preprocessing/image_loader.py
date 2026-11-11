"""Functions for loading source images into the OMR pipeline."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from omr_pipeline.models import SourceImage

SUPPORTED_FORMATS = {"PNG", "JPEG", "TIFF"}


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded into the OMR pipeline."""


def load_image(path: str | Path) -> SourceImage:
    """Load an image file and return a validated SourceImage.

    Supported formats:
        - PNG
        - JPEG
        - TIFF

    Colour images are normalized to RGB or RGBA.
    Grayscale images are normalized to L (8-bit grayscale).

    Args:
        path: Path to the image file.

    Returns:
        A validated SourceImage instance.

    Raises:
        ImageLoadError: If the file is missing, unsupported, corrupt,
            or cannot be decoded.
    """

    image_path = Path(path)

    if not image_path.exists():
        raise ImageLoadError(f"Image file does not exist: {image_path}")

    if not image_path.is_file():
        raise ImageLoadError(f"Path is not a file: {image_path}")

    try:
        with Image.open(image_path) as image:
            image_format = image.format

            if image_format not in SUPPORTED_FORMATS:
                raise ImageLoadError(f"Unsupported image format: {image_format}")

            if getattr(image, "n_frames", 1) > 1:
                raise ImageLoadError("Multi-frame images are not currently supported.")

            image.load()
            oriented_image = ImageOps.exif_transpose(image)

            if oriented_image.mode in ("1", "L"):
                normalized = oriented_image.convert("L")
            elif (
                "A" in oriented_image.getbands()
                or "transparency" in oriented_image.info
            ):
                normalized = oriented_image.convert("RGBA")
            else:
                normalized = oriented_image.convert("RGB")

            pixels: NDArray[np.uint8] = np.array(
                normalized,
                dtype=np.uint8,
                copy=True,
            )

    except ImageLoadError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise ImageLoadError(f"Failed to load image '{image_path}': {exc}") from exc

    return SourceImage(
        pixels=pixels,
        source_name=image_path.name,
        source_format=image_format,
    )
