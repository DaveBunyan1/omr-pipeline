import math

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from omr_pipeline.models.source_image import SourceImage
from omr_pipeline.preprocessing.binarization import apply_otsu_threshold


def calculate_horizontal_projection(
    binary_image: NDArray[np.uint8],
) -> NDArray[np.int64]:
    """Calculate the horizontal projection profile of a binary image."""
    if binary_image.ndim != 2:
        raise ValueError("Expected a 2D binary image.")
    return np.sum(binary_image, axis=1)


def create_foreground_mask(
    source: SourceImage,
) -> NDArray[np.uint8]:
    """Convert a grayscale SourceImage into a binary foreground mask."""
    if source.pixels.ndim != 2:
        raise ValueError(f"Expected grayscale format, got {source.pixels.ndim} array.")

    threshold = apply_otsu_threshold(source)

    foreground_mask: NDArray[np.uint8] = (threshold.pixels == 0).astype(np.uint8)

    return foreground_mask


def calculate_projection_score(
    projection: NDArray[np.integer],
) -> float:
    """Calculate the variance of a horizontal projection profile."""
    return float(np.var(projection))


def rotate_image(
    image: NDArray[np.uint8],
    angle: float,
) -> NDArray[np.uint8]:
    """Rotate an image by the specified angle."""

    if image.ndim != 2:
        raise ValueError("Expected a 2D image.")

    if not math.isfinite(angle):
        raise ValueError("Angle must be finite.")

    img = Image.fromarray(image)

    rotated = img.rotate(
        angle,
        resample=Image.Resampling.NEAREST,
        expand=True,
        fillcolor=0,
    )

    return np.array(rotated, dtype=np.uint8, copy=True)
