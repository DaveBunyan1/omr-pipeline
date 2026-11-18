import numpy as np
from numpy.typing import NDArray

from omr_pipeline.models import SourceImage
from omr_pipeline.preprocessing.morphological_operations import (
    morphological_opening,
)


def extract_horizontal_lines(
    source: SourceImage,
    kernel_length: int,
) -> NDArray[np.bool_]:
    """Extract long horizontal structures from a binary image."""

    if source.pixels.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image.")

    if not np.isin(source.pixels, [0, 255]).all():
        raise ValueError("Expected a binary image with pixel values 0 or 255.")

    foreground_mask = source.pixels == 0

    return morphological_opening(foreground_mask, kernel_length)


def remove_staff_lines(
    source: SourceImage,
    kernel_length: int,
) -> SourceImage:
    """Remove extracted horizontal lines from a binary image."""

    horizontal_lines = extract_horizontal_lines(
        source,
        kernel_length,
    )

    cleaned_pixels = source.pixels.copy()
    cleaned_pixels[horizontal_lines] = 255

    return SourceImage(
        pixels=cleaned_pixels,
        source_name=source.source_name,
        source_format=source.source_format,
    )
