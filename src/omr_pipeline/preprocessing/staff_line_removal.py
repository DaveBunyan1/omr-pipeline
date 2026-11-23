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


def repair_staff_line_intersections(
    source: SourceImage,
    horizontal_lines: NDArray[np.bool_],
) -> SourceImage:
    """Restore pixels at staff-line intersections with vertical structures."""

    if source.pixels.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image.")

    if not np.isin(source.pixels, [0, 255]).all():
        raise ValueError("Expected a binary image with pixel values 0 or 255.")

    if horizontal_lines.ndim != 2:
        raise ValueError("Expected a two-dimensional mask.")

    if horizontal_lines.dtype != np.bool_:
        raise TypeError("Expected a boolean horizontal-line mask.")

    if horizontal_lines.shape != source.pixels.shape:
        raise ValueError("Mask dimensions must match the source image.")

    foreground = source.pixels == 0

    # Remove the extracted horizontal structures.
    cleaned = foreground & ~horizontal_lines

    # Find pixels immediately above and below the extracted line mask.
    above = np.zeros_like(foreground)
    below = np.zeros_like(foreground)

    above[1:, :] = cleaned[:-1, :]
    below[:-1, :] = cleaned[1:, :]

    # Restore line-mask pixels with vertical support on both sides.
    repair_mask = horizontal_lines & above & below

    reconstructed = cleaned | repair_mask

    output_pixels = np.where(reconstructed, 0, 255).astype(np.uint8)

    return SourceImage(
        pixels=output_pixels,
        source_name=source.source_name,
        source_format=source.source_format,
    )
