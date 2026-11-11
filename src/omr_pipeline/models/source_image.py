"""Data model representing an ingested sheet music image."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SourceImage:
    """An ingested sheet music image.

    Pixel data uses uint8 values and one of the following
    representations:

    - Grayscale: (height, width)
    - RGB:       (height, width, 3)
    - RGBA:      (height, width, 4)
    """

    pixels: NDArray[np.uint8]
    source_name: str
    source_format: str

    def __post_init__(self) -> None:
        """Validate the image representation."""

        if self.pixels.dtype != np.uint8:
            raise ValueError("Image pixels must have dtype uint8.")

        if self.pixels.ndim not in (2, 3):
            raise ValueError("Image must have 2 or 3 dimensions.")

        if self.pixels.size == 0:
            raise ValueError("Image cannot be empty.")

        if self.pixels.ndim == 3 and self.pixels.shape[2] not in (3, 4):
            raise ValueError("Image must have 3 (RGB) or 4 (RGBA) channels.")

        if not self.source_name:
            raise ValueError("Source name cannot be empty.")

        if not self.source_format:
            raise ValueError("Source format cannot be empty.")

    @property
    def width(self) -> int:
        """Return image width in pixels."""
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        """Return image height in pixels."""
        return int(self.pixels.shape[0])

    @property
    def channels(self) -> int:
        """Return the number of image channels."""
        if self.pixels.ndim == 2:
            return 1

        return int(self.pixels.shape[2])
