import numpy as np
from numpy.typing import NDArray


def calculate_horizontal_projection(
    binary_image: NDArray[np.uint8],
) -> NDArray[np.int64]:
    """Calculate the horizontal projection profile of a binary image."""
    if binary_image.ndim != 2:
        raise ValueError("Expected a 2D binary image.")
    return np.sum(binary_image, axis=1)
