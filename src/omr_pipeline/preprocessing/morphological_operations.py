import numpy as np
from numpy.typing import NDArray


def erode_horizontal(
    image: NDArray[np.bool_],
    kernel_length: int,
) -> NDArray[np.bool_]:
    """Apply horizontal morphological erosion to a binary image."""

    if image.ndim != 2:
        raise ValueError("Expected a two-dimensional binary image.")

    if type(kernel_length) is not int:
        raise TypeError("kernel_length must be an integer.")

    if kernel_length < 1:
        raise ValueError("kernel_length must be greater than zero.")

    if kernel_length > image.shape[1]:
        raise ValueError("kernel_length cannot exceed image width.")

    result = np.ones(
        (image.shape[0], image.shape[1] - kernel_length + 1),
        dtype=np.bool_,
    )

    for offset in range(kernel_length):
        result &= image[:, offset : offset + result.shape[1]]

    return result
