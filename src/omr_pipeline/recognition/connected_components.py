"""Connected component analysis for binary images."""

import numpy as np
from numpy.typing import NDArray


def label_connected_components(
    binary_image: NDArray[np.bool_],
) -> tuple[NDArray[np.int32], int]:
    """Label connected foreground regions using 8-connectivity."""

    if binary_image.ndim != 2:
        raise ValueError("Expected a two-dimensional binary image.")

    if binary_image.dtype != np.bool_:
        raise TypeError("Expected a boolean binary image.")

    height, width = binary_image.shape

    labels = np.zeros((height, width), dtype=np.int32)

    component_count = 0

    for row in range(height):
        for col in range(width):
            if not binary_image[row, col] or labels[row, col] != 0:
                continue

            component_count += 1
            labels[row, col] = component_count

            stack: list[tuple[int, int]] = [(row, col)]

            while stack:
                current_row, current_col = stack.pop()

                for row_offset in (-1, 0, 1):
                    for col_offset in (-1, 0, 1):
                        if row_offset == 0 and col_offset == 0:
                            continue

                        neighbour_row = current_row + row_offset
                        neighbour_col = current_col + col_offset

                        if not (
                            0 <= neighbour_row < height and 0 <= neighbour_col < width
                        ):
                            continue

                        if (
                            binary_image[neighbour_row, neighbour_col]
                            and labels[neighbour_row, neighbour_col] == 0
                        ):
                            labels[neighbour_row, neighbour_col] = component_count
                            stack.append((neighbour_row, neighbour_col))

    return labels, component_count
