import numpy as np
from numpy.typing import NDArray

from omr_pipeline.preprocessing.deskew import calculate_horizontal_projection


def detect_horizontal_lines(
    binary_image: NDArray[np.uint8], min_line_coverage: float = 0.5
) -> list[int]:

    if not isinstance(min_line_coverage, float):
        raise TypeError("Coverage must be a number between 0 and 1")

    if min_line_coverage <= 0.0 or min_line_coverage > 1:
        raise ValueError("Coverage must be a number between 0 and 1")

    projections = calculate_horizontal_projection(binary_image)

    threshold = binary_image.shape[1] * min_line_coverage
    line_positions: list[int] = np.where(projections >= threshold)[0].tolist()

    return line_positions


def group_line_positions(
    line_positions: list[int],
) -> list[float]:
    if not line_positions:
        return []

    grouped_line_positions = []

    start = line_positions[0]
    prev = line_positions[0]

    for current in line_positions[1:]:
        if current == prev + 1:
            prev = current
            continue

        grouped_line_positions.append((start + prev) / 2)
        start = current
        prev = current

    grouped_line_positions.append((start + prev) / 2)

    return grouped_line_positions
