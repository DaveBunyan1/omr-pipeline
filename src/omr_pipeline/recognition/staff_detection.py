import math
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from omr_pipeline.models.staff import STAFF_LINES, Staff
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


def is_valid_staff_spacing(
    line_positions: tuple[float, ...], tolerance: float = 0.2
) -> bool:
    if len(line_positions) != STAFF_LINES:
        return False

    if isinstance(tolerance, bool):
        raise TypeError("Tolerance must be a finite number greater than 0.")

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be a finite number greater than 0.")

    gaps = [curr - prev for prev, curr in pairwise(line_positions)]

    if any(gap <= 0 for gap in gaps):
        return False

    avg_gap = sum(gaps) / len(gaps)
    return all(math.isclose(gap, avg_gap, rel_tol=tolerance) for gap in gaps)


def detect_staffs(
    line_positions: Sequence[float],
) -> list[Staff]:
    staffs = []
    index = 0

    while index <= len(line_positions) - STAFF_LINES:
        window = tuple(line_positions[index : index + STAFF_LINES])

        if is_valid_staff_spacing(window):
            staff = Staff(window)
            staffs.append(staff)

            index += STAFF_LINES
        else:
            index += 1

    return staffs
