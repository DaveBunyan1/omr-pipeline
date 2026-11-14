import numpy as np
import pytest
from numpy.typing import NDArray

from omr_pipeline.models import SourceImage


@pytest.fixture
def synthetic_staff_image() -> SourceImage:
    """Create a synthetic grayscale image containing five staff lines."""

    height = 240
    width = 400

    pixels: NDArray[np.uint8] = np.full(
        (height, width),
        255,
        dtype=np.uint8,
    )

    staff_spacing = 15
    staff_thickness = 2
    first_staff_line = 90

    for i in range(5):
        y = first_staff_line + (i * staff_spacing)

        pixels[
            y : y + staff_thickness,
            20 : width - 20,
        ] = 0

    return SourceImage(
        pixels=pixels,
        source_name="synthetic_staff.png",
        source_format="PNG",
    )
