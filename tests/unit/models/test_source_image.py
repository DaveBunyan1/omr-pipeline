from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from omr_pipeline.models import SourceImage


def test_rgb_image_properties() -> None:
    pixels = np.zeros((100, 200, 3), dtype=np.uint8)

    image = SourceImage(
        pixels=pixels,
        source_name="violin_score.png",
        source_format="PNG",
    )

    assert image.width == 200
    assert image.height == 100
    assert image.channels == 3
    assert image.pixels.dtype == np.uint8


def test_grayscale_image_has_one_channel() -> None:
    pixels = np.zeros((100, 200), dtype=np.uint8)

    image = SourceImage(
        pixels=pixels,
        source_name="score.png",
        source_format="PNG",
    )

    assert image.channels == 1


def test_invalid_dtype_raises_error() -> None:
    pixels = np.zeros((100, 200, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="uint8"):
        SourceImage(
            pixels=cast(NDArray[np.uint8], pixels),
            source_name="score.png",
            source_format="PNG",
        )
