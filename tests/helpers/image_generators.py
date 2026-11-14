import numpy as np
from PIL import Image

from omr_pipeline.models import SourceImage


def create_skewed_staff_image(
    source: SourceImage,
    angle: float,
) -> SourceImage:
    """Create a synthetic skewed staff image for testing."""

    image = Image.fromarray(source.pixels)

    rotated = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=255,
    )

    return SourceImage(
        pixels=np.array(rotated, dtype=np.uint8, copy=True),
        source_name=source.source_name,
        source_format=source.source_format,
    )
