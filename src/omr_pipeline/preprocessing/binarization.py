import numpy as np

from omr_pipeline.models.source_image import SourceImage

THRESHOLD_MIN = 0
THRESHOLD_MAX = 255


def apply_threshold(
    source: SourceImage,
    threshold: int = 127,
) -> SourceImage:
    if type(threshold) is not int:
        raise TypeError(f"threshold must be an int, got {type(threshold).__name__}.")

    if threshold < THRESHOLD_MIN or threshold > THRESHOLD_MAX:
        raise ValueError(
            f"Threshold must be between {THRESHOLD_MIN} and {THRESHOLD_MAX}."
        )
    return SourceImage(
        pixels=np.where(source.pixels > threshold, THRESHOLD_MAX, THRESHOLD_MIN).astype(
            np.uint8
        ),
        source_name=source.source_name,
        source_format=source.source_format,
    )
