import numpy as np

from omr_pipeline.models.source_image import SourceImage

THRESHOLD_MIN = 0
THRESHOLD_MAX = 255
GLOBAL_THRESHOLD_DEFAULT = 127


def apply_threshold(
    source: SourceImage,
    threshold: int = GLOBAL_THRESHOLD_DEFAULT,
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


def apply_otsu_threshold(source: SourceImage) -> SourceImage:
    if source.pixels.ndim != 2:
        raise ValueError(f"Expected grayscale format, got {source.pixels.ndim} array.")

    histogram, _ = np.histogram(
        source.pixels, bins=THRESHOLD_MAX + 1, range=(THRESHOLD_MIN, THRESHOLD_MAX + 1)
    )
    total_pixels = source.pixels.size

    bins = np.arange(THRESHOLD_MAX + 1)

    best_threshold = 0
    max_variance = 0.0

    for t in range(THRESHOLD_MAX + 1):
        bg_pixels = np.sum(histogram[: t + 1])
        fg_pixels = np.sum(histogram[t + 1 :])

        # Avoid dividing by 0
        if bg_pixels == 0 or fg_pixels == 0:
            continue

        w0 = bg_pixels / total_pixels
        w1 = fg_pixels / total_pixels

        mu0 = np.sum(bins[: t + 1] * histogram[: t + 1]) / bg_pixels
        mu1 = np.sum(bins[t + 1 :] * histogram[t + 1 :]) / fg_pixels

        variance = w0 * w1 * ((mu0 - mu1) ** 2)

        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    return SourceImage(
        np.where(source.pixels > best_threshold, THRESHOLD_MAX, THRESHOLD_MIN).astype(
            np.uint8
        ),
        source_name=source.source_name,
        source_format=source.source_format,
    )
