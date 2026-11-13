import math

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


def apply_adaptive_threshold(
    source: SourceImage,
    block_size: int = 11,
    constant: float | int = 2.0,
) -> SourceImage:
    """
    Applies adaptive mean thresholding to a 2D grayscale image using an integral image.
    """
    img = source.pixels

    if type(block_size) is not int:
        raise TypeError("block_size must be an integer.")

    if block_size < 3 or block_size % 2 == 0:
        raise ValueError(
            "block_size must be an odd integer greater than or equal to 3."
        )

    if img.ndim != 2:
        raise ValueError(f"Expected grayscale format, got {img.ndim} array.")

    if isinstance(constant, bool):
        raise TypeError("constant must be a number.")

    if not math.isfinite(constant):
        raise ValueError("constant must be finite.")

    h, w = img.shape
    r = block_size // 2

    ii = np.cumsum(np.cumsum(img, axis=0, dtype=np.float64), axis=1)
    padded_ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant")

    ones = np.ones((h, w), dtype=np.float64)
    count_ii = np.cumsum(np.cumsum(ones, axis=0), axis=1)
    padded_count = np.pad(count_ii, ((1, 0), (1, 0)), mode="constant")

    rows = np.arange(h)
    cols = np.arange(w)

    r1 = np.clip(rows - r, 0, h - 1)[:, None]
    r2 = np.clip(rows + r, 0, h - 1)[:, None] + 1
    c1 = np.clip(cols - r, 0, w - 1)[None, :]
    c2 = np.clip(cols + r, 0, w - 1)[None, :] + 1

    window_sums = (
        padded_ii[r2, c2] - padded_ii[r1, c2] - padded_ii[r2, c1] + padded_ii[r1, c1]
    )
    window_counts = (
        padded_count[r2, c2]
        - padded_count[r1, c2]
        - padded_count[r2, c1]
        + padded_count[r1, c1]
    )

    local_mean = window_sums / window_counts

    local_threshold = np.clip(
        local_mean - constant,
        THRESHOLD_MIN,
        THRESHOLD_MAX,
    )

    return SourceImage(
        pixels=np.where(img > local_threshold, THRESHOLD_MAX, THRESHOLD_MIN).astype(
            np.uint8
        ),
        source_format=source.source_format,
        source_name=source.source_name,
    )
