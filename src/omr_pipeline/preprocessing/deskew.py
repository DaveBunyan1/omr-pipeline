import math

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from omr_pipeline.models.source_image import SourceImage
from omr_pipeline.preprocessing.binarization import apply_otsu_threshold


def calculate_horizontal_projection(
    binary_image: NDArray[np.uint8],
) -> NDArray[np.int64]:
    """Calculate the horizontal projection profile of a binary image."""
    if binary_image.ndim != 2:
        raise ValueError("Expected a 2D binary image.")
    return np.sum(binary_image, axis=1)


def create_foreground_mask(
    source: SourceImage,
) -> NDArray[np.uint8]:
    """Convert a grayscale SourceImage into a binary foreground mask."""
    if source.pixels.ndim != 2:
        raise ValueError(f"Expected grayscale format, got {source.pixels.ndim} array.")

    threshold = apply_otsu_threshold(source)

    foreground_mask: NDArray[np.uint8] = (threshold.pixels == 0).astype(np.uint8)

    return foreground_mask


def calculate_projection_score(
    projection: NDArray[np.integer],
) -> float:
    """Calculate the variance of a horizontal projection profile."""
    return float(np.var(projection))


def rotate_image(
    image: NDArray[np.uint8],
    angle: float,
) -> NDArray[np.uint8]:
    """Rotate an image by the specified angle."""

    if image.ndim != 2:
        raise ValueError("Expected a 2D image.")

    if not math.isfinite(angle):
        raise ValueError("Angle must be finite.")

    img = Image.fromarray(image)

    rotated = img.rotate(
        angle,
        resample=Image.Resampling.NEAREST,
        expand=True,
        fillcolor=0,
    )

    return np.array(rotated, dtype=np.uint8, copy=True)


def generate_candidate_angles(
    max_skew: float,
    angle_step: float,
) -> NDArray[np.float64]:
    """Generate candidate correction angles within the specified range."""
    span = 2 * max_skew

    num_steps = int(np.floor((span / angle_step) + 1e-12))

    arr: NDArray[np.float64] = (
        -max_skew + np.arange(num_steps + 1, dtype=np.float64) * angle_step
    )

    arr = np.append(arr, [0.0, max_skew])

    return np.unique(arr)


def estimate_skew_angle(
    source: SourceImage,
    max_skew: object = 5.0,
    angle_step: object = 0.1,
) -> float:
    """Estimate the correction angle for a slightly skewed sheet music image."""

    if source.pixels.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")

    if isinstance(max_skew, bool) or not isinstance(max_skew, (float, int)):
        raise TypeError("max_skew must be a number.")

    if not math.isfinite(max_skew) or max_skew <= 0:
        raise ValueError("max_skew must be a positive, finite number.")

    if isinstance(angle_step, bool) or not isinstance(angle_step, (float, int)):
        raise TypeError("angle_step must be a number.")

    if not math.isfinite(angle_step) or angle_step <= 0:
        raise ValueError("angle_step must be a positive, finite number.")

    if angle_step > max_skew:
        raise ValueError("angle_step must not exceed max_skew.")

    best_angle = 0.0
    best_score = float("-inf")

    foreground_mask = create_foreground_mask(source)

    candidate_angles = generate_candidate_angles(max_skew, angle_step)

    for angle in candidate_angles:
        rotated = rotate_image(foreground_mask, angle)
        projection = calculate_horizontal_projection(rotated)
        score = calculate_projection_score(projection)

        if score > best_score:
            best_score = score
            best_angle = float(angle)

    if best_score <= 0:
        raise ValueError("Unable to estimate skew from image.")

    return best_angle


def rotate_source_image(
    source: SourceImage,
    angle: object,
) -> SourceImage:
    image = source.pixels
    """Rotate a source image by the specified angle."""

    if isinstance(angle, bool) or not isinstance(angle, (int, float)):
        raise TypeError("angle must be a number.")

    if not math.isfinite(angle):
        raise ValueError("Angle must be finite.")

    img = Image.fromarray(image)
    fillcolor: tuple[int, int, int] | tuple[int, int, int, int] | int

    if img.mode == "RGBA":
        fillcolor = (255, 255, 255, 255)
    elif img.mode == "RGB":
        fillcolor = (255, 255, 255)
    else:
        fillcolor = 255

    rotated = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=fillcolor,
    )

    return SourceImage(
        pixels=np.asarray(rotated, dtype=np.uint8, copy=True),
        source_name=source.source_name,
        source_format=source.source_format,
    )


def deskew_image(
    source: SourceImage,
    max_skew: float = 5.0,
    angle_step: float = 0.1,
) -> SourceImage:
    """Correct small rotational skew in a sheet music image."""
    angle = estimate_skew_angle(source, max_skew, angle_step)

    if angle == 0.0:
        return source

    return rotate_source_image(source, angle)
