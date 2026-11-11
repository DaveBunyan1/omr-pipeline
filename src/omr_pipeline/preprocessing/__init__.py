"""Image preprocessing and ingestion functionality."""

from omr_pipeline.preprocessing.image_loader import (
    ImageLoadError,
    load_image,
)

__all__ = ["ImageLoadError", "load_image"]
