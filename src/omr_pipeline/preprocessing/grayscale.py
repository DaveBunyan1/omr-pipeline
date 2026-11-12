import numpy as np
from PIL import Image

from omr_pipeline.models.source_image import SourceImage


def convert_to_grayscale(source: SourceImage) -> SourceImage:
    img = Image.fromarray(source.pixels)

    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    grayscale_img = img.convert("L")
    img_array = np.asarray(grayscale_img)

    return SourceImage(img_array, source.source_name, source.source_format)
