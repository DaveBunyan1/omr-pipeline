import numpy as np
import pytest

from omr_pipeline.models.source_image import SourceImage
from omr_pipeline.recognition.staff_detection import detect_horizontal_lines


class TestDetectHorizontalLines:
    def test_detect_five_staff_lines(
        self,
        synthetic_staff_image: SourceImage,
    ) -> None:
        binary_image = (synthetic_staff_image.pixels == 0).astype(np.uint8)

        result = detect_horizontal_lines(binary_image)

        assert result == [
            90,
            91,
            105,
            106,
            120,
            121,
            135,
            136,
            150,
            151,
        ]

    def test_returns_row_indices_not_projection_values(
        self,
        synthetic_staff_image: SourceImage,
    ) -> None:
        binary_image = (synthetic_staff_image.pixels == 0).astype(np.uint8)

        result = detect_horizontal_lines(binary_image)

        assert all(isinstance(position, int) for position in result)
        assert all(0 <= position < binary_image.shape[0] for position in result)

    def test_custom_min_line_coverage(
        self,
        synthetic_staff_image: SourceImage,
    ) -> None:
        binary_image = (synthetic_staff_image.pixels == 0).astype(np.uint8)

        result = detect_horizontal_lines(
            binary_image,
            min_line_coverage=0.9,
        )

        assert result == [
            90,
            91,
            105,
            106,
            120,
            121,
            135,
            136,
            150,
            151,
        ]

    def test_reject_non_2d_image(self) -> None:
        binary_image = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            detect_horizontal_lines(binary_image)

    @pytest.mark.parametrize(
        "coverage",
        [0.0, -0.1, 1.1, 2.0],
    )
    def test_reject_invalid_coverage(
        self,
        coverage: float,
    ) -> None:
        binary_image = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError):
            detect_horizontal_lines(
                binary_image,
                min_line_coverage=coverage,
            )

    @pytest.mark.parametrize(
        "coverage",
        ["0.5", None, True],
    )
    def test_reject_invalid_coverage_type(
        self,
        coverage: float,
    ) -> None:
        binary_image = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(TypeError):
            detect_horizontal_lines(
                binary_image,
                min_line_coverage=coverage,
            )

    def test_no_lines_detected(self) -> None:
        binary_image = np.zeros((100, 100), dtype=np.uint8)

        result = detect_horizontal_lines(binary_image)

        assert result == []

    def test_detect_line_at_image_boundary(self) -> None:
        binary_image = np.zeros((100, 100), dtype=np.uint8)
        binary_image[0, :] = 1

        result = detect_horizontal_lines(binary_image)

        assert result == [0]
