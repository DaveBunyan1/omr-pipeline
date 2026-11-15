import numpy as np
import pytest

from omr_pipeline.models.source_image import SourceImage
from omr_pipeline.models.staff import Staff
from omr_pipeline.recognition.staff_detection import (
    detect_horizontal_lines,
    detect_staffs,
    group_line_positions,
    is_valid_staff_spacing,
)


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


class TestGroupLinePositions:
    def test_group_multiple_adjacent_positions(self) -> None:
        line_positions = [90, 91, 105, 106, 120, 121, 135, 136, 150, 151]

        result = group_line_positions(line_positions)

        assert result == [90.5, 105.5, 120.5, 135.5, 150.5]

    def test_group_single_pair(self) -> None:
        line_positions = [90, 91]

        result = group_line_positions(line_positions)

        assert result == [90.5]

    def test_single_position_groups(self) -> None:
        line_positions = [90, 105, 120]

        result = group_line_positions(line_positions)

        assert result == [90.0, 105.0, 120.0]

    def test_single_position(self) -> None:
        line_positions = [90]

        result = group_line_positions(line_positions)

        assert result == [90.0]

    def test_empty_input(self) -> None:
        result = group_line_positions([])

        assert result == []

    def test_groups_non_uniform_widths(self) -> None:
        line_positions = [90, 91, 92, 105, 106, 107, 108]

        result = group_line_positions(line_positions)

        assert result == [91.0, 106.5]

    def test_input_is_not_modified(self) -> None:
        line_positions = [90, 91, 105, 106]
        original = line_positions.copy()

        group_line_positions(line_positions)

        assert line_positions == original

    def test_returns_float_positions(self) -> None:
        line_positions = [90, 91, 105]

        result = group_line_positions(line_positions)

        assert all(isinstance(position, float) for position in result)


class TestIsValidStaffSpacing:
    def test_returns_true_for_equal_spacing(self):
        line_positions = (90.0, 105.0, 120.0, 135.0, 150.0)

        assert is_valid_staff_spacing(line_positions)

    def test_returns_true_for_spacing_within_tolerance(self):
        line_positions = (90.0, 105.0, 120.5, 135.0, 150.0)

        assert is_valid_staff_spacing(line_positions)

    def test_returns_false_for_spacing_outside_tolerance(self):
        line_positions = (90.0, 105.0, 125.0, 135.0, 150.0)

        assert not is_valid_staff_spacing(line_positions)

    def test_returns_false_for_fewer_than_five_lines(self):
        line_positions = (90.0, 105.0, 120.0, 135.0)

        assert not is_valid_staff_spacing(line_positions)

    def test_returns_false_for_more_than_five_lines(self):
        line_positions = (90.0, 105.0, 120.0, 135.0, 150.0, 165.0)

        assert not is_valid_staff_spacing(line_positions)

    def test_returns_false_for_duplicate_positions(self):
        line_positions = (90.0, 105.0, 105.0, 135.0, 150.0)

        assert not is_valid_staff_spacing(line_positions)

    def test_returns_false_for_descending_positions(self):
        line_positions = (90.0, 105.0, 135.0, 120.0, 150.0)

        assert not is_valid_staff_spacing(line_positions)

    def test_returns_true_with_zero_tolerance_for_equal_spacing(self):
        line_positions = (90.0, 105.0, 120.0, 135.0, 150.0)

        assert is_valid_staff_spacing(line_positions, tolerance=0.0)

    def test_returns_false_with_zero_tolerance_for_unequal_spacing(self):
        line_positions = (90.0, 105.0, 120.5, 135.0, 150.0)

        assert not is_valid_staff_spacing(line_positions, tolerance=0.0)


class TestDetectStaffs:
    def test_detects_single_staff(self):
        line_positions = [
            90.0,
            105.0,
            120.0,
            135.0,
            150.0,
        ]

        staffs = detect_staffs(line_positions)

        assert len(staffs) == 1
        assert staffs[0].line_positions == tuple(line_positions)

    def test_detects_multiple_staffs(self):
        line_positions = [
            90.0,
            105.0,
            120.0,
            135.0,
            150.0,
            200.0,
            215.0,
            230.0,
            245.0,
            260.0,
        ]

        staffs = detect_staffs(line_positions)

        assert len(staffs) == 2
        assert staffs[0].line_positions == (90.0, 105.0, 120.0, 135.0, 150.0)
        assert staffs[1].line_positions == (200.0, 215.0, 230.0, 245.0, 260.0)

    def test_returns_empty_list_when_no_staff_is_detected(self):
        line_positions = [
            90.0,
            105.0,
            120.0,
            140.0,
            150.0,
        ]

        assert detect_staffs(line_positions) == []

    def test_returns_empty_list_for_insufficient_lines(self):
        line_positions = [90.0, 105.0, 120.0, 135.0]

        assert detect_staffs(line_positions) == []

    def test_skips_invalid_lines_and_detects_following_staff(self):
        line_positions = [
            50.0,
            90.0,
            105.0,
            120.0,
            135.0,
            150.0,
        ]

        staffs = detect_staffs(line_positions)

        assert len(staffs) == 1
        assert staffs[0].line_positions == (90.0, 105.0, 120.0, 135.0, 150.0)

    def test_does_not_modify_input(self):
        line_positions = [
            90.0,
            105.0,
            120.0,
            135.0,
            150.0,
        ]
        original = line_positions.copy()

        detect_staffs(line_positions)

        assert line_positions == original

    def test_returns_staff_instances(self):
        line_positions = [
            90.0,
            105.0,
            120.0,
            135.0,
            150.0,
        ]

        staffs = detect_staffs(line_positions)

        assert all(isinstance(staff, Staff) for staff in staffs)
