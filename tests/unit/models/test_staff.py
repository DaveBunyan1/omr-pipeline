import pytest

from omr_pipeline.models.staff import Staff


@pytest.fixture
def staff() -> Staff:
    return Staff(line_positions=(100.0, 120.0, 140.0, 160.0, 180.0))


class TestStaff:
    def test_create_staff(self, staff: Staff) -> None:
        assert isinstance(staff, Staff)

    def test_line_positions(self, staff: Staff) -> None:
        assert staff.line_positions == (
            100.0,
            120.0,
            140.0,
            160.0,
            180.0,
        )

    def test_top(self, staff: Staff) -> None:
        assert staff.top == 100.0

    def test_bottom(self, staff: Staff) -> None:
        assert staff.bottom == 180.0

    def test_spacing(self, staff: Staff) -> None:
        assert staff.spacing == 20.0

    def test_spacing_with_uneven_positions(self) -> None:
        staff = Staff(line_positions=(100.0, 121.0, 139.0, 162.0, 180.0))

        assert staff.spacing == 20.0

    def test_reject_incorrect_number_of_lines(self) -> None:
        with pytest.raises(ValueError, match="exactly 5 lines"):
            Staff(line_positions=(100.0, 120.0, 140.0, 160.0))  # type: ignore

    def test_reject_too_many_lines(self) -> None:
        with pytest.raises(ValueError, match="exactly 5 lines"):
            Staff(line_positions=(100.0, 120.0, 140.0, 160.0, 180.0, 200.0))  # type: ignore

    def test_reject_unsorted_positions(self) -> None:
        with pytest.raises(ValueError, match="ordered from top to bottom"):
            Staff(line_positions=(100.0, 140.0, 120.0, 160.0, 180.0))

    def test_reject_duplicate_positions(self) -> None:
        with pytest.raises(ValueError, match="distinct positions"):
            Staff(line_positions=(100.0, 120.0, 120.0, 160.0, 180.0))

    def test_staff_is_immutable(self, staff: Staff) -> None:
        with pytest.raises(AttributeError):
            staff.line_positions = (101.0, 121.0, 141.0, 161.0, 181.0)  # type: ignore
