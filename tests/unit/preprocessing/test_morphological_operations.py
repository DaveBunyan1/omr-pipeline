import numpy as np
import pytest

from omr_pipeline.preprocessing.morphological_operations import (
    erode_horizontal,
)


def test_erode_horizontal_preserves_complete_foreground_regions() -> None:
    image = np.array(
        [[False, True, True, True, True, False]],
        dtype=np.bool_,
    )

    result = erode_horizontal(image, kernel_length=3)

    expected = np.array(
        [[False, True, True, False]],
        dtype=np.bool_,
    )

    np.testing.assert_array_equal(result, expected)


def test_erode_horizontal_removes_regions_shorter_than_kernel() -> None:
    image = np.array(
        [[False, True, True, False, False]],
        dtype=np.bool_,
    )

    result = erode_horizontal(image, kernel_length=3)

    expected = np.zeros((1, 3), dtype=np.bool_)

    np.testing.assert_array_equal(result, expected)


def test_erode_horizontal_supports_multiple_rows() -> None:
    image = np.array(
        [
            [True, True, True, True, True],
            [True, True, False, True, True],
        ],
        dtype=np.bool_,
    )

    result = erode_horizontal(image, kernel_length=3)

    expected = np.array(
        [
            [True, True, True],
            [False, False, False],
        ],
        dtype=np.bool_,
    )

    np.testing.assert_array_equal(result, expected)


def test_erode_horizontal_with_kernel_length_one() -> None:
    image = np.array(
        [[True, False, True]],
        dtype=np.bool_,
    )

    result = erode_horizontal(image, kernel_length=1)

    np.testing.assert_array_equal(result, image)


@pytest.mark.parametrize(
    "kernel_length",
    [0, -1],
)
def test_erode_horizontal_rejects_invalid_kernel_length(
    kernel_length: int,
) -> None:
    image = np.ones((3, 5), dtype=np.bool_)

    with pytest.raises(ValueError):
        erode_horizontal(image, kernel_length)


def test_erode_horizontal_rejects_kernel_larger_than_width() -> None:
    image = np.ones((3, 5), dtype=np.bool_)

    with pytest.raises(ValueError):
        erode_horizontal(image, kernel_length=6)


def test_erode_horizontal_rejects_non_integer_kernel_length() -> None:
    image = np.ones((3, 5), dtype=np.bool_)

    with pytest.raises(TypeError):
        erode_horizontal(image, kernel_length=2.5)  # type: ignore[arg-type]


def test_erode_horizontal_rejects_non_2d_image() -> None:
    image = np.ones((3, 5, 3), dtype=np.bool_)

    with pytest.raises(ValueError):
        erode_horizontal(image, kernel_length=3)
