import numpy as np
import pytest

from omr_pipeline.recognition.connected_components import (
    label_connected_components,
)


def test_labels_single_component() -> None:
    image = np.array(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ],
        dtype=np.bool_,
    )

    labels, count = label_connected_components(image)

    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(labels, expected)
    assert count == 1


def test_labels_multiple_components() -> None:
    image = np.array(
        [
            [True, True, False, False, True],
            [True, True, False, False, True],
            [False, False, False, False, False],
            [False, True, True, False, False],
        ],
        dtype=np.bool_,
    )

    labels, count = label_connected_components(image)

    assert count == 3
    assert labels[0, 0] == labels[1, 1]
    assert labels[0, 4] == labels[1, 4]
    assert labels[3, 1] == labels[3, 2]

    assert labels[0, 0] != labels[0, 4]
    assert labels[0, 4] != labels[3, 1]


def test_labels_diagonally_connected_pixels() -> None:
    image = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=np.bool_,
    )

    labels, count = label_connected_components(image)

    assert count == 1
    assert labels[0, 0] == labels[1, 1]


def test_labels_empty_foreground() -> None:
    image = np.zeros((4, 4), dtype=np.bool_)

    labels, count = label_connected_components(image)

    assert count == 0
    assert not labels.any()


def test_labels_single_foreground_pixel() -> None:
    image = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ],
        dtype=np.bool_,
    )

    labels, count = label_connected_components(image)

    assert count == 1
    assert labels[1, 1] == 1


def test_labels_does_not_modify_input() -> None:
    image = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=np.bool_,
    )
    original = image.copy()

    label_connected_components(image)

    np.testing.assert_array_equal(image, original)


def test_rejects_non_2d_image() -> None:
    image = np.zeros((3, 3, 3), dtype=np.bool_)

    with pytest.raises(ValueError):
        label_connected_components(image)


def test_rejects_non_boolean_image() -> None:
    image = np.zeros((3, 3), dtype=np.uint8)

    with pytest.raises(TypeError):
        label_connected_components(image)  # type: ignore
