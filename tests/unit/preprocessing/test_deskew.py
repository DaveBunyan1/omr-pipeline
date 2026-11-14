import numpy as np

from omr_pipeline.preprocessing.deskew import calculate_horizontal_projection


def test_calculate_horizontal_projection():
    binary_image = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )

    result = calculate_horizontal_projection(binary_image)

    expected = np.array([0, 4, 0, 5])

    np.testing.assert_array_equal(result, expected)
