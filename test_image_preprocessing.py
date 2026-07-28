import numpy as np
import pytest

from image_preprocessing import crop_to_foreground


def _mask_with_box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def test_crops_to_bounding_box():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    mask = _mask_with_box(10, 10, 2, 5, 3, 6)
    cropped, box = crop_to_foreground(image, mask)
    assert box == (3, 2, 6, 5)
    assert cropped.shape == (3, 3)


def test_padding_expands_box_and_clips_to_bounds():
    image = np.zeros((10, 10), dtype=np.uint8)
    mask = _mask_with_box(10, 10, 4, 6, 4, 6)
    _, box = crop_to_foreground(image, mask, padding=2)
    assert box == (2, 2, 8, 8)


def test_padding_clips_at_image_edges():
    image = np.zeros((10, 10), dtype=np.uint8)
    mask = _mask_with_box(10, 10, 0, 2, 0, 2)
    _, box = crop_to_foreground(image, mask, padding=5)
    assert box == (0, 0, 7, 7)


def test_supports_multichannel_images():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = _mask_with_box(10, 10, 1, 4, 1, 4)
    cropped, _ = crop_to_foreground(image, mask)
    assert cropped.shape == (3, 3, 3)


def test_none_inputs_raise():
    with pytest.raises(ValueError):
        crop_to_foreground(None, None)


def test_non_2d_mask_raises():
    image = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop_to_foreground(image, np.zeros((5, 5, 1), dtype=np.uint8))


def test_shape_mismatch_raises():
    image = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop_to_foreground(image, np.zeros((6, 6), dtype=np.uint8))


def test_negative_padding_raises():
    image = np.zeros((5, 5), dtype=np.uint8)
    mask = _mask_with_box(5, 5, 1, 3, 1, 3)
    with pytest.raises(ValueError):
        crop_to_foreground(image, mask, padding=-1)


def test_empty_mask_raises():
    image = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop_to_foreground(image, np.zeros((5, 5), dtype=np.uint8))
