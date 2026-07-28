import numpy as np


def crop_to_foreground(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Crop an image to the bounding box of a foreground mask.

    Args:
        image: Input image (H, W) or (H, W, C)
        mask: Binary mask (H, W)
        padding: Extra pixels around the object

    Returns:
        Cropped image and bounding box (x_min, y_min, x_max, y_max)
    """

    if image is None or mask is None:
        raise ValueError("Image and mask must not be None")

    if mask.ndim != 2:
        raise ValueError("Mask must be 2D")

    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have same height and width")

    if padding < 0:
        raise ValueError("Padding must be non-negative")

    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        raise ValueError("Mask has no foreground")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding + 1, mask.shape[1])
    y_max = min(y_max + padding + 1, mask.shape[0])

    cropped = image[y_min:y_max, x_min:x_max]

    return cropped, (x_min, y_min, x_max, y_max)
