# cv-utils

![CI](https://github.com/slavagaidukjb-art/cv-utils/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Small, dependency-light computer-vision utilities for dataset preparation and
image preprocessing — clean, typed, and tested.

## Install

```bash
git clone https://github.com/slavagaidukjb-art/cv-utils.git
cd cv-utils
pip install -r requirements-dev.txt
```

The library itself only needs **NumPy**.

## Usage

### `crop_to_foreground`

Crop an image to the bounding box of a binary foreground mask, with optional
padding that is automatically clipped to the image bounds.

```python
import numpy as np
from image_preprocessing import crop_to_foreground

image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
mask = np.zeros((480, 640), dtype=np.uint8)
mask[100:300, 200:500] = 1  # foreground region

cropped, (x_min, y_min, x_max, y_max) = crop_to_foreground(image, mask, padding=10)
print(cropped.shape)          # (220, 320, 3)
print(x_min, y_min, x_max, y_max)
```

**Signature**

```
crop_to_foreground(image, mask, padding=0) -> (cropped_image, (x_min, y_min, x_max, y_max))
```

| Argument  | Type        | Description                                  |
|-----------|-------------|----------------------------------------------|
| `image`   | `np.ndarray`| `(H, W)` or `(H, W, C)` image                |
| `mask`    | `np.ndarray`| `(H, W)` binary mask (non-zero = foreground) |
| `padding` | `int`       | Extra pixels around the object (clipped)     |

Raises `ValueError` on `None` inputs, non-2D masks, shape mismatch, negative
padding, or an empty mask.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .        # lint
pytest -q           # run tests
```

CI runs linting and the test suite on every push and pull request across
Python 3.10–3.12.

## License

MIT — see [LICENSE](LICENSE).
