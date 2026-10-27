"""Data conversion helpers."""


import numpy as np


def image_to_matrix(image):
    """Convert an image to the 2D grayscale matrix used by the PCA engine."""
    return np.asarray(image.convert("L"), dtype=np.float64)
