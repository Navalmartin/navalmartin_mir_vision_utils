"""module statistics_utils Various utilities
used around the library. The first implementation is
taken from: https://github.com/ocampor/image-quality

"""
import numpy as np


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    """Normalize the given kernel

    Parameters
    ----------
    kernel

    Returns
    -------

    """
    return kernel / np.sum(kernel)


def gaussian_kernel2d(kernel_size, sigma: float):
    y, x = np.indices((kernel_size, kernel_size)) - int(kernel_size / 2)
    kernel = (
        1
        / (2 * np.pi * sigma ** 2)
        * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    )
    return normalize_kernel(kernel)
