from PIL import Image
import numpy as np


def pil2ndarray(image: Image) -> np.ndarray:
    """Convert the Pillow image into a numpy array

    Parameters
    ----------
    image: The Pillow image to transform

    Returns
    -------

    """
    return np.asarray(image)
