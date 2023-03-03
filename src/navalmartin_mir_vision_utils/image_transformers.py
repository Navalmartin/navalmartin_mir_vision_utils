from PIL import Image
import numpy as np
import io


def pil2ndarray(image: Image) -> np.ndarray:
    """Convert the Pillow image into a numpy array

    Parameters
    ----------
    image: The Pillow image to transform

    Returns
    -------

    """
    return np.asarray(image)


def pil_image_to_bytes_string(image: Image) -> bytes:
    """Returns the byte string of the provided image.
    Code adapted from https://stackoverflow.com/questions/33101935/convert-pil-image-to-byte-array
    Parameters
    ----------
    image: The PIL.Image

    Returns
    -------
    A string that represents the image bytes
    """

    img_byte_arr = io.BytesIO()
    # image.save expects a file-like as a argument
    image.save(img_byte_arr, format=image.format)
    # Turn the BytesIO object back into a bytes object
    imgByteArr = img_byte_arr.getvalue()
    return imgByteArr
