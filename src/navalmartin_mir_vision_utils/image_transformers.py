from PIL import Image
import numpy as np
import io

from PIL import ImageOps

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


def pil_to_rgb(image: Image) -> Image:
    """Convert the PIL.Image to RGB. This function
    can be used to convert a PNG image to JPG/JPEG
    formats. Note that this function simply returns the
    converted image. It does not save the newly formatted image

    Parameters
    ----------
    image: The Image to convert

    Returns
    -------

    An Instance of PIL.Image
    """

    # don't convert anything if the
    # image is in the right mode
    if image.mode == 'RGB':
        return image

    return image.convert("RGB")


def pil_to_grayscale(img: Image) -> Image:
    """Converts the given image to greyscale

    Parameters
    ----------
    img: The image to convert to grayscale

    Returns
    -------

    A grey-scaled image
    """
    # makes it greyscale
    return ImageOps.grayscale(img)
