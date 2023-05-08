from PIL.Image import Image as PILImage
from PIL import Image
from PIL import ImageOps
from typing import List
import numpy as np
import io

from navalmartin_mir_vision_utils.mir_vision_config import WITH_TORCH
from navalmartin_mir_vision_utils.mir_vision_types import TorchTensor
from navalmartin_mir_vision_utils.exceptions import InvalidConfiguration

if WITH_TORCH:
    import torch
    from torchvision import transforms


def pil_to_ndarray(image: PILImage) -> np.ndarray:
    """Convert the Pillow image into a numpy array

    Parameters
    ----------
    image: The Pillow image to transform

    Returns
    -------

    """
    return np.asarray(image)


def pil_to_torch_tensor(image: PILImage, unsqueeze_dim: int = 0) -> TorchTensor:
    """Convert the given Pillow image to a PyTorch tensor.
    Raises  InvalidConfiguration if PyTorch is not installed

    Parameters
    ----------
    unsqueeze_dim: The dimension to unsqueeze the produced torch tensor
    image: The Pillow image

    Returns
    -------

    an instance of torch.Tensor
    """
    if not WITH_TORCH:
        raise InvalidConfiguration(message="PyTorch is not installed so cannot use pil_to_torch_tensor")

    if unsqueeze_dim is None or unsqueeze_dim == -1:
        return transforms.ToTensor()(image)
    else:
        return transforms.ToTensor()(image).unsqueeze_(unsqueeze_dim)

    #return transforms.PILToTensor()(image)


def pils_to_torch_tensor(images: List[PILImage], unsqueeze_dim: int = 0) -> TorchTensor:
    """Convert the given Pillow image to a PyTorch tensor.
    Raises  InvalidConfiguration if PyTorch is not installed

    Parameters
    ----------
    unsqueeze_dim: The dimension to unsqueeze the produced torch tensor
    images: The list of Pillow images

    Returns
    -------

    an instance of torch.Tensor
    """
    if not WITH_TORCH:
        raise InvalidConfiguration(message="PyTorch is not installed so cannot use pil_to_torch_tensor")

    tensors = []
    for img in images:
        tensors.append(pil_to_torch_tensor(img, unsqueeze_dim))
    return torch.stack(tensors)


def pil_image_to_bytes_string(image: PILImage) -> bytes:
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


def pil_to_rgb(image: PILImage) -> PILImage:
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


def pil_to_grayscale(img: PILImage) -> PILImage:
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


def pil_image_from_array(img: List) -> PILImage:
    """Build a Pillow image from the given
    list values

    Parameters
    ----------
    img

    Returns
    -------

    An instance of Image
    """
    return Image.fromarray(np.uint8(img))
