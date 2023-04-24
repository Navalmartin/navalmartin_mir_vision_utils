import numpy as np
from PIL import Image
from pathlib import Path
from typing import Callable, List, Union, TypeVar
from io import BytesIO

from navalmartin_mir_vision_utils import get_img_files

from navalmartin_mir_vision_utils.mir_vision_config import WITH_TORCH, WITH_CV2
from navalmartin_mir_vision_utils.mir_vision_types import TorchTensor
from navalmartin_mir_vision_utils.exceptions import InvalidConfiguration
from navalmartin_mir_vision_utils.image_enums import (ImageLoadersEnumType, IMAGE_LOADERS_TYPES_STR, IMAGE_STR_TYPES)
from navalmartin_mir_vision_utils.utils.messages import ERROR

if WITH_TORCH:
    import torch
    import torchvision
    from torchvision import transforms


if WITH_CV2:
    import cv2


def load_img(path: Path, transformer: Callable = None,
             loader: ImageLoadersEnumType = ImageLoadersEnumType.PIL):
    """Load the image from the given path

    Parameters
    ----------
    path: The path to the image
    transformer: Callable object that applies transformations on the image
    loader: How to read the image currently only CV2 or PIL

    Returns
    -------

    If not transform is used it returns an PIL.Image object.
    Otherwise, it returns the datasets type supported by the
    transform
    """

    if loader.name.upper() not in IMAGE_LOADERS_TYPES_STR:
        raise ValueError(f"Invalid image loader={loader.name.upper()} not in {IMAGE_LOADERS_TYPES_STR}")

    if loader == ImageLoadersEnumType.PIL:
        return load_image_as_pillow(path=path, transformer=transformer)

    if loader == ImageLoadersEnumType.PIL_NUMPY:
        return load_image_as_numpy(path=path, transformer=transformer)

    if loader == ImageLoadersEnumType.CV2:
        if WITH_CV2:
            return load_image_cv2(path=path, transformer=transformer)
        else:
            raise InvalidConfiguration(message="opencv-python was not found. Cannot import cv2")

    if loader == ImageLoadersEnumType.PYTORCH_TENSOR:
        if WITH_TORCH:
            return load_image_pytorch_tensor(path=path, transformer=transformer)
        else:
            raise InvalidConfiguration(message="PyTorch was not found.")

    return None


def load_pil_image_from_byte_string(image_byte_string: bytes,
                                    open_if_verify_success: bool = True) -> Image:
    """Loads a PIL.Image from the given byte string

    Parameters
    ----------
    image_byte_string: The byte string representing the image
    open_if_verify_success: Whether to reopen the Image after image.verify()
    is called

    Returns
    -------

    An instance of PIL.Image
    """
    try:
        image = Image.open(BytesIO(image_byte_string))
        image.verify()

        # we need to reopen after verify
        # see this:
        # https://stackoverflow.com/questions/3385561/python-pil-load-throwing-attributeerror-nonetype-object-has-no-attribute-rea
        if open_if_verify_success:
            image = Image.open(BytesIO(image_byte_string))

        return image
    except (IOError, SyntaxError) as e:
        print(f"{ERROR} the image_byte_string is corrupted")
        print(f"Exception message {str(e)}")
        return None


def load_images(path: Path, transformer: Callable = None,
                loader: ImageLoadersEnumType = ImageLoadersEnumType.PIL,
                img_formats: tuple = IMAGE_STR_TYPES) -> List:
    """Loads all the images in the specified path

    Parameters
    ----------
    path: The path to load the images from
    transformer: how to transform the images
    loader: The type of the laoder either PIL or CV2
    img_formats: The image format

    Returns
    -------
    A list of images. The actual type depends on the type of the
    """

    # get all the image files
    img_files = get_img_files(base_path=path, img_formats=img_formats)

    if len(img_files) == 0:
        raise ValueError(f"{path} does  not have images with formats {img_formats}")

    images = []
    # load every image in the Path
    for img in img_files:
        images.append(load_img(path=img,
                               transformer=transformer,
                               loader=loader))

    return images


def load_images_from_paths(imgs: List[Path], transformer: Callable,
                           loader: ImageLoadersEnumType = ImageLoadersEnumType.PIL):
    if len(imgs) == 0:
        raise ValueError("Empty images paths")

    if loader not in IMAGE_LOADERS_TYPES_STR:
        raise ValueError(f"Invalid loader. Loader={loader} not in {IMAGE_LOADERS_TYPES_STR}")

    imgs_data = []
    for img in imgs:
        imgs_data.append(load_img(path=img, transformer=transformer, loader=loader))

    return imgs_data  


def load_image_as_pillow(path: Path, transformer: Callable = None) -> Image:
    """Load the image in the specified path as Pillow.Image object

    Parameters
    ----------
    path
    transformer

    Returns
    -------

    """
    x = Image.open(path)

    if transformer is not None:
        x = transformer(x)
    return x


def load_image_as_numpy(path: Path, transformer: Callable = None) -> np.array:
    x = Image.open(path)
    if transformer is not None:
        x = transformer(x)
    return np.array(x)


def load_image_cv2(path: Path, transformer: Callable = None):
    """Load an image as OpenCV matrix. If WITH_CV2 is False
    throws InvalidConfiguration

    Parameters
    ----------
    path: Path to the image
    transformer: Transformer to apply on loading

    Returns
    -------
    OpenCV image matrix
    """

    if not WITH_CV2:
        print("opencv-python is not installed.")
        raise InvalidConfiguration(message="opencv-python is not installed.")

    image = cv2.imread(str(path))

    if transformer is not None:
        image = transformer(image)

    return image


def load_image_pytorch_tensor(path: Path, transformer: Callable = None) -> TorchTensor:
    """Load the image from the specified path.  If WITH_TORCH is False
    throws InvalidConfiguration

    Parameters
    ----------
    path
    transformer

    Returns
    -------


    """

    if not WITH_TORCH:
        print("PyTorch is not installed.")
        raise InvalidConfiguration(message="PyTorch is not installed.")

    with Image.open(path) as image:

        if transformer is None:
                transform_to_torch = transforms.Compose([transforms.ToTensor()])
                return transform_to_torch(image)

        x = image
        x = transformer(x)

        if isinstance(x, torch.Tensor):
            return x

        transform_to_torch = transforms.Compose([transforms.ToTensor()])
        return transform_to_torch(x)
   

def load_images_as_torch(x: List[Path], y_train: List[int],
                         transformer: Callable) -> tuple:
    """Load the images in the path as torch tensors

    Parameters
    ----------
    x: A list of image files
    y_train: The lebels associated with evey image
    transformer: Transform to apply when loading the images.
    Usually this will be transforms.Compose

    Returns
    -------
    A tuple of torch.Tensors
    """

    if not WITH_TORCH:
        print("load_image_pytorch_tensor is not available as PyTorch was not found.")
        raise
        return None

    data = [load_img(img_path, transformer) for img_path in x]
    return torch.stack(data), torch.tensor(y_train, dtype=torch.uint8)