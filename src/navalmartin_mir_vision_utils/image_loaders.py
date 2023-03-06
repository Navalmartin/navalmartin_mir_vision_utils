        
from PIL import Image
from pathlib import Path
from typing import Callable, List, Union, TypeVar

from navalmartin_mir_vision_utils import WITH_TORCH, TorchTensor
from navalmartin_mir_vision_utils.exceptions import InvalidPILImageMode
from navalmartin_mir_vision_utils.image_enums import (ImageFileEnumType, ImageLoadersEnumType,
                                                      IMAGE_LOADERS_TYPES_STR, IMAGE_STR_TYPES, VALID_PIL_MODES_STR)
from navalmartin_mir_vision_utils.mir_vison_io.file_utils import ERROR

if WITH_TORCH:
    import torch
    import torchvision
    from torchvision import transforms



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
    elif loader == ImageLoadersEnumType.CV2:
        return load_image_cv2(path=path, transformer=transformer)
    elif loader == ImageLoadersEnumType.PIL_NUMPY:
        return load_image_as_numpy(path=path, transformer=transformer)
    elif loader == ImageLoadersEnumType.PYTORCH_TENSOR:
        return load_image_pytorch_tensor(path=path, transformer=transformer)
    
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
    img_files = get_img_files(img_dir=path, img_formats=img_formats)

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
    """Load an image as OpenCV matrix

    Parameters
    ----------
    path: Path to the image
    transformer: Transformer to apply on loading

    Returns
    -------
    OpenCV image matrix
    """

    try:
        import cv2
        image = cv2.imread(str(path))

        if transformer is not None:
            image = transformer(image)
    except ModuleNotFoundError as e:
        print(f"An exception was raised in load_image_cv2. Message {str(e)}")
        print(f"navalmartin-mir-vision-utils has not been set up with OpenCV suppert")
        return None

    return image


def load_image_pytorch_tensor(path: Path, transformer: Callable = None) -> TorchTensor:
    """Load the image from the specified path

    Parameters
    ----------
    path
    transformer

    Returns
    -------


    """

    if not WITH_TORCH:
        print("PyTorch is not installed.")
        return None

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
                         transformer: transforms.Compose) -> tuple:
    """Load the images in the path as torch tensors

    Parameters
    ----------
    x: A list of image files
    y_train: The lebels associated with evey image
    transformer: Transform to apply when loading the images

    Returns
    -------
    A tuple of torch.Tensors
    """

    if not WITH_TORCH:
        print("PyTorch is not installed.")
        return None

    data = [load_img(img_path, transformer) for img_path in x]
    return torch.stack(data), torch.tensor(y_train, dtype=torch.uint8)