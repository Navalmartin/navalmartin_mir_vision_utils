"""
Various utilities for working with images in the _mir_ project.
"""
__version__ = "0.0.7"
from typing import TypeVar
from .image_utils import (load_img,
                          ImageLoadersEnumType,
                          is_valid_pil_image_from_bytes_string,
                          is_valid_pil_image_file,
                          show_pil_image,
                          get_img_files, get_pil_image_size)
from .image_transformers import pil2ndarray, pil_image_to_bytes_string
from .image_enums import (ImageFileEnumType,
                          IMAGE_STR_TYPES,
                          ImageLoadersEnumType,
                          IMAGE_LOADERS_TYPES_STR,
                          ValidPillowEnumType,
                          VALID_PIL_MODES_STR)

TorchTensor = TypeVar("TorchTensor")

WITH_TORCH = False

try:
    import torch
    import torchvision
    from torchvision import transforms
    WITH_TORCH = True   
except ModuleNotFoundError as e:
        print(f"An exception was raised whilst importing torch, torchvision, torchvision.transforms. Message {str(e)}")
        pass
