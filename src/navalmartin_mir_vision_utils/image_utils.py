"""module img_utils. Various image utilities

"""
import os
import shutil
from pathlib import Path
from io import BytesIO
from typing import Callable, List, Union, TypeVar
from PIL import Image

import numpy as np


from navalmartin_mir_vision_utils.exceptions import InvalidPILImageMode
from navalmartin_mir_vision_utils.image_enums import (ImageFileEnumType, ImageLoadersEnumType,
                                                      IMAGE_LOADERS_TYPES_STR, IMAGE_STR_TYPES, VALID_PIL_MODES_STR)
from navalmartin_mir_vision_utils.mir_vison_io.file_utils import ERROR


def is_valid_pil_image_from_bytes_string(image_byte_string: str) -> Image:
    """Check if the provided bytes correspond to a valid
    PIL.Image. If IOError or  SyntaxError is raised it returns None

    Parameters
    ----------
    image_byte_string: The string bytes that correspond to an image

    Returns
    -------

    An instance of a PIL.Image if the image is valid or None
    """
    try:
        image = Image.open(BytesIO(image_byte_string))
        image.verify()
        return image
    except (IOError, SyntaxError) as e:
        print(f"{ERROR} the image_byte_string is corrupted")
        print(f"Exception message {str(e)}")
        return None


def is_valid_pil_image_file(image: Path) -> Image:
    """Check if the given image is a valid Pillow image

    Parameters
    ----------
    image: The image filename

    Returns
    -------
    An instance of a PIL.Image if the image is valid or None
    """

    try:
        img = Image.open(image)
        img.verify()
        return img
    except (IOError, SyntaxError) as e:
        print(f"{ERROR} the file {image} is corrupted")
        return None


def get_pil_image_size(image: Image) -> tuple:
    """Returns the width, height of the given Pillow image

    Parameters
    ----------
    image: The PIL.Image

    Returns
    -------
    A tuple representing width, height
    """

    if image is None:
        raise ValueError("The provided image is None")

    return image.size




def list_image_files(base_path: Path,
                     valid_exts: Union[List , tuple] = IMAGE_STR_TYPES,
                     contains: str = None) -> Path:
    """Generator that returns all the images in the given
    base_path

    Parameters
    ----------
    base_path: Base path to look for image files
    valid_exts: Extensions to use
    contains: String that the filename should contain

    Returns
    -------

    """

    if isinstance(valid_exts, tuple):
        valid_exts = list(valid_exts)

    for i, item in enumerate(valid_exts):
        if isinstance(item, ImageFileEnumType):
            valid_exts[i] = f'.{item.name.lower()}'

    if not isinstance(valid_exts, tuple):
        valid_exts = tuple(valid_exts)

    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(base_path):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if valid_exts is None or ext.endswith(valid_exts):
                # construct the path to the image and yield it
                image_path = os.path.join(rootDir, filename)
                yield image_path


def get_img_files(base_path: Path,
                  img_formats: Union[List , tuple] = IMAGE_STR_TYPES) -> List[Path]:
    """Get the image files in the given image directory that have
    the specified image format.

    Parameters
    ----------
    img_dir: The image directory
    img_formats: The image formats

    Returns
    -------
    An instance of List[Path]
    """

    return list(list_image_files(base_path=base_path, valid_exts=img_formats))


def remove_metadata_from_image(image: Image, new_filename: Path) -> None:
    """Remove the metadata from the given image
    and saves it to the new location

    Parameters
    ----------
    image: The image to use
    new_filename: Where to store the new image

    Returns
    -------

    """

    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)
    image_without_exif.save(new_filename)


def delete_image_if(img_path: Path, size: tuple, direction: str) -> None:
    """Delete the image specified in the path that its file does not
    satisfy the direction. Direction can be any of the following:
    - >_both: Both coordinates are greater than
    - >_x: x coordinate is greater than
    - >_y: y coordinate is greater than
    - >_any: Any of the coordinates is less than
    - <_both: Both coordinates are less than
    - <_x: x coordinate is less than
    - <_y: y coordinate is less than
    - <_any: Any of the coordinates is less than


    Parameters
    ----------
    img_path
    size

    Returns
    -------

    """

    img = load_img(path=img_path, loader='PIL')

    img_size_width = img.width
    img_size_height = img.height

    if direction == ">_any" or direction == "<_any":

        if img_size_width < size[0] or \
                img_size_width > size[0] or \
                img_size_height < size[1] or \
                img_size_width > size[1]:
            del img
            os.remove(path=img_path)
    else:
        raise ValueError(f"Direction {direction} not implemented")


def copy_img_from_to(source: Path, dst: Path) -> None:
    """Copy the source image  to the dst image

    Parameters
    ----------
    source: The source image
    dst: The destination image

    Returns
    -------

    """
    shutil.copy(src=source, dst=dst)


def save_img(image: Image, filename: Path, img_format: str = None) -> None:
    """Save the given image on the given path

    Parameters
    ----------
    image: The image to save
    filename: The name of the image file
    img_format: The format of the image
    Returns
    -------

    """

    if format is None or format == "":
        image.save(filename)
    else:
        image.save(filename, format=img_format)


def show_pil_image(image: Image) -> None:
    """Show the image depending on the
    type

    Parameters
    ----------
    image
    img: The image to show

    Returns
    -------

    """

    image.show()


def save_img_from_str(img_str: str, encoding: str,
                      img_format: str, path: Path,
                      img_height: int, img_width: int,
                      mode: str = 'RGB') -> None:
    """

    Parameters
    ----------
    img_str: The decoded image string
    encoding: The encoding used to decode the image to a string
    img_format: The format of the image
    path: The path to store the image
    img_height: The height of the image
    img_width: The width of the image
    mode: The mode of an image is a string which defines the
    type and depth of a pixel in the image.
    Each pixel uses the full range of the bit depth.
    So a 1-bit pixel has a range of 0-1, an 8-bit pixel has a range of 0-255 and so on.
    The current release supports the following standard modes:
    For more information check https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes

    Returns
    -------

    None
    """

    # convert bytes data to PIL Image object

    if mode not in VALID_PIL_MODES_STR:
        raise InvalidPILImageMode(mode=mode)

    if encoding is None:
        img_bytes = str.encode(img_str)
    else:
        img_bytes = img_str.encode(encoding)

    img = Image.frombytes(mode, (img_width, img_height), img_bytes, 'raw')

    img.save(fp=path, format=img_format)
