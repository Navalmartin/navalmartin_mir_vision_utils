"""module img_utils. Various image utilities

"""
import os
from io import BytesIO
from pathlib import Path
from typing import List, Union

from PIL import Image

from navalmartin_mir_vision_utils.exceptions import InvalidPILImageMode
from navalmartin_mir_vision_utils.image_enums import (ImageFileEnumType, IMAGE_STR_TYPES, VALID_PIL_MODES_STR)
from navalmartin_mir_vision_utils.utils.messages import ERROR
from navalmartin_mir_vision_utils.mir_vision_config import WITH_CV2


def is_valid_pil_image_from_bytes_string(image_byte_string: bytes,
                                         open_if_verify_success: bool = True) -> Image:
    """Check if the provided bytes correspond to a valid
    PIL.Image. If IOError or  SyntaxError is raised it returns None

    Parameters
    ----------
    open_if_verify_success
    image_byte_string: The string bytes that correspond to an image

    Returns
    -------

    An instance of a PIL.Image if the image is valid or None
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


def is_valid_pil_image_file(image: Path, open_if_verify_success: bool = True) -> Image:
    """Check if the given image is a valid Pillow image

    Parameters
    ----------
    open_if_verify_success
    image: The image filename

    Returns
    -------
    An instance of a PIL.Image if the image is valid or None
    """

    try:
        img = Image.open(image)
        img.verify()
        if open_if_verify_success:
            img = Image.open(image)
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
                     valid_exts: Union[List, tuple] = IMAGE_STR_TYPES,
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
                yield Path(image_path)


def get_img_files(base_path: Path,
                  img_formats: Union[List, tuple] = IMAGE_STR_TYPES) -> List[Path]:
    """Get the image files in the given image directory that have
    the specified image format.

    Parameters
    ----------
    base_path: The image directory
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


def create_thumbnail_from_pil_image(max_size: tuple,
                                    image_filename: Path = None,
                                    image: Image = None) -> Image:
    """Create a thumbnail from the given image filename or the
    given image. If both are None it raises ValueError.
    If image_filename is not None it opens the image
    specified from the file and verifies its integrity.
    If image is not None it creates a thumbnail inplace

    Parameters
    ----------
    max_size: The max siz of the thumbnail
    image_filename: Image filename to read the image
    image: An Image instance

    Returns
    -------
    An instance of Image class
    """
    if image_filename is not None:
        image_thub = is_valid_pil_image_file(image=image_filename,
                                             open_if_verify_success=True)

        image_thub.thumbnail(max_size)
        return image_thub
    elif image is not None:
        image.thumbnail(max_size)
        return image

    raise ValueError("Both image_filename and image are None")


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

    if image is None:
        raise ValueError("The provided image is None")

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


def chuckify_img(img: Path, chunk_size: tuple, output_dir: Path,
                 create_output_dir: bool = False):
    """Creates chuncks of the image in the given Path
    and stores it in the given output directory
    under the same name appended with the chuck counter

    Parameters
    ----------

    img: The image path
    chunk_size: The size of the chunck
    output_dir: The output directory
    create_output_dir: Flag indicating if the output directory should be created
    Returns
    -------

    """

    if not WITH_CV2:
        raise NotImplementedError("The function chuckify_img requires OpenCV supprt. "
                                  "But OpenCV is not detected")

    import cv2

    height = chunk_size[1]
    width = chunk_size[0]

    im = cv2.imread(str(img))
    imgheight, imgwidth, channels = im.shape

    file, ext = os.path.splitext(str(img))
    file = file.split('/')[-1]

    if os.listdir(output_dir/file):
        pass
    elif not os.listdir(output_dir/file) and create_output_dir == False:
        raise ValueError("Specified output directory does not exist and cannot create it")
    elif not os.listdir(output_dir/file) and create_output_dir:
        os.mkdir(output_dir/file)

    save_at = output_dir / file
    counter = 0
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            a = im[i:i + height, j:j + width]
            filename = file + '_' + str(counter) + f'{ext}'
            filename = save_at / filename
            cv2.imwrite(str(filename), a)
            counter += 1
