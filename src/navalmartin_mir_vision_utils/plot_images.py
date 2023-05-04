import os
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
import textwrap
from PIL.Image import Image as PILImage
from navalmartin_mir_vision_utils.mir_vision_config import WITH_TORCH
from navalmartin_mir_vision_utils.mir_vision_types import TorchTensor
from navalmartin_mir_vision_utils.exceptions import InvalidConfiguration

if WITH_TORCH:
    import torchvision


def plot_pytorch_tensor_images(images: TorchTensor, title: str,
                               images_per_row: int) -> None:
    """Plot the images represented as PyTorch.Tensor tensors

    Parameters
    ----------
    images: The images to plot. 4D mini-batch Tensor of shape (B x C x H x W)
    title: The title of the plot
    images_per_row: Number of images in each row of the grid

    Returns
    -------
    None
    """

    if not WITH_TORCH:
        raise InvalidConfiguration(message="PyTorch is not installed so cannot use plot_pytorch_tensor_images")

    plt.figure()
    plt.title(title)
    plt.imshow(torchvision.utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0))


def plot_pytorch_image(image: TorchTensor,
                       title: str = None,
                       title_font_size: int = 8,
                       title_wrap_length: int = 50) -> None:
    """Plot the given Pillow image

    Parameters
    ----------
    image: The image to plot
    title: The title for the plot
    title_font_size: The font size for the title
    title_wrap_length: The wrap length for the title

    Returns
    -------

    """

    if not WITH_TORCH:
        raise InvalidConfiguration(message="PyTorch is not installed so cannot use plot_pytorch_image")

    if image is None:
        raise ValueError("The provided image is None")

    plt.imshow(image.permute(1, 2, 0))

    if title is None:
        pass
    else:
        title = textwrap.wrap(title, title_wrap_length)
        title = "\n".join(title)
        plt.title(title, fontsize=title_font_size)

    plt.show()


def plot_pil_image(image: PILImage,
                   title: str = None,
                   title_font_size: int = 8,
                   label_wrap_length: int = 50) -> None:
    """Show the image depending on the
    type

    Parameters
    ----------

    label_wrap_length: The length before wrapping any text
    title_font_size: font size for the title
    title: The title for the plot
    image: The image to show

    Returns
    -------

    """

    if image is None:
        raise ValueError("The provided image is None")

    plt.imshow(image)

    if title is None:
        pass
    elif title == "":
        if hasattr(image, 'filename'):
            title = image.filename
            title = textwrap.wrap(title, label_wrap_length)
            title = "\n".join(title)

        plt.title(title, fontsize=title_font_size)
        plt.show()
    else:
        title = textwrap.wrap(title, label_wrap_length)
        title = "\n".join(title)
        plt.title(title, fontsize=title_font_size)

    plt.show()


def plot_pil_images(images: List[PILImage],
                    columns: int = 5,
                    width: int = 20, height: int = 8,
                    max_images: int = 15,
                    title_wrap_length: int = 50,
                    title_font_size: int = 8):
    """Plots a grid of images from the given list of PIL images

    Parameters
    ----------
    images: The list of images to plot
    columns: How many columns the grid shall have
    width
    height
    max_images: Maximum number of images to plot from the given list
    title_wrap_length: The length to wrap the title
    title_font_size: The title font size

    Returns
    -------

    """
    if not images:
        print("No images to display.")
        return

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images = images[0:max_images]

    height = max(height, int(len(images) / columns) * height)
    plt.figure(figsize=(width, height))

    for i, image in enumerate(images):

        image = image
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image)

        if hasattr(image, 'filename'):
            title = image.filename
            if title.endswith("/"): title = title[0:-1]
            title = os.path.basename(title)
            title = textwrap.wrap(title, title_wrap_length)
            title = "\n".join(title)
            plt.title(title, fontsize=title_font_size)
    plt.show()


def plot_pil_images_with_label(images: List[Tuple[PILImage, Union[int, str]]],
                               columns: int = 5, width: int = 20,
                               height: int = 8, max_images: int = 15,
                               label_wrap_length: int = 50,
                               label_font_size: int = 8):
    if not images:
        print("No images to display.")
        return

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images = images[0:max_images]

    height = max(height, int(len(images) / columns) * height)
    plt.figure(figsize=(width, height))

    for i, image in enumerate(images):

        pil_image = image[0]
        image_label = image[1]
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(pil_image)

        if hasattr(pil_image, 'filename'):
            title = pil_image.filename
            if title.endswith("/"): title = title[0:-1]
            title = os.path.basename(title)
            title += f", label={image_label}"
            title = textwrap.wrap(title, label_wrap_length)
            title = "\n".join(title)
            plt.title(title, fontsize=label_font_size)
    plt.show()
