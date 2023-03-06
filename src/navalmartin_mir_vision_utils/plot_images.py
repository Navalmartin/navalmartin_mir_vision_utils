import matplotlib.pyplot as plt
import torch
import torchvision

def plot_pytorch_tensor_images(images: torch.Tensor, title: str,
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

    plt.figure()
    plt.title(title)
    plt.imshow(torchvision.utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0))
