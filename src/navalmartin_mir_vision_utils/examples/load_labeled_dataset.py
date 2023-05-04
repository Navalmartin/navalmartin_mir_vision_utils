from pathlib import Path
import pprint
from torchvision import transforms
from navalmartin_mir_vision_utils.mir_vision_io import LabeledImageDataset
from navalmartin_mir_vision_utils import plot_pil_images_with_label
from navalmartin_mir_vision_utils import pil_to_rgb

if __name__ == '__main__':

    base_path = Path("/home/alex/qi3/mir_datasets/vessels_currated")
    dataset = LabeledImageDataset(labels=[("catamaran", 0),
                                          ("rib", 1),
                                          ("single_hull", 2),
                                          ("trimaran", 3)],
                                  base_path=base_path)

    pprint.pprint(f"Number of images {len(dataset)}")

    pprint.pprint(f"Number of images per class {dataset.n_images_per_label}")

    # what type of images to we have
    pprint.pprint(f"Image formats {dataset.image_formats}")

    pprint.pprint(f"Image labels {dataset.labels}")

    # shuffle the images
    dataset.shuffle()

    images = dataset.images[0:10]

    # let's see the first 10 images and their labels
    plot_pil_images_with_label(images=images)

    # if an image is png it has 4 channels so won't
    # be possible to stack the tenosrs. Thus we need
    # to convert to rgb
    transformer = transforms.Compose([pil_to_rgb,
                                      transforms.Resize((256, 256))])
    # get PyTorch tensor with images and labels
    tensor_images, labels = LabeledImageDataset.as_pytorch_tensor(dataset, transformer)

    pprint.pprint(f"Tensor size={tensor_images.size()}")