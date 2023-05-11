from pathlib import Path
import pprint
from torchvision import transforms
from navalmartin_mir_vision_utils.mir_vision_io import LabeledImageDataset
from navalmartin_mir_vision_utils import plot_pil_images_with_label, ImageLoadersEnumType
from navalmartin_mir_vision_utils import pil_to_rgb

if __name__ == '__main__':
    base_path = Path("/home/alex/qi3/mir_datasets/vessel_classification")
    dataset = LabeledImageDataset(unique_labels=[("catamaran", 0),
                                                 ("rib", 1),
                                                 ("single_hull", 2),
                                                 ("trimaran", 3)],
                                  base_path=base_path,
                                  transformer=None)

    pprint.pprint(f"Number of images {len(dataset)}")

    pprint.pprint(f"Number of images per class {dataset.n_images_per_label}")

    # what type of images to we have
    pprint.pprint(f"Image formats {dataset.image_formats}")
    pprint.pprint(f"Image labels {dataset.unique_labels}")

    assert len(dataset) == len(dataset.image_labels), "Number of images not equal to the number of labels"

    # shuffle the images
    dataset.shuffle()

    images = dataset.images[0:10]

    # let's see the first 10 images and their labels
    plot_pil_images_with_label(images=images)

    # some images are of PNG format i.e. they
    # have 4 channels instead of 3. torch.stack
    # only accepts images of the same form so we need
    # to convert images from PNG to RGB
    dataset.apply_transform(transformer=pil_to_rgb)

    # if an image is png it has 4 channels so won't
    # be possible to stack the tenosrs. Thus we need
    # to convert to rgb
    transformer = transforms.Compose([transforms.Resize((256, 256))])

    # get PyTorch tensor with images and labels
    tensor_images, labels = LabeledImageDataset.as_pytorch_tensor(dataset, transformer)

    pprint.pprint(f"Tensor size={tensor_images.size()}")

    # the following will fail as the dataset already
    # contains data
    try:
        dataset.load(loader_type=ImageLoadersEnumType.FILEPATH,
                     transformer=None, force_load=False)
    except ValueError as e:
        print(str(e))

    # let's clear the dataset and reload with a new loader
    dataset.load(loader_type=ImageLoadersEnumType.FILEPATH,
                 transformer=None, force_load=True)

    # this should be the same as above
    pprint.pprint(f"Number of images {len(dataset)}")
    pprint.pprint(f"Number of images per class {dataset.n_images_per_label}")

    # get PyTorch tensor with images and labels
    transformer = transforms.Compose([pil_to_rgb, transforms.Resize((256, 256))])
    tensor_images, labels = LabeledImageDataset.as_pytorch_tensor(dataset, transformer)
    pprint.pprint(f"Tensor size={tensor_images.size()}")
