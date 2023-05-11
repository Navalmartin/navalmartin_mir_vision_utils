from pathlib import Path
import pprint
import numpy as np
from torchvision import transforms

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError as e:
    print("This example requires sklearn support")

from navalmartin_mir_vision_utils.mir_vision_io import LabeledImageDataset
from navalmartin_mir_vision_utils import plot_pil_images_with_label, ImageLoadersEnumType
from navalmartin_mir_vision_utils import pil_to_rgb

if __name__ == '__main__':

    # you will have to use your own path
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

    dataset.clear(full_clear=False)
    dataset.load(loader_type=ImageLoadersEnumType.FILEPATH,
                 transformer=None, force_load=False)

    # we want to split the dataset into train-validation-test sets
    x_train, x_test, x_train_labels, x_test_labels = train_test_split(dataset.images,
                                                                      dataset.image_labels,
                                                                      test_size=0.25,
                                                                      random_state=42,
                                                                      shuffle=True)

    unique_test_labels_idxs = np.unique(x_test_labels)
    unique_test_labels = [(dataset.get_label_name(index), index) for index in unique_test_labels_idxs]

    # create the test set
    test_set = LabeledImageDataset.build_from_list(images=x_test,
                                                   unique_labels=unique_test_labels,
                                                   image_labels=x_test_labels,
                                                   loader_type=ImageLoadersEnumType.FILEPATH,
                                                   transformer=None)

    pprint.pprint(f"Test set number of images {len(test_set)}")
    pprint.pprint(f"Number of images per class {test_set.n_images_per_label}")
