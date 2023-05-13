# Overfiting is a common problem when training machine learning models.
# Regularization techniques can be used in order to mitigate this problem.
# Another approach is to add more data. However, this is not always possible.
# One simple and inexpensive way to obtain more data is by generating new instances of the same
# data with some transformations. We call this data augmentation.

import torch
import torchvision
from pathlib import Path
import random

# ignore beta version warnings
torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as transforms
from navalmartin_mir_vision_utils.mir_vision_io import LabeledImageDataset

if __name__ == '__main__':

    # set the seed for torch engine
    torch.manual_seed(42)
    random.seed(42)

    proportion = 0.35

    # use your own path
    base_path = Path("/home/alex/qi3/mir_datasets/vessel_classification")
    classes = ["single_hull", "catamaran", "rib", "trimaran"]

    for class_name in classes:
        augmented_base_path = Path(f"/home/alex/qi3/mir_datasets/vessel_classification/augmented/{class_name}")

        dataset = LabeledImageDataset(unique_labels=[(class_name, 0)],
                                      base_path=base_path,
                                      transformer=None)

        print(f"Number of images={len(dataset)}")

        dataset.shuffle()

        n_augment_images = int(proportion * len(dataset))
        print(f"Number of images to augment {n_augment_images}")

        image_indices = dataset.random_selection(size=n_augment_images)

        # create a transformation object
        transformer = transform = transforms.Compose([transforms.ColorJitter(contrast=0.5),
                                                      transforms.RandomRotation(30)])

        for img_idx in image_indices:
            image = dataset[img_idx]

            full_image_filename = image[0].filename
            print(f"Transforming image {full_image_filename}")
            image_filename = full_image_filename.split("/")[-1]

            augmented_image_filename = "augmented_" + image_filename
            img_transformed = transformer(image[0])

            img_transformed.save(augmented_base_path / augmented_image_filename)
