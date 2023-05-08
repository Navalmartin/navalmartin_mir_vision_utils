"""module labeled_image_dataset. Simple
wrapper to load a labeled image dataset

"""
import random
from pathlib import Path
from typing import List, Any, Callable, Tuple
import os

from navalmartin_mir_vision_utils.image_utils import get_img_files
from navalmartin_mir_vision_utils.image_loaders import load_img
from navalmartin_mir_vision_utils.image_enums import (ImageLoadersEnumType, IMAGE_STR_TYPES)
from navalmartin_mir_vision_utils.mir_vision_config import WITH_TORCH
from navalmartin_mir_vision_utils.exceptions import InvalidConfiguration
from navalmartin_mir_vision_utils.mir_vision_types import TorchTensor
from navalmartin_mir_vision_utils.image_transformers import pil_to_torch_tensor

if WITH_TORCH:
    import torch


class LabeledImageDataset(object):
    """Simple class to load from a specified
    directory images that are organised into
    subdirectories that represent the labels

    """

    @staticmethod
    def as_pytorch_tensor(dataset: "LabeledImageDataset",
                          transformer: Callable = None) -> Tuple[TorchTensor, List[int]]:
        """Return

        Parameters
        ----------
        transformer
        dataset

        Returns
        -------

        """
        if not WITH_TORCH:
            raise InvalidConfiguration(message="PyTorch is not installed so cannot use pil_to_torch_tensor")

        labels = []
        data: List[TorchTensor] = []

        if dataset.loader_type == ImageLoadersEnumType.PIL:
            for image in dataset:
                img = image[0]
                label = image[1]

                if transformer is not None:
                    img = pil_to_torch_tensor(image=img)
                    img = transformer(img)
                    data.append(img)
                else:
                    data.append(pil_to_torch_tensor(image=img))
                labels.append(label)
            return torch.stack(data), labels
        elif dataset.loader_type == ImageLoadersEnumType.PYTORCH_TENSOR:
            for image in dataset:
                img = image[0]
                label = image[1]

                if transformer is not None:
                    img = transformer(img)

                data.append(img)
                labels.append(label)
            return torch.stack(data), labels
        else:
            raise ValueError(f"Invalid loader type {dataset.loader_type} "
                             f"not in [ImageLoadersEnumType.PIL, ImageLoadersEnumType.PYTORCH_TENSOR]")

    def __init__(self, labels: List[tuple], base_path: Path,
                 do_load: bool = True, *,
                 image_formats: List[Any] = IMAGE_STR_TYPES,
                 loader: ImageLoadersEnumType = ImageLoadersEnumType.PIL,
                 transformer: Callable = None):

        self.labels: List[tuple] = labels
        self.base_path = base_path
        self.image_formats = image_formats
        self.images: List[tuple] = []
        self.loader_type = loader
        self._images_per_label = {}
        self._current_pos: int = -1

        if do_load:
            self.load(loader=loader, transformer=transformer)

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self):
        self._current_pos = 0
        return self

    def __next__(self) -> tuple:
        if len(self.images) == 0:
            raise StopIteration

        if self._current_pos < len(self.images):
            result = self.images[self._current_pos]
            self._current_pos += 1
            return result
        else:
            self._current_pos = -1
            raise StopIteration

    def __getitem__(self, key: int) -> tuple:
        """Returns the image, label pair  that corresponds to the given key
        Parameters
        ----------
        key: The index of the image-label to retrieve
        Returns
        -------
        A tuple of the image-label
        """

        return self.images[key]

    def __del__(self) -> None:
        self.clean()

    @property
    def n_images_per_label(self) -> dict:
        """Get a dictionary with the number of images
        per label

        Returns
        -------

        """
        return self._images_per_label

    def clean(self) -> None:
        """Invalidate the dataset

        Returns
        -------

        """
        self.labels = []
        self.base_path = None
        self.image_formats = []
        self.images = []
        self.loader_type = None
        self._images_per_label = {}
        self._current_pos: int = -1

    def shuffle(self) -> None:
        """Randomly shuffle  the contents of the dataset

        Returns
        -------

        """
        random.shuffle(self.images)

    def load(self, loader: ImageLoadersEnumType = ImageLoadersEnumType.PIL,
             transformer: Callable = None) -> None:

        self.loader_type = loader
        tmp_img_formats = []

        for label in self.labels:

            label_name = label[0]
            label_idx = label[1]
            base_path = Path(str(self.base_path)) / label_name

            # get all the image files
            img_files: List[Path] = get_img_files(base_path=base_path,
                                                  img_formats=self.image_formats)

            label_images = []
            # load every image in the Path
            for img in img_files:

                suffix = img.suffix

                if suffix.lower() not in tmp_img_formats:
                    tmp_img_formats.append(suffix.lower())

                label_images.append((load_img(path=img,
                                              transformer=transformer,
                                              loader=loader), label_idx))

            self._images_per_label[label] = len(label_images)
            self.images.extend(label_images)

            self.image_formats = tmp_img_formats



