"""module labeled_image_dataset. Simple
wrapper to load a labeled image dataset

"""
import random
from pathlib import Path
from PIL.Image import Image as PILImage
from typing import List, Any, Callable, Tuple, Union, Dict
import os

from navalmartin_mir_vision_utils.image_utils import get_img_files
from navalmartin_mir_vision_utils.image_loaders import load_img
from navalmartin_mir_vision_utils.image_enums import (ImageLoadersEnumType, IMAGE_STR_TYPES)
from navalmartin_mir_vision_utils.mir_vision_config import WITH_TORCH, DUMMY_PATH
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
        """Returns a tuple with the images in the dataset as
        PyTorch tensors. If the images were loaded using ImageLoadersEnumType.PIL
        then the image is first converted into a PyTorch tensor.
        Acceptable loaders for the given dataset are ImageLoadersEnumType.PIL or
        ImageLoadersEnumType.PYTORCH_TENSOR
        If WITH_TORCH == False, it raises InvalidConfiguration.
        If the dataset loader is not of the right type it raises ValueError

        Parameters
        ----------
        transformer: A list of operations to apply on the returned tensors
        dataset: The dataset to work on

        Returns
        -------

        A tuple of PyTorch tensor, List[int]
        """
        if not WITH_TORCH:
            raise InvalidConfiguration(message="PyTorch is not installed so cannot use pil_to_torch_tensor")

        labels = []
        data: List[TorchTensor] = []

        if dataset.loader_type == ImageLoadersEnumType.PIL:
            for image in dataset:
                img = image[0]
                label = image[1]

                if isinstance(img, torch.Tensor):
                    raise ValueError(f"Dataset loader type is {ImageLoadersEnumType.PIL.name} "
                                     f"but image point is of type torch.Tensor. "
                                     f"Have you applied any transformation to the images?")

                # first convert to pytorch tensor and
                # then apply the transformations
                img = pil_to_torch_tensor(image=img, unsqueeze_dim=None)
                if transformer is not None:
                    img = transformer(img)
                data.append(img)
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
        elif dataset.loader_type == ImageLoadersEnumType.FILEPATH:
            for image in dataset:
                img = image[0]
                label = image[1]

                pytorch_image = load_img(path=img, transformer=transformer,
                                         loader=ImageLoadersEnumType.PYTORCH_TENSOR)

                data.append(pytorch_image)
                labels.append(label)
            return torch.stack(data), labels
        else:
            raise ValueError(f"Invalid loader type {dataset.loader_type} "
                             f"not in [ImageLoadersEnumType.PIL, ImageLoadersEnumType.PYTORCH_TENSOR]")

    @classmethod
    def build_from_list(cls, images: List[Tuple[Union[Path, PILImage, TorchTensor], Union[int, str]]],
                        unique_labels: List[tuple],
                        image_labels: List[int],
                        loader_type: ImageLoadersEnumType,
                        image_formats: List[str] = IMAGE_STR_TYPES,
                        transformer: Callable = None) -> "LabeledImageDataset":

        dataset = LabeledImageDataset(unique_labels=unique_labels,
                                      base_path=DUMMY_PATH,
                                      do_load=False,
                                      transformer=transformer,
                                      loader_type=loader_type,
                                      image_formats=image_formats)

        dataset.images = images
        dataset.image_labels = image_labels
        images_per_label = {}

        for img_label in image_labels:
            for item in unique_labels:
                if img_label == item[1]:
                    if item[0] in images_per_label:
                        images_per_label[item[0]] += 1
                    else:
                        images_per_label[item[0]] = 1

        dataset._images_per_label = images_per_label
        return dataset

    def __init__(self, unique_labels: List[tuple], base_path: Path,
                 do_load: bool = True, *,
                 image_formats: List[str] = IMAGE_STR_TYPES,
                 loader_type: ImageLoadersEnumType = ImageLoadersEnumType.PIL,
                 transformer: Callable = None):
        """Constructor. Initialize the dataset by providing
        the unique labels for the images in a form of [("label_name", idx)]
        and provide the base path to load the images from.
        The path should arrange the images into directories that each
        directory has the 'label_name' from the unique_labels.
        Depending on the loader_type the class will hold images into one of the
        following three options

        - Path -> loader_type == ImageLoadersEnumType.FILENAME
        - PILImage -> loader_type == ImageLoadersEnumType.PIL
        - TorchTensor -> loader_type == ImageLoadersEnumType.PYTORCH_TENSOR

        Parameters
        ----------
        unique_labels: The unique labels for the images
        base_path: The base path to pull images from
        do_load: Flag indicating if the images should be loaded on construction
        image_formats: The formats of the images to consider
        loader_type: What loader to use
        transformer: Whether any transformation should be applied whilst loading the images
        """

        self.unique_labels: List[Tuple[str, int]] = unique_labels
        self.base_path = base_path
        self.image_formats: List[str] = image_formats
        self.images: List[Tuple[Union[Path, PILImage, TorchTensor], Union[int, str]]] = []
        self.image_labels: List[int] = []
        self.loader_type: ImageLoadersEnumType = loader_type
        self._images_per_label: Dict = {}
        self._current_pos: int = -1

        if do_load:
            self.load(loader_type=loader_type, transformer=transformer)

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
        self.clear()

    @property
    def n_images_per_label(self) -> dict:
        """Get a dictionary with the number of images
        per label

        Returns
        -------

        """
        return self._images_per_label

    @property
    def label_names(self) -> List[str]:
        """Returns the names of the labels

        Returns
        -------

        A python list with the names of the labels
        """
        return [label[0] for label in self.unique_labels]

    def get_label_name(self, index: int) -> str:
        """Returns the label name associated with
        the given label index. If the index is not found
        raise ValueError

        Parameters
        ----------
        index: The label index to look for its label

        Returns
        -------

        The label name corresponding to this index
        """

        for label in self.unique_labels:
            if label[1] == index:
                return label[0]
        raise ValueError(f"Index={index} not found in dataset")

    def get_label_idx(self, label_name: str) -> int:
        """Returns the index that corresponds to the given
        label name. Returns -1 if the label name is not found
        in the list of self.unique_labels

        Parameters
        ----------
        label_name: The name of the label to look for its index

        Returns
        -------

        Integer representing the index of the label
        """
        for label in self.unique_labels:
            if label[0] == label_name:
                return label[1]

        return -1

    def clear(self, full_clear: bool = True) -> None:
        """Invalidate the dataset

        Returns
        -------

        """

        if full_clear:
            self.unique_labels = []
            self.base_path = DUMMY_PATH
            self.image_formats = []
            self._current_pos: int = -1

        self.loader_type = ImageLoadersEnumType.INVALID
        self.images = []
        self.image_labels = []
        self._images_per_label = {}

    def shuffle(self) -> None:
        """Randomly shuffle  the contents of the dataset

        Returns
        -------

        """
        random.shuffle(self.images)

    def apply_transform(self, transformer: Callable) -> None:
        """Apply the given transformation on all images
        in the dataset. This eventually will transform all the
        images.

        Parameters
        ----------
        transformer: Callable to apply on the images

        Returns
        -------

        """
        self.images = [(transformer(img[0]), img[1]) for img in self.images]

    def load(self, loader_type: ImageLoadersEnumType = ImageLoadersEnumType.PIL,
             transformer: Callable = None, force_load: bool = False) -> None:

        if not self.__can_load() and not force_load:
            raise ValueError("Dataset is not empty. Have you called clear()?")
        elif not self.__can_load() and force_load:
            self.clear(full_clear=False)

        if str(self.base_path) == str(DUMMY_PATH):
            raise ValueError(f"Cannot load dataset from DUMMY_PATH={str(DUMMY_PATH)}. Specify a correct data path.")

        self.loader_type = loader_type
        tmp_img_formats = []

        for label in self.unique_labels:

            label_name = label[0]
            label_idx = label[1]
            base_path = Path(str(self.base_path)) / label_name

            # get all the image files
            img_files: List[Path] = get_img_files(base_path=base_path,
                                                  img_formats=self.image_formats)

            label_images = []
            labels = []
            # load every image in the Path
            for img in img_files:

                suffix = img.suffix

                if suffix.lower() not in tmp_img_formats:
                    tmp_img_formats.append(suffix.lower())

                label_images.append((load_img(path=img,
                                              transformer=transformer,
                                              loader=loader_type), label_idx))
                labels.append(label_idx)

            self._images_per_label[label] = len(label_images)
            self.images.extend(label_images)
            self.image_labels.extend(labels)
            self.image_formats = tmp_img_formats

    def __can_load(self) -> bool:
        """Checks if the right conditions are met to laod
        a dataset

        Returns
        -------
        A flag indicating if the right conditions are met to load
        """

        if len(self.images) != 0:
            return False

        if len(self.image_labels) != 0:
            return False

        return True
