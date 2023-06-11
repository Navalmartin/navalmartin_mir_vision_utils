"""module labeled_image_dataset. Simple
wrapper to load a labeled image dataset

"""
import random
from pathlib import Path

from PIL.Image import Image as PILImage
from typing import List, Any, Callable, Tuple, Union, Dict
import csv
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

    @classmethod
    def load_from_csv(cls, csv_filename: Path, base_path: Path, path_creator: Callable = None) -> "LabeledImageDataset":
        """Load the dataset from the given CSV file. The file should
        have the following format ('image_filename', image_label_index, 'image_label_name')

        Parameters
        ----------
        csv_filename: The CSV file to load the dataset from
        base_path: The base (i.e. root) path that dataset resides
        path_creator: Adapt the path formed from the CSV rows
        Returns
        -------

        """

        images: List[Tuple[Union[Path], Union[int]]] = []
        image_labels: List[int] = []
        unique_labels = []
        image_formats = []
        with open(csv_filename, 'r', newline='\n') as csvfile:
            filereader = csv.reader(csvfile, delimiter=",")

            for row in filereader:

                if len(row) != 3:
                    raise ValueError(f"Invalid format. File should have 3 columns but has {len(row)}")

                if path_creator is not None:
                    img_path = path_creator(row)
                else:
                    img_path = Path(str(base_path) + "/" + row[2] + "/" + row[0])
                images.append((img_path, int(row[1])))
                image_labels.append(int(row[1]))
                format_ = Path(row[0]).suffix

                if format_ not in image_formats:
                    image_formats.append(format_)

                if (row[2], int(row[1])) not in unique_labels:
                    unique_labels.append((row[2], int(row[1])))

        dataset = LabeledImageDataset.build_from_list(images=images,
                                                      image_labels=image_labels,
                                                      unique_labels=unique_labels,
                                                      loader_type=ImageLoadersEnumType.FILEPATH,
                                                      image_formats=image_formats)
        dataset.base_path = base_path
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

    def get_class_images(self, class_name: str) -> List[Tuple[Union[Path, PILImage, TorchTensor], Union[int, str]]]:
        """Get the images corresponding to the class name.
        It raises ValueError if the class_name is not in the dataset

        Parameters
        ----------
        class_name: The class name to get the images from

        Returns
        -------

        Instance of: List[Tuple[Union[Path, PILImage, TorchTensor], Union[int, str]]]
        """
        if class_name not in self._images_per_label:
            raise ValueError(f"Class name {class_name} not in dataset")

        class_idx = self.get_label_idx(class_name)
        return [item for item in self.images if item[1] == class_idx]

    def random_selection(self, size: int) -> List[int]:
        """Returns a list of indices of the given size
        randomly selected

        Parameters
        ----------
        size: The size of the random sample

        Returns
        -------

        Instance of List[int]
        """

        if size >= len(self.images):
            raise ValueError(f"Invalid size parameter. size should be in [0,{len(self.images)}) but is {size}")

        indices = [i for i in range(len(self.images))]
        return random.sample(indices, size)

    def random_selection_for_class(self, size: int, class_name: str) -> List[Tuple[Union[Path, PILImage, TorchTensor], Union[int, str]]]:
        """Returns a random selection of images corresponding to the given
        class name. Raises ValueError if the class_name is not in the dataset.
        Raises ValueError if the given size is greater than or equal to the number
        of images in the dataset that correspond to the given dataset.

        Parameters
        ----------
        size: The size of the random sample to return
        class_name: The class name to get the images from

        Returns
        -------

        Instance of: List[Tuple[Union[Path, PILImage, TorchTensor], Union[int, str]]]
        """

        if class_name not in self._images_per_label:
            raise ValueError(f"Class name {class_name} not in dataset")

        n_images_for_class = self._images_per_label[class_name]

        if size >= n_images_for_class:
            raise ValueError(f"Invalid size parameter. size should be in [0,{n_images_for_class}) but is {size}")

        # get the images for the class
        class_images = self.get_class_images(class_name)
        return random.sample(class_images, size)

    def remove_images(self, images: List[Tuple[Union[Path, PILImage], Union[int, str]]]) -> None:
        """Remove the images specified in the list

        Parameters
        ----------
        images: The images to remove

        Returns
        -------
        """

        for img in images:

            image = img[0]
            if isinstance(image, Path):
                self.images.remove(img)
                label = img[1]
                label_name = self.get_label_name(label)
                self._images_per_label[label_name] -= 1
            elif isinstance(image, PILImage):
                self.images.remove(img)
                label = img[1]
                label_name = self.get_label_name(label)
                self._images_per_label[label_name] -= 1
            else:
                raise ValueError("Cannot remove image. Image should be either Path or PIL.Image")


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

    def add_to_class(self, images: List[Tuple[Union[Path, PILImage, TorchTensor], Union[int, str]]],
                     class_name: str):

        """Append the given images to the class. Raises ValueError
        if the class_name is not in the unique_labels

        Parameters
        ----------
        images: Images to append
        class_name: The class name to append the images

        Returns
        -------

        """

        if len(images) == 0:
            return

        class_item = self.get_label_idx(label_name=class_name)
        if class_item == -1:
            raise ValueError(f"Label {class_name} not in dataset")

        self.images.extend(images)
        self.image_labels.extend([class_item]*len(images))
        self._images_per_label[class_name] += len(images)

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

            self._images_per_label[label_name] = len(label_images)
            self.images.extend(label_images)
            self.image_labels.extend(labels)
            self.image_formats = tmp_img_formats

    def save_to_csv(self, filename: Path) -> None:
        """Save the given dataset in a CSV format.
        Currently, only a dataset that has
        loader_type == ImageLoadersEnumType.FILENAME

        Parameters
        ----------
        filename: The file to save the dataset

        Returns
        -------
        """

        if self.loader_type != ImageLoadersEnumType.FILEPATH:
            raise ValueError(f"Cannot save a dataset loaded with {self.loader_type.name}. "
                             f"Load dataset using {ImageLoadersEnumType.FILEPATH.name}. ")

        with open(filename, 'w', newline='\n') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=",")

            for img, label in zip(self.images, self.image_labels):
                label_name = self.get_label_name(label)
                image = str(img[0]).split("/")[-1]
                row = [image, label, label_name]

                filewriter.writerow(row)

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
