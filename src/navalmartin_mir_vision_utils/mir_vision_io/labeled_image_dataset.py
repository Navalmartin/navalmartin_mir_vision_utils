"""module labeled_image_dataset. Simple
wrapper to load a labeled image dataset

"""
import random
from pathlib import Path
from typing import List, Any, Callable

from navalmartin_mir_vision_utils.image_utils import get_img_files
from navalmartin_mir_vision_utils.image_loaders import load_img
from navalmartin_mir_vision_utils.image_enums import (ImageLoadersEnumType, IMAGE_STR_TYPES)


class LabeledImageDataset(object):

    def __init__(self, labels: List[Any], base_path: Path,
                 do_load: bool = True, *,
                 image_formats: List[Any] = IMAGE_STR_TYPES,
                 loader: ImageLoadersEnumType = ImageLoadersEnumType.PIL,
                 transformer: Callable = None):

        self.labels = labels
        self.base_path = base_path
        self.image_formats = image_formats
        self.images: List[tuple] = []
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
        """Returns the image, label pari  that corresponds to the given key
        Parameters
        ----------
        key: The index of the image-label to retrieve
        Returns
        -------
        A tuple of the image-label
        """

        return self.images[key]

    def shuffle(self) -> None:
        """Randomly shuffle  the contents of the dataset

        Returns
        -------

        """
        random.shuffle(self.images)

    def load(self, loader: ImageLoadersEnumType = ImageLoadersEnumType.PIL,
             transformer: Callable = None) -> None:

        for label in self.labels:

            base_path = Path(str(self.base_path)) / label
            # get all the image files
            img_files = get_img_files(base_path=base_path,
                                      img_formats=self.image_formats)

            label_images = []
            # load every image in the Path
            for img in img_files:
                label_images.append(load_img(path=img,
                                             transformer=transformer,
                                             loader=loader))
            self.images.extend(label_images)
