import unittest

from pathlib import Path

from src.navalmartin_mir_vision_utils.statistics.image_statistics import compute_image_channels_means
from src.navalmartin_mir_vision_utils.statistics.image_statistics import compute_image_channels_variance
from src.navalmartin_mir_vision_utils.image_utils import load_img
from src.navalmartin_mir_vision_utils.image_enums import ImageLoadersEnumType


class TestImageStatistics(unittest.TestCase):

    def test_compute_image_channels_means(self):

        image = load_img(path=Path('/imutils/test_data/corrosion_4.png'),
                         loader=ImageLoadersEnumType.PIL)
        means = compute_image_channels_means(image=image)
        self.assertEqual(len(means), 4)

    def test_compute_image_channels_variance(self):

        image = load_img(path=Path('/imutils/test_data/corrosion_4.png'),
                         loader=ImageLoadersEnumType.PIL)
        means = compute_image_channels_variance(image=image)
        self.assertEqual(len(means), 4)


if __name__ == '__main__':
    unittest.main()