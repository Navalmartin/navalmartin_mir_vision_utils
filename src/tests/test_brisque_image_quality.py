import unittest
from pathlib import Path

from src.navalmartin_mir_vision_utils.image_enums import ImageLoadersEnumType
from src.navalmartin_mir_vision_utils.image_utils import load_img
from src.navalmartin_mir_vision_utils.image_quality.brisque import score


class TestBrisqueImageQuality(unittest.TestCase):

    def test_get_img_files(self):
        image = load_img(path=Path('/home/alex/qi3/mir_vision_utils/test_data/img_18_3.jpg'),
                         loader=ImageLoadersEnumType.PIL,
                         transformer=None)

        score(image=image)


if __name__ == '__main__':
    unittest.main()
