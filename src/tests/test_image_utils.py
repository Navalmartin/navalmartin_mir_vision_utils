import unittest
from pathlib import Path
from src.navalmartin_mir_vision_utils.image_utils import get_img_files
from src.navalmartin_mir_vision_utils.image_enums import ImageFileEnumType


class TestImageUtils(unittest.TestCase):

    def test_get_img_files(self):
        files = get_img_files(img_dir=Path('/imutils/test_data'),
                              img_formats=(ImageFileEnumType.PNG,
                                           ImageFileEnumType.JPG))

        self.assertEqual(2, len(files))


if __name__ == '__main__':
    unittest.main()
