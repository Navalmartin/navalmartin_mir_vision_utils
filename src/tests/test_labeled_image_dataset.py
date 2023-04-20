import unittest
from pathlib import Path
from navalmartin_mir_vision_utils.mir_vision_io.labeled_image_dataset import LabeledImageDataset
from navalmartin_mir_vision_utils.image_enums import ImageFileEnumType


class TestImageUtils(unittest.TestCase):

    def test_create_dataset(self):

        dataset = LabeledImageDataset(labels=["1", "2", "3"],
                                      base_path=Path("./test_data/labeled_dataset"),
                                      do_load=True,
                                      image_formats=[".jpg", ".jpeg", ".png"])

        self.assertEqual(3, len(dataset))


if __name__ == '__main__':
    unittest.main()
