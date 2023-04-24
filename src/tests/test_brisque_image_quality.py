import unittest
from pathlib import Path
from navalmartin_mir_vision_utils.mir_vision_io.file_utils import rename_files_with_counter


class TestBrisqueImageQuality(unittest.TestCase):

    def test_rename_files_with_counter(self):
        rename_files_with_counter(base_path_src=Path("./test_data"),
                                  base_path_dist=Path("./test_data/copy_rename_images"),
                                  file_formats=["png"])





if __name__ == '__main__':
    unittest.main()
