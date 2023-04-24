from pathlib import Path
from navalmartin_mir_vision_utils.mir_vision_io import LabeledImageDataset


if __name__ == '__main__':

    base_path = Path("/home/alex/qi3/mir_datasets/vessels")
    dataset = LabeledImageDataset(labels=["catamaran", "rib",
                                          "single_hull", "trimaran"],
                                  base_path=base_path)

    print(f"Numebr of images {len(dataset)}")