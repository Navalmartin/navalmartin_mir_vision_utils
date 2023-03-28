from pathlib import Path
from navalmartin_mir_vision_utils import load_img, ImageLoadersEnumType

from navalmartin_mir_vision_utils.image_quality import brisque

if __name__ == '__main__':

    image_path = Path("/home/alex/qi3/mir_vision_utils/test_data/img_2_6.jpg")
    image = load_img(path=image_path, loader=ImageLoadersEnumType.PIL)
    brisque_score = brisque.score(image)
    print(brisque_score)

    image_path = Path("/home/alex/qi3/mir_vision_utils/test_data/corrosion_4.png")

    image = load_img(path=image_path, loader=ImageLoadersEnumType.PIL)
    brisque_score = brisque.score(image)
    print(brisque_score)
