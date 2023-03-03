from pathlib import Path
from navalmartin_mir_vision_utils import (is_valid_pil_image_file,
                                          get_pil_image_size,
                                          get_img_files)

if __name__ == '__main__':

    image = is_valid_pil_image_file(
        image=Path("/home/alex/qi3/mir-engine/datasets/cracks_v_3_id_8/train/cracked/img_9_9.jpg"))
    if image is not None:
        print("The provided image is OK")
        image_size = get_pil_image_size(image=image)
        print(f"Image size is {image_size}")
    else:
        print("The provided image is NOT OK")

    base_path = Path("/home/alex/qi3/mir-engine/datasets/cracks_v_3_id_8/train/cracked/")
    image_files = get_img_files(base_path=base_path)
    print(f"There are {len(image_files)} in {base_path}")
