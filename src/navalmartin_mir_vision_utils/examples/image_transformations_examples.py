from pathlib import Path

from navalmartin_mir_vision_utils import (show_pil_image,
                                          is_valid_pil_image_from_bytes_string,
                                          load_img,
                                          ImageLoadersEnumType,
                                          pil_image_to_bytes_string)


if __name__ == '__main__':

    image_path = Path("/home/alex/qi3/mir-engine/datasets/cracks_v_3_id_8/train/cracked/img_9_9.jpg")
    image = load_img(path=image_path, loader=ImageLoadersEnumType.PIL)

    show_pil_image(image=image)

    image_bytes = pil_image_to_bytes_string(image=image)
    image = is_valid_pil_image_from_bytes_string(image_byte_string=image_bytes)
    show_pil_image(image=image)
