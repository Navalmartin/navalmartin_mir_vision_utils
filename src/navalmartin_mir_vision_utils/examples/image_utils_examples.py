from pathlib import Path
from navalmartin_mir_vision_utils import (is_valid_pil_image_file,
                                          get_pil_image_size,
                                          get_img_files,
                                          pil_image_to_bytes_string,
                                          create_thumbnail_from_pil_image)

from navalmartin_mir_vision_utils.mir_vision_io import get_md5_checksum

if __name__ == '__main__':

    image_file = Path("/home/alex/qi3/mir-engine/datasets/cracks_v_3_id_8/train/cracked/img_9_9.jpg")
    image = is_valid_pil_image_file(image=image_file)
    if image is not None:
        print("The provided image is OK")
        image_size = get_pil_image_size(image=image)
        print(f"Image size is {image_size}")
    else:
        print("The provided image is NOT OK")

    base_path = Path("/home/alex/qi3/mir-engine/datasets/cracks_v_3_id_8/train/cracked/")
    image_files = get_img_files(base_path=base_path)
    print(f"There are {len(image_files)} in {base_path}")

    # calculate file checksum
    image_checksum = get_md5_checksum(file=image_file)
    print(f"Calculated MD5 checksum {image_checksum}")

    image_checksum = get_md5_checksum(file=image.tobytes())
    print(f"Calculated MD5 checksum {image_checksum}")

    # create a thumbnail
    image = create_thumbnail_from_pil_image(max_size=(50, 50),
                                            image_filename=image_file)

    image.show()
