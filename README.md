# navalmartin_mir_vision_utils

Various utilities for working with images in the _mir_ project. 

## Acknowledgements 

The project incorporates the following repositories

- ```image-quality``` https://github.com/ocampor/image-quality (for BRISQUE)
- ```imutils```: https://github.com/PyImageSearch/imutils (for various utilities with OpenCV)

## Dependencies

- torch
- torchvision
- numpy
- dataclasses
- Pillow
- matplotlib
- opencv-python
- scipy
- scikit-image
- libsvm

## Installation

Installing the utilities via ```pip```

```
pip install navalmartin-mir-vision-utils
```

For a specific version use

```
pip install navalmartin-mir-vision-utils==x.x.x
```

You can uninstall the project via

```commandline
pip3 uninstall navalmartin_mir_vision_utils
```

## How to use

Below are some use-case samples. You can find more in the <a href="./src/navalmartin_mir_vision_utils/examples">examples</a>.

### Using ```image_utils```

```
from pathlib import Path
from navalmartin_mir_vision_utils.image_utils import (is_valid_pil_image_file, get_pil_image_size,
                                                      get_img_files)


if __name__ == '__main__':

    image = is_valid_pil_image_file(image=Path("/home/alex/qi3/mir-engine/datasets/cracks_v_3_id_8/train/cracked/img_9_9.jpg"))
    if image is not None:
        print("The provided image is OK")
        image_size = get_pil_image_size(image=image)
        print(f"Image size is {image_size}")
    else:
        print("The provided image is NOT OK")

    base_path = Path("/home/alex/qi3/mir-engine/datasets/cracks_v_3_id_8/train/cracked/")
    image_files = get_img_files(base_path=base_path)
    print(f"There are {len(image_files)} in {base_path}")
```
