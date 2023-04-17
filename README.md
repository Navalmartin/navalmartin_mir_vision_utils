# navalmartin_mir_vision_utils

A collection of various utilities for working with images in the _mir_ project. 
The provided utilities use to a large extent the python <a href="https://pillow.readthedocs.io/en/stable/">PIL</a> library.

## Acknowledgements 

The project incorporates the following repositories

- ```image-quality``` https://github.com/ocampor/image-quality (for BRISQUE)
- ```imutils```: https://github.com/PyImageSearch/imutils (for various utilities with OpenCV)

## Dependencies

The general dependencies are:

- numpy
- Pillow
- scipy
- scikit-image
- libsvm

In addition, utilities for working to PyTorch and OpenCV exists but 
you need to install these dependencies yourself. The ```mir_vision_config``` 
provides various configuration flags to customize the API. These are

- ```WITH_TORCH``` 
- ```WITH_CV2``` 
- ```WITH_SKIMAGE_VERSION```

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

```
pip3 uninstall navalmartin-mir-vision-utils
```

## How to use

Below are some use-case samples. You can find more in the <a href="./src/navalmartin_mir_vision_utils/examples">examples</a>.

### Using ```image_utils```

Various image utilities are provided in this module. The example below
showcases some.

```
from pathlib import Path
from navalmartin_mir_vision_utils import (is_valid_pil_image_file,
                                          get_pil_image_size,
                                          get_img_files,
                                          pil_image_to_bytes_string,
                                          create_thumbnail_from_pil_image)

from navalmartin_mir_vision_utils.mir_vison_io import get_md5_checksum

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
```

### Using ```image_transformers```

```
from pathlib import Path
from navalmartin_mir_vision_utils.image_transformers import pil_image_to_bytes_string
from navalmartin_mir_vision_utils.image_utils import load_img
from navalmartin_mir_vision_utils.image_enums import ImageLoadersEnumType
from navalmartin_mir_vision_utils.image_utils import is_valid_pil_image_from_bytes_string
from navalmartin_mir_vision_utils.image_utils import show_pil_image

if __name__ == '__main__':

    image_path = Path("/home/alex/qi3/mir-engine/datasets/cracks_v_3_id_8/train/cracked/img_9_9.jpg")
    image = load_img(path=image_path, loader=ImageLoadersEnumType.PIL)

    show_pil_image(image=image)

    image_bytes = pil_image_to_bytes_string(image=image)
    image = is_valid_pil_image_from_bytes_string(image_byte_string=image_bytes)
    show_pil_image(image=image)
```

### Compute basic image statistics

```
from pathlib import Path
from navalmartin_mir_vision_utils import load_img, ImageLoadersEnumType
from navalmartin_mir_vision_utils.statistics import compute_image_statistics, fit_gaussian_distribution_on_image

if __name__ == '__main__':

    image_path = Path("/home/alex/qi3/mir-engine/datasets/cracks_v_3_id_8/train/cracked/img_9_9.jpg")
    image = load_img(path=image_path, loader=ImageLoadersEnumType.PIL)

    print(f"Image size {image.size}")
    print(f"Image bands {image.getbands()}")

    image_stats = compute_image_statistics(image)
    print(f"Image channel mean {image_stats.mean}")
    print(f"Image channel var {image_stats.var}")
    print(f"Image channel median {image_stats.median}")

    channels_fit = fit_gaussian_distribution_on_image(image=image)
    print(f"Gaussian distribution channel fit: {channels_fit}")
```

### Compute image quality calculation

Currently, only the BRISQUE algorithm is supported. The implementation
from https://github.com/ocampor/image-quality has been integrated into the 
utilities.

```
from pathlib import Path

from navalmartin_mir_vision_utils import load_img, ImageLoadersEnumType
from navalmartin_mir_vision_utils.image_quality import brisque

if __name__ == '__main__':
    image_path = Path("/home/alex/qi3/mir-engine/datasets/cracks_v_3_id_8/train/cracked/img_9_9.jpg")
    image = load_img(path=image_path, loader=ImageLoadersEnumType.PIL)
    brisque_score = brisque.score(image)
    print(brisque_score)
```

### Issues

- ```rescale() got an unexpected keyword argument 'multichannel'```

This issue may be related to the version of the skimage package you have installed. You can check
with version is installed on your system by using

```
import skimage
print(skimage.__version__)

```

or use the variable ```WITH_SKIMAGE_VERSION``` in ```mir_vision_config```. 