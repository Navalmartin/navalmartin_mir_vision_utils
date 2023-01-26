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
pip install -i https://test.pypi.org/simple/ navalmartin-mir-vision-utils==0.0.1
```

Notice that the project is pulled from ```TestPyPi``` which does not have the same packages
as the official PyPi index. This means that dependencies may fail to install. It is advised therefore
to manually install the dependencies mentioned above.

You can uninstall the project via

```commandline
pip3 uninstall navalmartin_mir_vision_utils
```

## How to use
