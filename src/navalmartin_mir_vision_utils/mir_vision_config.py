import skimage
from pathlib import Path

WITH_SKIMAGE_VERSION = skimage.__version__
WITH_TORCH = False
WITH_CV2 = False
WITH_MATPLOTLIB = False
DUMMY_PATH = Path("/this/is/a/dummy/path")

try:
    import torch
    import torchvision
    from torchvision import transforms

    WITH_TORCH = True
except ModuleNotFoundError as e:
    print(
        f"WARNING: An exception was raised whilst importing torch, torchvision. Message {str(e)}")
    print("WARNING: mir-vision-utils will not use torch and torchvision")
    pass

try:
    import cv2
    WITH_CV2 = True
except ModuleNotFoundError as e:
    print(
        f"WARNING: An exception was raised whilst importing cv2 for OpenCV. Message {str(e)}")
    print("WARNING: mir-vision-utils will not use OpenCV")
    pass

try:
    import matplotlib.pyplot as plt
    WITH_MATPLOTLIB = True

except ModuleNotFoundError as e:
    print(
        f"WARNING: An exception was raised whilst importing matplotlib. Message {str(e)}")
    print("WARNING: mir-vision-utils will not use matplotlib")
    pass
