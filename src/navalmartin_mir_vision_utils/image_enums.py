from enum import Enum


class ImageFileEnumType(Enum):

    INVALID = 0
    JPG = 1
    JPEG = 2
    PNG = 3
    BMP = 4
    TIF = 5
    TIFF = 6


IMAGE_STR_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class ImageLoadersEnumType(Enum):

    INVALID = 0
    PIL = 1
    CV2 = 2
    PIL_NUMPY = 3
    PYTORCH_TENSOR = 4


IMAGE_LOADERS_TYPES_STR = ('PIL', 'CV2', 'PIL_NUMPY', 'PYTORCH_TENSOR')


class ValidPillowEnumType(Enum):

    INVALID = 0
    ONE = 1
    CMYK = 2
    F = 3
    HSV = 4
    I = 5
    L = 6
    LAB = 7
    P = 8
    RGB = 9
    RGBA = 10
    RGBX = 11
    YCbCr = 12


VALID_PIL_MODES_STR = ("1", "CMYK", "F", "HSV", "I", "L", "LAB", "P", "RGB", "RGBA", "RGBX", "YCbCr")