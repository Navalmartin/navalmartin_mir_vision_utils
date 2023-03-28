"""module brisque. Implementation of the
BRISQUE algorithm. Original implementation is
taken from: https://github.com/ocampor/image-quality

"""
import os
import pickle
import typing
import warnings
from enum import Enum

import PIL.Image
import numpy
import scipy.signal
import skimage.color
import skimage.transform
from libsvm import svmutil

from navalmartin_mir_vision_utils.statistics.distributions.asymmetric_generalized_gaussian import AsymmetricGeneralizedGaussian
from navalmartin_mir_vision_utils.statistics.statistics_utils import gaussian_kernel2d
from navalmartin_mir_vision_utils.image_transformers import pil2ndarray
from navalmartin_mir_vision_utils.image_quality.models import MODELS_PATH
from navalmartin_mir_vision_utils.mir_vision_config import WITH_SKIMAGE_VERSION

VALID_BRISQUE_IMAGE_FORMATS = ['JPG', 'JPEG', 'PNG', 'MPO']

with open(os.path.join(MODELS_PATH, "normalize.pickle"), "rb") as file:
    scale_parameters = pickle.load(file)

model = svmutil.svm_load_model(os.path.join(MODELS_PATH, "brisque_svm.txt"))


class MscnType(Enum):
    mscn = 1
    horizontal = 2
    vertical = 3
    main_diagonal = 4
    secondary_diagonal = 5


class Brisque:
    _local_mean = None
    _local_deviation = None
    _mscn = None
    _features = None

    def __init__(
        self,
        image: typing.Union[PIL.Image.Image, numpy.ndarray],
        kernel_size: int = 7,
        sigma: float = 7 / 6,
    ):
        self.image = pil2ndarray(image)

        if self.image.shape[-1] == 3:
            self.image = skimage.color.rgb2gray(self.image)
        elif self.image.shape[-1] == 4:
            self.image = skimage.color.rgb2gray(skimage.color.rgba2rgb(self.image))

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = gaussian_kernel2d(kernel_size, sigma)

    @property
    def local_mean(self):
        if self._local_mean is None:
            self._local_mean = scipy.signal.convolve2d(self.image, self.kernel, "same")
        return self._local_mean

    @property
    def local_deviation(self):
        if self._local_deviation is None:
            sigma = numpy.square(self.image)
            sigma = scipy.signal.convolve2d(sigma, self.kernel, "same")
            self._local_deviation = numpy.sqrt(
                numpy.abs(numpy.square(self.local_mean) - sigma)
            )
        return self._local_deviation

    @property
    def mscn(self):
        if self._mscn is None:
            c = 1 / 255
            self._mscn = (self.image - self.local_mean) / (self.local_deviation + c)
        return self._mscn

    @property
    def mscn_horizontal(self):
        return self.mscn[:, :-1] * self.mscn[:, 1:]

    @property
    def mscn_vertical(self):
        return self.mscn[:-1, :] * self.mscn[1:, :]

    @property
    def mscn_diagonal(self):
        return self.mscn[:-1, :-1] * self.mscn[1:, 1:]

    @property
    def mscn_secondary_diagonal(self):
        return self.mscn[1:, :-1] * self.mscn[:-1, 1:]

    @property
    def features(self):
        return numpy.concatenate(
            [self.calculate_features(mscn_type) for mscn_type in MscnType]
        )

    def get_coefficients(self, mscn_type: MscnType):
        coefficients = {
            MscnType.mscn: self.mscn,
            MscnType.horizontal: self.mscn_horizontal,
            MscnType.vertical: self.mscn_vertical,
            MscnType.main_diagonal: self.mscn_diagonal,
            MscnType.secondary_diagonal: self.mscn_secondary_diagonal,
        }
        return coefficients[mscn_type]

    def calculate_features(self, mscn_type: MscnType):
        agg = AsymmetricGeneralizedGaussian(self.get_coefficients(mscn_type)).fit()
        if mscn_type == MscnType.mscn:
            var = numpy.mean(
                [numpy.square(agg.sigma_left), numpy.square(agg.sigma_right)]
            )
            return numpy.array([agg.alpha, var])

        return numpy.array(
            [
                agg.alpha,
                agg.mean,
                numpy.square(agg.sigma_left),
                numpy.square(agg.sigma_right),
            ]
        )


def scale_features(features: numpy.ndarray) -> numpy.ndarray:
    _min = numpy.array(scale_parameters["min_"])
    _max = numpy.array(scale_parameters["max_"])
    return -1 + (2.0 / (_max - _min) * (features - _min))


def calculate_features(image: PIL.Image, kernel_size, sigma) -> numpy.ndarray:
    brisque = Brisque(image, kernel_size=kernel_size, sigma=sigma)
    # WARNING: The algorithm is very sensitive to rescale
    # FIXME: this is empirically the best configuration; however, scikit-image warns about bi-quadratic implementation.
    #    Fix this warning error in version of scikit-image 0.16.0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if WITH_SKIMAGE_VERSION == "0.19.3":
            downscaled_image = skimage.transform.rescale(
                brisque.image,
                1 / 2,
                order=2,
                mode="constant",
                anti_aliasing=False,
                multichannel=False,
            )
        else:
            downscaled_image = skimage.transform.rescale(
                brisque.image,
                1 / 2,
                order=2,
                mode="constant",
                anti_aliasing=False,
            )

    downscaled_brisque = Brisque(downscaled_image, kernel_size=kernel_size, sigma=sigma)
    features = numpy.concatenate([brisque.features, downscaled_brisque.features])
    scaled_features = scale_features(features)
    return scaled_features


def predict(features: numpy.ndarray) -> float:
    x, idx = svmutil.gen_svm_nodearray(
        features, isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED)
    )
    nr_classifier = 1
    prob_estimates = (svmutil.c_double * nr_classifier)()
    return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)


def score(image: PIL.Image.Image, kernel_size=7, sigma=7 / 6) -> float:
    """Calculate the BRISQUE score for the given image. Currently, only
    JPG images are supported

    Parameters
    ----------
    image: The Pillow image to calculate its quality
    kernel_size
    sigma

    Returns
    -------

    The BRISQUE score
    """

    if image is None:
        raise ValueError("Pillow Image is None")

    if image.format not in VALID_BRISQUE_IMAGE_FORMATS:
        raise ValueError(f"Current image format {image.format} is not in {VALID_BRISQUE_IMAGE_FORMATS}")

    print(f"INFO: Using {WITH_SKIMAGE_VERSION} version of skimage")

    scaled_features = calculate_features(image, kernel_size, sigma)
    return predict(scaled_features)
