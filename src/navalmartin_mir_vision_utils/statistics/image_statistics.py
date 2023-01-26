from typing import List
import numpy as np
from PIL import Image
from PIL.ImageStat import Stat
from scipy.stats import norm as normal_distribution
from scipy.stats import entropy
from scipy.special import rel_entr
from math import log2


def gamma_correction(img: np.ndarray, gamma_val) -> np.ndarray:
    """Implements gamma correction on the given image

    Parameters
    ----------
    img: The image to perform the correction
    gamma_val: The gamma value

    Returns
    -------

    """

    return np.power(img, gamma_val)


def compute_image_statistics(image: Image) -> Stat:
    return Stat(image_or_list=image)


def compute_image_channels_means(image: Image) -> List[float]:

    stats = Stat(image_or_list=image)
    return stats.mean


def compute_image_channels_variance(image: Image) -> List[float]:
    stats = Stat(image_or_list=image)
    return stats.var


def compute_image_channels_median(image: Image) -> List[float]:
    stats = Stat(image_or_list=image)
    return stats.median


def fit_gaussian_distribution(data: List[float]) -> List[float]:
    """Fits a Gaussian distribution on the given data and
    returns the mean and standard deviation of the resulting
    distribution

    Parameters
    ----------
    data: The data to fit

    Returns
    -------

    A list containing the mean and standard deviation of
    the fitted distribution
    """
    return normal_distribution.fit(data)


def kl_divergence(q: List[float], p: List[float],
                  do_test_sum: bool = True) -> float:
    """Compute the KL divergence between the two given mass functions
    Note that the elements in each array must sum to one.
    The SciPy library provides the kl_div() function for calculating the KL divergence,
    although with a different definition as defined here.
    It also provides the rel_entr() function for calculating the relative entropy,
    which matches the definition of KL divergence here.
    This is odd as “relative entropy” is often used as a synonym for “KL divergence.”

    Parameters
    ----------
    q: The first mass function
    p: The second mass function
    do_test_sum: Test if the elements in the two arrays
    sum to one
    Returns
    -------

    The difference between the two mass functions
    """

    if do_test_sum:
        assert sum(q) == 1 and sum(p) == 1

    return rel_entr(q, p)


def jensen_shannon_distance(q: List[float], p: List[float],
                            do_test_sum: bool = True) -> float:
    """Calculate the Jensen-Shannon divergence
    The JS divergence quantifies the difference (or similarity) between two probability distributions.
    Note that the elements in q and p should sum up to one.

    Parameters
    ----------
    q: The first mass function
    p: The second mass function
    do_test_sum: Test if the elements in the two arrays
    sum to one

    Returns
    -------

    """

    if do_test_sum:
        assert sum(q) == 1 and sum(p) == 1

    # calculate m
    m = (p + q) / 2.

    # compute Jensen Shannon Divergence
    divergence = (entropy(p, m) + entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)
    return distance
