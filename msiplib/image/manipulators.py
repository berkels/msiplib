"""
Collection of image manipulation utilities.
"""
import numpy as np

from scipy import ndimage




def image_is_gray(img):
    if len(img.shape) < 3:
        return True

    # One channel means gray scale.
    if img.shape[2] == 1:
        return True

    # The logic below only handles RGB and RGBA images.
    if img.shape[2] != 3 and img.shape[2] != 4:
        raise ValueError("Unsupported image shape.")

    # Check for non-constant alpha channel.
    if img.shape[2] == 4 and not (img[:, :, 3] == img[0, 0, 3]).all():
        return False

    # Check if all three color channels are the same.
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    if (b == g).all() and (b == r).all():
        return True

    return False


def assert_image_is_gray(img):
    if img.ndim > 2:
        if not image_is_gray(img):
            raise ValueError("Only grayscale input images are supported.")
        else:
            return img[..., 0]
    else:
        return img
