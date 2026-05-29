""" module contains implementations of MS segmentation related functionals """

import numpy as np
from msiplib.finite_differences import gradient_FD2D


def ms_functional(seg, f, lambda_):
    r"""
    function computes the functional value of the MS segmentation functional
    for a segmentation (partition) and given indicator function values

    Args:
        seg: segmentation (point) where the functional is evaluated
        f: indicator function values
        lambda\_: regularization parameter
    Returns:
        the resulting functional value J_{MS}[seg]
    """

    # get number of segments in given segmentation
    k = f.shape[-1]

    # convert segmentation to a one-hot encoding (characteristic function)
    u = np.zeros(seg.shape + (k,), like=f)
    for s in range(k):
        u[seg == s, s] = 1.0

    # compute data term
    data = np.sum(f * u)

    # compute perimeter that equals the total variation of the characteristic function
    reg = np.sum(np.sqrt(np.sum(np.square(gradient_FD2D(u)), axis=0)))

    return data + lambda_ * reg


def zach_functional(u, f, lambda_):
    r"""
    function computes the functional value for a u and given indicator function values
    Args:
        u: point where the functional is evaluated
        f: indicator function values
        lambda\_: regularization parameter
    Returns:
        the resulting functional value J_{Zach}[u]
    """

    # compute data term
    data = np.sum(f * u)

    # compute total variation
    reg = np.sum(np.sqrt(np.sum(np.square(gradient_FD2D(u)), axis=0)))

    return data + lambda_ * reg
