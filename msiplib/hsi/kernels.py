# -*- coding: utf-8 -*-

""" collection of HSI kernels """

from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel


def direct_summation(
    x_spat,
    x_spec,
    y_spat,
    y_spec,
    spat_kernel=rbf_kernel,
    spat_args=None,
    spec_kernel=polynomial_kernel,
    spec_args=None,
):
    r"""
    Implements the direct summation kernel for HSI proposed in
    Composite Kernels for Hyperspectral Image Classification (2006) by Camps-Valls et al.

    .. math:: K(x_i, x_j) = K_s(x_i^s, x_j^s) + K_\omega(x_i^{\omega}, x_j^{\omega})

    :param x_spat: **Spatial** information of pixels in x (num_pixels x num_features)
    :param x_spec: **Spectral** information of pixels in x (num_pixels x num_features)
    :param y_spat: **Spatial** information of pixels in y (num_pixels x num_features)
    :param y_spec: **Spectral** information of pixels in y (num_pixels x num_features)
    :param spat_kernel: Kernel function to be used for **spatial** information (default: RBF kernel)
    :param spec_kernel: Kernel function to be used for **spectral** information (default: polynomial kernel)
    :param spat_args: Arguments of spatial kernels, given as a list.
    :param spec_args: Arguments of spectral kernels, given as a list.
    :returns: the kernel matrix being the sum of the spatial and the spectral kernel matrices for pixels in x and y
    """

    return spat_kernel(x_spat, y_spat, spat_args) + spec_kernel(
        x_spec, y_spec, spec_args[0], spec_args[1], spec_args[2]
    )


def weighted_summation(
    x_spat,
    x_spec,
    y_spat,
    y_spec,
    alpha=1.0,
    spat_kernel=rbf_kernel,
    spat_args=None,
    spec_kernel=polynomial_kernel,
    spec_args=None,
):
    r"""
    Implements the weighted summation kernel for HSI proposed in
    Composite Kernels for Hyperspectral Image Classification (2006) by Camps-Valls et al.

    .. math:: K(x_i, x_j) = \mu K_s(x_i^s, x_j^s) + (1 - \mu) K_\omega(x_i^{\omega}, x_j^{\omega})

    :param x_spat: **Spatial** information of pixels in x (num_pixels x num_features)
    :param x_spec: **Spectral** information of pixels in x (num_pixels x num_features)
    :param y_spat: **Spatial** information of pixels in y (num_pixels x num_features)
    :param y_spec: **Spectral** information of pixels in y (num_pixels x num_features)
    :param alpha: weight spatial kernel with alpha and spectral kernel with 1 - alpha
    :param spat_kernel: Kernel function to be used for **spatial** information (default: RBF kernel)
    :param spec_kernel: Kernel function to be used for **spectral** information (default: polynomial kernel)
    :param spat_args: Arguments of spatial kernels, given as a list.
    :param spec_args: Arguments of spectral kernels, given as a list.
    :returns: the kernel matrix being the sum of the spatial and the spectral kernel matrices for pixels in x and y
    """

    return alpha * spat_kernel(x_spat, y_spat, spat_args) + (1 - alpha) * spec_kernel(
        x_spec, y_spec, spec_args[0], spec_args[1], spec_args[2]
    )


def cross_information(
    x_spat,
    x_spec,
    y_spat,
    y_spec,
    spat_kernel=rbf_kernel,
    spat_args=None,
    spec_kernel=polynomial_kernel,
    spec_args=None,
):
    r"""
    Implements the cross-information kernel for HSI proposed in
    Composite Kernels for Hyperspectral Image Classification (2006) by Camps-Valls et al.

    .. math::

        K(x_i, x_j) =& K_s(x_i^s, x_j^s) + K_\omega(x_i^{\omega}, x_j^{\omega}) \\
                     & + K_{s\omega}(x_i^s, x_j^{\omega}) + K_{\omega s}(x_i^{\omega}, x_j^s)

    The spectral kernel is used for the spectral information as well as cross-information.

    :param x_spat: **Spatial** information of pixels in x (num_pixels x num_features)
    :param x_spec: **Spectral** information of pixels in x (num_pixels x num_features)
    :param y_spat: **Spatial** information of pixels in y (num_pixels x num_features)
    :param y_spec: **Spectral** information of pixels in y (num_pixels x num_features)
    :param spat_kernel: Kernel function to be used for **spatial** information (default: RBF kernel)
    :param spec_kernel: Kernel function to be used for **spectral** information (default: polynomial kernel)
    :param spat_args: Arguments of spatial kernels, given as a list.
    :param spec_args: Arguments of spectral kernels, given as a list.
    :returns: the kernel matrix being the sum of the spatial and the spectral kernel matrices for pixels in x and y
    """

    return (
        spat_kernel(x_spat, y_spat, spat_args)
        + spec_kernel(x_spec, y_spec, spec_args[0], spec_args[1], spec_args[2])
        + spec_kernel(x_spat, y_spec, spec_args[0], spec_args[1], spec_args[2])
        + spec_kernel(x_spec, y_spat, spec_args[0], spec_args[1], spec_args[2])
    )
