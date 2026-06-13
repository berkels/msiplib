#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=not-an-iterable

""" Collection of indicator functions used for image segmentation """

import logging
import numpy as np
from numba import jit, prange
from scipy.ndimage import uniform_filter
from scipy.stats import trimboth, trim_mean
from sklearn.metrics.pairwise import pairwise_kernels
from spectral import calc_stats, noise_from_diffs, mnf
from msiplib.decomposition import pca


from msiplib.hsi.kernels import cross_information, direct_summation, weighted_summation
from msiplib.metrics import anisotropic_2norm as m_anisotropic_2norm


def compute_segment_mean(image, seg_mask, label):
    """
    computes mean feature vector of a segment in an image

    Args:
        image: an image as a two- or three-dimensional tensor

        seg_mask: a matrix of the same size as the image with integer entries being the labels
                  of the corresponding pixels

        label: the label of the segment of which the mean feature vector shall be computed

    Returns:
        a vector of the same dimension as the feature vectors of the pixels in the image being the mean feature of
        the segment
    """
    return np.mean(image[seg_mask == label], axis=0)


def epsAMS(image, seg_mask, label, eps, means, pcs, weights, tol=1e-05, max_iter=100, valid_mask=None):
    """
    computes for a given segment the indicator function based on the non-squared anisotropic 2-norm epsAMS that is
    regularized with 1 / epsilon to ensure invertibility of the covariance matrix

    Args:
        image: an image as a two- or three-dimensional tensor

        seg_mask: a matrix of the same size as the image with integer entries being the labels
                  of the corresponding pixels

        label: the label of the segment to be processed

        eps: regularization parameter used to take care of directions with very low standard deviation

        means: initial guess for mean feature vector

        pcs: initial guess for principal components

        weights: initial guess for weights (regularized standard deviations) of indicator function

        tol: tolerance for stopping criterion

        max_iter: maximum number of iterations to find mean and covariance

        valid_mask: a matrix of the same size as the image with an entry being true if the corresponding pixel
                    shall contribute to the computation of the segment mean

    Returns:
        a vector of the same dimension as the feature vectors of the pixels in the image being the mean feature of
        the segment
    """

    logger = logging.getLogger("indicator")

    # extract pixels belonging to segment
    if valid_mask is not None:
        # remove pixels that shall not contribute to computation of segment's mean, components and standard deviations
        valid_pixels = image[valid_mask]
        valid_segmentation_mask = seg_mask[valid_mask]
        valid_segment_pixels = valid_pixels[valid_segmentation_mask == label]
    else:
        valid_pixels = image.reshape((image.shape[0] * image.shape[1], image.shape[-1]))
        valid_segmentation_mask = np.ravel(seg_mask)
        valid_segment_pixels = valid_pixels[valid_segmentation_mask == label]

    # initialize mean values for segment and allocate necessary memory
    tau = 1e-02
    seg_mean_old = np.full_like(means, np.finfo(np.float32).max, shape=image.shape[-1], dtype=image.dtype)
    weights_old = np.full_like(weights, np.finfo(np.float32).max, shape=image.shape[-1], dtype=image.dtype)
    seg_pcs_old = np.full_like(
        pcs, np.finfo(np.float32).max, shape=(image.shape[-1], image.shape[-1]), dtype=image.dtype
    )
    dists_inv = np.empty_like(image, shape=valid_segment_pixels.shape[0], dtype=image.dtype)
    pxs_centered = np.empty_like(image, shape=valid_segment_pixels.shape, dtype=image.dtype)
    pxs_scaled = np.empty_like(image, shape=valid_segment_pixels.shape, dtype=image.dtype)
    cov = np.empty_like(image, shape=(image.shape[-1], image.shape[-1]), dtype=image.dtype)

    # TODO: Can we work with references here instead of copying the data? Would that make returning weights unnecessary?
    t_weights = weights[label].copy()
    seg_mean = means[label].copy()
    seg_pcs = pcs[label].copy()

    it = 0
    logger.info("Regularize 2-norm with tau: %s", tau)
    logger.info("Maximum number of fixed point iterations: %s", max_iter)
    logger.info("Stopping threshold: %s", tol)
    while (
        np.linalg.norm(seg_mean - seg_mean_old)
        + np.linalg.norm(t_weights - weights_old)
        + np.linalg.norm(seg_pcs - seg_pcs_old)
    ) > tol and it < max_iter:
        # store old values of mean, standard deviations and PCs
        np.copyto(seg_mean_old, seg_mean)
        np.copyto(weights_old, t_weights)
        np.copyto(seg_pcs_old, seg_pcs)

        # compute distances wrt to current mean, standard deviations and PCs and store the reciprocals
        dists_inv[...] = m_anisotropic_2norm(
            (valid_segment_pixels - seg_mean[np.newaxis]).T, seg_pcs, t_weights, squared=False, tau=tau
        )
        np.reciprocal(dists_inv, out=dists_inv)

        # compute estimate of mean
        np.multiply(valid_segment_pixels, dists_inv[np.newaxis].T / np.sum(dists_inv), out=pxs_scaled)
        np.sum(pxs_scaled, axis=0, out=seg_mean)

        # compute estimate of covariance
        np.subtract(valid_segment_pixels, seg_mean[np.newaxis], out=pxs_centered)
        np.divide(dists_inv, 2.0, out=dists_inv)
        np.multiply(pxs_centered, dists_inv[np.newaxis].T, out=pxs_scaled)
        np.matmul(pxs_scaled.T, pxs_centered, out=cov)
        np.divide(cov, valid_segment_pixels.shape[0], out=cov)

        # eigenvalue decomposition of cov
        seg_std, seg_pcs = np.linalg.eigh(cov)
        np.maximum(seg_std, 0.0, out=seg_std)
        np.sqrt(seg_std, out=seg_std)

        # compute the weights for anisotropic 2-norm with current iterates
        np.maximum(seg_std, eps, out=t_weights)
        np.reciprocal(t_weights, out=t_weights)

        it += 1

    logger.info("Number of iterations needed to find mean and covariance: %s", it)
    logger.info("Components with standard deviation smaller than epsilon: %s", np.sum(t_weights == 1 / eps))

    # compute logarithm of determinant of covariance matrix
    log_det_cov = -2 * np.sum(np.log(t_weights))

    return (
        (
            m_anisotropic_2norm(
                (image.reshape((image.shape[0] * image.shape[1], image.shape[2])) - seg_mean[np.newaxis]).transpose(),
                seg_pcs,
                t_weights,
                squared=False,
                tau=tau,
            ).reshape((image.shape[0], image.shape[1]))
            + log_det_cov
        ),
        seg_mean,
        seg_pcs,
        t_weights,
    )


def phiAMS(
    image, seg_mask, label, means, pcs, weights, valid_mask=None, tol=1e-05, max_iter=100, alpha=1, beta=1
):
    """
    computes for a given segment the indicator function based on the parameter-free non-squared anisotropic 2-norm
    (phiAMS), i.e., weighted trace where the weighting factor for the 1/std depends on the variation
    in the corresponding direction.

    Args:
        image: an image as a two- or three-dimensional tensor

        seg_mask: a matrix of the same size as the image with integer entries being the labels
                  of the corresponding pixels

        label: the label of the segment to be processed

        means: initial guess for mean feature vector

        pcs: initial guess for principal components

        weights: initial guess for weights (standard deviations) of indicator function

        valid_mask: a matrix of the same size as the image with an entry being true if the corresponding pixel
                    shall contribute to the computation of the segment mean

        tol: tolerance for stopping criterion

        max_iter: maximum number of iterations to find mean and covariance

        alpha: scaling factor of weighted trace term

        beta: scaling factor of log det Sigma term (volume regularizer)

    Returns:
        a vector of the same dimension as the feature vectors of the pixels in the image being the mean feature of
        the segment
    """

    logger = logging.getLogger("indicator")

    def compute_indicators(spectra, mean, orth_mat, weights, scaling_func, alpha, beta):
        """
        computes indicator values for given spectra based on mean, orthogonal matrix and weights
        """
        # compute distances wrt to current mean, standard deviations and PCs (Mahalanobis part or data term)
        mahalanobis_term = np.empty_like(spectra, dtype=spectra.dtype, shape=len(spectra))
        spectra_centered = np.subtract(spectra, mean[np.newaxis])
        spectra_centered_orth = np.transpose(np.matmul(orth_mat.T, spectra_centered.T))
        del spectra_centered
        spectra_centered_orth_square = np.square(spectra_centered_orth)
        del spectra_centered_orth
        np.sum(np.multiply(spectra_centered_orth_square, weights), axis=1, out=mahalanobis_term)

        spectra_scaling_factor = scaling_func.eval(spectra_centered_orth_square)
        del spectra_centered_orth_square

        np.multiply(spectra_scaling_factor, weights, out=spectra_scaling_factor)
        
        trace_reg = np.multiply(alpha, np.sum(spectra_scaling_factor, axis=1))
        del spectra_scaling_factor

        data_term = np.sqrt(np.add(mahalanobis_term, trace_reg))
        del mahalanobis_term
        del trace_reg

        # compute volume regularizer
        eigvals = np.reciprocal(weights)
        log_det_cov = np.sum(np.log(eigvals))
        del eigvals
        log_det_cov = beta * log_det_cov

        return data_term + log_det_cov

    def step_on_stiefel(orth, tau, A):
        if tau == 0.0:
            return orth

        # compute (I + tau / 2 * A)
        mat_Yk = np.empty_like(A, dtype=A.dtype, shape=A.shape)
        np.add(np.identity(A.shape[0], dtype=A.dtype, like=A), tau / 2 * A, out=mat_Yk)

        # compute (I + tau / 2 * A)^{-1} (I - tau / 2 * A) = (I + tau / 2 * A)^{-1} (I + tau / 2 * A)^T
        np.matmul(np.linalg.inv(mat_Yk), np.transpose(mat_Yk), out=mat_Yk)
        # compute step from orth with step size tau
        return np.matmul(mat_Yk, orth)

    def phi(tau, seg_spectra, mean, orth_mat, A, weights, scaling_f, alpha, beta):
        return np.sum(compute_indicators(seg_spectra, mean, step_on_stiefel(orth_mat, tau, A), weights, scaling_f, alpha, beta))
    

    def find_stepsize(A, seg_spectra, mean, orth_mat, weights, scaling_f, alpha, beta):
        """
        This function implements Algorithm 3.1 (Backtracking Line Search) 
        from Numerical Optimization by Nocedal and Wright.
        """
        tau_bar = 1
        rho = 0.75
        c = 0.5

        der_f_tau_0 = -1 / 2 * np.linalg.norm(A, ord='fro')**2

        tau = tau_bar
        while (phi(tau, seg_spectra, mean, orth_mat, A, weights, scaling_f, alpha, beta) >
            phi(0.0, seg_spectra, mean, orth_mat, A, weights, scaling_f, alpha, beta) + c * tau * der_f_tau_0):

            tau = rho * tau

        return tau


    # extract pixels belonging to segment
    if valid_mask is not None:
        # remove pixels that shall not contribute to computation of segment's mean, components and standard deviations
        valid_pixels = image[valid_mask]
        valid_segmentation_mask = seg_mask[valid_mask]
        valid_segment_pixels = valid_pixels[valid_segmentation_mask == label]
    else:
        valid_pixels = image.reshape((image.shape[0] * image.shape[1], image.shape[-1]))
        valid_segmentation_mask = np.ravel(seg_mask)
        valid_segment_pixels = valid_pixels[valid_segmentation_mask == label]

    # initialize mean values for segment and allocate necessary memory
    seg_mean_old = np.full_like(means, np.finfo(np.float32).max, shape=image.shape[-1], dtype=image.dtype)
    weights_old = np.full_like(weights, np.finfo(np.float32).max, shape=image.shape[-1], dtype=image.dtype)
    seg_pcs_old = np.full_like(
        pcs, np.finfo(np.float32).max, shape=(image.shape[-1], image.shape[-1]), dtype=image.dtype
    )
    dists_inv = np.empty_like(image, shape=valid_segment_pixels.shape[0], dtype=image.dtype)
    pxs_centered = np.empty_like(image, shape=valid_segment_pixels.shape, dtype=image.dtype)
    pxs_centered_orth = np.empty_like(image, shape=valid_segment_pixels.shape, dtype=image.dtype)
    pxs_centered_orth_square = np.empty_like(image, shape=valid_segment_pixels.shape, dtype=image.dtype)
    pxs_scaling_factor = np.empty_like(image, shape=valid_segment_pixels.shape, dtype=image.dtype)
    pxs_der_scaling_func = np.empty_like(image, shape=valid_segment_pixels.shape, dtype=image.dtype)
    pcs_scaling_factors = np.empty_like(image, shape=valid_segment_pixels.shape[1], dtype=image.dtype)

    pxs_scaled = np.empty_like(image, shape=valid_segment_pixels.shape, dtype=image.dtype)
    mat_A = np.empty_like(image, shape=(image.shape[-1], image.shape[-1]), dtype=image.dtype)
    t_mat = np.empty_like(image, shape=(image.shape[-1], image.shape[-1]), dtype=image.dtype)
    grad_orth = np.empty_like(image, shape=(image.shape[-1], image.shape[-1]), dtype=image.dtype)

    # TODO: Can we work with references here instead of copying the data? Would that make returning weights unnecessary?
    t_weights = weights[label].copy()
    seg_mean = means[label].copy()
    seg_pcs = pcs[label].copy()

    # set scaling function
    scaling_func = inverse_exp()

    it = 0
    logger.info("Maximum number of fixed point iterations: %s", max_iter)
    logger.info("Stopping threshold: %s", tol)
    logger.info("Scaling factor of weighted trace (alpha): %s", alpha)
    logger.info("Scaling factor of volume regularizer (beta): %s", beta)
    while (
        np.linalg.norm(seg_mean - seg_mean_old)
        + np.linalg.norm(t_weights - weights_old)
        + np.linalg.norm(seg_pcs - seg_pcs_old)
    ) > tol and it < max_iter:
        # store old values of mean, standard deviations and PCs
        np.copyto(seg_mean_old, seg_mean)
        np.copyto(weights_old, t_weights)
        np.copyto(seg_pcs_old, seg_pcs)

        # compute distances wrt to current mean, standard deviations and PCs (term in square root)
        # and store the reciprocals
        np.subtract(valid_segment_pixels, seg_mean[np.newaxis], out=pxs_centered)
        pxs_centered_orth[...] = np.transpose(np.matmul(seg_pcs.T, pxs_centered.T))
        np.square(pxs_centered_orth, out=pxs_centered_orth_square)
        pxs_scaling_factor[...] = scaling_func.eval(pxs_centered_orth_square)
        np.sqrt(
            np.sum(np.multiply(np.add(pxs_centered_orth_square, alpha * pxs_scaling_factor), t_weights), axis=1),
            out=dists_inv,
        )
        np.reciprocal(dists_inv, out=dists_inv)

        ### update mean
        # compute weighted sum of principal components
        pxs_der_scaling_func[...] = scaling_func.derivative(pxs_centered_orth_square)
        np.multiply(pxs_der_scaling_func, pxs_centered_orth, out=pxs_der_scaling_func)
        np.multiply(pxs_der_scaling_func, dists_inv[np.newaxis].T, out=pxs_der_scaling_func)
        np.divide(np.sum(pxs_der_scaling_func, axis=0), np.sum(dists_inv), out=pcs_scaling_factors)
        np.multiply(np.sum(np.multiply(seg_pcs, pcs_scaling_factors), axis=1), alpha, out=seg_mean)

        # compute weighted sum of pixels and add to weighted sum of pcs
        np.multiply(valid_segment_pixels, dists_inv[np.newaxis].T / np.sum(dists_inv), out=pxs_scaled)
        np.add(np.sum(pxs_scaled, axis=0), seg_mean, out=seg_mean)

        ### update eigenvalues/weights lambda (be careful as t_weights stores the reciprocals of the eigenvalues)
        np.add(pxs_centered_orth_square, alpha * pxs_scaling_factor, out=pxs_scaling_factor)
        np.multiply(pxs_scaling_factor, dists_inv[np.newaxis].T, out=pxs_scaling_factor)
        np.sum(pxs_scaling_factor, axis=0, out=t_weights)
        np.divide(t_weights, 2 * beta * valid_segment_pixels.shape[0], out=t_weights)
        np.reciprocal(t_weights, out=t_weights)

        ### update principal components
        # second part of gradient wrt to orthogonal matrix
        np.multiply(pxs_der_scaling_func, t_weights, out=pxs_der_scaling_func)
        # the next broadcast_to command produces a view of pxs_centered, containing for each centered pixel
        # an LxL matrix having the centered pixel in each row
        pxs_centered_broad = np.transpose(
            np.broadcast_to(
                pxs_centered[:, np.newaxis], (pxs_centered.shape[0], pxs_centered.shape[1], pxs_centered.shape[1])
            ),
            (0, 2, 1),
        )
        pxs_der_scaling_func_broad = np.broadcast_to(
            pxs_der_scaling_func[:, np.newaxis],
            (
                pxs_der_scaling_func.shape[0],
                pxs_der_scaling_func.shape[1],
                pxs_der_scaling_func.shape[1],
            ),
        )
        np.sum(pxs_centered_broad * pxs_der_scaling_func_broad, axis=0, out=grad_orth)

        # first part of gradient wrt to orthogonal matrix
        np.multiply(pxs_centered, dists_inv[np.newaxis].T, out=pxs_scaled)
        np.matmul(pxs_scaled.T, pxs_centered, out=t_mat)
        np.matmul(t_mat, seg_pcs_old, out=t_mat)
        np.multiply(t_mat, t_weights, out=t_mat)

        # add both parts of gradient to obtain the final gradient wrt to orthogonal matrix, stored in grad_orth
        np.add(t_mat, alpha * grad_orth, out=grad_orth)

        ### update seg_pcs
        # compute matrix A = grad_orth seg_pcs^T - seg_pcs grad_orth^T, stored in mat_A
        np.matmul(grad_orth, seg_pcs.T, out=mat_A)
        np.subtract(mat_A, mat_A.T, out=mat_A)

        tau = find_stepsize(mat_A, valid_segment_pixels, seg_mean, seg_pcs, t_weights, scaling_func, alpha, beta)
        seg_pcs[...] = step_on_stiefel(seg_pcs, tau, mat_A)
        
        it += 1

    logger.info("Number of iterations needed to find mean, standard deviations and orthogonal matrix: %s", it)
    logger.info("||A in Cayley transformation||_2: %s", np.linalg.norm(mat_A))
    logger.info("det(A) = %s", np.linalg.det(mat_A))
    logger.info("||grad_V on manifold||_2: %s", np.linalg.norm(np.matmul(mat_A, seg_pcs)))
    logger.info("det(V) = %s", np.linalg.det(seg_pcs))

    # delete obsolete variables and free space
    del seg_mean_old
    del weights_old
    del seg_pcs_old
    del dists_inv
    del pxs_centered
    del pxs_centered_orth
    del pxs_centered_orth_square
    del pxs_scaling_factor
    del pxs_der_scaling_func
    del pcs_scaling_factors
    del pxs_scaled
    del mat_A
    del t_mat
    del grad_orth

    ### compute indicator values for all pixels by evaluating the indicator function
    # compute distances wrt to current mean, standard deviations and PCs (Mahalanobis part or data term)
    mahalanobis_term = np.empty_like(image, dtype=image.dtype, shape=image.shape[0] * image.shape[1])
    pxs_centered_im = np.subtract(
        image.reshape((image.shape[0] * image.shape[1], image.shape[2])), seg_mean[np.newaxis]
    )
    pxs_centered_orth_im = np.transpose(np.matmul(seg_pcs.T, pxs_centered_im.T))
    del pxs_centered_im
    pxs_centered_orth_square_im = np.square(pxs_centered_orth_im)
    del pxs_centered_orth_im
    np.sum(np.multiply(pxs_centered_orth_square_im, t_weights), axis=1, out=mahalanobis_term)
    logger.info("min(Mahalanobis term) = %s", np.min(mahalanobis_term))
    logger.info("mean(Mahalanobis term) = %s", np.mean(mahalanobis_term))
    logger.info("max(Mahalanobis term) = %s", np.max(mahalanobis_term))
    logger.info("std(Mahalanobis term) = %s", np.std(mahalanobis_term))

    pxs_scaling_factor_im = scaling_func.eval(pxs_centered_orth_square_im)
    del pxs_centered_orth_square_im
    logger.info("min(Scaling factors weighted trace) = %s", np.min(pxs_scaling_factor_im))
    logger.info("mean(Scaling factors weighted trace) = %s", np.mean(pxs_scaling_factor_im))
    logger.info("max(Scaling factors weighted trace) = %s", np.max(pxs_scaling_factor_im))
    logger.info("std(Scaling factors weighted trace) = %s", np.std(pxs_scaling_factor_im))

    np.multiply(pxs_scaling_factor_im, t_weights, out=pxs_scaling_factor_im)
    logger.info("min(Scaling factors weighted trace / lambda_{l,n}) = %s", np.min(pxs_scaling_factor_im))
    logger.info("mean(Scaling factors weighted trace / lambda_{l,n}) = %s", np.mean(pxs_scaling_factor_im))
    logger.info("max(Scaling factors weighted trace / lambda_{l,n}) = %s", np.max(pxs_scaling_factor_im))
    logger.info("std(Scaling factors weighted trace / lambda_{l,n}) = %s", np.std(pxs_scaling_factor_im))
    
    trace_reg = np.multiply(alpha, np.sum(pxs_scaling_factor_im, axis=1))
    del pxs_scaling_factor_im
    logger.info("min(weighted trace) = %s", np.min(trace_reg))
    logger.info("mean(weighted trace) = %s", np.mean(trace_reg))
    logger.info("max(weighted trace) = %s", np.max(trace_reg))
    logger.info("std(weighted trace) = %s", np.std(trace_reg))

    data_term = np.sqrt(np.add(mahalanobis_term, trace_reg))
    del mahalanobis_term
    del trace_reg

    # compute volume regularizer
    eigvals = np.reciprocal(t_weights)
    log_det_cov = np.sum(np.log(eigvals))
    logger.info("log det Sigma_l = %s", log_det_cov)
    log_det_cov = beta * log_det_cov
    logger.info("beta * log det Sigma_l = %s", log_det_cov)

    logger.info("Smallest eigenvalue of Sigma: %s", np.min(eigvals))
    logger.info("Largest eigenvalue of Sigma: = %s", np.max(eigvals))
    del eigvals
    
    return (data_term.reshape((image.shape[0], image.shape[1])) + log_det_cov), seg_mean, seg_pcs, t_weights


class inverse_exp(object):
    r"""Implementation of the inverse exponential and its derivative

    .. math::

        f(x) = \frac{1}{\exp(x)}
    """

    def __init__(self):
        pass

    def eval(self, x):
        return np.exp(-x)

    def derivative(self, x):
        return -self.eval(x)


def euclidean_norm(image, segmentation_mask, label, valid_mask=None):
    """
    computes for a given segment the indicator function based on the euclidean norm

    Args:
        image: an image as a two- or three-dimensional tensor

        segmentation_mask: a matrix of the same size as the image with integer entries being the labels of
                           the corresponding pixels

        label: the label of the segment of which the mean feature vector shall be computed

        valid_mask: a matrix of the same size as the image with an entry being true if the corresponding pixel
                    shall contribute to the computation of the segment mean

    Returns:
        a matrix of the size as the image where at entry the indicator value of the pixel with respect to the
        considered segment is stored
    """

    if valid_mask is not None:
        # remove pixels that shall not contribute to the computation of the segment's mean
        valid_pixels = image[valid_mask]
        valid_segmentation = segmentation_mask[valid_mask]
    else:
        valid_pixels = image
        valid_segmentation = segmentation_mask

    segment_mean = compute_segment_mean(valid_pixels, valid_segmentation, label)

    return np.sum(np.square(image - segment_mean), axis=2)


def anisotropic_2norm(
    image,
    segmentation_mask,
    label,
    eps,
    eps_inverse=True,
    valid_mask=None,
    algo_means="arithmetic",
    algo_comps="pca",
    algo_vars="sdm",
    trim_proportion=0.5,
):
    """
    computes for a given segment the indicator function based on the anisotropic 2-norm that is regularized with
    epsilon or its reciprocal 1 / epsilon.

    Args:
        image: an image as a two- or three-dimensional tensor

        segmentation_mask: a matrix of the same size as the image with integer entries being the labels of
                           the corresponding pixels
        label: the label of the segment of which the mean feature vector shall be computed

        eps: regularization parameter used to take care of directions with very low standard deviation

        eps_inverse: if true, weight directions with standard deviation lower than epsilon with 1 / epsilon.
                     weight with epsilon otherwise.

        valid_mask: a matrix of the same size as the image with an entry being true if the corresponding pixel
                    shall contribute to the computation of the segment mean and standard deviations

        algo_means: method to compute the mean feature vector of the segment

        algo_comps: method with which the components are computed

        algo_vars: method to compute the variances in directions of the components

        trim_proportion: amount of points that will be considered outliers when trimming is applied

    Returns:
        a matrix of the size as the image where at entry the indicator value of the pixel with respect to the
        considered segment is stored
    """
    logger = logging.getLogger("indicator")

    if valid_mask is not None:
        # remove pixels that shall not contribute to computation of segment's mean, components and standard deviations
        valid_pixels = image[valid_mask]
        valid_segmentation_mask = segmentation_mask[valid_mask]
        valid_segment_pixels = valid_pixels[valid_segmentation_mask == label]
    else:
        valid_pixels = image.reshape((image.shape[0] * image.shape[1], image.shape[-1]))
        valid_segmentation_mask = np.ravel(segmentation_mask)
        valid_segment_pixels = valid_pixels[valid_segmentation_mask == label]

    # compute mean feature vector
    if algo_means == "trimmed":
        # compute (trimmed) mean of segment
        logger.info("Mean feature vector: trimmed")
        logger.info("Trim proportion for mean: %s", trim_proportion)
        segment_mean = trim_mean(valid_segment_pixels, proportiontocut=trim_proportion, axis=0)
    elif algo_means == "iterative":
        # compute (iterative) mean of segment
        logger.info("Mean feature vector: iterative")
        segment_mean = compute_segment_mean(valid_pixels, valid_segmentation_mask, label)

        segment_mean = trim_mean(valid_segment_pixels, proportiontocut=trim_proportion, axis=0)
    else:
        # compute mean of segment
        logger.info("Mean feature vector: full")
        segment_mean = compute_segment_mean(valid_pixels, valid_segmentation_mask, label)

    # compute components
    if algo_comps == "tga":
        logger.info("Components: TGA")
        logger.info("Trim proportion for TGA: %s", trim_proportion)
    elif algo_comps == "mnf":
        logger.info("Components: MNF")
        # compute MNF components
        logger.info("Noise estimation for MNF: shift differences within bands")
        logger.info("Shift direction: lower")
        cov_noise = noise_from_diffs(valid_segment_pixels[:, np.newaxis], direction="lower")
        cov_data = calc_stats(valid_segment_pixels[:, np.newaxis])
        components = mnf(cov_data, cov_noise).napc.eigenvectors
        valid_pixels_t = np.matmul(valid_segment_pixels, components)
        variances = np.var(valid_pixels_t, axis=0)
    else:
        # compute PCA components
        logger.info("Components: PCA")
        variances, components = pca(valid_segment_pixels.transpose())

    # if variances are computed on trimmed data, overwrite given variances by algorithms for components
    if algo_vars == "trimmed":
        logger.info("Variances: on trimmed data")
        logger.info("Trim proportion for variances: %s", trim_proportion)
        # project data onto components
        segment_pixels_c = valid_segment_pixels - trim_mean(valid_segment_pixels, trim_proportion, axis=0)
        segment_pixels_t = np.matmul(segment_pixels_c, components)
        # trim transformed data for each component
        segment_pixels_t_trim = trimboth(segment_pixels_t, trim_proportion, axis=0)
        variances = np.var(segment_pixels_t_trim, axis=0)
    else:
        logger.info("Variances: full")

    # compute standard deviations from variances
    std_devs = np.sqrt(np.maximum(variances, 0.0))

    components_mask = std_devs < eps
    logger.info("Components with standard deviation smaller epsilon: %s", components_mask.sum())
    logger.info("Explained variances: %s", variances)

    # weight directions with standard deviation smaller than epsilon with 1 / epsilon or epsilon
    weights = 1.0 / np.maximum(std_devs, eps)
    if not eps_inverse:
        weights[std_devs < eps] = eps

    # compute logarithm of determinant of covariance matrix
    log_det_cov = np.log(np.square(1 / np.prod(weights)))

    return (
        m_anisotropic_2norm(
            (image.reshape((image.shape[0] * image.shape[1], image.shape[2])) - segment_mean).transpose(),
            components,
            weights,
            squared=True,
        ).reshape((image.shape[0], image.shape[1]))
        + log_det_cov
    )


def pca_eps_discard_norm(image, segmentation_mask, label, eps, valid_mask=None):
    """
    computes for a given segment the indicator function based on the PCA norm that discards principal components
    with variance smaller than epsilon

    Args:
        image: an image as a two- or three-dimensional tensor

        segmentation_mask: a matrix of the same size as the image with integer entries being the labels of
                           the corresponding pixels
        label: the label of the segment of which the mean feature vector shall be computed

        valid_mask: a matrix of the same size as the image with an entry being true if the corresponding pixel
                    shall contribute to the computation of the segment mean and standard deviations

    Returns:
        a matrix of the size as the image where at entry the indicator value of the pixel with respect to
        the considered segment is stored
    """
    logger = logging.getLogger("indicator")

    if valid_mask is not None:
        # remove pixels that shall not contribute to computation of segment's mean, PCs and standard deviations
        valid_pixels = image[valid_mask]
        valid_segmentation_mask = segmentation_mask[valid_mask]
    else:
        valid_pixels = image
        valid_segmentation_mask = segmentation_mask

    segment_mean = compute_segment_mean(valid_pixels, valid_segmentation_mask, label)
    variances, components = pca(valid_pixels[valid_segmentation_mask == label].transpose())
    std_devs = np.sqrt(np.maximum(variances, 0))
    components_mask = std_devs > eps

    logger.info(
        "Components with standard deviation smaller than epsilon: %s", image[0, 0].shape[0] - components_mask.sum()
    )
    if components_mask.sum() == 0:
        logger.info("Keep components with largest standard deviations.")

        return m_anisotropic_2norm(
            (image.reshape((image.shape[0] * image.shape[1], image.shape[2])) - segment_mean).transpose(),
            components[0][np.newaxis].transpose(),
            1 / std_devs[0][np.newaxis],
            squared=True,
        ).reshape((image.shape[0], image.shape[1]))
    else:
        return m_anisotropic_2norm(
            (image.reshape((image.shape[0] * image.shape[1], image.shape[2])) - segment_mean).transpose(),
            components[components_mask].transpose(),
            1 / std_devs[components_mask],
            squared=True,
        ).reshape((image.shape[0], image.shape[1]))


def compute_kernel_matrix(image, kernel_func="gaussian", params=0.1):
    """
    Computes the kernel (Gram) matrix of the pixels of an image with respect to a kernel function

    Args:
        image: image [im_height x im_width x num_channels]

        kernel: the chosen kernel function, choices:
                ['gaussian', 'polynomial', 'chi-squared', 'laplacian', 'cosine', 'sigmoid', 'direct-summation',
                'weighted-summation', 'cross-information']

        params: parameters of kernel function

    Returns:
        kernel matrix of the pairwise kernel evaluations
    """
    logger = logging.getLogger("indicator")

    def compute_spatial_features(image, params):
        # compute spatial features as channel-wise average of a t x t window with the pixel as the center
        # padding is done by replicating the nearest pixel value
        logger.info("Spatial features: mean of local neighborhood of size %s x %s", params, params)
        logger.info("Spatial features: pad by replicating the value of the last pixel.")
        return uniform_filter(image, (params, params, 1), mode="nearest")

    # compute kernel (Gram) matrix g (num pixels x num pixels) dependent on chosen kernel
    if kernel_func == "gaussian":
        logger.info("Kernel: gaussian")
        logger.info("Kernel parameter gamma: %s", params[0])
        g = pairwise_kernels(
            image.reshape((-1, image.shape[-1])),
            image.reshape((-1, image.shape[-1])),
            metric="rbf",
            n_jobs=-1,
            gamma=params[0],
        )
    elif kernel_func == "polynomial":
        logger.info("Kernel: polynomial")
        logger.info("Kernel parameter gamma: %s", params[0])
        logger.info("Kernel parameter c_0: %s", params[1])
        logger.info("Kernel parameter d: %s", params[2])
        # parameters of polynomial kernel named according to
        # https://scikit-learn.org/stable/modules/metrics.html#polynomial-kernel
        # parameters named different in thesis
        g = pairwise_kernels(
            image.reshape((-1, image.shape[-1])),
            image.reshape((-1, image.shape[-1])),
            metric="polynomial",
            n_jobs=-1,
            gamma=params[0], # in thesis: \delta
            coef0=params[1], # in thesis: c
            degree=params[2], # in thesis: p
        )
    elif kernel_func == "chi-squared":
        logger.info("Kernel: chi-squared")
        logger.info("Kernel parameter gamma: %s", params[0])
        g = pairwise_kernels(
            image.reshape((-1, image.shape[-1])),
            image.reshape((-1, image.shape[-1])),
            metric="chi2",
            n_jobs=-1,
            gamma=params[0],
        )
    elif kernel_func == "laplacian":
        logger.info("Kernel: laplacian")
        logger.info("Kernel parameter gamma: %s", params[0])
        g = pairwise_kernels(
            image.reshape((-1, image.shape[-1])),
            image.reshape((-1, image.shape[-1])),
            metric="laplacian",
            n_jobs=-1,
            gamma=params[0],
        )
    elif kernel_func == "cosine":
        logger.info("Kernel: cosine")
        g = pairwise_kernels(
            image.reshape((-1, image.shape[-1])), image.reshape((-1, image.shape[-1])), metric="cosine", n_jobs=-1
        )
    elif kernel_func == "sigmoid":
        logger.info("Kernel: sigmoid")
        logger.info("Kernel parameter gamma: %s", params[0])
        logger.info("Kernel parameter c_0: %s", params[1])
        g = pairwise_kernels(
            image.reshape((-1, image.shape[-1])),
            image.reshape((-1, image.shape[-1])),
            metric="sigmoid",
            n_jobs=-1,
            gamma=params[0],
            coef0=params[1],
        )
    elif kernel_func == "direct-summation":
        logger.info("Kernel: direct summation")

        # compute spatial features to input into the kernel
        im_spat = compute_spatial_features(image, params[0])

        logger.info("Spatial kernel: gaussian")
        logger.info("Spatial kernel parameter gamma: %s", params[1])
        logger.info("Spectral kernel: polynomial")
        logger.info("Spectral kernel parameter delta: %s", params[2])
        logger.info("Spectral kernel parameter c: %s", params[3])
        logger.info("Spectral kernel parameter p: %s", params[4])

        # compute the direct summation kernel
        g = direct_summation(
            im_spat.reshape((-1, im_spat.shape[-1])),
            image.reshape((-1, image.shape[-1])),
            im_spat.reshape((-1, im_spat.shape[-1])),
            image.reshape((-1, image.shape[-1])),
            spat_args=params[1],
            spec_args=params[2:], # in thesis: delta, c, p, in scikit-learn: gamma, coeff0, degree
        )

    elif kernel_func == "weighted-summation":
        logger.info("Kernel: weighted summation")

        # compute spatial features to input into the kernel
        im_spat = compute_spatial_features(image, params[0])

        logger.info("Weight mu: %s", params[1])
        logger.info("Spatial kernel: gaussian")
        logger.info("Spatial kernel parameter gamma: %s", params[2])
        logger.info("Spectral kernel: polynomial")
        logger.info("Spectral kernel parameter gamma: %s", params[3])
        logger.info("Spectral kernel parameter c_0: %s", params[4])
        logger.info("Spectral kernel parameter d: %s", params[5])

        # compute the weighted summation kernel
        g = weighted_summation(
            im_spat.reshape((-1, im_spat.shape[-1])),
            image.reshape((-1, image.shape[-1])),
            im_spat.reshape((-1, im_spat.shape[-1])),
            image.reshape((-1, image.shape[-1])),
            alpha=params[1],
            spat_args=params[2],
            spec_args=params[3:],
        )

    elif kernel_func == "cross-information":
        logger.info("Kernel: cross-information")

        # compute spatial features to input into the kernel
        im_spat = compute_spatial_features(image, params[0])

        logger.info("Spatial kernel: gaussian")
        logger.info("Spatial kernel parameter gamma: %s", params[1])
        logger.info("Spectral kernel: polynomial")
        logger.info("Spectral kernel parameter gamma: %s", params[2])
        logger.info("Spectral kernel parameter c_0: %s", params[3])
        logger.info("Spectral kernel parameter d: %s", params[4])

        # compute the cross information kernel
        g = cross_information(
            im_spat.reshape((-1, im_spat.shape[-1])),
            image.reshape((-1, image.shape[-1])),
            im_spat.reshape((-1, im_spat.shape[-1])),
            image.reshape((-1, image.shape[-1])),
            spat_args=params[1],
            spec_args=params[2:],
        )

    else:
        logger.info("Kernel function not implemented!")
        raise NotImplementedError("Kernel function not implemented!")

    return g


def kMS(g, segmentation_mask, label, valid_mask=None):
    """
    Implements the kernelized version of the 2-norm as indicator function, called kMS.
    It applies the kernel trick to the 2-norm as indicator function and hence computes the indicator function
    in the feature space that is associated to the chosen kernel (the reproducing kernel Hilbert space)

    Args:
        g: the kernel (Gram) matrix corresponding to the image pixels

        segmentation_mask: a matrix of the same size as the image with integer entries being the labels of
                           the corresponding pixels

        label: the label of the segment of which the mean feature vector shall be computed

        valid_mask: a matrix of the same size as the image with an entry being true if the corresponding pixel
                    shall contribute to the computation of the segment mean

    Returns:
        a matrix of the size as the image where at each entry the indicator value of the pixel with respect to the
        considered segment is stored
    """

    # create mask that is true for every pixel that belongs to the segment and
    # shall contribute to computation of the mean of the segment
    if valid_mask is not None:
        valid_segment_pixel_mask = np.zeros_like(g, shape=segmentation_mask.shape, dtype=bool)
        valid_segment_pixel_mask[np.logical_and(valid_mask, segmentation_mask == label)] = True
    else:
        valid_segment_pixel_mask = segmentation_mask == label

    valid_segment_pixel_mask = valid_segment_pixel_mask.reshape(np.prod(valid_segment_pixel_mask.shape))

    # determine number of pixels contributing to mean of segment
    n = float((valid_segment_pixel_mask).sum())

    # compute first term: k(x, x)
    # copying the diagonal is necessary since otherwise the following operations would manipulate
    # the diagonal of the kernel matrix when working in-place
    h = np.diagonal(g).copy()

    # compute second term: -2/n sum_{i=1}^n k(x_i, x)
    np.subtract(h, 2 / n * np.sum(g[valid_segment_pixel_mask], axis=0), out=h)

    # compute third term: 1/n^2 sum_{i,j=1}^n k(x_i, x_j)
    np.add(h, 1 / n ** 2 * np.sum(g[valid_segment_pixel_mask][:, valid_segment_pixel_mask]), out=h)

    return h.reshape((segmentation_mask.shape[0], segmentation_mask.shape[1]))

