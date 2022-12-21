#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=not-an-iterable

''' Collection of indicator functions used for image segmentation '''

import logging
import numpy as np
from numba import jit, prange
from scipy.stats import trimboth, trim_mean
from sklearn.metrics.pairwise import pairwise_kernels
from spectral import calc_stats, noise_from_diffs, mnf
from torch.cuda import is_available
from msiplib.decomposition import pca
from msiplib.metrics import anisotropic_2norm as m_anisotropic_2norm


def compute_segment_mean(image, seg_mask, label):
    '''
    computes mean feature vector of a segment in an image

    Args:
        image: an image as a two- or three-dimensional tensor

        seg_mask: a matrix of the same size as the image with integer entries being the labels
                  of the corresponding pixels

        label: the label of the segment of which the mean feature vector shall be computed

    Returns:
        a vector of the same dimension as the feature vectors of the pixels in the image being the mean feature of
        the segment
    '''
    return np.mean(image[seg_mask == label], axis=0)


def nonsquared_anistropic_2norm(image, seg_mask, label, eps, means, pcs, weights, tol=1e-05, max_iter=100, valid_mask=None):
    '''
    computes for a given segment the indicator function based on the non-squared anisotropic 2-norm that is
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
    '''

    logger = logging.getLogger('indicator')

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
    seg_mean_old = np.full(image.shape[-1], np.finfo(np.float32).max, dtype=image.dtype)
    weights_old = np.full(image.shape[-1], np.finfo(np.float32).max, dtype=image.dtype)
    seg_pcs_old = np.full((image.shape[-1], image.shape[-1]), np.finfo(np.float32).max, dtype=image.dtype)
    dists_inv = np.empty(valid_segment_pixels.shape[0], dtype=image.dtype)
    pxs_centered = np.empty(valid_segment_pixels.shape, dtype=image.dtype)
    pxs_scaled = np.empty(valid_segment_pixels.shape, dtype=image.dtype)
    cov = np.empty((image.shape[-1], image.shape[-1]), dtype=image.dtype)

    # TODO: Can we work with references here instead of copying the data? Would that make returning weights unnecessary?
    t_weights = weights[label].copy()
    seg_mean = means[label].copy()
    seg_pcs = pcs[label].copy()

    it = 0
    logger.info('Regularize 2-norm with tau: %s', tau)
    logger.info('Maximum number of fixed point iterations: %s', max_iter)
    logger.info('Stopping threshold: %s', tol)
    # TODO: can we take into account the standard deviations in the stopping criterion?
    # Unclear since these are regularized and might therefore cause large changes in criterion
    while ((np.linalg.norm(seg_mean - seg_mean_old)
           + np.linalg.norm(t_weights - weights_old)
           + np.linalg.norm(seg_pcs - seg_pcs_old)) > tol
           and it < max_iter):

        # store old values of mean, standard deviations and PCs
        np.copyto(seg_mean_old, seg_mean)
        np.copyto(weights_old, t_weights)
        np.copyto(seg_pcs_old, seg_pcs)

        # compute distances wrt to current mean, standard deviations and PCs and store the reciprocals
        dists_inv[...] = m_anisotropic_2norm((valid_segment_pixels - seg_mean[np.newaxis]).T, seg_pcs, t_weights,
                                             squared=False, tau=tau)
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

    logger.info('Number of iterations needed to find mean and covariance: %s', it)
    logger.info('Components with standard deviation smaller than epsilon: %s', np.sum(t_weights == 1 / eps))

    # compute logarithm of determinant of covariance matrix
    log_det_cov = -2 * np.sum(np.log(t_weights))

    return (m_anisotropic_2norm((image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
                                 - seg_mean[np.newaxis])
            .transpose(), seg_pcs, t_weights, squared=False, tau=tau).reshape((image.shape[0], image.shape[1]))
            + log_det_cov
           ), seg_mean, seg_pcs, t_weights


def euclidean_norm(image, segmentation_mask, label, valid_mask=None):
    '''
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
    '''

    if valid_mask is not None:
        # remove pixels that shall not contribute to the computation of the segment's mean
        valid_pixels = image[valid_mask]
        valid_segmentation = segmentation_mask[valid_mask]
    else:
        valid_pixels = image
        valid_segmentation = segmentation_mask

    segment_mean = compute_segment_mean(valid_pixels, valid_segmentation, label)

    return np.sum(np.square(image - segment_mean), axis=2)


def anisotropic_2norm(image, segmentation_mask, label, eps, eps_inverse=True, valid_mask=None, algo_means='arithmetic',
                      algo_comps='pca', algo_vars='sdm', trim_proportion=0.5):
    '''
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
    '''
    logger = logging.getLogger('indicator')

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
    if algo_means == 'trimmed':
        # compute (trimmed) mean of segment
        logger.info('Mean feature vector: trimmed')
        logger.info('Trim proportion for mean: %s', trim_proportion)
        segment_mean = trim_mean(valid_segment_pixels, proportiontocut=trim_proportion, axis=0)
    elif algo_means == 'iterative':
        # compute (iterative) mean of segment
        logger.info('Mean feature vector: iterative')
        segment_mean = compute_segment_mean(valid_pixels, valid_segmentation_mask, label)

        segment_mean = trim_mean(valid_segment_pixels, proportiontocut=trim_proportion, axis=0)
    else:
        # compute mean of segment
        logger.info('Mean feature vector: full')
        segment_mean = compute_segment_mean(valid_pixels, valid_segmentation_mask, label)

    # compute components
    if algo_comps == 'tga':
        logger.info('Components: TGA')
        logger.info('Trim proportion for TGA: %s', trim_proportion)
    elif algo_comps == 'mnf':
        logger.info('Components: MNF')
        # compute MNF components
        logger.info('Noise estimation for MNF: shift differences within bands')
        logger.info('Shift direction: lower')
        cov_noise = noise_from_diffs(valid_segment_pixels[:, np.newaxis], direction='lower')
        cov_data = calc_stats(valid_segment_pixels[:, np.newaxis])
        components = mnf(cov_data, cov_noise).napc.eigenvectors
        valid_pixels_t = np.matmul(valid_segment_pixels, components)
        variances = np.var(valid_pixels_t, axis=0)
    else:
        # compute PCA components
        logger.info('Components: PCA')
        variances, components = pca(valid_segment_pixels.transpose())

    # if variances are computed on trimmed data, overwrite given variances by algorithms for components
    if algo_vars == 'trimmed':
        logger.info('Variances: on trimmed data')
        logger.info('Trim proportion for variances: %s', trim_proportion)
        # project data onto components
        segment_pixels_c = valid_segment_pixels - trim_mean(valid_segment_pixels, trim_proportion, axis=0)
        segment_pixels_t = np.matmul(segment_pixels_c, components)
        # trim transformed data for each component
        segment_pixels_t_trim = trimboth(segment_pixels_t, trim_proportion, axis=0)
        variances = np.var(segment_pixels_t_trim, axis=0)
    else:
        logger.info('Variances: full')

    # compute standard deviations from variances
    std_devs = np.sqrt(np.maximum(variances, 0.0))

    components_mask = (std_devs < eps)
    logger.info('Components with standard deviation smaller epsilon: %s', components_mask.sum())
    logger.info('Explained variances: %s', variances)

    # weight directions with standard deviation smaller than epsilon with 1 / epsilon or epsilon
    weights = 1.0 / np.maximum(std_devs, eps)
    if not eps_inverse:
        weights[std_devs < eps] = eps

    # compute logarithm of determinant of covariance matrix
    log_det_cov = np.log(np.square(1 / np.prod(weights)))

    return (m_anisotropic_2norm((image.reshape((image.shape[0] * image.shape[1], image.shape[2])) - segment_mean)
            .transpose(), components, weights, squared=True).reshape((image.shape[0], image.shape[1]))
            + log_det_cov
           )

def pca_eps_discard_norm(image, segmentation_mask, label, eps, valid_mask=None):
    '''
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
    '''
    logger = logging.getLogger('indicator')

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
    components_mask = (std_devs > eps)

    logger.info('Components with standard deviation smaller than epsilon: %s',
                image[0, 0].shape[0] - components_mask.sum())
    if components_mask.sum() == 0:
        logger.info('Keep components with largest standard deviations.')

        return m_anisotropic_2norm((image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
                                          - segment_mean).transpose(), components[0][np.newaxis].transpose(),
                                         1 / std_devs[0][np.newaxis], squared=True
                                        ).reshape((image.shape[0], image.shape[1]))
    else:
        return m_anisotropic_2norm((image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
                                          - segment_mean).transpose(), components[components_mask].transpose(),
                                         1 / std_devs[components_mask], squared=True
                                        ).reshape((image.shape[0], image.shape[1]))

