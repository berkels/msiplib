# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ,unused-argument

''' collection of decompositions '''

import numbers
import warnings

import numpy as np
from scipy import linalg, stats
from scipy.sparse.linalg import eigsh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer
from sklearn.utils.extmath import randomized_svd, stable_cumsum, svd_flip
from sklearn.utils.validation import (as_float_array, check_array,
                                      check_is_fitted, check_random_state)


def pca(samples, weights=None):
    '''
        function computes the Principal Components Analysis for a set of samples

        Input:
            samples: array of vectors given in shape num_features x num_samples
            weights: if weights are given, weight every sample accordingly in mean and covariance

        Returns:
            Variances in the directions of principal components as 1D array
            and the principal components as 2D array with PCs as columns
    '''
    assert samples.shape[1] > 1, 'PCA needs a list of at least 2 samples as input.'

    # TODO: Test correctness
    if weights is not None:
        assert (weights < 0.0).sum() == 0, 'Weights have to be non-negative.'

        w_sum = weights.sum()
        if w_sum != 1.0:
            w = weights / w_sum
        else:
            w = weights

    # center samples around origin and compute covariance matrix
    if weights is not None:
        x_centered = samples - np.average(samples, axis=1, weights=w)[:, np.newaxis]
        cov = np.matmul(w * x_centered, x_centered.transpose())
    else:
        x_centered = samples - samples.mean(axis=1)[:, np.newaxis]
        cov = 1.0 / (samples.shape[1] - 1.0) * np.matmul(x_centered, x_centered.transpose())

    # eigenvalue decomposition of cov
    variances, principal_components = np.linalg.eigh(cov)

    # return eigenvalues as variances and eigenvectors as principal components
    return np.maximum(0.0, variances), principal_components

