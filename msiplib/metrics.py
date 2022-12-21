'''
    Module for metrics used in signal and image processing
'''

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score
from numba import jit
from msiplib.segmentation import convert_segmentation_to_partition


def anisotropic_2norm(x, directions, weights, squared=False, tau=None):
    '''
        computes an anisotropic 2-norm that weights the given directions with the given weights.

        Input:
            x: a list of vectors (vectors build the columns of the numpy array) that shall be measured
            directions: a list of dim(x) orthonormal directions with the directions being the columns of a numpy array
            weights: a vector of dim(x) nonnegative scalars providing the weights of the directions
            squared: flag. If true, the squared anisotropic-2-norm is computed.
            tau: a non-negative scalar that is added under the square root to regularize the 2-norm in 0

        Returns:
            weighted lengths of the vectors in x
    '''

    norm_squared = np.sum(np.square((np.matmul(np.transpose(directions), x).transpose() * weights).transpose()), axis=0)

    if squared:
        return norm_squared
    else:
        if tau is None or tau == 0.0:
            return np.sqrt(norm_squared)
        else:
            if tau < 0.0:
                raise ValueError('Tau has to be non-negative.')
            return np.sqrt(norm_squared + tau)

def mae(k, l):
    '''
        Computes the mean absolute error (MAE) of two matrices

        Args:
            k: first matrix
            l: second matrix of the same dimension
        Returns:
            a nonnegative scalar
    '''

    return np.mean(np.abs(k - l))

def rmse(k, l):
    '''
        Computes the overall root mean square error (RMSE) of two matrices

        Args:
            two matrices of the same dimension
        Returns:
            a nonnegative scalar
    '''

    return np.mean(np.sqrt(np.mean(np.square(k - l), axis=0)))


def sad(v, w):
    '''
        Computes the spectral angle distance between two vectors.

        Args:
            two vectors of the same dimension
        Returns:
            a scalar in :math:`[0, \\pi]`
    '''

    return np.arccos(np.dot(v / np.linalg.norm(v), w / np.linalg.norm(w)))

@jit(nopython=True, cache=True)
def angular_similarity(v, w):
    '''
        Computes the angular similarity between two vectors v and w
        as defined by Sun et al. in their paper
        'Band Selection Using Improved Sparse Subspace Clustering for Hyperspectral Imagery Classification'

        Args:
            two (non-zero) vectors of the same dimension
        Returns:
            non-negative scalar
    '''
    return np.square(np.dot(v, w) / (np.square(np.linalg.norm(v)) * np.square(np.linalg.norm(w))))


def asam(k, l):
    '''
        Computes the mean spectral angle distance (aSAD), also called mean spectral angle mapper (aSAM),
        for two collections of vectors of the same dimension. It is assumed that the vectors are the
        columns of the matrices.

        Args:
            two matrices of the same dimension
        Returns:
            the mean SAD or SAM in the interval :math:`[0, \\pi]`
    '''

    # normalize columns (the vectors) of both matrices
    normalized_k = k / np.sqrt(np.sum(np.square(k), axis=0))[np.newaxis]
    normalized_l = l / np.sqrt(np.sum(np.square(l), axis=0))[np.newaxis]

    # compute inner products of columns of matrices, compute the arccos and return the resulting mean
    return np.mean(np.arccos(np.sum(normalized_k * normalized_l, axis=0)))


def variation_of_information(x, y):
    '''
        Compares two partitions x and y of a set A using the variation of information.
        for more information see https://en.wikipedia.org/wiki/Variation_of_information.

        Args:
            x: a list of disjunct unique lists representing a partition of a set
            y: a list of disjunct unique lists representing another partition of the same set
        Returns :
            variation of information of the two partitions. the variation of information lies in the interval
            :math:`[0, \\log(|A|)]`
    '''

    n = sum([len(xi) for xi in x])

    assert n == sum([len(yj) for yj in y]), 'x and y are partitions of different sets.'

    pi = np.array([len(xi) / n for xi in x])
    qj = np.array([len(yj) / n for yj in y])
    rij = np.array([len(np.intersect1d(xi, yj, assume_unique=True)) / n for xi in x for yj in y])
    rij = np.reshape(rij, (len(x), len(y)))
    rij_pi = rij / pi[np.newaxis].T
    rij_qj = rij / qj

    mask_rij = (rij > 0)
    return -np.sum(rij[mask_rij] * (np.log2(rij_pi[mask_rij]) + np.log2(rij_qj[mask_rij])))


def segmentation_scores(segmentation, groundtruth, ignore_label=None, return_perm=False):
    '''
        Computes the segmentation scores overall accuracy, mean class accuracy, mean intersection over union (mIoU),
        mean dice coefficient, kappa coefficient and variation of information
        for a given segmentation and a corresponding ground truth.

        Args:
            segmentation: a segmentation of an image as a matrix where each entry contains an integer
                          indicating the segment number of the pixel
            groundtruth: the corresponding ground truth, also given as a matrix with each entry
                         being the number of the segment of the pixel
            ignore_label: an integer. every pixel carrying this label in the ground truth is ignored
                          when computing the segmentation scores
            return_perm: boolean to specify whether permutation to get optimal matching of labelings should be returned
        Returns:
            a list containing the computed scores
    '''

    # remove the pixels that shall be ignored and change labels of ground truth such that it has labels starting
    # from 0 up to 'number of segments in groundtruth - 1'
    # the labels in the segmentation must not be rebuild here because this permutes the labeling of the segmentation
    # with the consequence that the returned permutation does not correspond to the original input segmentation
    if ignore_label is not None:
        mask = (groundtruth != ignore_label)
        seg = segmentation[mask]
        t_gt = groundtruth[mask]
        gt_labels = np.unique(t_gt)
        gt = np.zeros(t_gt.shape[0], dtype=groundtruth.dtype)
        for i in range(gt_labels.shape[0]):
            gt[t_gt == gt_labels[i]] = i
    else:
        seg = segmentation
        gt = groundtruth

    # get number of segments in the ground truth
    gt_labels = np.unique(gt.astype(np.uint8))
    num_gt_segments = gt_labels.shape[0]

    # get the confusion matrix and get the permutation
    y_pred = seg.reshape(-1)
    y = gt.reshape(-1)
    conf_mat = confusion_matrix(y, y_pred)

    # apply the Hungarian method to find the permutation of columns that yields the highest sum on the diagonal
    ind = linear_sum_assignment(conf_mat, maximize=True)

    # permute the columns according to the permutation given by the Hungarian method
    # ind[1] contains the indices of the columns (labels) in order to maximize the sum on the diagonal
    # hence, entry ind[1][j] contains the label that has to mapped to label j, in other words
    # the permutation is given as ind[1][j] -> j
    hist = conf_mat[:, ind[1]]

    # compute final scores
    final_scores = \
            {'overallAcc': np.diag(hist).sum() / hist.sum(),
             'meanClassAcc': np.nanmean(np.diag(hist) / hist.sum(axis=1)),
             'mIoU': \
                 np.nanmean(np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0)[:num_gt_segments] - np.diag(hist))),
             'mDice': np.nanmean(2 * np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0)[:num_gt_segments]))
            }

    # kappa coefficient needs some extra lines to be computed
    n_0 = np.sum(hist, axis=1)
    n_1 = np.sum(hist, axis=0)
    l = min(len(n_0), len(n_1))
    p_e = 1 / (np.sum(n_0)**2) * np.sum(n_0[: l] * n_1[: l])
    final_scores['kappa'] = (final_scores['overallAcc'] - p_e) / (1 - p_e)

    # compute variation of information, the lower the VI score the better
    # minimum is 0, maximum is log2(num_pixels)
    seg_partition = convert_segmentation_to_partition(seg)
    gt_partition = convert_segmentation_to_partition(gt)
    final_scores['VI'] = variation_of_information(seg_partition, gt_partition)
    final_scores['NMI'] = normalized_mutual_info_score(y, y_pred)
    final_scores['ARS'] = adjusted_rand_score(y, y_pred)

    if return_perm:
        # the inverse permutation of ind[1] has to be returned because ind[1] gives the permutation as indices, meaning
        # that the j-th entry contains the index that has to be mapped onto j.
        return final_scores, inverse_permutation(ind[1])
    else:
        return final_scores

def inverse_permutation(perm):
    '''
        returns the inverse permutation
        Args:
            perm: a permutation given as j -> perm[j]
        Returns:
            inverse permutation inverting perm
    '''
    inverse = np.zeros(perm.shape[0], dtype=np.uint16)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse
