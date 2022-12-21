'''
    Hyperspectral image segmentation

    Jan-Christopher Cohrs, Chandrajit Bajaj and Benjamin Berkels. A distribution-dependent
    Mumford-Shah model for unsupervised hyperspectral image segmentation. Accepted for publication
    in IEEE Transactions on Geoscience and Remote Sensing, 2022. [https://arxiv.org/abs/2203.15058]
'''

import os
from scipy.io import loadmat
from requests import get
from msiplib.segmentation.hsi.hsi_ms import ms_segmentation
from msiplib.io.segmentation import saveColoredSegmentation

def main():
    ''' main function '''

    # download data file and store it as netCDF
    url_data = 'https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat'
    dest_data = './IndianPines.mat'
    response = get(url_data, timeout=10)
    open(dest_data, "wb").write(response.content)


    # download ground truth and store it as png file
    url_gt = 'https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'
    dest_gt = './IndianPines_gt.mat'
    response_gt = get(url_gt, timeout=10)
    open(dest_gt, "wb").write(response_gt.content)

    # store ground truth as png file
    # if in args below the gt_file is chosen as './IndianPines_gt.png',
    # the colors given by the png file are used to visualize the segmentation
    gt = loadmat(dest_gt)['indian_pines_gt']
    saveColoredSegmentation(gt, './IndianPines_gt.png', None, 0, False, gt)

    # choose parameter setting to run algorithm with.
    args = {
        # path to configuration file (str). if provided, overrides all following choices
        'config_path': None,
        # path to image file (str)
        'filename': os.path.expandvars('./IndianPines.mat'),
        # path to ground truth file (str)
        'gt_file': os.path.expandvars('./IndianPines_gt.mat'),
        # number of segments to be sought (int)
        'k': 16,
        # regularization parameter, balances data and regularization term (float)
        'reg_par': 0.24,
        # if true, add additional segment to collect irregular points
        'irreg': False,
        # radius of balls used to find irregular segments (int)
        'r': 1,
        # epsilon to compute indicator values for irregular region (float)
        'irreg_eps': 1e-06,
        # indicator function (str)
        # choices: ['2', 'anisotropic-eps-inverse', 'anisotropic-eps-normal',
        # 'anisotropic-eps-discard', 'non-squared-anisotropic-eps-inverse']
        'ind_func': 'non-squared-anisotropic-eps-inverse',
        # epsilon to regularize indicator function (float)
        'ind_eps': 0.125,
        # kernel function (str).
        'kernel': 'None',
        # list of parameters for indicator function
        # if indicator function has several parameters, give them with a list [value1, value2,...]
        # for indicator function proposed in the above mentioned paper,
        # this specifies the maximum number of fixed point iterations
        'ind_params': [20],
        # convexification of Mumford-Shah functional, only Zach implemented (str)
        'convexification': 'zach',
        # type of primal dual hybrid gradient (PDHG) algorithm used for optimization (int)
        'pdhgtype': 1,
        # set the precision of computations (str). choices: ['float', 'double']
        'precision': 'float',
        # maximum number of PDHG iterations (int)
        'max_iter': 1000,
        # maximum number of outer iterations (int)
        'outer_iter': 20,
        # threshold of stopping criterion of PDHG method (float)
        'stop_eps': 1e-06,
        # threshold of outer stopping criterion (float)
        'outer_stop_eps': 1e-06,
        # band selection method (str). choices: ['full', 'issc']
        'band_selection': 'full',
        # parameter of band selection method (float)
        'band_select_param': 0.0001,
        # reduce dimensionality of data before processing it (bool)
        'reduce_dimensionality': True,
        # dimensionality reduction method (str). choices: ['pca', 'tsne', 'umap', 'isomap', 'mnf', 'ica']
        'dim_red_method': 'mnf',
        # number of features to keep (float)
        'n_features': 8,
        # initialization method. choices: ['kmeans', 'random', 'dbscan', 'gmm', 'hierarchical', 'optics', 'birch', 'gt']
        'init': 'kmeans',
        # should the dimensionality be reduced before initialization is applied? (bool)
        # useful when initialization method should run on reduced data, but the segmentation method on the full data
        'reduce_dim_init': False,
        # computation of estimates of segment means (str). choices: ['arithmetic', 'trimmed']
        # does not apply to non-squared-anisotropic-eps-inverse
        'means': 'arithmetic',
        # computation of estimates of segment variances (str). choices: ['sdm', 'trimmed']
        # sdm: squared differences from mean
        # does not apply to non-squared-anisotropic-eps-inverse
        'variances': 'sdm',
        # computation of estimates of principal components of the segments (str). choices: ['pca', 'mnf']
        # does not apply to non-squared-anisotropic-eps-inverse
        'components': 'pca',
        # proportion of points trimmed at both sides when computing trimmed means or variances (float)
        'trim_proportion': 0.5,
        # seed for randomness to make results reproducible (int)
        'seed': 42,
        # ignore pixels that carry this label in the ground truth (int)
        'ignore_label': 0,
        # if true, ignore pixels with ignore label in computations (bool)
        'ignore_pixels': False,
        # if true, ignore pixels with ignore label in data term (bool)
        'ignore_pixels_data_term': False,
        # if true, try to solve empty segments when they occur (bool)
        'solve_empty_segment': False,
        # if true, save found indermediate segmentations (bool)
        'save_intermediate_segs': False
    }

    # set environment variable for output
    os.environ["OUTPUT_DIR"] = "."

    ms_segmentation(args)

if __name__ == '__main__':
    main()
