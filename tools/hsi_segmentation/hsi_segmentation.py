'''
    Hyperspectral image segmentation

    Method epsAMS published in

    [1]
    Jan-Christopher Cohrs, Chandrajit Bajaj and Benjamin Berkels.
    A distribution-dependent Mumford-Shah model for unsupervised hyperspectral image segmentation.
    IEEE Transactions on Geoscience and Remote Sensing, 2022. [https://ieeexplore.ieee.org/document/9970756]
    DOI: 10.1109/TGRS.2022.3227061]

    Methods phiAMS and kMS published in

    [2]
    Jan-Christopher Cohrs.
    Mumford-Shah type models for unsupervised hyperspectral image segmentation.
    Dissertation. December 2025. [https://publications.rwth-aachen.de/record/1023351]
    DOI: 10.18154/RWTH-2025-10634

    All three models can be run partly on the GPU if a GPU device is available.
    To use the GPU, please install the package cupy via `mamba install cupy` and set the parameter 'use_gpu' in line 141 to True.
'''

import os
import requests
from scipy.io import loadmat
from msiplib.segmentation.hsi.hsi_ms import ms_segmentation
from msiplib.io.segmentation import saveColoredSegmentation

def main():
    ''' main function '''

    # download data file and store it as netCDF
    url_data = 'https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat'
    dest_data = './IndianPines.mat'
    response = requests.get(url_data)
    with open(dest_data, 'wb') as f:
        f.write(response.content)
    print("Downloaded Indian Pines dataset.")

    # download ground truth and store it as png file
    url_gt = 'https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'
    dest_gt = './IndianPines_gt.mat'
    response = requests.get(url_gt)
    with open(dest_gt, 'wb') as f:
        f.write(response.content)
    print("Downloaded ground truth.")

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
        'gt_file': os.path.expandvars('./IndianPines_gt.png'),
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
        # 'anisotropic-eps-discard', 'epsAMS', 'phiAMS', 'kernel']
        'ind_func': 'epsAMS',
        # epsilon to regularize indicator function (float)
        'ind_eps': 0.125,
        # kernel function (str).
        # choices: ['direct-summation', 'weighted-summation', 'cross-information']
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
        'precision': 'double',
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
        # does not apply to epsAMS
        'means': 'arithmetic',
        # computation of estimates of segment variances (str). choices: ['sdm', 'trimmed']
        # sdm: squared differences from mean
        # does not apply to epsAMS
        'variances': 'sdm',
        # computation of estimates of principal components of the segments (str). choices: ['pca', 'mnf']
        # does not apply to epsAMS
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
        'save_intermediate_segs': False,
        # if true, run computations on GPU if device and CUDA available
        'use_gpu': False
    }

    # set environment variable for output. determines where the output is stored
    os.environ['OUTPUT_DIR'] = '.'

    # set an environment variable pointing to the parent folder of the msiplib repository
    os.environ['REPOS_DIR'] = os.path.abspath(__file__).split('msiplib')[0][:-1]

    # run segmentation algorithm with epsAMS as published in [1]
    # it runs with the parameters set above in args
    ms_segmentation(args)
    print("Segmentation with epsAMS finished.")

    # run segmentation algorithm with phiAMS as published in [2]
    # run with args as set above, overriding the following parameters
    args['ind_func'] = 'phiAMS'
    args['ind_params'] = [50, 7.88, 9.38]
    args['reg_par'] = 0.296
    ms_segmentation(args)
    print("Segmentation with phiAMS finished.")

    # run segmentation algorithm with kMS with direct-summation kernel as published in [2]
    # run with args as set above, overriding the following parameters
    args['ind_func'] = 'kMS'
    args['kernel'] = 'direct-summation'
    args['ind_params'] = [3, 8.5, 1, 1, 8] # params: r, gamma (RBF kernel), delta, c, p (polynomial kernel)
    args['reg_par'] = 0.26
    ms_segmentation(args)
    print("Segmentation with kMS and direct-summation kernel finished.")
    
    # run segmentation algorithm with kMS with weighted-summation kernel as published in [2]
    # run with args as set above, overriding the following parameters
    args['ind_func'] = 'kMS'
    args['kernel'] = 'weighted-summation'
    args['ind_params'] = [3, 0.2, 45, 1, 1, 7] # params: r, alpha, gamma (RBF kernel), delta, c, p (polynomial kernel)
    args['reg_par'] = 0.29
    ms_segmentation(args)
    print("Segmentation with kMS and weighted-summation kernel finished.")

    # run segmentation algorithm with kMS with cross-information kernel as published in [2]
    # run with args as set above, overriding the following parameters
    args['ind_func'] = 'kMS'
    args['kernel'] = 'cross-information'
    args['ind_params'] = [7, 30, 1, 1, 7] # params: r, gamma (RBF kernel), delta, c, p (polynomial kernel)
    args['reg_par'] = 0.91
    ms_segmentation(args)
    print("Segmentation with kMS and cross-information kernel finished.")

if __name__ == '__main__':
    main()
