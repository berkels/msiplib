#!/opt/anaconda3/envs/hsu/bin python
# -*- coding: utf-8 -*-

'''
    Script to apply the Mumford-Shah model for image segmentation. Code is adapted from Benjamin's MS.py script.
'''

import logging
import signal
import sys
import timeit
from shutil import copy, rmtree

# import cupy as cp
import numpy as np

from msiplib.segmentation.hsi.args_processing import create_config, initialize_logger, parse_args
from msiplib.segmentation.hsi.dimensionality_reduction import band_selection, reduce_number_features
from msiplib.segmentation.hsi.initialization import initialize_segmentation
from msiplib.segmentation.hsi.input_output import read_inputimage, save_segmentation

from msiplib.decomposition import pca
from msiplib.segmentation import indicator_functions

from msiplib.finite_differences import gradient_FD2D, divergence_FD2D
from msiplib.io import saveArrayAsNetCDF
from msiplib.optimization import pd_hybrid_grad_alg
from msiplib.proximal_mappings import project_canonical_simplex, project_unit_ball2D
from msiplib.segmentation import get_segmentation_mean_values
from msiplib import metrics


def evaluate_segmentation(segmentation, seg_gt=None, ignore_label=None):
    ''' function that evaluates found segmentation and writes scores and configuration to log file '''

    logger = logging.getLogger('eval')
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)

    # log scores
    logger.info('Evaluation')

    # If no ground truth is provided, use internal statistics to evaluate segmentation (clustering)
    if seg_gt is None:
        logger.info('No ground truth available. Only internal evaluation.')
    else:
        # compute segmentation scores
        scores = metrics.segmentation_scores(segmentation, seg_gt, ignore_label=ignore_label)

        if np.unique(segmentation).shape[0] > np.unique(seg_gt).shape[0]:
            logger.warning('More segments in segmentation than in ground truth!')
        logger.info('Overall accuracy: %s', scores['overallAcc'])
        logger.info('Mean class accuracy: %s', scores['meanClassAcc'])
        logger.info('mIoU: %s', scores['mIoU'])
        logger.info('mean Dice: %s', scores['mDice'])
        logger.info('Kappa coefficient: %s', scores['kappa'])
        logger.info('Variation of information: %s', scores['VI'])
        logger.info('Normalized mutual information: %s', scores['NMI'])
        logger.info('Adjusted rand score: %s', scores['ARS'])
        return scores['overallAcc']

def compute_indicator_functions(image, segmentation, k, args, valid_mask=None, means=None,
                                pcs=None, weights=None, kernel_matrix=None):

    ''' computes indicator function '''

    logger = logging.getLogger('indicator')

    # initialize array for storing the indicator values
    f = np.zeros((image.shape[0], image.shape[1], k), dtype=image.dtype)

    # if an anisotropic indicator function is chosen, compute minimum of log det cov and subtract it to ensure
    # that the minimum of the indicator functions is 0
    # variants with a kernel involved need extra handling as in this case dimensionality of covariance matrix may vary
    if 'anisotropic-eps' in args['ind_func']:
        min_log_det_cov = 2 * image.shape[-1] * np.log(args['ind_eps'])
        if ('normal' in args['ind_func']) and (args['ind_eps'] > 1):
            min_log_det_cov = -1 * min_log_det_cov

    # compute indicator values for all pixels and segments
    for l in range(k):

        # irregular region is added at the end of array f (last segment)
        if args['irreg'] and l == k - 1:
            # handle irregular regions
            # omega_hat is computed outside of the function to allow application of numba
            omega_hat = np.argmin(f[:, :, :k - 1], axis=2).astype(np.uint32)
        else:
            # check if valid pixels for segment l are available
            if image[np.logical_and(segmentation == l, valid_mask)].size == 0:
                logger.warning('No valid pixels in segment! Stop iteration.')
            elif args['ind_func'] != '2' and image[np.logical_and(segmentation == l, valid_mask)].shape[0] == 1:
                logger.warning('Only one valid pixel in segment! Stop iteration.')
            if args['ind_func'] == '2':
                # the typical 2-norm is used as indicator function
                f[:, :, l] = indicator_functions.euclidean_norm(image, segmentation, l, valid_mask)
            elif args['ind_func'] == 'anisotropic-eps-inverse':
                logger.info('Segment %s', l)
                # the anisotropic 2-norm is used as indicator function where components with a standard deviation
                # smaller than epsilon are scaled by 1/eps
                f[:, :, l] = indicator_functions.anisotropic_2norm(
                    image, segmentation, l, args['ind_eps'], True, valid_mask,
                    args['means'], args['components'], args['variances'],
                    args['trim_proportion']) - min_log_det_cov
            elif args['ind_func'] == 'anisotropic-eps-normal':
                logger.info('Segment %s', l)
                # the anisotropic 2-norm is used as indicator function where components with a standard deviation
                # smaller than epsilon are scaled by eps
                f[:, :, l] = indicator_functions.anisotropic_2norm(
                    image, segmentation, l, args['ind_eps'], False, valid_mask,
                    args['means'], args['components'], args['variances'],
                    args['trim_proportion']) - min_log_det_cov
            elif args['ind_func'] == 'anisotropic-eps-discard':
                logger.info('Segment %s', l)
                # the anisotropic 2-norm is used as indicator function where components with a standard deviation
                f[:, :, l] = indicator_functions.pca_eps_discard_norm(
                    image, segmentation, l, args['ind_eps'], valid_mask)
            elif args['ind_func'] == 'non-squared-anisotropic-eps-inverse':
                logger.info('Segment %s', l)
                # the non-squared anisotropic 2-norm is used as indicator function where components with a standard
                # deviation smaller than epsilon are scaled by 1/eps
                f[:, :, l], means[l], pcs[l], weights[l] = indicator_functions.nonsquared_anistropic_2norm(
                    image, segmentation, l, args['ind_eps'], means, pcs, weights, tol=1e-05,
                    max_iter=args['ind_params'][0], valid_mask=valid_mask)
                np.subtract(f[:, :, l], min_log_det_cov, out=f[:, :, l])

    if args['ind_func'] == 'non-squared-anisotropic-eps-inverse':
        # in the case of the non-squared anisotropic 2-norm return means, PCs and weights to reuse them
        # as initial guesses in the next iteration
        return f, means, pcs, weights
    else:
        return f


def update_u(u, f, lambda_, max_iter=1000, eps=1e-06, convexification='zach', pdhgtype=1):
    ''' functions performs an update of u, i.e., one inner iteration '''
    inner_logger = logging.getLogger('inner')

    # u = cp.array(u)
    # f = cp.array(f)

    tstart = timeit.default_timer()
    if convexification == 'zach':
        u, _, it = \
            pd_hybrid_grad_alg(u, lambda a, t: project_canonical_simplex(a - t * f),
                               project_unit_ball2D, gradient_FD2D, divergence_FD2D,
                               lambda_, max_iter, eps, PDHGAlgType=1)

    time_elapsed = timeit.default_timer() - tstart
    inner_logger.info('Time elapsed: %ss', time_elapsed)
    inner_logger.info('Number of iterations: %s', it)

    # u = cp.asnumpy(u)

    return u


def stopping_criterion(segmentation, last_means, image, num_segments, threshold=1e-05):
    ''' implements the stopping criterion that stops outer MS iteration '''
    logger = logging.getLogger('outer')

    # get number of pixels
    n = float(image.shape[0] * image.shape[1])

    # compute current mean features
    means = get_segmentation_mean_values(image, segmentation, num_segments)
    #means_dev = np.sqrt(np.square(means - last_means).sum(axis=1)) # 2-norm
    means_dev = np.max(np.abs(means - last_means), axis=1)          # infinity norm

    # if means_dev contains NaNs because of empty segments, set values to maximum
    means_dev[np.isnan(means_dev)] = np.finfo(image.dtype).max

    # compute weights for weighted average based on number of pixels in segment
    weights = np.zeros(num_segments, dtype=image.dtype)
    for l in range(num_segments):
        weights[l] = (segmentation == l).sum() / n

    # compute t as the weighted average over the norms of the differences of last and current mean features
    t = (means_dev * weights).sum()
    logger.info('Stopping criterion value: %s', t)

    return (t < threshold), means


def zach_functional(u, f, lambda_):
    r'''
        function computes the functional value for a u and given indicator function values
        Args:
            u: point where the functional is evaluated
            f: indicator function values
            lambda\_: regularization parameter
        Returns:
            the resulting functional value J_{Zach}[u]
    '''

    # compute data term
    data = np.sum(f * u)

    # compute total variation
    # reg = np.sum(np.sqrt(np.sum(np.square(cp.asnumpy(gradient_FD2D(cp.array(u)))), axis=0)))
    reg = np.sum(np.sqrt(np.sum(np.square(gradient_FD2D(u)), axis=0)))

    return data + lambda_ * reg


def ms_segmentation(args):
    ''' main function '''

    # initialize logger
    ms_logger, filepath_name = initialize_logger(args)
    logger_path = (filepath_name[:filepath_name.rfind('/') + 1])

    # copy config file to folder if file was given
    # if only args dictionary was given, create a config file based on that and copy it to output folder
    logger_name = filepath_name[filepath_name.rfind('/') + 1:]
    cfg_path = f'{logger_path}/{logger_name}.cfg'
    if args['config_path'] is not None:
        copy(args['config_path'], cfg_path)
    else:
        create_config(cfg_path, args)

    # initialize signal handler
    def signal_handler(sig, frame): # pylint: disable=unused-argument
        ''' Signal handler to handle SIGINT '''
        print('Abort by user!')
        rmtree(logger_path)
        print('Removed created folder and files.')
        sys.exit(0)

    # catch and handle SIGINT. remove all created files and folders if SIGINT was sent
    signal.signal(signal.SIGINT, signal_handler)

    # read input image
    inputimage, seg_gt, k_gt, nc_gt_flag = read_inputimage(args['filename'], args['gt_file'], args['precision'])

    # rescale lambda to make it independent of the resolution of the input image.
    h = 1 / (max(inputimage.shape[:2]) - 1)
    lambda_ = args['reg_par'] / h

    # if input image has less than three channels, use it to compute the resulting mean value colored segmentation
    if inputimage.shape[-1] <= 3:
        original_image = inputimage.copy()

    k = args['k']

    # check whether the number of segments is given by the ground truth. if so, give a warning when number of segments
    # is different from the number of segments in the ground truth
    if (nc_gt_flag or (args['gt_file'] is not None)) and (args['ignore_label'] is None):
        if k != k_gt:
            ms_logger.warning('Number of segments %s differs from the actual number of segments %s '
                              'given by the ground truth!', k, k_gt)

    ######################################## initialization #####################################
    init_logger = logging.getLogger('init')
    init_logger.info('Initialization')

    # if ignore label is given, generate mask to indicate the valid (non-ignored) pixels
    # if pixels carrying the ignore label shall be ignored in computations, define mask with valid pixels.
    if (args['ignore_label'] is not None) and (seg_gt is not None):
        m = seg_gt != args['ignore_label']
        if args['ignore_pixels']:
            valid_mask_comps = m
        else:
            valid_mask_comps = np.ones((inputimage.shape[0], inputimage.shape[1]), dtype=bool)

        if args['ignore_pixels_data_term']:
            valid_mask_dataterm = m

    else:
        valid_mask_comps = np.ones((inputimage.shape[0], inputimage.shape[1]), dtype=bool)
        valid_mask_dataterm = valid_mask_comps

    # select relevant bands if image has more than three bands and band selection method is provided
    if (inputimage.shape[-1] > 3) and (args['band_selection'] != 'full'):
        inputimage = band_selection(inputimage, args['band_selection'], args['band_select_param'], args['seed'])

    # Dimensionality reduction
    if args['reduce_dimensionality']:
        inputimage = reduce_number_features(inputimage, args['dim_red_method'], args['n_features'],
                                            args['trim_proportion'], args['seed'])

    # initialize segmentation using unsupervised clustering on valid pixels
    segmentation = initialize_segmentation(inputimage, k, valid_mask_comps, args, seg_gt, args['seed'])

    # save initial segmentation if flag is set
    if args['save_intermediate_segs']:
        f_name = f'{filepath_name}_it{"init"}'
        saveArrayAsNetCDF(segmentation, f'{f_name}.nc')
        if 'original_image' in locals():
            save_segmentation(f_name, segmentation, original_image, args['gt_file'], args['ignore_label'], True,
                              args['irreg'])
        else:
            save_segmentation(f_name, segmentation, None, args['gt_file'], args['ignore_label'], True,
                              args['irreg'])


    # compute means of segments based on initial segmentation for stopping criterion
    st_means = get_segmentation_mean_values(inputimage, segmentation, k)

    # non-squared anisotropic 2-norm needs estimates for means, principal components and weights
    if args['ind_func'] == 'non-squared-anisotropic-eps-inverse':
        means = st_means.copy()
        pcs = np.empty((k, inputimage.shape[-1], inputimage.shape[-1]), dtype=inputimage.dtype)
        weights = np.empty((k, inputimage.shape[-1]), dtype=inputimage.dtype)

        # compute initial values for weights and principal components for every segment
        for l in range(k):
            weights[l, ...], pcs[l, ...] = pca(inputimage[segmentation == l].T)

        # regularize standard deviations to ensure invertibility
        np.maximum(weights, 0.0, out=weights)
        np.sqrt(weights, out=weights)
        np.maximum(weights, args['ind_eps'], out=weights)
        np.reciprocal(weights, out=weights)


    ######################################## main loop ##########################################
    tstart = timeit.default_timer()

    # log mIoU score after initialization if ground truth is available
    if seg_gt is not None:
        scores = metrics.segmentation_scores(segmentation, seg_gt, args['ignore_label'])
        miou = scores['mIoU']
        print('Scores after initialization:')
        print(scores)
        init_logger.info('OA: %s', scores['overallAcc'])
        init_logger.info('mIoU: %s', miou)

    outer_logger = logging.getLogger('outer')
    console_handler = logging.StreamHandler(sys.stdout)
    outer_logger.addHandler(console_handler)


    for i in range(args['outer_iter']):
        it = i + 1
        outer_logger.info('Iteration %s', it)

        # initialize u based on the current segmentation
        u = np.zeros((inputimage.shape[0], inputimage.shape[1], k), dtype=inputimage.dtype)
        for l in range(k):
            u[segmentation == l, l] = 1.0

        # compute the indicator functions
        # for every pixel in the image, f contains a k-dimensional vector that consists of the distances of
        # the pixel's feature vector to the different segments
        if args['ind_func'] == 'non-squared-anisotropic-eps-inverse':
            f, means[...], pcs[...], weights[...] = compute_indicator_functions(inputimage, segmentation, k, args,
                valid_mask_comps, means, pcs, weights, None)
        else:
            f = compute_indicator_functions(inputimage, segmentation, k, args, valid_mask_comps,
                                            None, None, None, kernel_matrix)
        print(f'Minimum of indicators: {f.min()}')

        # if pixels with ignore label should not contribute to data term, set indicator values to 0.
        if args['ignore_pixels_data_term']:
            f[np.invert(valid_mask_dataterm)] = 0

        # with this values of the indicator functions, solve for u
        u = update_u(u, f, lambda_, args['max_iter'], args['stop_eps'], args['convexification'],
                     args['pdhgtype'])

        # compute segment labels with current u
        segmentation_new = np.argmax(u, axis=2).astype(np.uint8)

        # log the current functional value
        outer_logger.info('Current functional value: %s', zach_functional(u, f, lambda_))


        # log score after each iteration if ground truth is available
        if seg_gt is not None:
            scores = metrics.segmentation_scores(segmentation_new, seg_gt, args['ignore_label'])
            outer_logger.info('OA: %s', scores['overallAcc'])

        # check if there is an empty segment. if so, stop iterating
        if np.unique(segmentation_new).shape[0] != k and not args['irreg']:
            if args['solve_empty_segment']:
                #segmentation = solve_empty_segment(10, inputimage, segmentation, k, args['ind_func'],
                #                                   args['ind_eps'])
                outer_logger.warning('Empty segment! Try to solve problem.')
            else:
                outer_logger.warning('Empty segment! Stop iteration.')
                return 0

        # save intermediate segmentations if flag is set
        if args['save_intermediate_segs']:
            f_name = f'{filepath_name}_it{it}'

            # TODO: Find better solution for the case when means, pcs and weights are None.
            # TODO: Also consider returning and storing statistics for other indicator functions
            if 'original_image' in locals():
                if 'means' in locals() and 'pcs' in locals() and 'weights' in locals():
                    save_segmentation(f_name, segmentation_new, original_image, args['gt_file'], args['ignore_label'],
                                      True, args['irreg'], means, pcs, np.reciprocal(weights))
                else:
                    save_segmentation(f_name, segmentation_new, original_image, args['gt_file'], args['ignore_label'],
                                      True, args['irreg'])
            else:
                if 'means' in locals() and 'pcs' in locals() and 'weights' in locals():
                    save_segmentation(f_name, segmentation_new, None, args['gt_file'], args['ignore_label'],
                                      True, args['irreg'], means, pcs, np.reciprocal(weights))
                else:
                    save_segmentation(f_name, segmentation_new, None, args['gt_file'], args['ignore_label'],
                                      True, args['irreg'])

        # check the stopping criterion
        stop, st_means = stopping_criterion(segmentation_new, st_means, inputimage, k,
                                            threshold=args['outer_stop_eps'])
        if stop:
            print(f"Stopping after {it} iterations.")
            break

        # remember found segmentation
        np.copyto(segmentation, segmentation_new)

    time_elapsed = timeit.default_timer() - tstart
    outer_logger.info('Total time elapsed: %fs', time_elapsed)
    outer_logger.info('Number of outer iterations: %s', it)

    # save the found minimizer u of the Zach functional
    saveArrayAsNetCDF(u, f'{filepath_name}_u.nc')

    # compute and log the resulting value of the minimized functional
    outer_logger.info('Final functional value: %s', zach_functional(u, f, lambda_))

    ######################## evaluation ##########################
    if (args['gt_file'] is not None) or nc_gt_flag:
        score = evaluate_segmentation(segmentation_new, seg_gt, args['ignore_label'])

    # shutdown logger and release all files
    logging.shutdown()

    # save segmentation as an image file
    # TODO: Find better solution for the case when means, pcs and weights are None.
    if 'original_image' in locals():
        if 'means' in locals() and 'pcs' in locals() and 'weights' in locals():
            save_segmentation(filepath_name, segmentation_new, original_image, args['gt_file'], args['ignore_label'],
                              True, args['irreg'], means, pcs, np.reciprocal(weights))
        else:
            save_segmentation(filepath_name, segmentation_new, original_image, args['gt_file'], args['ignore_label'],
                              True, args['irreg'])

    else:
        if 'means' in locals() and 'pcs' in locals() and 'weights' in locals():
            save_segmentation(filepath_name, segmentation_new, None, args['gt_file'], args['ignore_label'],
                              True, args['irreg'], means, pcs, np.reciprocal(weights))
        else:
            save_segmentation(filepath_name, segmentation_new, None, args['gt_file'], args['ignore_label'],
                              True, args['irreg'])

    if (args['gt_file'] is not None) or nc_gt_flag:
        return -1 * score
    else:
        return 1


def main():
    ''' main function '''

    # parse arguments from command line and config file
    hyper_params = parse_args()

    # call MS segmentation function with given hyperparameters
    ms_segmentation(hyper_params)

if __name__ == "__main__":
    main()
