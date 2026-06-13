#!/opt/anaconda3/envs/hsu/bin python
# -*- coding: utf-8 -*-

"""
    Script to apply the Mumford-Shah model for image segmentation. Code is adapted from Benjamin's MS.py script.
"""

import logging
import signal
import sys
import timeit
from shutil import copy, rmtree

import numpy as np

from msiplib.segmentation.hsi.args_processing import create_config, initialize_logger, parse_args
from msiplib.segmentation.hsi.dimensionality_reduction import band_selection, reduce_number_features
from msiplib.segmentation.hsi.initialization import initialize_segmentation
from msiplib.segmentation.hsi.input_output import (
    plot_iteration_stats,
    read_inputimage,
    save_segmentation,
    save_variables,
)

from msiplib.decomposition import pca
from msiplib.segmentation.functional import ms_functional, zach_functional
from msiplib.segmentation.indicator_functions import compute_kernel_matrix

from msiplib.io import saveArrayAsNetCDF
from msiplib.metrics import segmentation_scores
from msiplib.segmentation import get_segmentation_mean_values
from msiplib.segmentation.hsi.utils import compute_indicator_functions, evaluate_segmentation, update_u


def stopping_criterion(segmentation, last_means, image, num_segments, threshold=1e-05):
    """implements the stopping criterion that stops outer MS iteration"""
    logger = logging.getLogger("outer")

    # get number of pixels
    n = float(image.shape[0] * image.shape[1])

    # compute current mean features
    means = get_segmentation_mean_values(image, segmentation, num_segments)
    # means_dev = np.sqrt(np.square(means - last_means).sum(axis=1)) # 2-norm
    means_dev = np.max(np.abs(means - last_means), axis=1)  # infinity norm

    # if means_dev contains NaNs because of empty segments, set values to maximum
    means_dev[np.isnan(means_dev)] = np.finfo(image.dtype).max

    # compute weights for weighted average based on number of pixels in segment
    weights = np.zeros_like(image, shape=num_segments, dtype=image.dtype)
    for l in range(num_segments):
        weights[l] = (segmentation == l).sum() / n

    # compute t as the weighted average over the norms of the differences of last and current mean features
    t = (means_dev * weights).sum()
    logger.info("Stopping criterion value: %s", t)

    return (t < threshold), means, t


def ms_segmentation(args):
    """
    Function to perform an image segmentation with the multiphase Mumford-Shah segmentation functional.

    Args:
        args: a dictionary specifying the hyperparameters of the model (see below how to define it).

    Returns:
        The negative overall accuracy if a ground truth is given.
        If no ground truth is given, the function returns the segmentation and u.

    Notes:
        The following gives an exemplary args dictionary with the possible choices for the different parameters.
        Given arguments that are not needed for the chosen parameters are ignored. E.g. if 'reduce_dimensionality'
        is set to 'False', arguments like 'dim_red_method' and 'n_features' are ignored.

        args =
            | { # path to configuration file (str). if provided, overrides all following choices
            | 'config_path': None,
            | # image file (numpy array) or path to file containing the data (str)
            | 'filename': os.path.expandvars('${DATA_DIR}/path-to-file.nc'),
            | # path to ground truth file (str)
            | 'gt_file': os.path.expandvars('${DATA_DIR}/path-to-gt.png'),
            | # number of segments to be sought (int)
            | 'k': 16,
            | # regularization parameter, balances data and regularization term (float)
            | 'reg_par': 0.1,
            | # if true, add additional segment to collect irregular points
            | 'irreg': False,
            | # radius of balls used to find irregular segments (int)
            | 'r': 1,
            | # epsilon to compute indicator values for irregular region (float)
            | 'irreg_eps': 1e-06,
            | # indicator function (str)
            | # choices: ['2', 'anisotropic-eps-inverse', 'anisotropic-eps-normal',
            | # 'anisotropic-eps-discard', 'epsAMS', 'phiAMS', 'kMS']
            | 'ind_func': 'kMS',
            | # epsilon to regularize indicator function (float)
            | 'ind_eps': 0.125,
            | # kernel function (str)
            | # choices: ['gaussian', 'polynomial', 'chi-squared', 'laplacian', 'cosine', 'sigmoid', 'direct-summation',
            | # 'weighted-summation', 'cross-information']
            | 'kernel': 'cross-information',
            | # list of parameters for indicator function
            | # if indicator function has several parameters, give them with a list [value1, value2,...]
            | # rbf kernel: [gamma]
            | # polynomial kernel: [delta, c, p]
            | # direct summation kernel: [window size spat features, rbf params, polynomial params]
            | # weighted summation kernel:
            | # [window size spat features, weight alpha in [0, 1], rbf params, polynomial params]
            | # cross-information kernel: [window size spat features, rbf params, polynomial params]
            | 'ind_params': [3, 10.0, 1.0, 1.0, 3],
            | # convexification of Mumford-Shah functional, for multiphase segmentation only Zach is implemented (str)
            | # for binary segmentation: ['zach', 'binaryUC']
            | 'convexification': 'zach',
            | # type of primal dual hybrid gradient (PDHG) algorithm used for optimization (int)
            | 'pdhgtype': 1,
            | # set the precision of computations (str). choices: ['float', 'double']
            | 'precision': 'float',
            | # maximum number of PDHG iterations (int)
            | 'max_iter': 1000,
            | # maximum number of outer iterations (int)
            | 'outer_iter': 20,
            | # threshold of stopping criterion of PDHG method (float)
            | 'stop_eps': 1e-06,
            | # threshold of outer stopping criterion (float)
            | 'outer_stop_eps': 1e-06,
            | # band selection method (str). choices: ['full', 'issc']
            | 'band_selection': 'full',
            | # parameter of band selection method (float)
            | 'band_select_param': 0.0001,
            | # reduce dimensionality of data before processing it (bool)
            | 'reduce_dimensionality': True,
            | # dimensionality reduction method (str). choices: ['pca', 'tsne', 'umap', 'isomap', 'mnf', 'ica']
            | 'dim_red_method': 'mnf',
            | # number of features to keep (float)
            | 'n_features': 8,
            | # initialization method. choices:
            | # ['kmeans', 'random', 'dbscan', 'gmm', 'hierarchical', 'optics', 'birch', 'gt']
            | 'init': 'kmeans',
            | # should the dimensionality be reduced before initialization is applied? (bool)
            | # useful when init method should run on reduced data, but the segmentation method on the full data
            | 'reduce_dim_init': False,
            | # computation of estimates of segment means (str). choices: ['arithmetic', 'trimmed']
            | # does not apply to epsAMS, phiAMS, and kMS
            | 'means': 'arithmetic',
            | # computation of estimates of segment variances (str). choices: ['sdm', 'trimmed']
            | # sdm: squared differences from mean
            | # does not apply to epsAMS, phiAMS, and kMS
            | 'variances': 'sdm',
            | # computation of estimates of principal components of the segments (str). choices: ['pca', 'mnf']
            | # does not apply to epsAMS, phiAMS, and kMS
            | 'components': 'pca',
            | # proportion of points trimmed at both sides when computing trimmed means or variances (float)
            | 'trim_proportion': 0.5,
            | # seed for randomness to make results reproducible (int)
            | 'seed': 42,
            | # ignore pixels that carry this label in the ground truth (int)
            | 'ignore_label': 0,
            | # if true, ignore pixels with ignore label in computations (bool)
            | 'ignore_pixels': False,
            | # if true, ignore pixels with ignore label in data term (bool)
            | 'ignore_pixels_data_term': False,
            | # if true, try to solve empty segments when they occur (bool)
            | 'solve_empty_segment': False,
            | # if true, save found indermediate segmentations (bool)
            | 'save_intermediate_segs': False,
            | # if true, use GPU to run computations when available
            | 'use_gpu': True}
    """

    # initialize logger
    ms_logger, filepath_name = initialize_logger(args)
    logger_path = filepath_name[: filepath_name.rfind("/") + 1]

    # copy config file to folder if file was given
    # if only args dictionary was given, create a config file based on that and copy it to output folder
    logger_name = filepath_name[filepath_name.rfind("/") + 1 :]
    cfg_path = f"{logger_path}/{logger_name}.cfg"
    if args["config_path"] is not None:
        copy(args["config_path"], cfg_path)
    else:
        create_config(cfg_path, args)

    # initialize signal handler
    def signal_handler(sig, frame):  # pylint: disable=unused-argument
        """Signal handler to handle SIGINT"""
        print("Abort by user!")
        rmtree(logger_path)
        print("Removed created folder and files.")
        sys.exit(0)

    # catch and handle SIGINT. remove all created files and folders if SIGINT was sent
    signal.signal(signal.SIGINT, signal_handler)

    # read input image
    inputimage, seg_gt, k_gt, nc_gt_flag = read_inputimage(args["filename"], args["gt_file"], args["precision"])

    # rescale lambda to make it independent of the resolution of the input image.
    h = 1 / (max(inputimage.shape[:2]) - 1)
    lambda_ = args["reg_par"] / h

    # if input image has less than three channels, use it to compute the resulting mean value colored segmentation
    if inputimage.shape[-1] <= 3:
        original_image = inputimage.copy()

    k = args["k"]
    kernel_matrix = None

    # check whether the number of segments is given by the ground truth. if so, give a warning when number of segments
    # is different from the number of segments in the ground truth
    if (nc_gt_flag or (args["gt_file"] is not None)) and (args["ignore_label"] is None):
        if k != k_gt:
            ms_logger.warning(
                "Number of segments %s differs from the actual number of segments %s " "given by the ground truth!",
                k,
                k_gt,
            )

    ######################################## initialization #####################################
    init_logger = logging.getLogger("init")
    init_logger.info("Initialization")

    # if ignore label is given, generate mask to indicate the valid (non-ignored) pixels
    # if pixels carrying the ignore label shall be ignored in computations, define mask with valid pixels.
    if (args["ignore_label"] is not None) and (seg_gt is not None):
        m = seg_gt != args["ignore_label"]
        if args["ignore_pixels"]:
            valid_mask_comps = m
        else:
            valid_mask_comps = np.ones_like(inputimage, shape=(inputimage.shape[0], inputimage.shape[1]), dtype=bool)

        if args["ignore_pixels_data_term"]:
            valid_mask_dataterm = m

    else:
        valid_mask_comps = np.ones_like(inputimage, shape=(inputimage.shape[0], inputimage.shape[1]), dtype=bool)
        valid_mask_dataterm = valid_mask_comps

    # select relevant bands if image has more than three bands and band selection method is provided
    if (inputimage.shape[-1] > 3) and (args["band_selection"] != "full"):
        inputimage = band_selection(inputimage, args["band_selection"], args["band_select_param"], args["seed"])

    # Dimensionality reduction
    if args["reduce_dimensionality"]:
        inputimage = reduce_number_features(
            inputimage, args["dim_red_method"], args["n_features"], args["trim_proportion"], args["seed"]
        )

    # initialize segmentation using unsupervised clustering on valid pixels
    segmentation = initialize_segmentation(inputimage, k, valid_mask_comps, args, seg_gt, args["seed"])

    # save initial segmentation if flag is set
    if args["save_intermediate_segs"]:
        f_name = f'{filepath_name}_it{"init"}'
        orig_save = original_image if "original_image" in locals() else None
        saveArrayAsNetCDF(segmentation, f"{f_name}.nc")
        save_segmentation(f_name, segmentation, orig_save, args["gt_file"], args["ignore_label"], True, args["irreg"])

    # normalize image after dimensionality reduction since chi-squared kernel needs non-negative values as input
    if args["kernel"] == "chi-squared":
        np.subtract(inputimage, inputimage.min(), out=inputimage)

    # compute means of segments based on initial segmentation for stopping criterion
    st_means = get_segmentation_mean_values(inputimage, segmentation, k)

    # epsAMS and phiAMS need estimates for means, principal components and weights
    if "AMS" in args["ind_func"]:
        means = st_means.copy()
        pcs = np.empty((k, inputimage.shape[-1], inputimage.shape[-1]), dtype=inputimage.dtype)
        weights = np.empty((k, inputimage.shape[-1]), dtype=inputimage.dtype)

        # allocate memory to save old means, pcs and weights in case empty segments occur
        means_old = st_means.copy()
        pcs_old = np.empty((k, inputimage.shape[-1], inputimage.shape[-1]), dtype=inputimage.dtype)
        weights_old = np.empty((k, inputimage.shape[-1]), dtype=inputimage.dtype)

        # compute initial values for weights and principal components for every segment
        for l in range(k):
            weights[l, ...], pcs[l, ...] = pca(inputimage[segmentation == l].T)

        # regularize standard deviations to ensure invertibility
        if args["ind_func"] != "epsAMS":
            args["ind_eps"] = 1
        np.maximum(weights, 0.0, out=weights)
        np.sqrt(weights, out=weights)
        np.maximum(weights, args["ind_eps"], out=weights)
        np.reciprocal(weights, out=weights)


    ######################################## main loop ##########################################
    tstart = timeit.default_timer()

    # log mIoU score after initialization if ground truth is available
    if seg_gt is not None:
        scores = segmentation_scores(segmentation, seg_gt, args["ignore_label"])
        miou = scores["mIoU"]
        print("Scores after initialization:")
        print(scores)
        init_logger.info("OA: %s", scores["overallAcc"])
        init_logger.info("mIoU: %s", miou)

    outer_logger = logging.getLogger("outer")
    console_handler = logging.StreamHandler(sys.stdout)
    outer_logger.addHandler(console_handler)

    # if kMS is chosen as the indicator function, precompute kernel matrix
    if "kernel" in args["ind_func"] or args["ind_func"] == "kMS":
        kernel_matrix = compute_kernel_matrix(inputimage, args["kernel"], args["ind_params"])
        print("Kernel matrix precomputed.")

    # if computations should be run on GPU, copy data to GPU
    if args["use_gpu"]:
        # check the availability of GPU computations by importing cupy
        try:
            import cupy as cp  # pylint: disable=import-outside-toplevel

            cp.cuda.Device(args["device"] if "device" in args else 0).use()
        except ModuleNotFoundError:
            init_logger.warning("Cupy not available. Fall back to CPU.")
            args["use_gpu"] = False
        else:
            # if import of cupy succeeds, data can be copied to GPU
            inputimage = cp.array(inputimage)
            segmentation = cp.array(segmentation)
            st_means = cp.array(st_means)
            valid_mask_comps = cp.array(valid_mask_comps)
            if args["ignore_pixels_data_term"]:
                valid_mask_dataterm = cp.array(valid_mask_dataterm)
            if 'AMS' in args['ind_func']:
                means = cp.array(means)
                pcs = cp.array(pcs)
                weights = cp.array(weights)
            elif "kernel" in args["ind_func"] or args["ind_func"] == "kMS":
                kernel_matrix = cp.array(kernel_matrix)

    # initialize u
    if args["convexification"] == "binaryUC":
        u = np.zeros_like(inputimage, shape=inputimage.shape[:-1], dtype=inputimage.dtype)
    else:
        u = np.zeros_like(inputimage, shape=(inputimage.shape[0], inputimage.shape[1], k), dtype=inputimage.dtype)

    # create structures to log oa, functional value, MS functional value, stopping
    # criterion, smallest and largest eigenvalues for each iteration
    ms_vals = []
    func_vals = []
    oas = []
    stopping = []
    if "AMS" in args["ind_func"]:
        smallest_ev = []
        largest_ev = []
    
    # empty = False # used when run with several seeds
    for i in range(args["outer_iter"]):
        it = i + 1
        outer_logger.info("Iteration %s", it)

        # set u based on the current segmentation to one of the vertices of the simplex
        if args["convexification"] != "binaryUC":
            for l in range(k):
                u[segmentation == l, l] = 1.0

        # compute the indicator functions
        # for every pixel in the image, f contains a k-dimensional vector that consists of the distances of
        # the pixel's feature vector to the different segments
        if "AMS" in args["ind_func"]:
            # compute indicator values and update estimates of means, pcs and weights
            f, means[...], pcs[...], weights[...] = compute_indicator_functions(
                inputimage, segmentation, k, args, valid_mask_comps, means, pcs, weights, None
            )
            # save smallest and largest eigenvalue to plot them
            Sigma_eigvals = np.square(np.reciprocal(weights))
            if args["use_gpu"]:
                smallest_ev.append(np.min(Sigma_eigvals, axis=1).get())
                largest_ev.append(np.max(Sigma_eigvals, axis=1).get())
            else:
                smallest_ev.append(np.min(Sigma_eigvals, axis=1))
                largest_ev.append(np.max(Sigma_eigvals, axis=1))
        else:
            f = compute_indicator_functions(
                inputimage, segmentation, k, args, valid_mask_comps, None, None, None, kernel_matrix
            )
        print(f"Minimum of indicators: {f.min()}")

        # if pixels with ignore label should not contribute to data term, set indicator values to 0.
        if args["ignore_pixels_data_term"]:
            f[np.invert(valid_mask_dataterm)] = 0

        # with this values of the indicator functions, solve for u
        u[...] = update_u(u, f, lambda_, args["max_iter"], args["stop_eps"], args["convexification"], args["pdhgtype"])

        # compute segment labels with current u
        if args["convexification"] == "binaryUC":
            segmentation_new = (u > 0.5).astype(np.uint8)
        else:
            segmentation_new = np.argmax(u, axis=2).astype(np.uint8)

        # log the current MS functional value
        ms_val = (
            ms_functional(segmentation_new, f, lambda_).get()
            if args["use_gpu"]
            else ms_functional(segmentation_new, f, lambda_)
        )
        ms_vals.append(ms_val)
        outer_logger.info("Current MS functional value: %s", ms_vals[-1])

        # if Zach's convexification is used, log the current functional value
        if args["convexification"] == "zach":
            zach_val = zach_functional(u, f, lambda_).get() if args["use_gpu"] else zach_functional(u, f, lambda_)
            func_vals.append(zach_val)
            outer_logger.info("Current functional value: %s", func_vals[-1])


        # log score after each iteration if ground truth is available
        if seg_gt is not None:
            if args["use_gpu"]:
                scores = segmentation_scores(cp.asnumpy(segmentation_new), seg_gt, args["ignore_label"])
            else:
                scores = segmentation_scores(segmentation_new, seg_gt, args["ignore_label"])
            oas.append(scores["overallAcc"])
            outer_logger.info("OA: %s", scores["overallAcc"])
        else:
            oas.append(0.0)

        # check if there is an empty segment. if so, stop iterating
        if np.unique(segmentation_new).shape[0] != k and not args["irreg"]:
            if args["solve_empty_segment"]:
                # segmentation = solve_empty_segment(10, inputimage, segmentation, k, args['ind_func'],
                #                                   args['ind_eps'])
                outer_logger.warning("Empty segment! Try to solve problem.")
            else:
                outer_logger.warning("Empty segment! Stop iteration.")
                orig_save = original_image if "original_image" in locals() else None
                mean_save = means if "means" in locals() else None
                pcs_save = pcs if "pcs" in locals() else None
                weights_save = weights if "weights" in locals() else None
                save_variables(filepath_name, u, segmentation_new, args, orig_save, mean_save, pcs_save, weights_save)
                stopping.append(0.0)

                # empty = True # used when run with several seeds
                # break # used when run with several seeds
                # in case of a run with several seeds comment out the following lines

                # remove all file handlers from loggers
                logger = logging.getLogger()
                while logger.hasHandlers():
                    logger.removeHandler(logger.handlers[0])
                loggers = ["MS", "init", "outer", "indicator", "inner", "eval"]
                for log_name in loggers:
                    log = logging.getLogger(log_name)
                    log.handlers.clear()
                return 0

        # save intermediate segmentations if flag is set
        if args["save_intermediate_segs"]:
            f_name = f"{filepath_name}_it{it}"
            orig_save = original_image if "original_image" in locals() else None
            mean_save = means if "means" in locals() else None
            pcs_save = pcs if "pcs" in locals() else None
            weights_save = weights if "weights" in locals() else None
            save_variables(filepath_name, u, segmentation_new, args, orig_save, mean_save, pcs_save, weights_save)

        # check the stopping criterion
        stop, st_means, t = stopping_criterion(
            segmentation_new, st_means, inputimage, k, threshold=args["outer_stop_eps"]
        )
        stop_val = t.get() if args["use_gpu"] else t
        stopping.append(stop_val)
        if stop:
            print(f"Stopping after {it} iterations.")
            break

        # remember found segmentation
        np.copyto(segmentation, segmentation_new)

    time_elapsed = timeit.default_timer() - tstart
    outer_logger.info("Total time elapsed: %fs", time_elapsed)
    outer_logger.info("Number of outer iterations: %s", it)

    # log the resulting functional value of the Mumford-Shah segmentation functional
    outer_logger.info("Final MS functional value: %s", ms_functional(segmentation_new, f, lambda_))

    # if Zach's convexification is used, compute and log the resulting value of the minimized functional
    if args["convexification"] == "zach":
        outer_logger.info("Final functional value: %s", zach_functional(u, f, lambda_))

    ######################## evaluate and save results ##########################
    # save segmentation as an image file
    orig_save = original_image if "original_image" in locals() else None
    mean_save = means if "means" in locals() else None
    pcs_save = pcs if "pcs" in locals() else None
    weights_save = weights if "weights" in locals() else None
    save_variables(filepath_name, u, segmentation_new, args, orig_save, mean_save, pcs_save, weights_save)

    # plot statistics against outer iterations
    if args["ind_func"] in ["2", "kMS"]:
        smallest_ev = [np.zeros(k) for i in range(it)]
        largest_ev = [np.zeros(k) for i in range(it)]
    plot_iteration_stats(
        logger_path,
        np.array([range(1, it + 1), ms_vals, func_vals, oas, stopping]),
        np.vstack(smallest_ev),
        np.vstack(largest_ev),
    )

    # copy data back to CPU to return results
    if args["use_gpu"]:
        u = cp.asnumpy(u)
        segmentation_new = cp.asnumpy(segmentation_new)

    # evaluation of results
    if (args["gt_file"] is not None) or nc_gt_flag:
        score = evaluate_segmentation(segmentation_new, seg_gt, args["ignore_label"])

    # remove all file handlers from loggers
    logger = logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    loggers = ["MS", "init", "outer", "indicator", "inner", "eval"]
    for log_name in loggers:
        log = logging.getLogger(log_name)
        log.handlers.clear()

    # return score, empty # used when run with several seeds, in that case comment out the following lines
    if (args["gt_file"] is not None) or nc_gt_flag:
        return -1 * score
    else:
        return segmentation_new, u


def main():
    """main function"""

    # parse arguments from command line and config file
    hyper_params = parse_args()

    # call MS segmentation function with given hyperparameters
    ms_segmentation(hyper_params)


if __name__ == "__main__":
    main()
