""" module contains utility functions that are needed to run the hsi segmentation code """

import logging
import sys
import timeit

import numpy as np

from msiplib.finite_differences import gradient_FD2D, divergence_FD2D
from msiplib.segmentation.indicator_functions import (
    anisotropic_2norm,
    euclidean_norm,
    epsAMS,
    pca_eps_discard_norm,
)
from msiplib.metrics import segmentation_scores
from msiplib.optimization import pd_hybrid_grad_alg
from msiplib.proximal_mappings import project_canonical_simplex, project_unit_ball2D, ProxMapBinaryUCSegmentation



def evaluate_segmentation(segmentation, seg_gt=None, ignore_label=None):
    """function that evaluates found segmentation and writes scores and configuration to log file"""

    logger = logging.getLogger("eval")
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)

    # log scores
    logger.info("Evaluation")

    # If no ground truth is provided, use internal statistics to evaluate segmentation (clustering)
    if seg_gt is None:
        logger.info("No ground truth available. Only internal evaluation.")
    else:
        # compute segmentation scores
        scores = segmentation_scores(segmentation, seg_gt, ignore_label=ignore_label)

        if np.unique(segmentation).shape[0] > np.unique(seg_gt).shape[0]:
            logger.warning("More segments in segmentation than in ground truth!")
        logger.info("Overall accuracy: %s", scores["overallAcc"])
        logger.info("Mean class accuracy: %s", scores["meanClassAcc"])
        logger.info("mIoU: %s", scores["mIoU"])
        logger.info("mean Dice: %s", scores["mDice"])
        logger.info("Kappa coefficient: %s", scores["kappa"])
        logger.info("Variation of information: %s", scores["VI"])
        logger.info("Normalized mutual information: %s", scores["NMI"])
        logger.info("Adjusted rand score: %s", scores["ARS"])
        return scores["overallAcc"]
        # return scores # used when run with several seeds, in that case comment out the other return statement


def compute_indicator_functions(
    image, segmentation, k, args, valid_mask=None, means=None, pcs=None, weights=None, kernel_matrix=None
):
    """computes indicator function"""

    logger = logging.getLogger("indicator")

    # initialize array for storing the indicator values
    f = np.zeros_like(image, shape=(image.shape[0], image.shape[1], k), dtype=image.dtype)

    # if an anisotropic indicator function is chosen, compute minimum of log det cov and subtract it to ensure
    # that the minimum of the indicator functions is 0
    # variants with a kernel involved need extra handling as in this case dimensionality of covariance matrix may vary
    if "eps" in args["ind_func"]:
        min_log_det_cov = 2 * image.shape[-1] * np.log(args["ind_eps"])
        if ("normal" in args["ind_func"]) and (args["ind_eps"] > 1):
            min_log_det_cov = -1 * min_log_det_cov

    # compute indicator values for all pixels and segments
    for l in range(k):
        # irregular region is added at the end of array f (last segment)
        if args["irreg"] and l == k - 1:
            # handle irregular regions
            # omega_hat is computed outside of the function to allow application of numba
            omega_hat = np.argmin(f[:, :, : k - 1], axis=2).astype(np.uint32)
        else:
            # check if valid pixels for segment l are available
            if image[np.logical_and(segmentation == l, valid_mask)].size == 0:
                logger.warning("No valid pixels in segment! Stop iteration.")
            elif args["ind_func"] != "2" and image[np.logical_and(segmentation == l, valid_mask)].shape[0] == 1:
                logger.warning("Only one valid pixel in segment! Stop iteration.")
            if args["ind_func"] == "2":
                # the typical 2-norm is used as indicator function
                f[:, :, l] = euclidean_norm(image, segmentation, l, valid_mask)
            elif args["ind_func"] == "anisotropic-eps-inverse":
                logger.info("Segment %s", l)
                # the anisotropic 2-norm is used as indicator function where components with a standard deviation
                # smaller than epsilon are scaled by 1/eps
                f[:, :, l] = (
                    anisotropic_2norm(
                        image,
                        segmentation,
                        l,
                        args["ind_eps"],
                        True,
                        valid_mask,
                        args["means"],
                        args["components"],
                        args["variances"],
                        args["trim_proportion"],
                    )
                    - min_log_det_cov
                )
            elif args["ind_func"] == "anisotropic-eps-normal":
                logger.info("Segment %s", l)
                # the anisotropic 2-norm is used as indicator function where components with a standard deviation
                # smaller than epsilon are scaled by eps
                f[:, :, l] = (
                    anisotropic_2norm(
                        image,
                        segmentation,
                        l,
                        args["ind_eps"],
                        False,
                        valid_mask,
                        args["means"],
                        args["components"],
                        args["variances"],
                        args["trim_proportion"],
                    )
                    - min_log_det_cov
                )
            elif args["ind_func"] == "anisotropic-eps-discard":
                logger.info("Segment %s", l)
                # the anisotropic 2-norm is used as indicator function where components with a standard deviation
                f[:, :, l] = pca_eps_discard_norm(image, segmentation, l, args["ind_eps"], valid_mask)
            elif args["ind_func"] == "epsAMS":
                logger.info("Segment %s", l)
                # the non-squared anisotropic 2-norm epsAMS is used as indicator function where components with a standard
                # deviation smaller than epsilon are scaled by 1/eps
                f[:, :, l], means[l], pcs[l], weights[l] = epsAMS(
                    image,
                    segmentation,
                    l,
                    args["ind_eps"],
                    means,
                    pcs,
                    weights,
                    tol=1e-05,
                    max_iter=args["ind_params"][0],
                    valid_mask=valid_mask,
                )
                np.subtract(f[:, :, l], min_log_det_cov, out=f[:, :, l])

    if "AMS" in args["ind_func"]:
        # in the case of epsAMS return means, PCs and weights to reuse them
        # as initial guesses in the next iteration
        return f, means, pcs, weights
    else:
        return f


def update_u(u, f, lambda_, max_iter=1000, eps=1e-06, convexification="zach", pdhgtype=1):
    """functions performs an update of u, i.e., one inner iteration"""
    inner_logger = logging.getLogger("inner")

    tstart = timeit.default_timer()
    if convexification == "zach":
        u, _, it = pd_hybrid_grad_alg(
            u,
            lambda a, t: project_canonical_simplex(a - t * f),
            project_unit_ball2D,
            gradient_FD2D,
            divergence_FD2D,
            lambda_,
            max_iter,
            eps,
            L=8,
            PDHGAlgType=1,
        )
    elif convexification == "binaryUC":
        prox = ProxMapBinaryUCSegmentation(f)
        # multiplication of lambda_ by 2 to make the scaling of lambda_ similar to Zach
        u, _, it = pd_hybrid_grad_alg(
            u,
            prox.eval,
            project_unit_ball2D,
            gradient_FD2D,
            divergence_FD2D,
            2 * lambda_,
            max_iter,
            eps,
            PDHGAlgType=pdhgtype,
            gamma=prox.gamma(),
        )

    time_elapsed = timeit.default_timer() - tstart
    inner_logger.info("Time elapsed: %ss", time_elapsed)
    inner_logger.info("Number of iterations: %s", it)

    return u
