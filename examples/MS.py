#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_float32
import sklearn.cluster as skcl
from msiplib.optimization import pd_hybrid_grad_alg
from msiplib.proximal_mappings import project_canonical_simplex, project_unit_ball2D
from msiplib.finite_differences import gradient_FD2D, divergence_FD2D
from msiplib.io import saveArrayAsNetCDF
from msiplib.io.segmentation import saveColoredSegmentation
from msiplib.io.terminal import Timer
from msiplib.segmentation import get_segmentation_mean_values, ProxMapBinaryUCSegmentation


def MumfordShah_segmentation(inputimage, K, reg_par, max_iter, stop_eps, use_cupy=False):
    # If the image is grayscale, add a third dimension of size 1.
    if inputimage.ndim == 2:
        inputimage = np.expand_dims(inputimage, axis=2)

    # Rescale lambda to make it independent of the resolution of the input image.
    h = 1 / (max(inputimage.shape[:2]) - 1)
    lambda_ = reg_par / h

    # Determine initial estimates for the average colors using K-Means.
    reshapedimage = np.reshape(inputimage, newshape=(inputimage.shape[0]*inputimage.shape[1], inputimage.shape[2]))
    Kmeans = skcl.KMeans(n_clusters=K, random_state=42).fit(reshapedimage)
    meanValues = Kmeans.cluster_centers_

    # Compute the indicator functions.
    f = np.sum(np.square(inputimage[:, :, np.newaxis, :] - meanValues[np.newaxis, np.newaxis, :, :]), axis=3)
    f = f.astype(np.float32)

    # To use cupy, we need to convert our numpy array to a cupy array (which copies the data to the GPU)
    if use_cupy:
        import cupy as cp
        f = cp.array(f)

    # For the binary case, use the unconstrained strongly convex model instead of Zach.
    if K == 2:
        u = np.zeros_like(f, shape=inputimage.shape[:-1])
        prox = ProxMapBinaryUCSegmentation(f)

        # To be equivalent to Zach's model, we need to scale lambda by two.
        with Timer():
            u = pd_hybrid_grad_alg(u, prox.eval,
                                   project_unit_ball2D, gradient_FD2D, divergence_FD2D,
                                   2*lambda_, max_iter, stop_eps, PDHGAlgType=2, gamma=prox.gamma())[0]

        segmentation = (u > 0.5).astype(int)
    else:
        u = np.zeros_like(f)

        with Timer():
            u = pd_hybrid_grad_alg(u, lambda a, t: project_canonical_simplex(a - t * f),
                                   project_unit_ball2D, gradient_FD2D, divergence_FD2D,
                                   lambda_, max_iter, stop_eps)[0]

        segmentation = np.argmax(u, axis=2)

    # Convert the cupy arrays to numpy arrays (which copies the data from the GPU back to the CPU)
    if use_cupy:
        u = cp.asnumpy(u)
        segmentation = cp.asnumpy(segmentation)

    return segmentation, u


def main():
    parser = argparse.ArgumentParser(description="Multiphase Mumford-Shah with Zach's convexification.")
    parser.add_argument('filename', help='Input image file')
    parser.add_argument('--numsegments', default=2, dest='K', type=int,
                        help='Number of segments.')
    parser.add_argument('--lambda', default=0.0025, dest='reg_par', type=float,
                        help='Weight of the regularizer.')
    parser.add_argument('--maxiter', default=1000, dest='max_iter', type=int,
                        help='Maximal number of iterations.')
    parser.add_argument('--stopeps', default=1e-4, dest='stop_eps', type=float,
                        help='Threshold of the stopping criterion.')
    parser.add_argument('--usecupy', action='store_true', default=False, dest='use_cupy',
                        help='Use cupy to do the computation on the GPU.')
    args = parser.parse_args()

    inputimage = img_as_float32(imread(args.filename))

    segmentation, u = MumfordShah_segmentation(inputimage, args.K, args.reg_par, args.max_iter, args.stop_eps, args.use_cupy)

    saveColoredSegmentation(segmentation+1, 'segmentation_{}_{}.png'.format(args.K, args.reg_par))

    meanValues = get_segmentation_mean_values(inputimage, segmentation, args.K)
    colImg = (255 * meanValues[segmentation[:]]).astype('uint8')
    imsave('segmentation_col_{}_{}.png'.format(args.K, args.reg_par), colImg)

    saveArrayAsNetCDF(segmentation, 'segmentation_{}_{}.nc'.format(args.K, args.reg_par))
    saveArrayAsNetCDF(u, 'u_{}_{}.nc'.format(args.K, args.reg_par))


if __name__ == '__main__':
    main()
