#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from skimage.io import imread, imsave
from skimage import img_as_float32
from msiplib.io import saveArrayAsNetCDF
from msiplib.io.segmentation import saveColoredSegmentation
from msiplib.segmentation import get_segmentation_mean_values, MumfordShah_segmentation


def main():
    parser = argparse.ArgumentParser(description="Multiphase Mumford-Shah with Zach's convexification.")
    parser.add_argument("filename", help="Input image file")
    parser.add_argument("--numsegments", default=2, dest="K", type=int, help="Number of segments.")
    parser.add_argument("--lambda", default=0.0025, dest="reg_par", type=float, help="Weight of the regularizer.")
    parser.add_argument("--maxiter", default=1000, dest="max_iter", type=int, help="Maximal number of iterations.")
    parser.add_argument(
        "--stopeps", default=1e-4, dest="stop_eps", type=float, help="Threshold of the stopping criterion."
    )
    parser.add_argument(
        "--usecupy",
        action="store_true",
        default=False,
        dest="use_cupy",
        help="Use cupy to do the computation on the GPU.",
    )
    args = parser.parse_args()

    input_image = img_as_float32(imread(args.filename))

    segmentation, u = MumfordShah_segmentation(
        input_image, args.K, args.reg_par, args.max_iter, args.stop_eps, args.use_cupy
    )

    saveColoredSegmentation(segmentation + 1, "segmentation_{}_{}.png".format(args.K, args.reg_par))

    meanValues = get_segmentation_mean_values(input_image, segmentation, args.K)
    colImg = (255 * meanValues[segmentation[:]]).astype("uint8")
    imsave("segmentation_col_{}_{}.png".format(args.K, args.reg_par), colImg)

    saveArrayAsNetCDF(segmentation, "segmentation_{}_{}.nc".format(args.K, args.reg_par))
    saveArrayAsNetCDF(u, "u_{}_{}.nc".format(args.K, args.reg_par))


if __name__ == "__main__":
    main()
