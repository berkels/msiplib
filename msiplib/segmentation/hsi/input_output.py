# pylint: disable=import-outside-toplevel,unsupported-membership-test

""" Module handling all functions processing input and output of MS segmentation framework """

import logging
import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity

from msiplib.segmentation.hsi.args_processing import get_image_index
from msiplib.io import saveArrayAsNetCDF
from msiplib.io.segmentation import (
    append_statistics_to_netcdf,
    load_seg_mask_from_image_file,
    saveColoredSegmentation,
    save_gt_colored_segmentation,
    save_mean_values_colored_segmentation,
    visualize_correct_wrong_pixels,
)
from msiplib.segmentation import rebuild_segment_numbering


def read_inputimage(data, gt_file=None, precision="float"):
    """
    function reads input image and ground truth if provided
    """
    ms_logger = logging.getLogger("MS")
    index, _ = get_image_index(data)
    if isinstance(data, str):
        ms_logger.info("Image path: %s", data)
    else:
        ms_logger.info("Image data given as %s.", type(data))
    ms_logger.info("Image index: %s", index)

    # set precision in form of used data type
    if precision == "float":
        dtype = np.float32
    else:
        dtype = np.float64

    # load image
    if isinstance(data, str):
        if data.endswith(".nc"):
            # case: data stored in a netcdf file
            # It is necessary to access the attribute data of the read array
            # since some functions cannot deal with masked arrays
            import netCDF4 as nc

            ncfile = nc.Dataset(data, "r")
            inputimage = ncfile["data"][:].data.astype(dtype)
            nc_gt_flag = "groundtruth" in ncfile.groups
        elif data.endswith(".mat"):
            # case: data stored in mat file
            from scipy.io import loadmat

            inputimage_mat = loadmat(data)
            key = list(inputimage_mat.keys())[-1]
            inputimage = inputimage_mat[key].astype(dtype)
            nc_gt_flag = False
        else:
            # case: data stored in an image file
            from imageio import imread
            # only keep the first three channels
            # in case RGB images have an alpha channel
            inputimage = (imread(data).astype(dtype))[:, :, :3]
            nc_gt_flag = False
    else:
        # assume that data is given as a numpy array
        from skimage import img_as_float32

        inputimage = img_as_float32(data).astype(dtype)
        nc_gt_flag = False

    # normalize image to have values between [0, 1]
    inputimage = rescale_intensity(inputimage, out_range=(0.0, 1.0))

    print(f"Image size: {inputimage.shape}")
    print(f"min(image) = {inputimage.min()}, max(image) = {inputimage.max()}")

    # if the image is grayscale, add a third dimension of size 1
    if inputimage.ndim == 2:
        inputimage = np.expand_dims(inputimage, axis=2)

    # load ground truth if available
    if nc_gt_flag or (gt_file is not None):
        if isinstance(gt_file, str):
            if nc_gt_flag:
                # load ground truth from netCDF file
                seg_gt = ncfile["groundtruth"]["segmentation_mask"][:].data
                k_gt = ncfile["groundtruth"]["num_segments"][:].data.item()
                ms_logger.info("Ground truth read from netCDF file.")
            elif gt_file.endswith(".mat"):
                # load ground truth from a given mat file
                gt_mat = loadmat(gt_file)
                gt_key = list(gt_mat.keys())[-1]
                seg_gt = gt_mat[gt_key].astype(np.uint8)
                k_gt = np.unique(seg_gt).shape[0]
                ms_logger.info("Ground truth read from mat file.")
            elif gt_file.endswith(".png"):
                # load ground truth from a png file assuming that each colors describes one class
                seg_gt = load_seg_mask_from_image_file(gt_file)
                k_gt = np.unique(seg_gt).shape[0]
                ms_logger.info("Ground truth read from png file.")
            else:
                # handle all other data types
                seg_gt = None
                k_gt = None
                ms_logger.info("Format of ground truth is not understood.")
        else:
            # assume ground truth is given as a numpy array
            seg_gt = gt_file
            k_gt = np.unique(seg_gt).shape[0]
            ms_logger.info("Ground truth given as numpy array.")

        # to avoid problems with segment numbering, rebuild it if ground truth is given
        if seg_gt is not None:
            seg_gt = rebuild_segment_numbering(seg_gt)
    else:
        seg_gt = None
        k_gt = None
        ms_logger.info("No ground truth provided.")

    return inputimage, seg_gt, k_gt, nc_gt_flag


def save_variables(f_name, u, segmentation, args, image=None, means=None, pcs=None, std_devs=None):
    """function to save variables on disk"""

    if std_devs is not None:
        weights = np.reciprocal(std_devs)
    else:  # if std_devs is None
        weights = std_devs
    if args["use_gpu"]:
        from cupy import asnumpy  # pylint:disable=import-error

        u = asnumpy(u)
        segmentation = asnumpy(segmentation)
        if image is not None:
            image = asnumpy(image)
        if means is not None:
            means = asnumpy(means)
        if pcs is not None:
            pcs = asnumpy(pcs)
        if weights is not None:
            weights = asnumpy(weights)

    saveArrayAsNetCDF(u, f"{f_name}_u.nc")
    save_segmentation(
        f_name, segmentation, image, args["gt_file"], args["ignore_label"], True, args["irreg"], means, pcs, weights
    )


def save_segmentation(
    filepath_name,
    seg,
    image=None,
    gt_file=None,
    ignore_label=None,
    seg_boundaries=False,
    irreg=False,
    means=None,
    pcs=None,
    std_devs=None,
):
    """save resulting segmentation as a png image and as netCDF4"""

    if image is not None:
        if gt_file is not None:
            gt = load_seg_mask_from_image_file(gt_file)
            save_mean_values_colored_segmentation(f"{filepath_name}.png", image, seg, seg_boundaries, gt, irreg)
        else:
            save_mean_values_colored_segmentation(f"{filepath_name}.png", image, seg)
    else:
        if (
            gt_file is not None
            and isinstance(gt_file, str)
            and gt_file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
        ):
            # save gt colored segmentation with boundaries
            save_gt_colored_segmentation(
                f"{filepath_name}.png",
                seg,
                gt_file,
                ignore_label=ignore_label,
                add_gt_bounds=seg_boundaries,
                bounds_col=None,
                irreg=irreg,
                ignore_pixels=False,
            )
            # save gt colored segmentation without boundaries and masked pixels with ignore label
            save_gt_colored_segmentation(
                f"{filepath_name}_masked.png",
                seg,
                gt_file,
                ignore_label=ignore_label,
                add_gt_bounds=False,
                bounds_col=None,
                irreg=irreg,
                ignore_pixels=True,
            )
            # save png with correctly and wrongly classfied pixels
            visualize_correct_wrong_pixels(f"{filepath_name}.png", seg, gt=None, gt_file=gt_file, ignore_label=0)

        else:
            saveColoredSegmentation(seg, f"{filepath_name}.png")

    saveArrayAsNetCDF(seg, f"{filepath_name}.nc")

    if (means is not None) or (pcs is not None) or (std_devs is not None):
        append_statistics_to_netcdf(f"{filepath_name}.nc", means, pcs, std_devs)


def plot_iteration_stats(out_path, data, smallest_ev, largest_ev):
    """ plots functional value, overall accuracy and value of stopping criterion against iteration """
    import matplotlib.pyplot as plt

    k = smallest_ev[0].shape[0]
    height = 8
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(height / 9 * 16 + 1, height))
    fig.suptitle('Plot iteration against MS functional value, functional value, OA and stopping criterion')

    # plot MS functional value
    ax1.plot(data[0], data[1])
    ax1.grid()
    ax1.set(xlabel="Iteration", ylabel="MS functional value", title="MS functional value")

    # plot Zach functional value
    ax2.plot(data[0], data[2])
    ax2.grid()
    ax2.set(xlabel="Iteration", ylabel="Functional value", title="Functional value")

    # plot overall accuracy
    ax3.plot(data[0], data[3])
    ax3.grid()
    ax3.set(xlabel="Iteration", ylabel="Overall accuracy", title="Overall accuracy")

    # plot stopping criterion
    ax4.plot(data[0], data[4])
    ax4.grid()
    ax4.set(xlabel="Iteration", ylabel="Stopping criterion", title="Stopping criterion")

    fig.savefig(f'{out_path}/plot_it_vs_stats.png')

    # also store the data used to generate the plots
    df = pd.DataFrame(data.T, columns=['iteration', 'ms_val', 'zach_val', 'oa', 'stop_crit'])
    df['iteration'] = df['iteration'].astype(int)       # convert iteration column to integer
    df.to_csv(f'{out_path}/plot_it_vs_stats_data.csv', sep='\t', index=False)

    # plot smallest eigenvalues
    fig_smallest_ev, ax_smallest = plt.subplots(1, 1, figsize=(height / 9 * 16 + 1, height))
    for l in range(k):
        ax_smallest.plot(data[0], smallest_ev[:, l], label=f'Segment {l + 1}')
    ax_smallest.grid()
    ax_smallest.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    ax_smallest.set(xlabel='Iteration', ylabel='Smallest eigenvalue', title='Smallest eigenvalues')
    fig_smallest_ev.savefig(f'{out_path}/plot_it_vs_smallest_evs.png')

    # plot largest eigenvalues
    fig_largest_ev, ax_largest = plt.subplots(1, 1, figsize=(height / 9 * 16 + 1, height))
    for l in range(k):
        ax_largest.plot(data[0], largest_ev[:, l], label=f'Segment {l + 1}')
    ax_largest.grid()
    ax_largest.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    ax_largest.set(xlabel='Iteration', ylabel='Largest eigenvalue', title='Largest eigenvalues')
    fig_largest_ev.savefig(f'{out_path}/plot_it_vs_largest_evs.png')

    plt.close()
