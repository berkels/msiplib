# pylint: disable=import-outside-toplevel,unsupported-membership-test

''' Module handling all functions processing input and output of MS segmentation framework '''

import logging
import numpy as np
from msiplib.segmentation.hsi.args_processing import get_image_index
from skimage.exposure import rescale_intensity
from msiplib.io import saveArrayAsNetCDF
from msiplib.io.segmentation import (append_statistics_to_netcdf, load_seg_mask_from_image_file,
                                     saveColoredSegmentation, save_gt_colored_segmentation,
                                     save_mean_values_colored_segmentation, visualize_correct_wrong_pixels)
from msiplib.segmentation import rebuild_segment_numbering


def read_inputimage(filename, gt_file=None, precision='float'):
    '''
        function reads input image and ground truth if provided
    '''
    ms_logger = logging.getLogger('MS')
    index = get_image_index(filename)
    ms_logger.info('Image path: %s', filename)
    ms_logger.info('Image index: %s', index)

    # set precision in form of used data type
    if precision == 'float':
        dtype = np.float32
    else:
        dtype = np.float64

    # load image
    if filename.endswith('.nc'):
        # case: data stored in a netcdf file. It is necessary to access the attribute data of the read array since some
        # functions cannot deal with masked arrays
        import netCDF4 as nc
        ncfile = nc.Dataset(filename, 'r')
        inputimage = ncfile['data'][:].data.astype(dtype)
        nc_gt_flag = ('groundtruth' in ncfile.groups)
    elif filename.endswith('.mat'):
        # case: data stored in mat file
        from scipy.io import loadmat
        inputimage_mat = loadmat(filename)
        key = list(inputimage_mat.keys())[-1]
        inputimage = inputimage_mat[key].astype(dtype)
        nc_gt_flag = False
    else:
        # case: data stored in an image file
        from imageio import imread
        inputimage = imread(filename).astype(dtype)
        nc_gt_flag = False

    # normalize image to have values between [0, 1]
    inputimage = rescale_intensity(inputimage, out_range=(0.0, 1.0))

    print(f'min(image) = {inputimage.min()}, max(image) = {inputimage.max()}')

    # if the image is grayscale, add a third dimension of size 1
    if inputimage.ndim == 2:
        inputimage = np.expand_dims(inputimage, axis=2)

    # load ground truth if available
    if nc_gt_flag or (gt_file is not None):
        if nc_gt_flag:
            seg_gt = ncfile['groundtruth']['segmentation_mask'][:].data
            k_gt = ncfile['groundtruth']['num_segments'][:].data.item()
            ms_logger.info('Ground truth read from netCDF file.')
        elif gt_file.endswith('.mat'):
            gt_mat = loadmat(gt_file)
            gt_key = list(gt_mat.keys())[-1]
            seg_gt = gt_mat[gt_key].astype(np.uint8)
            k_gt = np.unique(seg_gt).shape[0]
            ms_logger.info('Ground truth read from mat file.')
        elif gt_file.endswith('.png'):
            seg_gt = load_seg_mask_from_image_file(gt_file)
            k_gt = np.unique(seg_gt).shape[0]
            ms_logger.info('Ground truth read from png file.')
        else:
            seg_gt = None
            k_gt = None
            ms_logger.info('Format of ground truth is not understood.')

        # to avoid problems with segment numbering, rebuild it if ground truth is given
        if seg_gt is not None:
            seg_gt = rebuild_segment_numbering(seg_gt)
    else:
        seg_gt = None
        k_gt = None
        ms_logger.info('No ground truth provided.')

    return inputimage, seg_gt, k_gt, nc_gt_flag


def save_segmentation(filepath_name, seg, image=None, gt_file=None, ignore_label=None, seg_boundaries=False,
                      irreg=False, means=None, pcs=None, std_devs=None):

    ''' save resulting segmentation as a png image and as netCDF4 '''

    if image is not None:
        if gt_file is not None:
            gt = load_seg_mask_from_image_file(gt_file)
            save_mean_values_colored_segmentation(f'{filepath_name}.png', image, seg, seg_boundaries, gt, irreg)
        else:
            save_mean_values_colored_segmentation(f'{filepath_name}.png', image, seg)
    else:
        if gt_file is not None and gt_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # save gt colored segmentation with boundaries
            save_gt_colored_segmentation(f'{filepath_name}.png', seg, gt_file, ignore_label=ignore_label,
                                         add_gt_bounds=seg_boundaries, bounds_col=None, irreg=irreg,
                                         ignore_pixels=False)
            # save gt colored segmentation without boundaries and masked pixels with ignore label
            save_gt_colored_segmentation(f'{filepath_name}_masked.png', seg, gt_file, ignore_label=ignore_label,
                                         add_gt_bounds=False, bounds_col=None, irreg=irreg,
                                         ignore_pixels=True)
            # save png with correctly and wrongly classfied pixels
            visualize_correct_wrong_pixels(f'{filepath_name}.png', seg, gt=None, gt_file=gt_file, ignore_label=0)

        else:
            saveColoredSegmentation(seg, f'{filepath_name}.png')

    saveArrayAsNetCDF(seg, f'{filepath_name}.nc')

    if (means is not None) or (pcs is not None) or (std_devs is not None):
        append_statistics_to_netcdf(f'{filepath_name}.nc', means, pcs, std_devs)
