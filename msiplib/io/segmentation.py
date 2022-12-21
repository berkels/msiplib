'''
    Functions for input and output of segmentation related tasks.
'''

import warnings
import netCDF4 as nc4
import numpy as np
from imageio import imread, imwrite
from skimage.segmentation import find_boundaries
from msiplib.io import read_image
from msiplib.metrics import segmentation_scores
from msiplib.segmentation import convert_image_to_segmentation_labels, convert_segmentation_to_image,\
    create_segmentation_colormap, get_segmentation_mean_values, permute_labels, rebuild_segment_numbering


def append_gt_to_netcdf(filename, segmentation_mask, ignore_label=None):
    '''
        Appends a segmentation ground truth to an existing netcdf file containing the raw image data

        Args:
            filename: name of the netCDF file that contains the raw image
            segmentation_mask: a segmentation mask in the form of a matrix where the entries correspond to the
                               spatial positions of the pixels in the image and contain integers to indicate
                               the segments, the pixels belong to.
            ignore_label: pixels with ignore_label in the ground truth are considered to be unlabelled
    '''
    a = nc4.Dataset(filename, 'a')

    # check whether segmentation_mask and image already stored in the file have the same dimensions
    if (a['data'][:].shape[0] != segmentation_mask.shape[0]) and (a['data'][:].shape[1] != segmentation_mask.shape[1]):
        warnings.warn('Original image and segmentation masks have different sizes.')

    # add group for ground truth to the netcdf file
    g = a.createGroup('groundtruth')

    # add segmentation mask
    g.createDimension('x', segmentation_mask.shape[0])
    g.createDimension('y', segmentation_mask.shape[1])
    seg = g.createVariable('segmentation_mask', 'u1', ('x', 'y'))
    seg[:] = segmentation_mask

    # add number of segments
    # if ignore label is provided, create a variable and reduce the number of segments by 1.
    num_segments = g.createVariable('num_segments', 'u1')
    num_segs = np.unique(segmentation_mask).shape[0]
    if ignore_label is not None:
        ignore_l = g.createVariable('ignore_label', 'u1')
        ignore_l[:] = ignore_label
        num_segs -= 1

    num_segments[:] = num_segs

    a.close()

def read_gt_from_netcdf(filename):
    '''
        Reads the segmentation ground truth from an existing netcdf file containing the raw image data

        Args:
            filename: name of the netCDF file that contains the raw image and its ground truth
        Returns:

    '''
    d = nc4.Dataset(filename, 'r')
    gt = d['groundtruth']['segmentation_mask'][:].data

    if 'ignore_label' in d['groundtruth'].variables:
        return gt, d['groundtruth']['ignore_label'][:].data.item()
    else:
        return gt


def append_statistics_to_netcdf(filename, means, pcs=None, std_devs=None):
    '''
        Appends additional information (statistics) like mean, principal components and weights in the respective
        directions to an existing netcdf file containing a segmentation mask

        Args:
            filename: name of the netCDF file that contains the raw image
            means: an array containing the mean features of the segments
            pcs: a three-dimensional array containing the principal components for each segment
            std_devs: an array containing the standard deviations along each of the principal components
    '''
    a = nc4.Dataset(filename, 'a')

    # add group for ground truth to the netcdf file
    g = a.createGroup('statistics')

    # add means
    g.createDimension('k', means.shape[0])
    g.createDimension('L', means.shape[1])
    m = g.createVariable('means', means.dtype, ('k', 'L'))
    m[:] = means

    # add principal components
    if pcs is not None:
        p_comps = g.createVariable('principal_components', pcs.dtype, ('k', 'L', 'L'))
        p_comps[:] = pcs

    # add standard deviations
    if std_devs is not None:
        std = g.createVariable('standard_deviations', std_devs.dtype, ('k', 'L'))
        std[:] = std_devs

    a.close()


def load_seg_mask_from_image_file(path):
    '''
        Creates the segmentation mask from an RGB image by assuming that a specific RGB array
        stands for a specific segment

        Args:
            path: path to RGB image file containing the visualized segmentation
        Returns:
            an H x W array where each entry is an integer giving the label for the corresponding pixel
    '''
    im = imread(path)

    return convert_image_to_segmentation_labels(im)


def saveColoredSegmentation(segmentation_mask, filename, colormap=None, ignore_label=None, add_true_boundaries=False,
                            gt=None, boundary_color=None):

    col_img = convert_segmentation_to_image(segmentation_mask, colormap)

    # if ignore label for unlabeled pixels and ground truth are provided, mask these pixels by making them black
    if ignore_label is not None and gt is not None:
        col_img[gt == ignore_label] = np.array([0.0, 0.0, 0.0])

    # add true segment boundaries if ground truth provided
    if add_true_boundaries:
        if gt is None:
            raise ValueError('Unable to draw true segment boundaries. No ground truth provided.')

        # if no color for boundary is provided, set it to black
        if boundary_color is None:
            bd_col = np.array([0.0, 0.0, 0.0])
        else:
            bd_col = boundary_color

        # find and add boundaries to colored segmentation
        boundaries = find_boundaries(gt, connectivity=1, mode='thick')
        col_img[boundaries] = bd_col

    imwrite(filename, (255 * col_img).astype('uint8'))


def save_mean_values_colored_segmentation(filename, image, seg_mask, add_true_boundaries=False, gt=None,
                                          boundary_color=None, irreg=False):
    '''
        Save colored segmentation with the resulting mean values as color map

        Args:
            filename: path where file will be stored
            image: the image corresponding to the segmentation mask
            seg_mask: segmentation that shall be stored as a colored image
            gt: ground truth
            add_true_boundaries: if true and ground truth is provided, add boundaries of true segments to file
            boundary_color: specify color of boundaries
            irreg: if true, an extra segment with irregular pixels is specified by highest segment index
    '''

    # create color map from the image
    num_segments = np.unique(seg_mask).shape[0]
    seg_mask = rebuild_segment_numbering(seg_mask)
    colormap = get_segmentation_mean_values(image, seg_mask, num_segments)

    # if irregular pixels are marked, add a color for this segment to colormap
    if irreg:
        irreg_col = 1 / 2 * (colormap[0] + colormap[1])
        colormap = np.concatenate((colormap, irreg_col[np.newaxis]), axis=0)

    # save colored segmentation using this colormap
    saveColoredSegmentation(seg_mask, filename, colormap, add_true_boundaries, gt, boundary_color)


def save_gt_colored_segmentation(filename, seg, gt_file, ignore_label=None, add_gt_bounds=False, bounds_col=None,
                                 irreg=False, ignore_pixels=False):
    '''
        Save colored segmentation with matching colors from ground truth file as color map

        Args:
            filename: path where file will be stored
            seg: segmentation that shall be stored as a colored image
            gt_file: image file that contains the colored ground truth
            ignore_label: label in ground truth indicating
            add_gt_bounds: if true, add segment boundaries from ground truth to segmentation
            bounds_col: RGB color for the boundaries (values in :math:`[0, 1]`)
            irreg: if true, an extra segment with irregular pixels is specified by highest segment index
            ignore_pixels: if true, pixels carrying ignore label given by ground truth are colored in black
    '''
    gt_im = read_image(gt_file) / 255
    gt = convert_image_to_segmentation_labels(gt_im)

    # get different colors in ground truth
    if len(gt_im.shape) != 3:
        # grayscale image
        gt_colors = np.unique(gt_im.reshape(-1), axis=0)
    else:
        # RGB image
        gt_colors = np.unique(gt_im.reshape((-1, gt_im.shape[-1])), axis=0)

    # if ignore label is given, remove the corresponding color extracted from the ground truth file
    if ignore_label is not None:
        # if pixels with ignore label should be masked, remember color of ignored pixels in ground truth file
        # to add it to the color map after permutation of colors of segments
        if ignore_pixels:
            ignore_color = np.unique(gt_im[gt == ignore_label], axis=0)
        gt_colors = gt_colors[np.any(gt_colors != np.unique(gt_im[gt == ignore_label], axis=0), axis=1)]

    # get permutation of segmentation to correctly match the different classes of ground truth and segmentation
    _, perm = segmentation_scores(seg, gt, ignore_label=ignore_label, return_perm=True)
    seg_permuted = permute_labels(seg, perm)

    # if irregular pixels are marked, add a color for this segment to colormap
    if irreg:
        irreg_col = 1 / 2 * (gt_colors[0] + gt_colors[1])
        gt_colors = np.concatenate((gt_colors, irreg_col[np.newaxis]), axis=0)

    # if pixels with ignore label should be masked in colored segmentation, add the ignore color from the ground
    # truth file and also add the ignore label to the segmentation mask
    if ignore_pixels:
        if ignore_label is None:
            raise ValueError('Unable to mask ignored pixels. No ignore label provided.')
        gt_colors = np.insert(gt_colors, 0, ignore_color, axis=0)
        seg_permuted = add_ignore_label_to_seg_mask(seg_permuted, gt, ignore_label)

        saveColoredSegmentation(seg_permuted, filename, gt_colors, ignore_label, add_gt_bounds, gt, bounds_col)
    else:
        saveColoredSegmentation(seg_permuted, filename, gt_colors, None, add_gt_bounds, gt, bounds_col)


def visualize_correct_wrong_pixels(basename, seg, gt=None, gt_file=None, ignore_label=None):
    '''
        function that visualizes the correctly and wrongly classified pixels.

        Args:
            basename: filename where the visualizations should be stored.
                      Given name is extended by _wrong and _correct, resp.
            seg: segmentation map
            gt: ground truth
            gt_file: filename of ground truth visualization stored as an image.
                     if provided, colors for visualization will be taken from this visualization of ground truth.
            ignore_label: provide ignore label such that unlabeled pixels in the ground truth are ignored
    '''

    # either ground truth as np.array or gt file has to be given
    if gt is None and gt_file is None:
        raise RuntimeError('Grund truth has to be provided as either numpy.array or image file. None was given!')

    # extract colormap from gt_file if it is given. otherwise use default from msiplib
    if gt_file is not None:
        gt_im = read_image(gt_file) / 255
        gt = convert_image_to_segmentation_labels(gt_im)

        # get different colors in ground truth file
        if len(gt_im.shape) != 3:
            # grayscale image
            colormap = np.unique(gt_im.reshape(-1), axis=0)
        else:
            # RGB image
            colormap = np.unique(gt_im.reshape((-1, gt_im.shape[-1])), axis=0)
    else:
        colormap = create_segmentation_colormap()
        colormap = np.insert(colormap, 0, np.array([0.0, 0.0, 0.0]), axis=0)

    # permute segmentation labels according to ground truth
    _, perm = segmentation_scores(seg, gt, ignore_label, return_perm=True)
    seg_perm = permute_labels(seg, perm)
    if ignore_label is not None:
        seg_perm = add_ignore_label_to_seg_mask(seg_perm, gt, ignore_label)

    # after permuting the segmentation labels difference of seg and gt can be computed.
    # every non-zero entry is wrongly classified.
    wrong_pxs = np.zeros(seg_perm.shape, dtype=np.uint8)
    wrong_pxs[(seg_perm - gt) != 0] = seg_perm[(seg_perm - gt) != 0]
    wrong_pxs_im = convert_segmentation_to_image(wrong_pxs, colormap)
    imwrite(basename.replace('.png', '_wrong.png'), (255 * wrong_pxs_im).astype('uint8'))

    correct_pxs = np.zeros(seg_perm.shape, dtype=np.uint8)
    correct_pxs[(seg_perm - gt) == 0] = seg_perm[(seg_perm - gt) == 0]
    correct_pxs_im = convert_segmentation_to_image(correct_pxs, colormap)
    imwrite(basename.replace('.png', '_correct.png'), (255 * correct_pxs_im).astype('uint8'))


def add_ignore_label_to_seg_mask(seg, gt, ignore_label=0):
    '''
        Marks all ignored pixels indicated by the ground truth as 'ignored' by changing the labels of these pixels to 0
        and increasing the other labels by 1.
        Args:
            seg: segmentation mask
            gt: ground truth
            ignore_label: ignore label used in the ground truth
        Returns
            segmentation mask where in ground truth ignored pixels are masked with label 0
    '''
    t_mask = seg + 1
    t_mask[gt == ignore_label] = 0

    return t_mask
    # return rebuild_segment_numbering(t_mask)
