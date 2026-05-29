'''
    Functions for input and output of segmentation related tasks.
'''

import warnings
import netCDF4 as nc4
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread, imwrite
from pathlib import Path
from skimage.measure import find_contours
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


def saveOverlayedSegmentation(image, u, filepath, alpha=0.75):
    """
    Draws the individual segments given by np.argmax(u, 2) with transparency
    on top of the image and saves the result as a png image.

    """
    # TODO: Merge with the convert_segmentation_to_image function

    # Convert the soft segmentation u into a hard segmentation h
    h = np.argmax(u, 2)

    # Create a colormap
    colormap = create_segmentation_colormap()

    # Issue a warning if u contains more segments than colors are available in
    # colormap
    if u.shape[2] > colormap.shape[0]:
        warning("Requested a colored segmentation with " + str(u.shape[2])
                + " segments, but the colormap contains only "
                + str(colormap.shape[0]) + " colors. Some of the segments will "
                "not be colorized in the resulting image.")
        tmp = np.zeros((u.shape[2], 3))
        tmp[0:colormap.shape[0], :] = colormap
        colormap = tmp

    # Draw the colored segments with an opacity of 1-alpha on top of the image
    coloredSegmentation = np.array([[255*(alpha*image[x, y]
                                          + (1-alpha)*colormap[h[x, y]])
                                     for y in range(image.shape[1])]
                                    for x in range(image.shape[0])]
                                   ).astype(np.uint8)

    # Save the colored segmentation to file
    imwrite(filepath, coloredSegmentation)


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
        
        # if ignore_label is None, add black as color for correcty classified or wrongly classified pixels, resp.
        # TODO: could catch the case where black is present in the colormap and another color has to be added.
        if ignore_label is None:
            colormap = np.insert(colormap, 0, np.array([0.0, 0.0, 0.0]), axis=0)

    else:
        colormap = create_segmentation_colormap()
        colormap = np.insert(colormap, 0, np.array([0.0, 0.0, 0.0]), axis=0)

    # permute segmentation labels according to ground truth
    _, perm = segmentation_scores(seg, gt, ignore_label, return_perm=True)
    seg_perm = permute_labels(seg, perm)
    if ignore_label is not None:
        seg_perm = add_ignore_label_to_seg_mask(seg_perm, gt, ignore_label)

    # if no ignore label is given, increase every label by 1 to be able to compensate for adding black to colormap
    if ignore_label is None:
        seg_perm_increased = seg_perm + 1
    else:
        seg_perm_increased = seg_perm

    # after permuting the segmentation labels, the difference of seg and gt can be computed.
    # every non-zero entry is wrongly classified.
    wrong_pxs = np.zeros(seg_perm.shape, dtype=np.uint8)
    wrong_pxs[(seg_perm - gt) != 0] = seg_perm_increased[(seg_perm - gt) != 0]
    wrong_pxs_im = convert_segmentation_to_image(wrong_pxs, colormap)
    imwrite(basename.replace('.png', '_wrong.png'), (255 * wrong_pxs_im).astype('uint8'))

    correct_pxs = np.zeros(seg_perm.shape, dtype=np.uint8)
    correct_pxs[(seg_perm - gt) == 0] = seg_perm_increased[(seg_perm - gt) == 0]
    correct_pxs_im = convert_segmentation_to_image(correct_pxs, colormap)
    imwrite(basename.replace('.png', '_correct.png'), (255 * correct_pxs_im).astype('uint8'))


def add_ignore_label_to_seg_mask(seg, gt, ignore_label=0):
    """
    Marks all pixels in the segmentation that are indicated as ignored by the ground truth
    by changing the labels of these pixels to 0 and increasing the other labels by 1.

    Args:
        seg: segmentation mask
        gt: ground truth
        ignore_label: ignore label used in the ground truth

    Returns
        segmentation mask where in ground truth ignored pixels are masked with label 0
    """
    # TODO: processes only ignore label 0
    t_mask = seg + 1
    t_mask[gt == ignore_label] = 0

    return t_mask
    # return rebuild_segment_numbering(t_mask)


def plot_RGB_feature_distributions(image, labels, ignore_label=None, size=1.5, cmap=None, means=None, pcs=None,
                                   std_devs=None):
    """ plots the feature distribution of an RGB image given segmentation labels """
    import plotly.graph_objects as go

    diff_labels = np.unique(labels)
    num_segments = diff_labels.shape[0]

    if cmap is None:
        cmap = (255 * create_segmentation_colormap())[:len(diff_labels)]
    else:
        if num_segments > cmap.shape[0]:
            for i in range(num_segments - cmap.shape[0]):
                new_col = 1 / 2 * (cmap[i] + cmap[i + 1])
                cmap = np.concatenate((cmap, new_col[np.newaxis]), axis=0)

    fig = go.Figure()

    # if colormap contains grayscale values, convert the map to an rgb map
    if cmap.ndim == 1:
        cmap = np.transpose(np.broadcast_to(cmap, (3, cmap.shape[0])))

    # plot pixels with ignore label
    if ignore_label is not None:
        vectors = image[labels == ignore_label]
        color = 'rgb({}, {}, {})'.format(cmap[ignore_label, 0], cmap[ignore_label, 1], cmap[ignore_label, 2])
        fig.add_trace(go.Scatter3d(x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
                                   name=('{}'.format('Pxs with ignore label')), mode='markers',
                                   marker_color=(color)))
        # remove color used by ignored pixels
        cmap = cmap[diff_labels != ignore_label]
        diff_labels = diff_labels[diff_labels != ignore_label]

    # add a trace for every segment
    for i, l in enumerate(diff_labels):
        vectors = image[labels == l]
        color = 'rgb({}, {}, {})'.format(cmap[i, 0], cmap[i, 1], cmap[i, 2])
        fig.add_trace(go.Scatter3d(x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
                                   name=('Segment {}'.format(i + 1)), mode='markers',
                                   marker_color=(color)))

    fig.update_traces(mode='markers', marker_line_width=2, marker_size=size)

    if means is not None:
        for i in range(len(diff_labels)):
            color = 'rgb({}, {}, {})'.format(cmap[i, 0], cmap[i, 1], cmap[i, 2])
            fig.add_trace(go.Scatter3d(x=[means[i, 0]], y=[means[i, 1]], z=[means[i, 2]],
                                    name=('Mean of segment {}'.format(i + 1)), marker_color=(color)))

    # if pcs is not None and std_devs is not None:
    #     for i in range(len(diff_labels)):
    #         for j in range(3):
    #             fig = fig.add_trace(go.Cone(
    #                                 x=[means[i, 0]],
    #                                 y=[means[i, 1]],
    #                                 z=[means[i, 2]],
    #                                 u=[std_devs[i, j] * pcs[i, j, 0]],
    #                                 v=[std_devs[i, j] * pcs[i, j, 1]],
    #                                 w=[std_devs[i, j] * pcs[i, j, 2]],
    #                                 sizemode="absolute",
    #                                 sizeref=2,
    #                                 anchor="tail",
    #                                 name=('PC {} of segment {}'.format(j + 1, i + 1))))

    # fig.update_layout(
    #     scene=dict(domain_x=[0, 1],
    #                camera_eye=dict(x=-1.57, y=1.36, z=0.58)))

    # Set options common to all traces
    fig.update_layout(title='Feature distribution', yaxis_zeroline=False, xaxis_zeroline=False)

    fig.show()


def saveSegmentationContours(image, u, filepath):
    """Draws the outline of the segment boundaries given by np.argmax(u, 2) on
       top of the image and saves the result as a png image."""
    # Convert the soft segmentation u into a hard segmentation h
    h = np.argmax(u, 2)

    # This uses ideas from https://stackoverflow.com/a/34769840 to render the background
    # image exactly with its pixel resolution.

    # On-screen, things will be displayed at 80dpi regardless of what we set here
    # This is effectively the dpi for the saved figure. We need to specify it,
    # otherwise `savefig` will pick a default dpi based on your local configuration
    dpi = 80

    height = image.shape[0]
    width = image.shape[1]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Add the image as background so that the segmentation contours can be drawn
    # on top.
    ax.imshow(image*255, extent=[-0.5, image.shape[1]-0.5, -0.5,
                                 image.shape[0]-0.5],
              cmap='gray', vmin=0, vmax=255, interpolation='nearest')

    # Find the contours of all segments
    for i in range(u.shape[2]):
        # Create a binary image from the hard segmentation
        feature_i_segments = np.zeros(image.shape)
        feature_i_segments[h == i] = 1

        # Use the binary image to find the segment contours with find_contours
        # and plot the contours on top of the image
        feature_i_contours = find_contours(feature_i_segments, 0.5)
        for c in feature_i_contours:
            c[:, [0, 1]] = c[:, [1, 0]]
            c[:, 1] = image.shape[0] - c[:, 1] - 1
            plt.plot(*(c.T), linewidth=1, color='red')

    # Save the figure as an image (with the format given by the extension in
    # filepath)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)


def saveSoftSegmentationImages(image, u, directory):
    """Saves images of each segment in the soft segmentation u to the given
       directory."""
    # Remove a trailing "/" sign from directory if necessary
    if directory[-1] == "/":
        directory = directory[:-1]

    # Create the directory to which the images are saved
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Save images of each segment, either using u as the transparency or by
    # saving u directly
    for k in range(u.shape[2]):
        imwrite(directory + "/image_segment_" + str(k) + ".png",
                        np.uint8(255 * image * u[:, :, k]))
        imwrite(directory + "/noimage_segment_" + str(k) + ".png",
                        np.uint8(255 * u[:, :, k]))


def saveSegmentation(image, u, directory, filename_base, alpha=0.75):
    """
    Saves the soft segmentation of image given by u in various formats

    The input image is assumed to be a three dimensional ndarray, where the
    third dimension corresponds to the individual color channels.
    """
    # Convert the input image to a grayscale image
    if image.shape[2] == 3:
        img = image[:, :, 0]
        #img = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3
    else:
        img = image[:, :, 0]

    # Add a trailing "/" sign to the directory path if necessary
    if directory[-1] != "/":
        directory += "/"

    saveOverlayedSegmentation(img, u, directory + "colorized_" + filename_base
                            + ".png", alpha)
    for i in range(image.shape[-1]):
        Path(directory + str(i)).mkdir(parents=True, exist_ok=True)
        saveSegmentationContours(image[:, :, i], u,
                                 directory + str(i) + "/contours_" + filename_base + ".svg")
    saveSoftSegmentationImages(img, u, directory + "segments_"
                               + filename_base)
