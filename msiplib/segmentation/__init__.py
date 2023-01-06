'''
    collection of functions particularly useful for segmentation
'''

import numpy as np
import matplotlib.pyplot as plt
import imageio

from skimage.measure import find_contours
from pathlib import Path


def convert_image_to_segmentation_labels(image_labels):
    '''
        converts an image into segmentation labels by assuming that each appearing color indicates a segment

        Args:
            image_labels: a grayscale or (a)RGB image where each color stands for a specific segment

        Returns:
            a matrix of the same dimension as the image where each entry contains an integer indicating
            the segment of the pixel
    '''
    # recover class labels from the different colors
    if image_labels.ndim == 2:
        # case: grayscale image
        _, classlabels = np.unique(image_labels.reshape(-1), return_inverse=True, axis=0)
    else:
        # case: (a)RGB image
        _, classlabels = np.unique(image_labels.reshape((-1, image_labels.shape[-1])), return_inverse=True, axis=0)

    return classlabels.reshape((image_labels.shape[0], image_labels.shape[1])).astype(np.uint8)


def convert_segmentation_to_image(segmentation_mask, colormap=None):
    if colormap is None:
        colormap = create_segmentation_colormap()
    return colormap[segmentation_mask[:]]


def convert_segmentation_to_partition(segmentation, order='C'):
    '''
        convert a segmentation mask to a partition of the pixel domain by enumerating the pixels
        row-wise or column-wise.

        Args:
            segmentation: segmentation mask defining the segment labels
            order: enumerate pixels row-wise ('C') or column-wise ('F'), same as in numpy.reshape function

        Returns:
            a list of lists containing the pixel indices of the pixels that belong to the segment
    '''

    pixels = np.arange(np.prod(segmentation.shape), dtype=np.uint64)
    segmentation_flat = np.reshape(segmentation, (np.prod(segmentation.shape)), order=order)
    labels = np.unique(segmentation)
    partition = [pixels[segmentation_flat == l] for l in labels]

    return partition


def get_segmentation_mean_values(input_image, segmentation, num_segments):
    ''' yields mean feature vectors of segments '''
    mean_values = np.zeros_like(input_image, shape=(num_segments, input_image.shape[2] if input_image.ndim == 3 else 1), dtype=input_image.dtype)
    for k in range(num_segments):
        mean_values[k, :] = np.mean(input_image[segmentation == k], axis=0)
    return mean_values


def create_segmentation_colormap():
    '''
        yields a color map for the visualization of segmentations.

        Colors are taken from "A Colour Alphabet and the Limits of Colour Coding" by Paul Green-Armytage, 2010.
    '''
    return np.array([[0.94117647, 0.63921569, 1.],[0., 0.45882353, 0.8627451], [0.6, 0.24705882, 0.],
                     [0.29803922, 0., 0.36078431], [0., 0.36078431, 0.19215686], [0.16862745, 0.80784314, 0.28235294],
                     [1., 0.8, 0.6], [0.50196078, 0.50196078, 0.50196078], [0.58039216, 1., 0.70980392],
                     [0.56078431, 0.48627451, 0.], [0.61568627, 0.8, 0.], [0.76078431, 0., 0.53333333],
                     [0., 0.2, 0.50196078], [1., 0.64313725, 0.01960784], [1., 0.65882353, 0.73333333],
                     [0.25882353, 0.4, 0.], [1., 0., 0.0627451], [0.36862745, 0.94509804, 0.94901961],
                     [0., 0.6, 0.56078431], [0.87843137, 1., 0.4], [0.45490196, 0.03921569, 1.], [0.6, 0., 0.],
                     [1., 1., 0.], [1., 0.31372549, 0.01960784]])


def create_segmentation_colormap_no_gray():
    return np.delete(create_segmentation_colormap(), 7, axis=0)




def create_segmentation_colormap_dictionary():
    colormap = create_segmentation_colormap()
    keys = np.arange(colormap.shape[0])
    return dict(zip(keys, colormap))


def permute_labels(segmentation_mask, perm):
    '''
        function changes labels in segmentation mask according to given permutation.

        Args:
            segmentation_mask: an array containing unsigned integers as class labels

            perm: the permutation that is applied to the labels. an array with distinct unsigned integers is expected
                  where k at index l means that label l is send to label k or in short l -> perm[l].
    '''
    return perm[segmentation_mask]


def rebuild_segment_numbering(segmentation_mask):
    ''' returns segmentation with a continuous numbering starting at 0 '''
    labels = np.unique(segmentation_mask)
    num_segments = labels.shape[0]
    seg_new = np.array(segmentation_mask)

    for l in range(num_segments):
        seg_new[segmentation_mask == labels[l]] = l

    return seg_new


class ProxMapBinaryUCSegmentation(object):
    r"""Implementation of the proximal map corresponding to
    :math:`G[u] = \int_\Omega u^2f_1+(1-u^2)f_2\mathrm{d}x`
    """

    def __init__(self, f):
        self.shift = 0.5
        self.indicator1 = f[..., 0] + self.shift
        self.indicator1Plus2 = self.indicator1 + f[..., 1] + self.shift

    def eval(self, u, t):
        return np.divide(u + 2*t*self.indicator1, (1 + 2*t*self.indicator1Plus2))

    def gamma(self):
        # The uniform convexity constant of the objective should be 2, but smaller values seem to work better in practice.
        return 0.7

# ------------------------------------------------------------------------------
#                  Functions to save segmentation images
# ------------------------------------------------------------------------------

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


def saveColoredSegmentation(image, u, filepath, alpha=0.75):
    """
    Draws the individual segments given by np.argmax(u, 2) with transparency
    on top of the image and saves the result as a png image.

    """
    # TODO: Merge with the convert_segmentation_to_image function

    # Convert the soft segmentation u into a hard segmentation h
    h = np.argmax(u, 2)

    # Create a colormap
    colormap = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                         [1., 1., 0.], [1., 0., 1.], [0., 1., 1.],
                         [1., 0.5, 0.], [0., 1., 0.5], [0.5, 0., 1.],
                         [1., 0., 0.5], [0.5, 1., 0.], [0., 0.5, 1.],
                         [1., 0.5, 0.5], [0.5, 1., 0.5], [0.5, 0.5, 1.]])

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
    imageio.imwrite(filepath, coloredSegmentation)


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
        imageio.imwrite(directory + "/image_segment_" + str(k) + ".png",
                        np.uint8(255 * image * u[:, :, k]))
        imageio.imwrite(directory + "/noimage_segment_" + str(k) + ".png",
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

    saveColoredSegmentation(img, u, directory + "colorized_" + filename_base
                            + ".png", alpha)
    for i in range(image.shape[-1]):
        Path(directory + str(i)).mkdir(parents=True, exist_ok=True)
        saveSegmentationContours(image[:, :, i], u,
                                 directory + str(i) + "/contours_" + filename_base + ".svg")
    saveSoftSegmentationImages(img, u, directory + "segments_"
                               + filename_base)
