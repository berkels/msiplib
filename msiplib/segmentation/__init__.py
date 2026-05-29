"""
Collection of functions particularly useful for segmentation.
"""

import numpy as np
from sklearn.cluster import KMeans
from msiplib.optimization import pd_hybrid_grad_alg
from msiplib.proximal_mappings import project_canonical_simplex, project_unit_ball2D, ProxMapBinaryUCSegmentation
from msiplib.finite_differences import gradient_FD2D, divergence_FD2D
from msiplib.io.terminal import Timer


def convert_image_to_segmentation_labels(image_labels):
    """
    Converts an image into segmentation labels by assuming that each appearing color
    indicates a segment.

    Args:
        image_labels: a grayscale or (a)RGB image where each color stands for a specific segment

    Returns:
        a matrix of the same dimension as the image where each entry contains an integer indicating
        the segment of the pixel
    """
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


def convert_segmentation_to_partition(segmentation, order="C"):
    """
    Convert a segmentation mask to a partition of the pixel domain by enumerating the
    pixels row-wise or column-wise.

    Args:
        segmentation: segmentation mask defining the segment labels
        order: enumerate pixels row-wise ('C') or column-wise ('F'), same as in numpy.reshape function

    Returns:
        a list of lists containing the pixel indices of the pixels that belong to the segment
    """

    pixels = np.arange(np.prod(segmentation.shape), dtype=np.uint64)
    segmentation_flat = np.reshape(segmentation, (np.prod(segmentation.shape)), order=order)
    labels = np.unique(segmentation)
    partition = [pixels[segmentation_flat == l] for l in labels]

    return partition


def get_segmentation_mean_values(input_image, segmentation, num_segments):
    """
    Yields mean feature vectors of segments.
    """
    mean_values = np.zeros_like(
        input_image, shape=(num_segments, input_image.shape[2] if input_image.ndim == 3 else 1), dtype=input_image.dtype
    )
    for k in range(num_segments):
        mean_values[k, :] = np.mean(input_image[segmentation == k], axis=0)
    return mean_values


def create_segmentation_colormap():
    """
    Yields a color map for the visualization of segmentations.

    Colors are taken from "A Colour Alphabet and the Limits of Colour Coding" by Paul
    Green-Armytage, 2010.
    """
    return np.array(
        [
            [0.94117647, 0.63921569, 1.0],
            [0.0, 0.45882353, 0.8627451],
            [0.6, 0.24705882, 0.0],
            [0.29803922, 0.0, 0.36078431],
            [0.0, 0.36078431, 0.19215686],
            [0.16862745, 0.80784314, 0.28235294],
            [1.0, 0.8, 0.6],
            [0.50196078, 0.50196078, 0.50196078],
            [0.58039216, 1.0, 0.70980392],
            [0.56078431, 0.48627451, 0.0],
            [0.61568627, 0.8, 0.0],
            [0.76078431, 0.0, 0.53333333],
            [0.0, 0.2, 0.50196078],
            [1.0, 0.64313725, 0.01960784],
            [1.0, 0.65882353, 0.73333333],
            [0.25882353, 0.4, 0.0],
            [1.0, 0.0, 0.0627451],
            [0.36862745, 0.94509804, 0.94901961],
            [0.0, 0.6, 0.56078431],
            [0.87843137, 1.0, 0.4],
            [0.45490196, 0.03921569, 1.0],
            [0.6, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.31372549, 0.01960784],
        ]
    )


def create_segmentation_colormap_no_gray():
    return np.delete(create_segmentation_colormap(), 7, axis=0)




def create_segmentation_colormap_dictionary():
    colormap = create_segmentation_colormap()
    keys = np.arange(colormap.shape[0])
    return dict(zip(keys, colormap))


def permute_labels(segmentation_mask, perm):
    """
    Function changes labels in segmentation mask according to given permutation.

    Args:
        segmentation_mask: an array containing unsigned integers as class labels

        perm: the permutation that is applied to the labels. an array with distinct unsigned integers is expected
              where k at index l means that label l is send to label k or in short l -> perm[l].
    """
    return perm[segmentation_mask]


def rebuild_segment_numbering(segmentation_mask):
    """
    Returns segmentation with a continuous numbering starting at 0.
    """
    labels = np.unique(segmentation_mask)
    num_segments = labels.shape[0]
    seg_new = np.array(segmentation_mask)

    for l in range(num_segments):
        seg_new[segmentation_mask == labels[l]] = l

    return seg_new


def MumfordShah_segmentation(inputimage, K, reg_par, max_iter, stop_eps, use_cupy=False):
    # If the image is grayscale, add a third dimension of size 1.
    if inputimage.ndim == 2:
        inputimage = np.expand_dims(inputimage, axis=2)

    # Rescale lambda to make it independent of the resolution of the input image.
    h = 1 / (max(inputimage.shape[:2]) - 1)
    lambda_ = reg_par / h

    # Determine initial estimates for the average colors using K-Means.
    reshapedimage = np.reshape(inputimage, newshape=(inputimage.shape[0] * inputimage.shape[1], inputimage.shape[2]))
    Kmeans = KMeans(n_clusters=K, random_state=42).fit(reshapedimage)
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
            u = pd_hybrid_grad_alg(
                u,
                prox.eval,
                project_unit_ball2D,
                gradient_FD2D,
                divergence_FD2D,
                2 * lambda_,
                max_iter,
                stop_eps,
                PDHGAlgType=2,
                gamma=prox.gamma(),
            )[0]

        segmentation = (u > 0.5).astype(int)
    else:
        u = np.zeros_like(f)

        with Timer():
            u = pd_hybrid_grad_alg(
                u,
                lambda a, t: project_canonical_simplex(a - t * f),
                project_unit_ball2D,
                gradient_FD2D,
                divergence_FD2D,
                lambda_,
                max_iter,
                stop_eps,
            )[0]

        segmentation = np.argmax(u, axis=2)

    # Convert the cupy arrays to numpy arrays (which copies the data from the GPU back to the CPU)
    if use_cupy:
        u = cp.asnumpy(u)
        segmentation = cp.asnumpy(segmentation)

    return segmentation, u
