import re
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io as io
from scipy.optimize import curve_fit, minimize
from scipy import ndimage
from math import atan2, pi
from argparse import ArgumentParser
from numba import jit, prange
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
from skimage import img_as_float32
from matplotlib.backends.backend_pdf import PdfPages
from .io import read_image, write_image
from .segmentation import get_segmentation_mean_values, create_segmentation_colormap_no_gray, MumfordShah_segmentation
from .misc import plot_image_pixel_precise, rint
import matplotlib.cm as cm


def twoD_Gaussian(xy, x0, y0, x_sigma, y_sigma, A, b, r, np_mod=np):
    r"""
    .. math::
      g(x,y)=A \exp\left( \frac{- 1}{ 2 ( 1 - r^2 ) } \left( \left( \frac{x-x_0}{\sigma_x} \right)^2 + \left( \frac{y-y_0}{\sigma_y} \right)^2 - \frac{2r}{ \sigma_x \sigma_y} (x-x_0 ) (y-y_0) \right) \right) + b
    """
    x, y = xy
    c1 = -1 / (2 * (1 - r**2))
    gaussian = (
        A
        * np_mod.exp(
            c1
            * (
                ((x - x0) / x_sigma) ** 2
                + ((y - y0) / y_sigma) ** 2
                - 2 * r * (x - x0) * (y - y0) / (x_sigma * y_sigma)
            )
        )
        + b
    )
    return gaussian


def jacobian_twoD_Gaussian(xy, x0, y0, x_sigma, y_sigma, A, b, r):
    x, y = xy
    pos = [x - x0, y - y0]
    c1 = -1 / (2 * (1 - r**2))
    f = A * np.exp(
        c1 * (((x - x0) / x_sigma) ** 2 + ((y - y0) / y_sigma) ** 2 - 2 * r * (x - x0) * (y - y0) / (x_sigma * y_sigma))
    )
    c2 = 2 * r / (x_sigma * y_sigma)
    dx0 = f * c1 * (-2 * pos[0] / (x_sigma) ** 2 + c2 * pos[1])
    dy0 = f * c1 * (-2 * pos[1] / (y_sigma) ** 2 + c2 * pos[0])
    dA = f / A
    dx_sigma = f * c1 * (-2 * (pos[0] / x_sigma) ** 2 / x_sigma + c2 / x_sigma * pos[0] * pos[1])
    dy_sigma = f * c1 * (-2 * (pos[1] / y_sigma) ** 2 / y_sigma + c2 / y_sigma * pos[0] * pos[1])
    dr = f * (
        pos[0] * pos[1] / (x_sigma * y_sigma * (1 - r**2))
        - r / ((1 - r**2)) ** 2 * ((pos[0] / x_sigma) ** 2 + (pos[1] / y_sigma) ** 2 - c2 * pos[0] * pos[1])
    )
    db = np.ones_like(dr)
    # The extra "np.array(...)" are needed here so that works both if xy only contains two scalars,
    # i.e. one position or two arrays, i.e. multiple positions. This may create extra copies, but
    # is at least compatible with numba.
    return np.stack(
        (np.array(dx0), np.array(dy0), np.array(dx_sigma), np.array(dy_sigma), np.array(dA), db, np.array(dr)), axis=-1
    )


def fit_Gaussian_bump(cropped_image, x0, y0, approx_width, approx_height, bump_index=0):
    if (cropped_image.shape[0] == 1) or (cropped_image.shape[1] == 1):
        print("Can't fit on images with a width or height of only one pixel")
        return None

    x0_offset = x0
    y0_offset = y0

    # center coordinates of the cropped image
    x0ApproxCenter = 0
    y0ApproxCenter = 0
    initial_guess = [
        x0ApproxCenter,
        y0ApproxCenter,
        approx_width,
        approx_height,
        cropped_image.max() - cropped_image.mean(),
        cropped_image.min(),
        0,
    ]

    # create grid
    x = np.arange(int(-cropped_image.shape[1] / 2), int(cropped_image.shape[1] / 2 + 1))
    y = np.arange(int(-cropped_image.shape[0] / 2), int(cropped_image.shape[0] / 2 + 1))
    X, Y = np.meshgrid(x, y)
    bounds = (
        [
            x.min(),
            y.min(),
            0,
            0,
            cropped_image.max() - cropped_image.mean(),
            cropped_image.min(),
            -1,
        ],
        [
            x.max(),
            y.max(),
            max(x.max(), approx_width),
            max(y.max(), approx_height),
            cropped_image.max() - cropped_image.min(),
            cropped_image.mean(),
            1,
        ],
    )
    try:
        params = curve_fit(
            twoD_Gaussian,
            (X.ravel(), Y.ravel()),
            cropped_image.ravel(),
            initial_guess,
            bounds=bounds,
            jac=jacobian_twoD_Gaussian,
        )[0]
        # predicted parameters
        pred_x0 = params[0] + x0_offset
        pred_y0 = params[1] + y0_offset
        pred_width = params[2]
        pred_height = params[3]
        pred_h = params[4]
        pred_b = params[5]
        pred_r = params[6]
        predicted_parameters = [
            pred_x0,
            pred_y0,
            pred_width,
            pred_height,
            pred_h,
            pred_b,
            pred_r,
        ]

    except RuntimeError:
        print("Optimal parameters not found in curve_fit")
        print("skip current blob: " + str(bump_index))
        return None
    return predicted_parameters


def generate_crystal_image(v=None):
    np.random.seed(42)
    bumpWidth = 4.25 #width of gaussian points
    maxIntensity = 85 #Brighness of gaussian points
    offset = 40
    # produce ground truth
    n = 256 #size of image
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    pad = 42  #buffer for borders
    auxX, auxY = np.meshgrid(np.arange(-pad, n + pad), np.arange(-pad, n + pad))
    if v is None:
        v = np.array([[4 / np.sqrt(3) * np.pi, 0], [2 / np.sqrt(3) * np.pi, 2 * np.pi]]).T * 3.78

    sigma = bumpWidth
    u = np.zeros(auxX.shape)
    numY = int(np.ceil(auxY[-1, -1] / v[1, 1])) #number of unit cells along y-axis
    numX = int(np.ceil(auxX[-1, -1] / v[0, 0])) #number of unit cells along x-axis
    #positioning off the gaussian points in each unit cell
    for j in range(1, numY + 1):
        for i in range(1, numX + 1):
            x = auxX[0, 0] + np.mod(j * v[0, 1], v[0, 0]) + i * v[0, 0]
            y = auxY[0, 0] + j * v[1, 1]
            u = u + 85 * np.exp(-((auxX - x) ** 2 + (auxY - y) ** 2) / (2 * sigma**2)) #generate gaussian point, 85=max_intensity
    u = (maxIntensity - offset) * u / np.amax(u) + offset #normation

    uClean = ndimage.map_coordinates(u, [X + pad, Y + pad], order=3, mode="nearest") #crop to size n without buffer

    return uClean


def get_atoms_simple(image, min_center_value, erase_inf_radius, min_boundary_dist=0, plot=False):
    """
    Find the peaks in squares of side length 2*erase_inf_radius in the image, s.t the
    peak's intensity is not less than min_center_value.
    """

    atoms = []  # list of atom centers positions
    image_copy = np.array(image.copy())

    if min_boundary_dist > 0:  # remove the boundary pixel values
        image_copy[:, 0:min_boundary_dist] = 0
        image_copy[0:min_boundary_dist, :] = 0
        image_copy[:, -min_boundary_dist:] = 0
        image_copy[-min_boundary_dist:, :] = 0
    if plot:
        fig = plt.figure()
        plt.imshow(image_copy)
        plt.colorbar()

    while image_copy.max() > min_center_value:
        # find the highest value in the image
        max_point = np.max(image_copy)
        max_index = np.unravel_index(np.argmax(image_copy), image_copy.shape)
        atoms.append(max_index[::-1])
        # crop the box around the global maximum out
        min_x = max(max_index[0] - erase_inf_radius, 0)
        min_y = max(max_index[1] - erase_inf_radius, 0)
        max_x = min(max_index[0] + erase_inf_radius, image_copy.shape[0])
        max_y = min(max_index[1] + erase_inf_radius, image_copy.shape[1])

        image_copy[min_x:max_x, min_y:max_y] = 0
        if plot:
            plt.plot(max_index[1], max_index[0], "xr")
            plt.plot([max_y, max_y, min_y, min_y, max_y], [min_x, max_x, max_x, min_x, min_x], "-r")
        plt.show()
    return np.array(atoms)




def _get_approx_atom_positions(input_image, clear_handling=False):
    approx_centers = []
    approx_width = []
    approx_height = []

    segmentation = MumfordShah_segmentation(rescale_intensity(img_as_float32(input_image)), 2, 2e-5, 1000, 1e-6)[0]
    meanValues = get_segmentation_mean_values(input_image, segmentation, 2)
    original = meanValues[segmentation[:]]
    original = original[..., 0]

    # label image regions
    if clear_handling:
        thresh = threshold_otsu(original)
        bw = closing(original > thresh, square(1))

        # remove artifacts connected to image border
        cleared = clear_border(bw, 1)
        label_image = label(cleared)
    else:
        label_image = label(original)

    regions = regionprops(label_image)
    for region in regions:
        # take regions with large enough areas
        if region.area < 10:
            continue
        minr, minc, maxr, maxc = region.bbox
        if min(maxr - minr, maxc - minc) <= 2:
            continue
        # clean the region with the image size
        if (
            min(maxr - minr, maxc - minc) == input_image.shape[0]
            or min(maxr - minr, maxc - minc) == input_image.shape[1]
        ):
            continue
        label_i = region.label
        x0 = region["Centroid"][1]
        y0 = region["Centroid"][0]
        approx_centers.append([label_i, x0, y0])
        width = 1 + 2 * np.round(1.25 * max(np.abs(minc - x0), np.abs(maxc - x0)))
        height = 1 + 2 * np.round(1.25 * max(np.abs(minr - y0), np.abs(maxr - y0)))
        approx_width.append([label_i, width])
        approx_height.append([label_i, height])

    return approx_centers, approx_width, approx_height


def get_atom_positions(
    input_image,
    clear_handling=False,
    initial_centers=None,
    initial_diameter=12,
    plot_bump_fit_domains=False,
    clear_border_atoms=True,
):
    # 2  get approximate blob's centers, widths and heights
    if initial_centers is None:
        approx_centers, approx_widths, approx_heights = _get_approx_atom_positions(input_image, clear_handling)
    else:
        num_centers = initial_centers.shape[0]
        approx_centers = []
        for i in range(num_centers):
            approx_centers.append([i, initial_centers[i, 0], initial_centers[i, 1]])
        approx_widths = [[i, initial_diameter] for i in range(num_centers)]
        approx_heights = approx_widths

    predicted_parameters = []
    bump_fit_domains = []

    for current_blob in range(len(approx_centers)):
        x0 = approx_centers[current_blob][1]
        y0 = approx_centers[current_blob][2]
        approx_width = approx_widths[current_blob][1]
        approx_height = approx_heights[current_blob][1]

        x0 = rint(x0)
        y0 = rint(y0)
        half_width = int(min(x0, max((approx_width - 1) // 2, 1), input_image.shape[1] - x0 - 1))
        half_height = int(min(y0, max((approx_height - 1) // 2, 1), input_image.shape[0] - y0 - 1))

        pixel_intensity = input_image[y0, x0]
        cropped_blob = input_image[
            (y0 - half_height) : (y0 + half_height + 1),
            (x0 - half_width) : (x0 + half_width + 1),
        ]

        bump_fit_domains.append([x0, y0, half_width, half_height])

        predicted_params = fit_Gaussian_bump(
            cropped_blob,
            x0,
            y0,
            approx_width,
            approx_height,
            current_blob,
        )
        if predicted_params is not None:
            predicted_parameters.append(predicted_params)
        else:
            print(f"Skipping blob {current_blob}")

    if plot_bump_fit_domains:
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots()
        ax.imshow(input_image, cmap="Greys_r")
        for i_blob in range(len(bump_fit_domains)):
            ax.scatter(bump_fit_domains[i_blob][0], bump_fit_domains[i_blob][1], s=20, c="red", marker="x")
            rect = Rectangle(
                (
                    bump_fit_domains[i_blob][0] - bump_fit_domains[i_blob][2],
                    bump_fit_domains[i_blob][1] - bump_fit_domains[i_blob][3],
                ),
                2 * bump_fit_domains[i_blob][2],
                2 * bump_fit_domains[i_blob][3],
                fc="none",
                ec="g",
                lw=1.5,
            )
            ax.add_patch(rect)
        ax.title.set_text("Bump fit domains")
        plt.show()

    if len(predicted_parameters) == 0:
        print("Error: Didn't find any atoms.")
        print("Probably segmentation failed.")
        return None, None

    # clear border atoms
    if clear_border_atoms:
        predicted_params_arr_temp = np.array(predicted_parameters)
        average_half_width = np.mean(predicted_params_arr_temp[:, 2])
        average_half_height = np.mean(predicted_params_arr_temp[:, 3])
        approx_atom_radius = average_half_width + average_half_height

        cleared_predicted_parameters = []
        for current in range(len(predicted_parameters)):
            if not (
                predicted_parameters[current][0] - approx_atom_radius < 0
                or predicted_parameters[current][0] + approx_atom_radius >= input_image.shape[1]
                or predicted_parameters[current][1] - approx_atom_radius < 0
                or predicted_parameters[current][1] + approx_atom_radius >= input_image.shape[0]
            ):
                cleared_predicted_parameters.append(predicted_parameters[current])

        predicted_parameters = cleared_predicted_parameters

    return predicted_parameters




def get_atom_sigma_estimate(nm_per_pixel):
    # According to Spark, the diameter of an atom is roughly 50pm=0.05nm
    return 50 / (1000 * nm_per_pixel)

