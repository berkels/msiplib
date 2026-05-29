r"""
.. _real-space-unit-cell:

.. sectionauthor:: Amel Shamseldeen Ali Alhassan 2021/2022

The objective of the  :py:meth:`unit_cell_from_real_space <msiplib.unit_cell_from_real_space>` module is to find the unit cell vectors directly from the real space.

The main function of :py:meth:`unit_cell_from_real_space <msiplib.unit_cell_from_real_space>` is :py:meth:`get_primitive_unit_cell_vectors<msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`  and can be called from outside the module. Other functions are to be used for intermediate steps within the module.



Example of how to use this module
----------------------------------

    >>> #run this code in the msiplib folder and be aware that some files will be created
    >>> 
    >>> import imageio
    >>> from msiplib.unit_cell_from_real_space import get_primitive_unit_cell_vectors
    >>> img = imageio.imread('./msiplib/examples/images/grain_1_cropped.png')
    >>> img_name = 'grain_1'
    >>> path = './'
    >>> get_primitive_unit_cell_vectors(img,path,img_name)


Background
----------

The Radon transform computes the projection of an image onto a line given the angle along which the projection is made.
An ensemble of projection lines is called a sinogram.

:py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>` determines the Radon Transform (sinogram) of a crystal image. The standard deviation of the sinogram, known as the projective standard deviation, is then calculated, which determines the directions perpendicular to directions of high periodicity of the lattice.

The directions of high periodicity = 90 - std(sinogram) :cite:`MeBe15`

Full description of the exact implementation here is given in :cite:`AlBe23` section 3.2


**Empirical choices:**

1-  The threshold for the **PSD** peaks is chosen as the maximum between

    .. math:: \max \left( 2.5 \sigma_{\text{PSD}}, \quad 0.8 \left| \text{PSD}_{\text{max}} - \text{PSD}_{\text{min}} \right| + \text{PSD}_{\text{min}} \right)

2- The precision of the high periodicity angles is 0.5 degrees (given by the default value ``accuracy=2`` on the function :py:meth:`psd <psd>`)

3- The length threshold of a vector is empirically chosen to be 5.

4- smallest possible value for the variance is ``1e-8`` (this was chosen because it worked for the 6 test samples, 4 experimental samples from Spark, 4 simulation samples from Marvin and one ground truth sample)

5- Default value of ``energy_exclusion_factor`` is chosen as 1.15 because of the reason above. For experimental images, sometimes it could be useful to change this value.

**Important note**

If the image name contains mathematical symbols such as "_", if write_to_file is True then the user must change the "_" symbol in the output tex file manually to "\_".

References:
------------
    .. [akaike74] H. Akaike, A new look at the statistical model identification, IEEE
        Transactions on Automatic Control, https://ieeexplore.ieee.org/document/1100705

    .. [akaike81] H. Akaike, Likelihood of a model and information criteria, Journal of Econometrics,
        https://www.sciencedirect.com/science/article/pii/0304407681900713

--------------------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon
from skimage.exposure import rescale_intensity
from scipy.ndimage import affine_transform
from sklearn.cluster import KMeans
from scipy import interpolate
from tqdm import tqdm
import warnings
from pathlib import Path
from .misc import plot_image_pixel_precise
from .optimization import GaussNewtonAlgorithm, validateDerivative
from scipy.ndimage import uniform_filter
from skimage.transform import downscale_local_mean

np.set_printoptions(precision=2)  # show number values to 2 decimal points precision
VEC_LEN_LIM = 5

latex_output_file = None


def print_file(*messages):

    if latex_output_file is not None:
        print(*messages, file=latex_output_file)


def img_format(img, square=True):
    """
    Assures the image is grayscale.

    In case the image has a rectangular shape, ``img_format`` crops it into a square.
    """
    if img.ndim != 2:
        raise ValueError("Error: image of wrong format. Input image must be of gray scale")
    if not square:
        return img
    return img[: min(img.shape), : min(img.shape)]


def atomic_center(im, imaging_mode="DF"):
    """
    Locates the brightest pixel within the central area of an image.

    This would represent an atomic center if the atoms in the input image are bright
    in a dark background as in [S]TEM's [HA]ADF images.

    parameters:

        im : *2x2 ndarray* : Input image in which the atomic center is to be located.

        imaging_mode : *str, optional* : By default dark field imaging mode is assumed
                             it is important to use **"BF"** for bright field mode.

    returns:

        The position of the brightest pixel around the center.

    note:

        For images with opposite contrast (e.g. **bright field** images), the input should be transformed by using ``1-im``.
    """
    bound = int(min(im.shape) * 3 / 8)
    if imaging_mode in ["BF", "bf", "bright field", "Bright Field"]:
        square = 1 - im[bound:-bound, bound:-bound]
    else:
        square = im[bound:-bound, bound:-bound]
    center = np.unravel_index(np.argmax(square), square.shape)
    return tuple(i + bound for i in center[::-1])




# puvs= primitive unit cell vectors
def plot_puvs(v1, v2, im, output_dir: str | os.PathLike | None = None, name_prefix=None, imaging_mode="DF", quiver=True, plotlongline=False):
    """
    Plots the lattice and displays the primitive unit cell vectors (PUVs) on the input image.

    parameters:

        v1,v2 : *2D array* : Vectors representing the lattice primitive unit cell vectors

        im : *2x2 ndarray* : Input image for :py:meth:`atomic_center <atomic_center>`.

        path : *str* : The path to the directory where the output file will be saved.

        name : *str* : Name of the image file to be saved.

        imaging_mode : *str, optional* : Input for :py:meth:`atomic_center <atomic_center>`.

        quiver :*bool, optional* : If ``True``, arrows will be displayed to represent the unit cell vectors.

        plotlongline : *bool, optional* : If ``True``, long lines will be shown to demonstrate the direction of the primitive unit cell vectors.

    returns:

        The input image with the plotted lattice and primitive unit cell vectors overlaid.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
    v1, v2 = np.array(v1), np.array(v2)
    pixel_coord = np.array(atomic_center(im, imaging_mode=imaging_mode))

    fig, ax, dpi = plot_image_pixel_precise(im)

    vec1 = np.array([pixel_coord, pixel_coord + v1])
    vec2 = np.array([pixel_coord, pixel_coord + v2])
    if plotlongline:
        longline = int(3 * min(im.shape) / (4 * np.max(np.array([v1, v2]))))
        vec1 = []
        vec2 = []
        for i in range(-longline, longline):
            vec1.append(pixel_coord + (i * v1))
            vec2.append(pixel_coord + (i * v2))
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
    if quiver:
        plt.quiver(*pixel_coord, v1[0], v1[1], color="r", angles="xy", scale_units="xy", scale=1)
        plt.quiver(*pixel_coord, v2[0], v2[1], color="b", angles="xy", scale_units="xy", scale=1)
    else:
        ax.plot(vec1[..., 0], vec1[..., 1], "-r")
        ax.plot(vec2[..., 0], vec2[..., 1], "-b")
    if output_dir is not None:
        name_prefix = "" if name_prefix is None else f"{name_prefix}"
        plt.savefig(output_dir / f"{name_prefix}optimized.png", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.show()
    plt.close()


def find_peaks(x, height=float("-inf"), pbc=False, relative_height=False, factor=0.8, neighbors=1):
    """
    Locates the local maxima on a list. The maxima must be greater than a conditional
    base value that is given by the variable ``height`` (The default value of ``height`` is
    negative infinity ).

    Periodic boundary conditions may be applied. A condition can also be applied to the local
    maximum, which is determined by the relationship between the global maximum and minimum. In
    this case, a local maximum has to be greater than
    a certain ``factor`` of the global min-max distance (The default value of ``factor`` is ``0.8``).

    parameters:

        x :*1D array-like* : The input array where local maxima are to be found.

        height :*float, optional, default=float('-inf')* : The minimum value that a peak must exceed to be considered a local maximum. Default is negative infinity, meaning all peaks are considered.

        pbc :*bool, optional, default=False* : Whether to apply periodic boundary conditions when searching for local maxima.

        relative_height :*bool, optional, default=False* : If ``True``, the height threshold for a local maximum is determined by the relative distance between the global maximum and global minimum, rather than the absolute ``height``.

        factor :*float, optional, default=0.8* : A scaling factor for the global maximum-to-minimum distance that defines the height threshold for local maxima when ``relative_height=True``.


    returns:

        The indices of the local maxima in the input array ``x`` that meet the specified conditions.
    """
    peaks_indices = []
    max_x = max(x)
    min_x = min(x)
    l = len(x)
    if relative_height:  # threshold on least considerable hight of a beak
        better_height = factor * (max_x - min_x) + min_x
        height = max(height, better_height)
    for i in range(l):
        if x[i] > height:
            left_neighbors = np.array([x[i - n - 1] for n in range(neighbors)])
            right_neighbors = np.array([x[(i + n + 1) % l] for n in range(neighbors)])
            if pbc:
                if (x[i] > left_neighbors).sum() == neighbors and (x[i] > right_neighbors).sum() == neighbors:
                    peaks_indices.append(i)
            else:
                if i > neighbors - 1 and i < len(x) - neighbors:
                    if (x[i] > left_neighbors).sum() == neighbors and (x[i] > right_neighbors).sum() == neighbors:
                        peaks_indices.append(i)
    return np.array(peaks_indices)


def psd(img, output_dir: str | os.PathLike, name_prefix="", accuracy=1, height_std_factor=2.5, relative_height=False, factor=0.8, plot=True):
    """
    Finds the angles of high periodicity in an image using the projection standard deviation (PSD) method, as described in :cite:`MeBe15`.

    Angles with values exceeding either 2.5 times the standard deviation of the PSD or the global minimum plus 0.8 times the distance between the global max and min are considered significant.


    parameters:

        img : *2x2 ndarray* : The input 2D image in which periodicity angles are detected

        path : *str* : The directory path where the output files (such as the plot) will be saved.

        name : *str* : name of the image

        accuracy : *int, optional, default=1* :  Determines the sensitivity for detecting directions. The directions are detected with a precision of 1/accuracy.

        height_std_factor : *float, optional, default=2.5* : A threshold factor that determines how many times above the standard deviation of the PSD a peak must be to be considered significant.

        relative_height : *bool, optional, default=False* :If ``True``, the height threshold for a local maximum is determined by the relative distance between the global maximum and global minimum, rather than the absolute ``height``.

        factor : *float, optional, default=0.8* : The percentage of the distance global max- global min which defines the height boundary.

    returns:

        This function identifies significant periodicity angles in a 2D image using the projection standard deviation method and returns the angles along with a plot.
    """
    output_dir = Path(output_dir)
    if name_prefix.endswith("_"):
        print_file("\\section{", name_prefix[:-1], "}")
    else:
        print_file("\\section{", name_prefix, "}")
    print_file("\nimage shape", img.shape, "\\\\")

    if accuracy not in [1, 2, 4]:
        raise ValueError("Accuracy has to be either 1, 2 or 4")

    points = 180 * accuracy
    theta = np.linspace(0.0, 180.0, points, endpoint=False)  # projection angles to be evaluated
    normalizer = np.ones(img.shape)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Radon transform: image must be zero outside the reconstruction circle"
        )
        normalizer_sinogram = radon(normalizer, theta=theta)
        image_sinogram = radon(img, theta=theta)

    # avoid dividing by zero
    if 0 in normalizer_sinogram:
        mask = normalizer_sinogram == 0
        masked_image_sinogram = np.ma.masked_array(image_sinogram, mask)
        masked_norm_sinogram = np.ma.masked_array(normalizer_sinogram, mask)
    else:
        masked_image_sinogram = image_sinogram
        masked_norm_sinogram = normalizer_sinogram

    sinogram = masked_image_sinogram / masked_norm_sinogram
    print_file("sinogram shape", sinogram.shape, "\\\\")

    # find the significant peaks
    sinogram_std = np.std(sinogram, axis=0)
    standard_dev = np.std(sinogram_std)
    height = height_std_factor * standard_dev
    peaks = find_peaks(sinogram_std, height=height, pbc=True, relative_height=False, neighbors=2)
    while len(peaks) < 2 and height > 0.5 * standard_dev:
        height -= 0.5 * standard_dev
        peaks = find_peaks(sinogram_std, height=height, pbc=True, relative_height=False)
    if len(peaks) < 2:
        raise ValueError("Not enough periodic directions were found")
    print_file("\n{\\bf There are", len(peaks), "significant directions}", "\n")

    if plot:
        additional_plots_dir = output_dir / "Additional_Plots"
        additional_plots_dir.mkdir(parents=True, exist_ok=True)
        fig = plt.figure()
        plt.plot(theta, sinogram_std)
        plt.plot(peaks / accuracy, sinogram_std[peaks], "xr", label=str(90 - (peaks / accuracy)))
        plt.title("Projective Standard Deviation " + name_prefix)
        plt.legend(loc="best")
        plt.savefig(additional_plots_dir / f"{name_prefix}projected_std.png")
        # plt.show()
        plt.close(fig)

    return 90 - (peaks / accuracy)


def plot_lines(img, output_dir: str | os.PathLike, name_prefix, theta, labels, col=None):
    """
    Given an image and a set of angles, this function plots lines from the center of the
    image in the specified directions.

    parameters:

        img : *2x2 ndarray* : The input crystal image.

        path : *str* : The directory path for saving the output file.

        name : *str* : The name of the image file to be saved.

        theta : *list or ndarray* : 1D array or list of the angles to be plotted

        col :  *list, optional* : list of colors abbreviations of the lines.

    returns:

        Image with lines indicating the directions of high periodicity.
    """
    output_dir = Path(output_dir)
    additional_plots_dir = output_dir / "Additional_Plots"
    additional_plots_dir.mkdir(parents=True, exist_ok=True)
    y0, x0 = (np.array(img.shape) / 2).astype(int)
    x = np.arange(len(img[0]))
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.ylim((len(img), 0))
    plt.xlim((0, len(img[0])))
    plt.imshow(img, cmap="gray")
    for i in range(len(theta)):
        y = np.tan(np.radians(theta[i])) * (x - x0) + y0  # line passing through the center (x0, y0)
        if col is not None:
            ax.plot(x, y, color=col[i], label=(str(theta[i]) + "\N{DEGREE SIGN}\n" + str(labels[i])))
        else:
            ax.plot(x, y, label=(str(theta[i]) + "\N{DEGREE SIGN}\n" + str(labels[i])))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    plt.savefig(additional_plots_dir / f"{name_prefix}periodicity_directions.png")
    plt.close(fig)
    return None


def energy(img, theta, t=1):
    r"""
    Computes the minimization energy based on equation (5) from :cite:`MeBe15` at a given vector length ``t`` and direction ``theta``.

    The energy is computed according to the following equation:

    .. math::

        E_{\alpha}(t) = \int_{\Omega_{\alpha}(t)} \left( f(x + t e_{\alpha}) - f(x) \right)^2 \, dx

    parameters:

        img : *2D ndarray* : The input image for which the energy is calculated.

        theta : *float* : The direction of the lattice vector in degrees.

        t : *int, optional, default=1* : Length of the lattice vector.

    returns:

        The minimization energy as described previously.
    """
    shift_direction = np.array([np.sin(np.radians(theta)), np.cos(np.radians(theta))])
    identity = np.identity(2)
    crop_side = np.array(img.shape) // 4
    shifted_img = affine_transform(img, matrix=identity, offset=t * shift_direction, mode="constant", cval=0.0)
    residue = (shifted_img - img) ** 2
    residue = residue[crop_side[0] : -crop_side[0], crop_side[1] : -crop_side[1]]
    return np.mean(residue), shifted_img


def plot_deformed_image(shifted_img, path: str | os.PathLike, name, line=False, theta=0):
    """
    Plots a deformed image and optionally displays the direction of the deformation.

    parameters:

        shifted_img : *2x2 ndarray* : The input image showing the deformation.

        path : *str* : The directory path where the output file will be saved.

        name : *str* :  The name of the image file to be saved.

        line : *bool, optional, default=False* : If ``True``, a line indicating the direction of deformation will be shown.

        theta : *float, optional, default=0* : The angle along which the deformation energy is calculated.

    returns:

        Figure showing the displacement along the given direction and saves the plot as an image file in the specified directory.
    """
    path = Path(path)
    shifted_figure = plt.figure()
    plt.ylim((len(shifted_img), 0))
    plt.xlim((0, len(shifted_img[0])))
    plt.imshow(shifted_img)
    if line:
        y0, x0 = (np.array(shifted_img.shape) / 2).astype(int)
        x = np.arange(len(shifted_img[0]))
        y = np.tan(np.radians(theta)) * (x - x0) + y0
        plt.plot(x, y, label=str(theta))
    plt.savefig(path / f"{name}_shifted_{int(theta):03d}.png")
    plt.close(shifted_figure)


def energy_diagram(img, path: str | os.PathLike, name, theta, relative_height=False, factor=0.8, show_bad_energies=False, plot=True):
    """
    Plots the energy values over a range of one-quarter of the maximum dimension of the crystal image.

    **Important**: The image must contain at least 4 unit cells in each direction.

    parameters:

        img : *2x2 ndarray* : The input image of the crystal.

        path : *str* : The directory path where output files will be saved.

        name : *str* : The name used for saving the output files.

        theta : *float* : Input for :py:meth:`energy <energy>`.

        relative_height : *bool, optional, default=False* : If ``True``, it calculates the relative height from the global maximum using the specified ``factor`` value and rejects any peaks that fall below this threshold.


        factor : *float, optional, default=0.8* : Input for :py:meth:`find_peaks <find_peaks>`.

        show_bad_energies : *bool, optional, default=False* : If ``True``, displays energies even when they do not meet the regular criteria.


        plot : *bool, optional, default=True* :  If `True`, plots the energy diagram.


    returns:

        A *list* of energy minima along the direction given by the angle ``theta``, a *list* of corresponding lengths, a *figure* of energies vs the offset, and a *figure* showing the displacement along the given direction at 1/6 of the image's maximum dimension.    """
    path = Path(path)
    length = np.arange(1, int(min(img.shape) / 4))  # the set of values t (as per Mevenkamp paper's notation)
    E = np.zeros([length.shape[0]])
    # create folder when plot is True
    if plot:
        energy_plots_dir = path / "Additional_Plots" / "energy"
        energy_plots_dir.mkdir(parents=True, exist_ok=True)
    for i in length:
        integral, shifted_img = energy(img, theta, i)
        E[i - 1] = integral
        if plot and i == int(max(img.shape) / 6):
            plot_deformed_image(shifted_img, energy_plots_dir, name, line=True, theta=theta)
    local_minima = find_peaks(-E, relative_height=True, factor=factor)
    segments = local_minima + 1
    print_file("\n\n\\subsection{Angle:", theta, "}", "\n\n lengths and energies: \n\n ")
    fig = None
    if plot:
        fig = plt.figure()
        plt.xlabel("t")
        plt.ylabel("E(t)")
        plt.title("Energy along %.2f degree" % theta)
        plt.plot(length, E)
    if len(local_minima) == 0:
        print_file(f"\nThe direction theta = {theta} will be skipped as no reasonable minimum can be found!\n")
        if plot and show_bad_energies:
            plt.show()
        plt.clf()
        plt.close()
        return [None], [None], None
    if plot:
        plt.plot(segments, E[local_minima], "xr")
        energy_filename = f"{name}energy_along_{int(theta)}.png" if theta > 0 else f"{name}energy_along_nve_{str(abs(int(theta)))}.png"
        plt.savefig(energy_plots_dir / energy_filename, format="png")
        np.save(energy_plots_dir / f"energy_{theta}.npy", E[local_minima])
    print_file("\n\\begin{tabular}{ | l | l | }\n")
    print_file("\n\\hline\n")
    print_file("\n$\\tau_{%.2f }$ & $E(\\tau_{%.2f })$\\\\\n" % (theta, theta))
    print_file("\n\\hline\n")
    print_file("\n")
    print_file("\n\\hline\n")
    print_file("\n\\end{tabular}\n")
    print_file("\nminimum energy list:\n", E[local_minima])
    return E[local_minima], segments, fig


# Akike Information Criterion
def energy_clusters(X, maxsigma):
    """
    Clusters a list of energy minima values and finds the optimal number of clusters using the Akaike Information Criterion (AIC).

    This function applies KMeans clustering to the 1D array of energy minima values and determines the optimal number of clusters based on the Akaike Information Criterion (AIC) [akaike74]_ [akaike81]_. It then returns the best clustering configuration.

    parameters:
        X : *1D ndarray* : A 1D array of minimum energy values along a certain direction.

        maxsigma : *float* : A threshold for the minimum allowable variance for each cluster.

    returns:
        Returns a ``KMeans`` object representing the best clustering of the values in ``X``, with attributes such as ``cluster_centers_`` (coordinates of the cluster centers) and ``labels_`` (labels indicating each point's cluster).
    """
    X = X.reshape(-1, 1)

    def Akaike_1D(X, k, maxsigma=1e-8):
        """
        Finds Akaike Information Criterion using kmean clustering given a list of
        elements and the number of clusters.

        parameters:

            X : *1D list* : The list or array of elements to be clustered.

            k : *int* : The number of clusters to be used in the k-means clustering.

            maxsigma : *float, optional, default=1e-8* : The maximum allowable variance for each cluster.

        returns:

            The Akaike Information Criterion (AIC) for the given number of clusters `k`.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Number of distinct clusters ")
            kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto", tol=1e-10).fit(X)
            indices = kmeans.labels_
            mean_i = kmeans.cluster_centers_
            transform = kmeans.transform(X)
            variances = [(transform[j][i]) ** 2 for j, i in enumerate(indices)]  # first step to find var_i
            n_i = np.array([np.sum([indices == m]) for m in range(len(mean_i))])
            var_i = np.array(
                [
                    np.sum(np.ma.masked_array(variances, indices != m)) / np.sum([indices == m])
                    for m in range(len(mean_i))
                ]
            )  # actual variances
            var_i = np.maximum(var_i, maxsigma)
            if np.isnan(var_i).any():
                return False
            n = sum(n_i)
            print_file("\nk=", k, "\n\n$\\mu_i$ =", mean_i, "\n\n$\\sigma_i^2$  =", var_i, "\n\n")
            return -2.0 * sum(n_i * np.log(n_i / n) - (n_i / 2.0) * (np.log(2.0 * np.pi * var_i) + 1)) + 6.0 * k - 2

    AIC = []
    for k in range(1, np.unique(X).size + 1):
        aic = Akaike_1D(X, k, maxsigma)
        if aic:
            AIC.append(aic)
    AIC = np.array(AIC)
    print_file("\n\nAIC =", AIC, "\n")
    K = np.argmax(np.exp((min(AIC) - AIC) / 2) > 0.1) + 1
    print_file("\nK =", K, "\n")
    return KMeans(n_clusters=K, random_state=0, n_init="auto").fit(X)


def show_AIC(fig, output_dir: str | os.PathLike, name_prefix, theta, energy_list, distance_list, maxsigma):
    """
    Labels energy minima based on Akaike Information Criterion (AIC) clustering and saves the plot.

    This function clusters the energy minima values using AIC-based KMeans clustering and plots the energy minima along with their corresponding distances. The plot is saved in the specified directory.
    """
    output_dir = Path(output_dir)
    energy_plots_dir = output_dir / "Additional_Plots" / "energy"
    energy_plots_dir.mkdir(parents=True, exist_ok=True)
    cluster = energy_clusters(energy_list, maxsigma)
    indices = cluster.cluster_centers_
    for index in range(len(indices)):
        E_min = energy_list[cluster.labels_ == index]
        fundamental_period_list = distance_list[cluster.labels_ == index]
        plt.plot(fundamental_period_list, E_min, "x", label=E_min)
    plt.legend(loc="best")
    energy_filename = f"{name_prefix}energy_along_{int(theta)}.png" if theta > 0 else f"{name_prefix}energy_along_nve_{str(abs(int(theta)))}.png"
    plt.savefig(energy_plots_dir / energy_filename, format="png")
    plt.close()


def find_fundamental_period(energy_list, distance_list, maxsigma):
    """
    Finds the fundamental period (or length) along a high-periodicity direction based on the energy minima and corresponding distances.

    This function takes in a list of energy minima values and their corresponding distances along a certain direction and clusters them to determine the fundamental period based on the lowest energy value.

    parameters:

        energy_list : *1D ndarray* : Input for :py:meth:`energy_clusters <energy_clusters>`.

        distance_list : *float* : A list or array of corresponding distances associated with each energy minimum.

        maxsigma : *float* : Input for :py:meth:`energy_clusters <energy_clusters>`.


    returns:
        The fundamental period (*float*), which is the distance corresponding to the lowest energy minimum, and the energy value (*float*) at that period.
    
    note:
        The function clusters the energy minima using AIC-based KMeans clustering to identify the fundamental period.
    """
    cluster = energy_clusters(energy_list, maxsigma)
    index = np.argmin(cluster.cluster_centers_)
    E_min = energy_list[cluster.labels_ == index]
    fundamental_period_list = distance_list[cluster.labels_ == index]
    fundamental_period_index = np.argmin(abs(fundamental_period_list))
    fundamental_period = fundamental_period_list[fundamental_period_index]
    print_file(
        "\nminimum energies:\n\n ",
        str(E_min)[1:-1],
        "\n\nt list:\n\n ",
        str(fundamental_period_list)[1:-1],
        "\n\nfundamental period:\n\n ",
        fundamental_period,
        "\n",
    )
    return fundamental_period, E_min[fundamental_period_index]


def exclude_high_energies(vectors, energies, energy_exclusion_factor=1.15, normalize_img=True, plot=True):
    """
    Excludes vectors associated with excessively high energy values.

    This function filters out vectors corresponding to excessively high energy values, based on a set of conditions. If normalization is enabled or certain thresholds are exceeded, vectors with high energies are excluded from the returned set.

    parameters:
        vectors : *array-like* : An array of candidate unit cell vectors.

        energies : *array-like* : An array of energies corresponding to the unit cell vectors.

        energy_exclusion_factor : *float, optional, default=1.15* : Defines the upper boundary of energy for eligible vectors as a multiple of the minimal energy.

        normalize_img : *bool, optional, default=True* : Whether the image is normalized before analysis.

        plot : *bool, optional, default=True* : Whether to plot the energy values.
    returns:
        The eligible vectors with minimal energy.
    """
    print_file("\n\n\n\nEnergies", energies)
    if len(energies) <= 1:
        raise ValueError("Less than two potential lattice vector directions found in this image!\n" \
                         "Potential reasons:\n" \
                         "- image too small (not enough copies of the unit cell)\n" \
                         "- 'height_std_factor' too large\n  (not enough PSD peaks selected, check PSD plot with 'plot=true'\n" \
                         "- 'uc_find_peaks_factor' too large\n  (energy_diagram discards valid minima, check with 'show_bad_energies=true')")
    min_enr = np.min(energies)
    if plot:
        a = np.arange(0, len(energies))
        plt.figure()
        for i in range(len(energies)):
            plt.plot(a[i], energies[i], "o", label=vectors[i])
        if len(vectors[energies <= energy_exclusion_factor * min_enr]) < 2 and len(energies) >= 2:
            plt.axhline(y=np.sort(energies)[1], color="Orange")
        else:
            plt.axhline(y=min_enr * energy_exclusion_factor, color="r")
        plt.legend(loc="best")
        plt.show()
    if len(energies) > 2:
        if len(vectors[energies <= energy_exclusion_factor * min_enr]) < 2:
            return vectors[np.argsort(energies)[:2]]
        else:
            return vectors[energies <= energy_exclusion_factor * min_enr]
    return vectors


def choose_set(initial_vectors_list):
    """
    Selects the unit cell lattice vectors from a list of potential lattice vectors.
    
    The selection is based on vector length, prioritizing the shortest vectors.
    If multiple vectors share the same length, the vector with the smallest angle
    relative to the :math:`[1,0]` axis is chosen first.

    parameters:
        initial_vectors_list : *2x2 ndarray* : A 2D array containing all candidate lattice vectors, where each vector is represented by two components.
    returns:
        ``v1`` and ``v2``, two array-like vectors representing the unit lattice cell vectors in 2D space.
    Raises:
        ValueError : If no suitable vector set is found.

    note:
        The function filters and selects lattice vectors as follows:
            - Vectors with unrealistic lengths are excluded based on a predefined threshold ``VEC_LEN_LIM``.
            - The function sorts vectors by their lengths and checks for equal-length vectors to select the most appropriate lattice vectors.
    """
    initial_norms_list = np.linalg.norm(initial_vectors_list, axis=1)
    # ----------exclude vectors with unrealistic lengths
    vectors_list = np.array(initial_vectors_list[np.ceil(initial_norms_list) >= VEC_LEN_LIM])
    norms_list = np.linalg.norm(vectors_list, axis=1)
    sorting = np.argsort(norms_list)
    sorted_lengths = vectors_list[sorting]
    sorted_norms = norms_list[sorting]
    vectors = np.zeros((3, 2))
    vectors[0] = np.array([1, 0])
    if len(sorted_lengths) == 2:
        vectors[1:] = sorted_lengths
        v1 = vectors[1]
        v2 = vectors[2]
        print_file("\n\nbest lattice vectors are:\n(", v1, v2, ")\n")
        return v1, v2
    elif len(sorted_lengths) > 2:
        # check if any vectors are equal in length
        # is sorted_norms[i] (=>/<=) sorted_norms[i+1] by at most 2?
        diff1 = np.abs(sorted_norms - np.roll(sorted_norms, -1)) <= 2
        # is sorted_norms[i-1] (=>/<=) sorted_norms[i] by at most 2?
        diff2 = np.abs(sorted_norms - np.roll(sorted_norms, 1)) <= 2
        equal_lengths = np.array([])
        if True in diff1 and False in diff2[1:]:
            inds_right_sub = [i for i, x in enumerate(diff1) if x]
            inds_left_sub = [i for i, x in enumerate(diff2[1:]) if not x]
            equal_lengths = sorted_lengths[inds_right_sub[0] : inds_left_sub[0] + 1]
        for ind in range(1, 3):
            if len(equal_lengths) > 0 and sorted_lengths[ind - 1] in equal_lengths:
                a = vectors[ind - 1]
                angles_list = np.array(
                    [  # TODO: consider angles on the positive and negative directions.
                        np.arcsin(
                            (a[1] * equal_lengths[i, 0] - a[0] * equal_lengths[i, 1])
                            / (np.linalg.norm(a) * np.linalg.norm(equal_lengths[i]))
                        )
                        for i in range(len(equal_lengths))
                    ]
                )
                angle_sorting = np.argsort(angles_list)
                sorted_angles_list = angles_list[angle_sorting]
                sorted_angles = equal_lengths[angle_sorting]
                vectors[ind] = sorted_angles[0]
                if ind == 1:
                    angle_difference = np.degrees(sorted_angles_list[1] - sorted_angles_list[0])
                    if angle_difference < 1.5 and len(sorted_angles) > 2:
                        vectors[ind + 1] = sorted_angles[2]
                    else:
                        vectors[ind + 1] = sorted_angles[1]
                    break
            else:
                vectors[ind] = sorted_lengths[ind - 1]
        v1 = vectors[1]
        v2 = vectors[2]
        print_file("\n\nbest lattice vectors are:\n(", v1, v2, ")\n")
        return v1, v2
    raise ValueError(f"vector set not found, initial_vectors_list={initial_vectors_list}")


def optimize(v, im, confirm_gradient):
    r"""

    Refines the unit cell vectors to find the optimal values using the Gauss-Newton method, as described in equation (7) of :cite:`MeBe15`.

    .. math::

        E(v_1, v_2) = \sum_{(z_1, z_2) \in Z} \int_{\tilde{\Omega}} \left( f(x) - f\left(x + z_1 v_1 + z_2 v_2\right) \right)^2 \, dx, \quad Z = \{(1, 0), (0, 1), (1, 1)\}


    Where:

    .. math::
            \tilde{\Omega} = \{x \in \Omega | dist(x, \delta \Omega) > max\{|v_1 |, |v_2 |, |v_1 + v_2 |\} + \epsilon \}, \quad \epsilon = 3

    parameters:

        v : *list* : A list containing the initial primitive lattice vectors ``[v1, v2]``.

        im : *2D array* : The input 2D image representing the crystal or pattern.

        confirm_gradient : *bool* : If `True`, plots the function and its tangent at the minimum value to visually confirm the gradient.

    returns:

        Two arrays representing the optimized primitive unit cell vectors in 2D.
    """

    def residual_i(ref, f_u, xy, v, bound):
        r"""
        Given the integrand of eq(7) .. math::F(v)

         .. math::
            F(v) = \sum_{v_i} F_i

        This function calculates the sub-residuals F_i
        """
        xy = np.add(v, xy)
        res = ref[bound:-bound, bound:-bound] - f_u.ev(
            xy[..., 0][bound:-bound, bound:-bound], xy[..., 1][bound:-bound, bound:-bound]
        )
        return res

    def residual(img, v, bound, f_u, xy):
        """
        Calculates the integrand of eq(7) above.
        """

        v1 = v[:2]
        v2 = v[2:4]
        v3 = v1 + v2
        res = np.dstack(
            (
                residual_i(ref, f_u, xy, v1, bound),
                residual_i(ref, f_u, xy, v2, bound),
                residual_i(ref, f_u, xy, v3, bound),
            )
        )
        return res


    def residual_grad_i(phi_grad, f_u, xy, v, bound):
        """
        Calculates the gradient of the residual's sub-component.
        """
        xy = np.add(v, xy)
        coords = (xy[..., 0][bound:-bound, bound:-bound], xy[..., 1][bound:-bound, bound:-bound])
        f_grad = -np.dstack([f_u.ev(*coords, dx=1, dy=0), f_u.ev(*coords, dx=0, dy=1)])
        res = np.einsum("...ij,...j", phi_grad, f_grad)
        return res

    def residual_grad(img, v, bound, f_u, xy):
        """
        Calculates the gradient of the residual.
        """

        v1 = v[:2]
        v2 = v[2:4]
        v3 = v1 + v2
        mat1 = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
        mat2 = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        mat3 = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        res = np.stack(
            (
                residual_grad_i(mat1, f_u, xy, v1, bound),
                residual_grad_i(mat2, f_u, xy, v2, bound),
                residual_grad_i(mat3, f_u, xy, v3, bound),
            ),
            axis=2,
        )

        return res

    bound = int(np.ceil(max(max(np.abs(v[:2])), max(np.abs(v[2:])), max(np.abs(v[:2] + v[2:]) + 3))))
    if bound > min(im.shape) / 3:
        raise ValueError("image too small compared to the unit cell vectors")

    scale = 1 / np.sqrt(im.size)

    v = v

    ref = im.copy()
    x = np.arange(0, im.shape[0])
    y = np.arange(0, im.shape[1])
    f_u = interpolate.RectBivariateSpline(y, x, im.T, kx=3, ky=3)
    xy = np.dstack(np.meshgrid(y, x))

    v = GaussNewtonAlgorithm(
        v,
        F=lambda v: scale * residual(im, v, bound, f_u, xy).reshape(-1),
        DF=lambda v: scale * residual_grad(im, v, bound, f_u, xy).reshape(-1, v.shape[0]),
        maxIter=100,
        stopEpsilon=1e-8,
    )

    # the following lines checks the refinement process
    def residual_energy(im, v):
        f = residual(im, v, bound, f_u, xy)
        return 1 / 2 * np.sum(f**2)

    def residual_energy_grad(im, v):
        f = residual(im, v, bound, f_u, xy)
        Df = residual_grad(im, v, bound, f_u, xy)
        return np.sum(np.einsum("...ij,...i", Df, f), axis=(0, 1))

    def Eo(v):
        return residual_energy(im, v)  # optimization energy

    def DEo(v):
        return residual_energy_grad(im, v)

    if confirm_gradient:
        validateDerivative(Eo, DEo, v, h=0.1)

    return v[:2], v[2:4]


def reduce_uc_vectors(v1, v2, min_reduction=0, vec_len_lim=5):
    """
    Reduces the unit cell vectors to ensure that they are not collinear and remain within specified length limits. This is done by combining or subtracting vectors iteratively to minimize their norms.

    parameters:
        v1 : *array-like* : The first primitive unit cell vector.

        v2 : *array-like* : The second primitive unit cell vector.

        min_reduction : *float, optional, default=0* : The minimum amount by which the vector norm must decrease in each iteration.

        vec_len_lim : *float, optional, default=5* : The minimum allowed length for the reduced unit cell vectors. If the norm of either vector falls below this limit, an error is raised.

    returns:
        A tuple containing two reduced unit cell vectors (v1, v2) after the optimization.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    while True:
        if np.linalg.norm(v1 + v2) < v1_norm - min_reduction:
            v1 = v1 + v2
            v1_norm = np.linalg.norm(v1)
        elif np.linalg.norm(v1 - v2) < v1_norm - min_reduction:
            v1 = v1 - v2
            v1_norm = np.linalg.norm(v1)
        elif np.linalg.norm(v1 + v2) < v2_norm - min_reduction:
            v2 = v1 + v2
            v2_norm = np.linalg.norm(v2)
        elif np.linalg.norm(v1 - v2) < v2_norm - min_reduction:
            v2 = v1 - v2
            v2_norm = np.linalg.norm(v2)
        else:
            break
    if v1_norm < vec_len_lim or v2_norm < vec_len_lim:
        raise ValueError("Vectors pointing to similar direction")
    return v1, v2


def get_primitive_unit_cell_vectors(
    im,
    output_dir: str | os.PathLike,
    name_prefix="",
    save_vectors=True,
    write_to_file=False,
    energy_exclusion_factor=1.15,
    normalize_img=True,
    maxsigma=1e-8,
    accuracy=2,
    height_std_factor=2.5,
    relative_height=False,
    factor=0.8,
    imaging_mode="DF",
    show_bad_energies=False,
    confirm_gradient=False,
    plot=False,
    uniform_prefilter_size=0,  # smooths the input image
    config={},
    plot_final_vectors=False,  # TODO: Convert all relevant parameters to entries of config (and remove those arguments covered by both)
):
    r"""
    Identifies the primitive unit cell vectors from an atomic resolution crystal image.

    parameters:
        im : *2x2 ndarray* : The input 2D lattice image.

        output_dir : *str* : The output directory where results will be saved.

        name_prefix : *str* : Identifier or name of the image, used for naming output files.

        save_vectors : *bool, optional, default=True* :  Whether to save the primitive unit cell vectors to an npz file.

        write_to_file : *bool, optional, default=False* : If ``True``, the calculation details will be written to a file.

        energy_exclusion_factor : *float, optional, default=1.15* : Factor to exclude atoms based on their fit energy, used to filter out poorly fitted atomic centers.

        normalize_img : *bool, optional, default=True* : Whether the input image should be normalized.

        maxsigma : *float, optional, default=1e-8* : The minimum possible value for the standard deviation of energy clusters.

        accuracy : *int, optional, default=2* : The angle observation accuracy, where the minimum angle is `1/accuracy`. Acceptable values are 1, 2, or 4.

        height_std_factor : *float, optional, default=2.5* : Defines the upper threshold for considering a sinogram standard deviation peak as a periodic direction, based on multiples of the standard deviation.

        relative_height : *bool, optional, default=False* : Whether the minimum value for a local maximum should depend on the global max-min distance.

        factor : *float, optional, default=0.8* : The percentage of the global max-min distance used to define the height boundary for significant local maxima.

        imaging_mode : *str, optional, default="DF"* : The imaging mode. The default is dark field ``"DF"``. Use ``"BF"`` for bright field mode.

        show_bad_energies : *bool, optional, default=False* : If True, displays energy diagrams for non-periodic energies.

        confirm_gradient : *bool, optional, default=False* : If True, plots the minimization energy and its gradient at the minimum value.

        plot : *bool, optional, default=False* : If True, shows plots of the various steps of the process.

        uniform_prefilter_size : *int, optional, default=0* : The size of the uniform prefilter applied to smooth the input image.

        config : *dict, optional* : A dictionary that overrides function arguments like ``uniform_prefilter_size`` and ``energy_exclusion_factor`` if specified.

        plot_final_vectors : *bool, optional, default=False* : If True, creates a plot showing the final unit cell vectors.



    return:
        A sub-directory with the image name inside **path**, containing,
            - The projection standard deviation diagram.
            - The lattice image with annotations of high periodicity directions.
            - The energy diagram for each high periodicity direction.
            - The lattice image shifted by 1/6 of the image dimension along each high periodicity direction.

        - An image of the lattice with the identified lattice vectors.

        - A LaTeX `.tex` file detailing all calculated parameters (which can be included in other LaTeX documents).

        - An array containing the identified lattice vectors.

    """
    if "uniform_prefilter_size" in config:
        uniform_prefilter_size = config["uniform_prefilter_size"]

    if "energy_exclusion_factor" in config:
        energy_exclusion_factor = config["energy_exclusion_factor"]

    if "factor" in config:
        factor = config["factor"]

    if "write_to_file" in config:
        write_to_file = config["write_to_file"]

    downscale_factor = config["downscale_factor"] if "downscale_factor" in config else 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    im = img_format(im)
    if normalize_img:
        im = rescale_intensity(im, out_range=(0.0, 1.0))
    if write_to_file:
        global latex_output_file
        latex_output_file = open(output_dir / f"{name_prefix}unit_cell.tex", "w")
    img_filtered = uniform_filter(im, size=uniform_prefilter_size, mode="reflect") if uniform_prefilter_size > 0 else im
    if downscale_factor > 1:
        img_filtered = downscale_local_mean(img_filtered, downscale_factor)
    directions = psd(
        img_filtered,
        output_dir,
        name_prefix,
        accuracy=accuracy,
        height_std_factor=height_std_factor,
        relative_height=relative_height,
        factor=factor,
        plot=plot,
    )
    direction_labels = []
    vectors = []
    energies = []
    for j in tqdm(directions, unit="dirs"):
        enrg, lng, dia_fig = energy_diagram(
            img_filtered, output_dir, name_prefix, j, relative_height, factor, show_bad_energies=show_bad_energies, plot=plot
        )
        if None in enrg:
            direction_labels.append("NA")
            continue
        magnitude, eng = find_fundamental_period(enrg, lng, maxsigma)
        if plot:
            show_AIC(dia_fig, output_dir, name_prefix, j, enrg, lng, maxsigma)
        vec = [round(magnitude * np.cos(np.radians(j)), 1), round(magnitude * np.sin(np.radians(j)), 1)]
        direction_labels.append(np.array2string(np.array(vec)))
        vectors.append(vec)
        energies.append(eng)
        print_file("")
    if plot:
        plot_lines(img_filtered, output_dir, name_prefix, directions, direction_labels)
    energies = np.array(energies).reshape(-1)
    vectors = np.array(vectors)
    eligible_vectors = exclude_high_energies(
        vectors, energies, energy_exclusion_factor=energy_exclusion_factor, normalize_img=normalize_img, plot=plot
    )
    v1, v2 = choose_set(eligible_vectors)
    v1, v2 = reduce_uc_vectors(v1, v2, min_reduction=1, vec_len_lim=VEC_LEN_LIM)
    v1, v2 = choose_set(np.array([v1, v2]))  # make sure the shorter vector is v1
    if downscale_factor > 1:
        v1 *= downscale_factor
        v2 *= downscale_factor
    if plot:
        plot_puvs(v1, v2, im, output_dir, name_prefix, imaging_mode)
    primitive_lattice_vectors = optimize(np.concatenate((v1, v2)), im, confirm_gradient=confirm_gradient)
    if plot or plot_final_vectors:
        plot_puvs(primitive_lattice_vectors[0], primitive_lattice_vectors[1], im, output_dir, name_prefix, imaging_mode)

    print_file("\nprimitive lattice vectors: \n\n", primitive_lattice_vectors)
    if write_to_file:
        latex_output_file.close()
        latex_output_file = None
    if save_vectors:
        np.savez(output_dir / f"{name_prefix}vectors", primitive_lattice_vectors)
    return primitive_lattice_vectors

