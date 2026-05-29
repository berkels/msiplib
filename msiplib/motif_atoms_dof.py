"""
This is the `Direct motif extraction from high resolution crystalline STEM images` as described in :cite:`AlBe23`.

The main function of this section is :py:meth:`get_motif_atoms_dof<msiplib.motif_atoms_dof.get_motif_atoms_dof>`  and can be called from outside the library. 

Example of how to use this module
----------------------------------
>>> #run this code in the msiplib folder and be aware that some files will be created
>>> 
>>> from msiplib.io import read_image
>>> from msiplib.motif_atoms_dof import get_motif_atoms_dof
>>> from skimage.exposure import rescale_intensity
>>> import jax
>>> 
>>> np_mod = jax.numpy
>>> f = rescale_intensity(read_image("./images/right_grain.png"), out_range=(0.0, 1.0))
>>> name = "right_grain"
>>> n = 2
>>> compute_uv = True
>>> erase_inf_radius = 20
>>> initial_diameter = 16
>>> image_path = "./"
>>> output_dir = "./"
>>> v1, v2 = None,None
>>> num_sigma = 4.0
>>> separation = None
>>>
>>> atom_gaussians,v = get_motif_atoms_dof(f, name, output_dir, n, v1, v2, compute_uv, np_mod, num_sigma, initial_diameter, erase_inf_radius, separation,plot=False, max_iter=50000)
--------------------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from msiplib.io import read_image, write_image
from msiplib.motif import (
    get_motif,
    image_from_motif,
    plot_motif_on_image,
    project_positions_to_unit_cell,
)
from msiplib.emic import get_atom_positions, get_atoms_simple
from msiplib.optimization import GradientDescent
import sklearn.cluster as skcl
import hyperspy.api as hs
import atomap.api as am
import jax
from msiplib.emic import twoD_Gaussian
from msiplib.misc import plot_image_pixel_precise, plot_image_pixel_precise_to_pdf, rint
from scipy.ndimage import affine_transform
from msiplib.segmentation import create_segmentation_colormap
from msiplib.emic import get_atom_sigma_estimate
from pathlib import Path


def atoms_pos(reconstruction, erase_inf_radius=None, separation=None, min_center_value=None):
    """
    Designed to detect atomic positions from a **crystalline image**, commonly used for reconstructed images.  It provides two methods for identifying objects: by using a radius to erase detected positions or by using a separation value to locate the centers of objects.

    Parameters:

        reconstruction : *2D array* : The reconstructed gray-scale image of the crystal lattice.
        erase_inf_radius  : *int or "None", optional* : The radius within which atomic positions will be erased. If provided, atoms within this radius are removed from the list of detected positions.
        separation  : *int or "None", optional* : The minimum separation between detected atomic centers. This parameter is used for atom detection when ``erase_inf_radius`` is not provided.
        min_center_value : *float, optional* : The minimum intensity value to be considered for detecting atomic centers. If not provided, it defaults to the mean value of the ``reconstruction`` image.

    Returns:

       A 2D array containing the coordinates of the detected atoms within the image.
    note:
        The following conditions apply:
            - The values for both ``separation`` and ``erase_inf_radius`` must be integers greater than or equal to ``1``. These parameters are mutually exclusive, meaning that if one is specified, the other cannot be used and it is set as ``None``, and the specified value must be greater than ``1``.

            - The value for ``initial_diameter`` must be an integer greater than or equal to ``1`` and is a required parameter.
    """
    if erase_inf_radius is not None:
        if min_center_value is None:
            min_center_value = reconstruction.mean()
        return get_atoms_simple(
            reconstruction, min_center_value=min_center_value, erase_inf_radius=erase_inf_radius, min_boundary_dist=2
        )
    elif separation is not None:
        # Initial atom positions are detected using atomap.
        im = hs.signals.Signal2D(reconstruction)
        return am.get_atom_positions(im, separation=separation)
    else:
        raise ValueError("Either erase_inf_radius or separation must be set")


def periodic_distance(x, y, v1, v2):
    """
    Computes the minimum distance between two points in a crystal lattice with periodic boundary conditions, considering the lattice unit cell vectors.

    Parameters:

        x : *1D array* : Coordinates of the first point.

        y : *1D array* : Coordinates of the second point.

        v1 :*1D array* : The first unit cell vector defining the periodicity of the lattice.

        v2 :*1D array* : The second unit cell vector defining the periodicity of the lattice.

    Returns:

       The minimum distance between the two points, considering their periodic images in the crystal lattice.
    """
    xMinusY = x - y
    return np.array(
        [
            np.linalg.norm(xMinusY),
            np.linalg.norm(xMinusY + v1),
            np.linalg.norm(xMinusY + v2),
            np.linalg.norm(xMinusY - v1),
            np.linalg.norm(xMinusY - v2),
            np.linalg.norm(xMinusY - v1 - v2),
            np.linalg.norm(xMinusY + v1 + v2),
            np.linalg.norm(xMinusY - v1 + v2),
            np.linalg.norm(xMinusY + v1 - v2),
        ]
    ).min()


def atom_in_motif_pos(
    f,
    u,
    v,
    initial_diameter=5,
    erase_inf_radius=None,
    separation=None,
    project_to_uc=True,
    n=None,
    save_base_path: str | os.PathLike | None = None,
    name_prefix="",
    set_size_bound=True,
    show_initial=False,
    min_center_value=None,
    plot=False,
    use_kmedoids=False,
):
    """
    Designed to detect, cluster, and refine atomic positions within a crystalline motif. It takes an image ``f``, motif ``u``, and primitive unit cell vectors ``v`` 
    to reconstruct the atomic positions inside the primitive unit cell and cluster them. The function also supports saving various visualizations and outputs.    
    
    Parameters:

        f : *2D ndarray* : Input for :py:meth:`get_motif <msiplib.motif.get_motif>`.
        
        u : *2D ndarray* : Input for :py:meth:`get_motif <msiplib.motif.get_motif>`.
        
        v : *2x2 ndarray* : Input for :py:meth:`get_motif <msiplib.motif.get_motif>`.

        initial_diameter : *int, optional, default=5* : The initial guess for the diameter of the detected atomic centers. 
        
        erase_inf_radius : *int or None, optional* : Input for :py:meth:`atoms_pos <atoms_pos>`.
        
        separation : *int or None, optional* : Input for :py:meth:`atoms_pos <atoms_pos>`.

        project_to_uc : *bool, optional, default=True* : Input for :py:meth:`get_atom_positions <msiplib.emic.get_atom_positions>`.

        n : *int, optional, default=None* : The number of clusters (atoms per unit cell). If ``None``, the number of clusters is determined automatically.
        
        save_base_path : *Path or str, optional, default=None* : The base path (directory) for saving the output visualizations. 
            If ``None``, no images are saved.
        
        name_prefix : *str, optional, default=""* : The prefix for output filenames (e.g., ``"name_"`` or ``""``).
        
        set_size_bound : *bool, optional, default=True* : If ``True``, the image will be cropped to the size of the primitive unit cell, ensuring consistency. 
        
        show_initial : *bool, optional, default=False* : If ``True``, displays the initial detected atomic positions. 

    Returns:

       A 2D ndarray ``motif_atoms`` containing the detected and refined atomic positions with Gaussian parameters, along with the motif image ``u`` (*2D ndarray*) and the primitive unit cell vectors ``v`` (*2x2 ndarray*) used for projection and clustering.

    """
    if save_base_path is not None:
        save_base_path = Path(save_base_path)
    
    size_bound = int(np.ceil(max(np.linalg.norm(v[0]), np.linalg.norm(v[1]))) * 4.5) if set_size_bound else max(f.shape)
    reconstruction = image_from_motif(u, f, v)[: min(f.shape[0], size_bound), : min(f.shape[1], size_bound)]
    atom_positions = atoms_pos(reconstruction, erase_inf_radius=erase_inf_radius, separation=separation, min_center_value=min_center_value)

    if save_base_path is not None or show_initial is True:
        fig, ax, dpi = plot_image_pixel_precise(f)
        plt.scatter(atom_positions[:, 0], atom_positions[:, 1], s=initial_diameter)
        if save_base_path is not None:
            fig.savefig(save_base_path / f"{name_prefix}initial_atoms.pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)
        if show_initial is True:
            plt.show()
        plt.close()

    if project_to_uc:
        predicted_parameters = np.array(
            get_atom_positions(
                reconstruction,
                initial_centers=atom_positions,
                initial_diameter=initial_diameter,
                plot_bump_fit_domains=plot,
                clear_border_atoms=project_to_uc,
            )
        )

        if save_base_path is not None:
            fig, ax, dpi = plot_image_pixel_precise(reconstruction)
            plt.scatter(predicted_parameters[:, 0], predicted_parameters[:, 1], s=predicted_parameters[:, 2:4].mean())
            fig.savefig(save_base_path / f"{name_prefix}reconst_fit.pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)
            plt.close()

        means = predicted_parameters.mean(axis=0)
        print("means = ", means)
        projected_atoms = project_positions_to_unit_cell(v, predicted_parameters[:, :2])
        if n is not None and not use_kmedoids:
            Kmeans = skcl.KMeans(n_clusters=n, random_state=42).fit(projected_atoms)
            motif_atoms = np.zeros((n, 7))
            motif_atoms[:, :2] = (
                Kmeans.cluster_centers_
            )  # this will return a matrix with the shape (n,7) n: #of atoms per unit cell
        else:
            mean_sigma = means[2:4].mean()

            v1 = np.array(v)[0]
            v2 = np.array(v)[1]

            def similarity(x, y):
                return periodic_distance(x, y, v1, v2)

            if n is not None and use_kmedoids:
                v1 = np.array(v)[0]
                v2 = np.array(v)[1]

                def similarity(x, y):
                    return periodic_distance(x, y, v1, v2)
                num_projected_atoms = projected_atoms.shape[0]
                distance_matrix = np.zeros((num_projected_atoms, num_projected_atoms))

                for i in range(num_projected_atoms):
                    for j in range(i + 1, num_projected_atoms):
                        d = similarity(projected_atoms[i], projected_atoms[j])
                        distance_matrix[i, j] = distance_matrix[j, i] = d

                from kmedoids import KMedoids
                Kmedoids = KMedoids(n_clusters=n, method='fasterpam', random_state=42)
                medoid_indices_ = Kmedoids.fit(distance_matrix).medoid_indices_

                motif_atoms = np.zeros((n, 7))
                motif_atoms[:, :2] = (
                    projected_atoms[medoid_indices_]
                )
            else:
                clustering = skcl.DBSCAN(eps=1.5 * mean_sigma, min_samples=3, metric=similarity).fit(projected_atoms)
                labels = clustering.labels_
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                print(f"# clusters: {n_clusters_}")
                unique_labels = set(labels)
                core_samples_mask = np.zeros_like(labels, dtype=bool)
                core_samples_mask[clustering.core_sample_indices_] = True

                colors = create_segmentation_colormap()
                median_centers = []
                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        # Black used for noise.
                        col = [0, 0, 0, 1]

                    class_member_mask = labels == k

                    xy = projected_atoms[class_member_mask & core_samples_mask]
                    plt.plot(
                        xy[:, 0],
                        xy[:, 1],
                        "o",
                        markerfacecolor=tuple(col),
                        markeredgecolor="k",
                        markersize=14,
                    )

                    if k != -1:
                        median_centers.append(np.median(xy, axis=0))

                    xy = projected_atoms[class_member_mask & ~core_samples_mask]
                    plt.plot(
                        xy[:, 0],
                        xy[:, 1],
                        "o",
                        markerfacecolor=tuple(col),
                        markeredgecolor="k",
                        markersize=6,
                    )
                    plt.plot(median_centers[-1][0], median_centers[-1][1], "x")
                center = np.zeros((2))
                parallelogram = np.array([center, center + v[0], center + v[0] + v[1], center + v[1], center])
                plt.plot(parallelogram[:, 0], parallelogram[:, 1], "r--")

                plt.gca().invert_yaxis()
                plt.title(f"Estimated number of clusters: {n_clusters_}")
                if save_base_path is not None:
                    plt.savefig(save_base_path / f"{name_prefix}DBSCAN.pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)
                    plt.close()
                motif_atoms = np.zeros((n_clusters_, 7))
                motif_atoms[:, :2] = median_centers
        motif_atoms[:, 2:] = means[2:]
        return motif_atoms, u, v 
    else: #project_to_uc = false
        # return the detected atomic positions with unoptimized gaussian parameters
        atom_width = initial_diameter // 2
        atoms_parameters = np.zeros((len(atom_positions), 7))
        atoms_parameters[:, :2] = atom_positions
        atoms_parameters[:, 2] = atom_width
        atoms_parameters[:, 3] = atom_width
        atoms_parameters[:, 4] = 1  # h
        atoms_parameters[:, 5] = np.min(f)  # b
        atoms_parameters[:, 6] = 1  # r

        if save_base_path is not None:
            fig, ax, dpi = plot_image_pixel_precise(reconstruction)
            plt.scatter(atoms_parameters[:, 0], atoms_parameters[:, 1], s=atom_width)
            fig.savefig(save_base_path / f"{name_prefix}reconst_fit.pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)
            plt.close()
        return atoms_parameters, u, v


def gaussian_indices(f, v, g, n, num_sigma, np_mod=jax.numpy):
    """
    Calculates the regions (indices) in the image that correspond to the positions of Gaussian functions, which represent atomic centers in a 2D lattice. It returns the indices of Gaussian bumps and their supports in the image, useful for fitting a Gaussian function to the lattice.

    Parameters:

        f : *2D array* : The original gray-scale image representing the crystal lattice.

        v : *2x2 ndarray* : The primitive unit cell vectors defining the periodicity of the lattice.

        g : *2D array* : Parameters of the Gaussian functions representing the atoms (center, width, amplitude, etc.).

        n : *int* : The number of atoms per unit cell.

        num_sigma : *int* : Number of standard deviations to consider when calculating the support of the Gaussian bumps (i.e., the region around each atom).

        np_mod : *module, optional* : A NumPy-like module, defaulting to jax.numpy, used for vectorized computation.


    Returns:

        **bump_indices** (*tuple*), **bump_supports** (*tuple*), and **coords** (*2D array*) represent the indices of
        Gaussian bumps (atomic centers), the regions influenced by these Gaussian functions, and the projected pixel coordinates of the Gaussian bumps in the image, respectively.
    """
    x = np.arange(0, f.shape[1])
    y = np.arange(0, f.shape[0])
    xy = np.dstack(np.meshgrid(x, y))

    # TODO: Question: Why project the xy mesh to the unit cell? And why do we only move the bump one step? Stencil? 
    p = project_positions_to_unit_cell(v, xy, np_mod=jax.numpy)  # convert back to pixels
    coords = [p[..., 0], p[..., 1]]

    bump_indices = []
    bump_supports = []

    for i in [0, 1, -1]:
        for j in [0, 1, -1]: #move the bump in 2d plane in all directions, eg. i and j move along v foreward, backward, up and down
            for k in range(n):
                x0, y0, x_sigma, y_sigma, A, b, r = g[k] #extract parameters of gaussian bump k
                x1, y1 = i * v[0] #calculate new coordinates after moving the central point of the bump along the vectors
                x2, y2 = j * v[1]
                bump_indices.append((i, j, k)) 
                bump_supports.append(
                    np_mod.where(
                        np_mod.logical_and(
                            (np_mod.square(coords[0] - (x0 + x1 + x2)) <= (num_sigma * x_sigma) ** 2), #test if the surrounding is still in a sigma enviroment
                            (np_mod.square(coords[1] - (y0 + y1 + y2)) <= (num_sigma * y_sigma) ** 2),
                        )
                    )
                )

    bump_indices = tuple(bump_indices) #indices of all movements
    bump_supports = tuple(bump_supports) #indices of the movements, that are still in the sigma enviroment
    return bump_indices, bump_supports, coords


def gaussian_in2d_fromnd(xy, x0, y0, x_sigma, y_sigma, A, b, r, np_mod=np):
    r"""
    This function builds multiple 2D Gaussian distributions simultaneously by using lists of equal lengths for the input parameters:

    .. math::
        g(x,y)=A \exp\left( \frac{- 1}{ 2 ( 1 - r^2 ) } \left( \left( \frac{x-x_0}{\sigma_x} \right)^2 + \left( \frac{y-y_0}{\sigma_y} \right)^2 - \frac{2r}{ \sigma_x \sigma_y} (x-x_0 ) (y-y_0) \right) \right) + b

    Example:

    >>> f = #some nd image
    >>> g, v = get_motif_atoms_dof(f, name='test', output_dir='./')
    >>> xy = np.dstack(np.meshgrid(np.arange(f.shape[1]), np.arange(f.shape[0])))
    >>> x0, y0, x_sigma, y_sigma, A,_, r, np_mod=np = g.T
    >>> b = g[0,5]
    >>> f_motif_model = np.zeros_like(f)
    >>> f_motif_model += gaussian_in2d_fromnd(xy, x0, y0, x_sigma, y_sigma, A, b, r, np_mod=np)
    """
    x, y = xy[..., 0], xy[..., 1]
    c1 = -1 / (2 * (1 - r**2))
    gaussian = (
        sum(
            A[:, np.newaxis, np.newaxis]
            * np_mod.exp(
                c1[:, np.newaxis, np.newaxis]
                * (
                    ((x - x0[:, np.newaxis, np.newaxis]) / x_sigma[:, np.newaxis, np.newaxis]) ** 2
                    + ((y - y0[:, np.newaxis, np.newaxis]) / y_sigma[:, np.newaxis, np.newaxis]) ** 2
                    - 2
                    * r[:, np.newaxis, np.newaxis]
                    * (x - x0[:, np.newaxis, np.newaxis])
                    * (y - y0[:, np.newaxis, np.newaxis])
                    / (x_sigma[:, np.newaxis, np.newaxis] * y_sigma[:, np.newaxis, np.newaxis])
                )
            ),
        )
        + b
    )
    return gaussian


# @jax.jit
def image_from_gaussian(g, coords, v, bump_indices, bump_supports, np_mod=jax.numpy):
    """
    Calculates the regions (indices) in the image that correspond to the positions of Gaussian functions, which represent atomic centers in a 2D lattice. It returns the indices of Gaussian bumps and their supports in the image, useful for fitting a Gaussian function to the lattice.

    Parameters:

        g : *2D array* : Parameters of the Gaussian functions representing the atoms (center, width, amplitude, etc.).

        coords : *2D array* : Projected pixel coordinates corresponding to the positions of the Gaussian bumps in the image.

        v : *2x2 ndarray* : The primitive unit cell vectors defining the periodicity of the lattice.

        bump_indices : *tuple* : Indices corresponding to the positions of Gaussian bumps (atomic centers) in the image.

        bump_supports : *tuple* : Supports of the Gaussian bumps, indicating the regions in the image that are influenced by the atomic Gaussians.


        np_mod : *module, optional* : A NumPy-like module, defaulting to jax.numpy, used for vectorized computation.


    Returns:

        A reconstructed image generated from the Gaussian functions placed at atomic positions.
    """

    n = g.shape[0]
    b = g[0][5]

    res = np_mod.empty_like(coords[0])
    res = res.at[:].set(b)

    for m in range(len(bump_indices)):
        i, j, k = bump_indices[m]
        x0, y0, x_sigma, y_sigma, A, _, r = g[k]
        x1, y1 = i * v[0]
        x2, y2 = j * v[1]
        indices = bump_supports[m]
        res = res.at[indices].add(
            twoD_Gaussian(
                [coords[0][indices], coords[1][indices]],
                x0 + x1 + x2,
                y0 + y1 + y2,
                x_sigma,
                y_sigma,
                A,
                0,
                r,
                np_mod=jax.numpy,
            )
        )
    return res


# @jax.jit
def g_energy(g, f, v, bump_indices, bump_supports, coords):
    r"""
    Calculates the mean squared error (MSE) between a crystalline image ``f`` and a reconstructed image created from Gaussian functions, as described in equation (16) :cite:`AlBe23`. 
    
    .. math::
            E_{\text{motif}}^{\text{atoms}}[\Theta, b, v] := \int_{\Omega} \left( f(x) - \varrho[g_{\text{motif}}[\Theta], b](P_{EC}[v](x)) \right)^2 \, dx

    Parameters:

        g : *2D array* : Parameters of the Gaussian functions representing the atoms (center, width, amplitude, etc.).

        f : *2D array* : The original gray-scale image representing the crystal lattice.

        v : *2x2 ndarray* : The primitive unit cell vectors defining the periodicity of the lattice.

        bump_indices : *tuple* : Indices corresponding to the positions of Gaussian bumps (atomic centers) in the image.

        bump_supports : *tuple* : Supports of the Gaussian bumps, indicating the regions in the image that are influenced by the atomic Gaussians.

        coords : *2D array* : Projected pixel coordinates corresponding to the positions of the Gaussian bumps in the image.


    Returns:

        The mean squared error (MSE) between the original image ``f`` and the reconstructed image from Gaussians.
    """
    res = np.mean((f - image_from_gaussian(g, coords, v, bump_indices, bump_supports)) ** 2)
    return res


def get_motif_atoms_dof(
    f,
    name,
    output_dir: str | os.PathLike,
    n=None,
    v1=None,
    v2=None,
    compute_uv=True,
    np_mod=jax.numpy,
    num_sigma=4,
    initial_diameter=None,
    erase_inf_radius=None,
    separation=None,
    nm_per_pixel=None,
    max_iter=500,
    plot=False,
    energy_exclusion_factor=1.15,
    show_bad_energies=False,
    height_std_factor=2.5,
    show_motif_legend=True,
    UCE_config={},
    rfrac=1,
    plot_rotated_orientation=True,
    min_center_value=None,
    align_to_y_axis=False,
    fit_to_reconstruction=False,
    use_kmedoids=False
):
    """
    Determines the Guassian parameters (DOF) of the atoms within the motif of a crystal lattice. It uses Gaussian fitting, clustering, and optimization to detect atom positions and refine the motif's parameters.
    
    Parameters:
        f : *2D array* : The original crystal image.

        name : *str* : Name of the image, used as a prefix for saving output files.

        output_dir : *str* : Directory where output files (e.g., motif images, atomic positions) will be saved.

        n : *int, optional* : Number of atoms per unit cell. If not provided, the algorithm uses **DBSCAN** clustering to estimate it. *Input for* :py:meth:`atom_in_motif_pos <atom_in_motif_pos>`.

        v1, v2 : *1D arrays, optional* : Initial guesses for the unit cell vectors. These are used if ``compute_uv`` is set to True for initializing the lattice vectors ``v1`` and ``v2``.

        compute_uv : *bool, optional* : If ``True``, the function computes the unit cell vectors from the image using the get_motif function. If ``False``, the function reads unit cell vectors from a previously saved file.

        np_mod : *module, optional* : A NumPy-like module used for array computations. Defaults to ``jax.numpy`` for optimized computations.

        num_sigma :*int, optional* : Input for :py:meth:`gaussian_indices <gaussian_indices>`.

        initial_diameter : *int, optional* : Input for :py:meth:`atom_in_motif_pos <atom_in_motif_pos>`.

        erase_inf_radius :*int or "None", optional* : Input for :py:meth:`atom_in_motif_pos <atom_in_motif_pos>`.

        separation :*int or "None", optional* : Input for :py:meth:`atom_in_motif_pos <atom_in_motif_pos>`.

        nm_per_pixel :*float, optional* : The scaling factor to convert pixel distances to nanometers. Used for converting the detected atomic positions and unit cell vectors into physical units.

        max_iter :*int, optional*: Maximum number of iterations for the optimization procedure using Gradient Descent.

        plot :*bool, optional* : *Input for* :py:meth : `get_motif <msiplib.motif.get_motif>`.

        energy_exclusion_factor :*float, optional* : Input for :py:meth:`get_motif <msiplib.motif.get_motif>`.

        show_bad_energies :*bool, optional* : Input for :py:meth:`get_motif <msiplib.motif.get_motif>`.

        height_std_factor :*float, optional* : Input for :py:meth:`get_motif <msiplib.motif.get_motif>`.

        show_motif_legend :*bool, optional* : Input for :py:meth:`plot_motif_on_image <msiplib.motif.plot_motif_on_image>`.

        UCE_config :*dict, optional* : Input for :py:meth:`get_motif <msiplib.motif.get_motif>`.

        rfrac :*float, optional* : Input for :py:meth:`get_motif <msiplib.motif.get_motif>`.

        plot_rotated_orientation :*bool, optional, default=True* : If True, the function will plot the motif in its rotated orientation, aligning the shortest unit cell vector along the x-axis.

    Returns:

        **g** (*2D array*) contains the parameters of the fitted Gaussian functions representing atomic centers, and **v** (*2x2 ndarray*) represents the refined unit cell vectors after the optimization process.

    note:
        The following conditions apply:
            -The value for ``nm_per_pixel`` can be either a string or a float, allowing for flexible input formats.

            -``v1`` and ``v2`` must both be provided as lists containing two items (either strings or floats) and are mutually dependent, meaning they must be supplied together.

            -``compute_uv`` must be a boolean. If set to ``0`` (``False``), it excludes ``v1`` and ``v2``, but if set to ``1`` (``True``), both ``v1`` and ``v2`` must be used.

            -``crop_start`` and ``crop_size`` must both be lists of two integers, and they are mutually dependent, meaning that if one is provided, the other must also be provided.

            -``energy_exclusion_factor`` and ``height_std_factor`` must be floats with a minimum value of ``1`` and depend on ``compute_uv`` being provided.

        The parameters ``v1`` and ``v2`` can be specified directly. Otherwise they can be set to ``None`` while enabling ``compute_uv=True``. If neither option is set correctly, the program expects the output directory to contain two files:

            -An **npz** file with the vectors in the format:
            ``output_dir + name + "_motif_vectors.npz"``

            -An **nc** file with the motif ``u`` in the format:
            ``output_dir + name + "_motif.nc"``

        These files are typically outputs from:

            -:py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`.

            -:py:meth:`get_motif <msiplib.motif.get_motif>`.
    """
    

    if nm_per_pixel is not None:
        if initial_diameter is None:
            initial_diameter = get_atom_sigma_estimate(nm_per_pixel)
            print(f"Estimating initial_diameter={initial_diameter}.")
        if erase_inf_radius is None and separation is None:
            erase_inf_radius = rint(initial_diameter * 2.5)
            print(f"Estimating erase_inf_radius={erase_inf_radius}.")

    if initial_diameter is None:
        raise ValueError("initial_diameter not specified")

    # Set up output directory and name prefix for consistent file naming
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    name_prefix = "" if name is None else f"{name}_"

    reconstruction = None
    f_orig = None

    if v1 is not None and v2 is not None:
        if np.linalg.norm(v1) > np.linalg.norm(v2):
            v1, v2 = v2, v1
    if compute_uv:
        v_init = np.array([v1, v2]) if (v1 is not None and v2 is not None) else None
        u, reconstruction, v = get_motif(
            f,
            name_prefix,
            path=output_path,
            read_vectors=False,
            v=v_init,
            plot=plot,
            energy_exclusion_factor=energy_exclusion_factor,
            show_bad_energies=show_bad_energies,
            height_std_factor=height_std_factor,
            UCE_config=UCE_config,
            rfrac=rfrac,
        )
    else:
        u = read_image(output_path / f"{name_prefix}motif.nc")
        v = np.load(output_path / f"{name_prefix}motif_vectors.npz")["arr_0"]

    g, u, v = atom_in_motif_pos(
        f,
        u,
        v,
        n=n,
        erase_inf_radius=erase_inf_radius,
        **({"initial_diameter": initial_diameter} if initial_diameter is not None else {}),
        separation=separation,
        save_base_path=output_path,
        name_prefix=name_prefix,
        min_center_value=min_center_value,
        plot=plot,
        use_kmedoids=use_kmedoids,
    )
    n = g.shape[0]

    fig, dpi = plot_motif_on_image(
        u,
        v,
        f,
        show_outline=True,
        motif_atoms=g[:, :2],
        extend_motif_atoms=True,
        show_plot=False,
        dot_size=int((10 * max(f.shape)) / 512),
        show_motif_legend=True,
    )
    fig.savefig(output_path / f"{name_prefix}motif_initial.pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)

    if fit_to_reconstruction and reconstruction is not None:
        print("Replacing f by reconstruction")
        f_orig = f
        f = reconstruction

    print("<<<initialized>>>")

    # precompute indices and supports for the gaussian bumps
    bump_indices, bump_supports, coords = gaussian_indices(f, v, g, n, num_sigma, np_mod)
    # initial energy and gaussian compilation
    t = image_from_gaussian(g, coords, v, bump_indices, bump_supports)
    write_image(output_path / f"{name_prefix}t.nc", t)
    write_image(output_path / f"{name_prefix}f.nc", f)
    print("compiling ... ")
    # JIT compile the objective function and its gradient for efficiency
    objective_image_from_gaussian = jax.jit(g_energy)
    objective_image_from_gaussian_grad = jax.jit(jax.grad(g_energy))

    objective_image_from_gaussian(g, f, v, bump_indices, bump_supports, coords)
    objective_image_from_gaussian_grad(g, f, v, bump_indices, bump_supports, coords)
    print("done")

    # Optimize the Gaussian parameters using Gradient Descent
    for i in range(2):
        g = GradientDescent(
            g,
            E=lambda t: objective_image_from_gaussian(t, f, v, bump_indices, bump_supports, coords),
            DE=lambda t: objective_image_from_gaussian_grad(t, f, v, bump_indices, bump_supports, coords),
            maxIter=max_iter,
            NonlinearCG=True,
        )
        if i == 0:
            bump_indices, bump_supports, coords = gaussian_indices(f, v, g, n, num_sigma, np_mod)

    if fit_to_reconstruction and reconstruction is not None:
        print("Switching back to f")
        f = f_orig

    print(g)
    reconstruction = image_from_gaussian(g, coords, v, bump_indices, bump_supports)
    write_image(output_path / f"{name_prefix}gaussian_reconstruction.nc", reconstruction)
    write_image(output_path / f"{name_prefix}g.nc", g)

    # Create pdf names plots_precision.pdf
    pp = PdfPages(output_path / f"{name_prefix}motif_results.pdf")
    plot_image_pixel_precise_to_pdf(f, pp, vmin=0, vmax=1)
    # plot_image_pixel_precise_to_pdf(reconstruction, pp, vmin=0, vmax=1)
    plot_image_pixel_precise_to_pdf(reconstruction, pp, vmin=reconstruction.min(), vmax=reconstruction.max())

    fig, dpi = plot_motif_on_image(
        u,
        v,
        f,
        show_outline=True,
        motif_atoms=g[:, :2],
        extend_motif_atoms=True,
        show_plot=False,
        dot_size=int((10 * max(f.shape)) / 512),
        show_motif_legend=show_motif_legend,
    )
    fig.savefig(output_path / f"{name_prefix}motif_updated.pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)

    fig.savefig(pp, format="pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)

    # converting the motif information into crystallographic information
    v1norm = np.linalg.norm(v[0])
    c, s = v[0, 0] / v1norm, v[0, 1] / v1norm
    M = np.array([[c, s], [-s, c]])
    # Rotate by another 90 degrees to align to the y-axis instead of the x-axis.
    if align_to_y_axis:
        M = np.matmul(np.array([[0, -1], [1, 0]]), M)
    print("M,", M, "\n")

    # align the shortest uc vector to the x axis
    alignedv = np.array([np.matmul(M, v[0]), np.matmul(M, v[1])])
    print("alignedv", alignedv, "\n\n\n")

    if not align_to_y_axis:
        # make sure v2 is in the fourth quarter
        if alignedv[1, 1] <= 0:
            alignedv[1] = -alignedv[1]
            if alignedv[1, 0] < 0:
                alignedv[1] = alignedv[1] + alignedv[0]
    # rotate the atoms
    rotatedg = (np.matmul(M, g[:, :2].T)).T
    ordered_atoms = project_positions_to_unit_cell(alignedv, rotatedg[:, :2], np_mod=np)
    ordered_atoms = ordered_atoms[abs(ordered_atoms[:, 1]).argsort()]

    if plot_rotated_orientation:
        # show the rotated orientation
        rotu = affine_transform(u, M, mode="wrap")
        center = np.array([f.shape[1], f.shape[0]]) / 2
        disp = np.matmul(M.T, center) - center
        crystaldisp = rint(np.matmul(disp, np.linalg.inv(v)))
        newdisp = np.matmul(crystaldisp, v)
        rotf = affine_transform(f, M, offset=(-newdisp[1], -newdisp[0]), mode="constant", cval=f.mean())
        fig, dpi = plot_motif_on_image(
            None,
            alignedv,
            rotf,
            show_outline=True,
            motif_atoms=ordered_atoms,
            extend_motif_atoms=True,
            show_plot=False,
            dot_size=int((10 * max(f.shape)) / 512),
            show_motif_legend=show_motif_legend,
        )
        fig.savefig(output_path / f"{name_prefix}motif_rotated.pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)
        fig.savefig(pp, format="pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)
        plt.close(fig)

    if nm_per_pixel is not None:
        info_string = ""
        string_formatter = {"float_kind": lambda x: f"{x:.6f}nm"}
        for i in range(2):
            info_string += f"v{i+1} = {np.array2string(v[i]*nm_per_pixel, formatter=string_formatter)}\n"
        for i in range(g.shape[0]):
            info_string += f"atom {i+1} = {np.array2string(g[i, :2]*nm_per_pixel, formatter=string_formatter)}\n"

        info_string += "\nAligned coordinate system\n"

        for i in range(2):
            info_string += f"v{i+1} = {np.array2string( alignedv[i]*nm_per_pixel, formatter=string_formatter)}\n"
        for i in range(g.shape[0]):
            info_string += (
                f"atom {i+1} = {np.array2string( ordered_atoms[i]*nm_per_pixel, formatter=string_formatter)}\n"
            )

        print(info_string)

        info_fig = plt.figure()
        info_fig.text(0.1, 0.0, info_string)
        info_fig.savefig(pp, format="pdf")
        plt.close(info_fig)
    pp.close()
    return np.array(g), v