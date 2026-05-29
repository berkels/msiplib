r"""
This is the first step of the `Direct motif extraction from high resolution crystalline STEM images`.

Full explanation is covered in section 4 of :cite:`AlBe23`

The main function of this file is :py:meth:`get_motif <get_motif>` and can be called from outside the module.

Example of how to use this module
----------------------------------

    >>> #run this code in the msiplib folder and be aware that some files will be created
    >>> 
    >>> from msiplib.motif import get_motif
    >>> from msiplib.io import read_image
    >>> im = read_image('right_7.png',as_gray=True)
    >>> name = 'right_7'
    >>> path = './'
    >>> motif_img, reconstructed_full_img, vectors = get_motif(im, name, path)
    
--------------------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from msiplib.unit_cell_from_real_space import get_primitive_unit_cell_vectors
import jax
from msiplib.optimization import GradientDescent
from msiplib.io import write_image, check_path
import matplotlib.transforms as mtransforms
import matplotlib.cm as cm
from .misc import plot_image_pixel_precise, rint
from .segmentation import create_segmentation_colormap_no_gray
from .image.manipulators import assert_image_is_gray


def abseil(x):
    return abs(np.ceil(x)).astype(int)


def linint(u, coords):
    """
    Performs linear interpolation on the array ``u`` at specified coordinates using JAX's ``map_coordinates``.

    Parameters:

        u : *array-like* : The input array of the crystalline motif image.
        coords : *array-like* : Coordinates to interpolate at.

    Returns:

        Interpolated values of u at the specified coordinates.
    """

    return jax.scipy.ndimage.map_coordinates(u, coords, order=1, mode="wrap")


@jax.jit
def image_from_motif(u, f, v):
    """
    Generates a modeled image with the same dimensions as the input image ``f``, based on the motif ``u`` and the unit cell vectors ``v``.   

    Parameters:

        u : *2D array* : Model image.
        f : *2D array* : Original image for comparison.
        v : *2x2 array* : Unit cell vectors.


    Returns:

       The image reconstructed from the motif.
    """
    #Pad the image u
    upad = jax.numpy.zeros(np.array(u.shape) + 1)#setting the size (u+ additional row and column)
    upad = upad.at[:-1, :-1].set(u) 
    upad = upad.at[:-1, -1].set(u[:, 0]) #copy first column of u into the last column of the pad
    upad = upad.at[-1, :-1].set(u[0, :]) # copy first row of u into the last row of the pad
    upad = upad.at[-1, -1].set(u[0, 0]) # copy first value of u into the last value of the pad

    A = jax.numpy.linalg.inv(v)#inverse of v, containing the two unit cell vectors
    x = np.arange(0, f.shape[1])
    y = np.arange(0, f.shape[0])
    ind = np.dstack(np.meshgrid(x, y))#mesh of the image pixels
    uc_coord = jax.numpy.einsum("...i,...ij", ind, A) #T_E->C
    frac = uc_coord - jax.numpy.floor(uc_coord)#P_CC  (crytall coordinates)
    coords = jax.numpy.array([frac[..., 0] * (upad.shape[0] - 1), frac[..., 1] * (upad.shape[1] - 1)])
    return linint(upad, coords)


@jax.jit
def motif_energy(u, f, v):
    r"""
    Computes how well the transformed motif ``u`` matches the reference image ``f`` under the transformation matrix ``v``. It essentially evaluates the **energy** 
    (or error) by calculating the mean squared error (MSE) between the transformed motif (using ``u`` and ``v``) and the actual lattice image ``f``.    

    This energy is computed by minimizing the difference between the original image :math:`f(x)` and the transformed 
    motif ``u`` according to the equation (10) :cite:`AlBe23`:

    .. math::

        E_{\text{motif}}^{\text{image}}[u,v] = \int_{\Omega} \left(f(x) - u(P_{E \to C}[v](x))\right)^2 dx

    Here, :math:`P_{E \to C}[v](x)` is defined as the composition of two operations: transforming the Euclidean 
    coordinates to crystal coordinates based on ``v`` and projecting to the unit cell, equation (11) :cite:`AlBe23`:

    .. math::

        P_{E \to C}[v] : \mathbb{R}^2 \to [0, 1]^2, \quad x \mapsto \left( P_{CC} \circ T_{E \to C}[v] \right)(x) = (s, t)
        
    Parameters:

        u : *2D array* : Motif image.
        f : *2D array* : Original image for comparison.
        v : *2x2 array* : Unit cell vectors.
  
    Returns:     
        Mean squared error between the original image and the reconstructed image.
    """
    val = image_from_motif(u, f, v)
    energy = np.mean((f - val) ** 2)
    return energy


def objective(u, im, v):
    """
    Computes the energy between the motif ``u``, the image ``im``, and the unit cell vectors ``v``, where the objective is to minimize the energy with respect to the motif ``u``.    
    
    Parameters:

        u : *2D array* : Motif image.
        im : *2D array* : Original image for comparison.
        v : *2x2 array* : Unit cell vectors.


    Returns:

        Energy of the motif reconstruction.
    """
    return motif_energy(u, im, v)


def objective_v(v, im, u):
    """
    This is a variant of the previous function :py:meth:`objective`, focusing on computing the energy with respect to the **unit cell vectors** ``v``, 
    where the objective is to minimize the energy with respect to the unit cell vectors ``v``, given a motif ``u`` and image ``im``.
 
    
    Parameters:

        v : *2x2 array* : Unit cell vectors.
        im : *2D array* : Input image.
        u : *2D array* : Motif image.


    Returns:

        Energy of the motif reconstruction based on ``v``.
    """
    return motif_energy(u=u, f=im, v=v)


@jax.jit
def objective_full(u, v, im):
    """
    This function takes the motif ``u``, the unit cell vectors ``v``, and the lattice image ``im``, and computes the energy using the :py:meth:`motif_energy<motif_energy>`.

    Parameters:

        u : *2D array* : motif image.
        v : *2x2 array* : unit cell vectors.
        im : *2D array* : input image.


    Returns:

        Total energy of the motif and unit cell vectors reconstruction.
    """
    return motif_energy(u, f=im, v=v)

#gradient for objective dependend on u and v
objective_grad_full = jax.jit(jax.grad(objective_full, argnums=(0, 1)))


def get_motif(
    im,
    name_prefix="",
    path: str | os.PathLike = "./",
    read_vectors=False,
    height_std_factor=2.5,
    imaging_mode="DF",
    v=None,
    plot=False,
    energy_exclusion_factor=1.15,
    maxsigma=1e-8,
    show_bad_energies=False,
    UCE_config={},
    rfrac=1,
    factor=0.8,
    accuracy=1,
    uniform_prefilter_size=0,
):
    """
    The function is designed to find the **motif image** ``u`` and refine the **primitive unit cell vectors** ``v`` based on a provided **crystalline image** ``im``. Additionally, it reconstructs a model image ``reconstruction`` from the motif and calculates how well the model matches the original image. The process involves optimization techniques, such as **gradient descent**, to iteratively improve both the motif and the unit cell vectors.

    Parameters:

        im: *2D array* : Gray-scale, defects-free crystalline image,

        name_prefix : *str* : Prefix for the name of the image files,

        path: *str, optional, default="./"* : Directory to save output files and to read input files when applicable.,

        read_vectors: *bool, optional, default=False* : Use initial values for v from an input file with the format  **path** / **name** _vectors.npz.

        height_std_factor: *float, optional, default=2.5* : Input of :py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`.

        imaging_mode: *str, optional, default="DF"* : Input of :py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`.

        v: *None or 2x2 ndarray, optional, default=None* : Initial value of primitive unit cell vectors.

        plot : *bool, optional, default=False* : Input of :py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`.

        energy_exclusion_factor : *float, optional, default=1.15* : Input of :py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`.

        maxsigma : *float, optional, default=1e-8* : Input of :py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`.

        show_bad_energies : *bool, optional, default=False* : Input of :py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`.

        UCE_config : *dict, optional* : Input of :py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`.

        rfrac : *float, optional* : Fraction of the radius used for optimizing the Gaussian fit to atomic positions.

        factor : *float, optional, default=0.8* : Input of :py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`.

        accuracy: *int, optional, default=2* : Input of :py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`.

        uniform_prefilter_size: *int, optional, default=0* : Input of :py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`.

    Returns:

        **reconstruction** (*2D array*) is the image reconstructed from the average motif, and **v** (*2x2 array*) represents the further refined primitive unit cell vectors.

    This function also writes several files to the specified ``path`` directory, all starting with the prefix **name**:

        - ``path/name_motif_vectors.npz``: The refined unit cell vectors in NumPy format.
        - ``path/name_motif.nc``: The motif image in NetCDF format.
        - ``path/name_motifs_on_img.pdf``: A PDF representation of the motif image overlaid on the original image.
        - ``path/name_difference_motif.png``: The difference (residue) between the original image and the reconstructed image.
        - ``path/name_reconstructedimage_motif.nc``: The fully reconstructed image in NetCDF format.

    note:
        If **read_vectors** is ``True``, the precalculated vectors must be provided in the same output folder as the one for the current calculations.
    """
    path = Path(path)
    im = assert_image_is_gray(im)
    
    if v is not None:
        if np.issubdtype(v.dtype, np.integer):
            raise ValueError("v has to be floating point type")
        v1 = np.array([v[0], v[1]]).astype("f")
    elif read_vectors:
        v1 = np.load(path / f"{name_prefix}vectors.npz")["arr_0"].astype("f")
    else:
        v1 = np.array(
            get_primitive_unit_cell_vectors(
                im,
                path,
                name_prefix,
                energy_exclusion_factor=energy_exclusion_factor,
                maxsigma=maxsigma,
                accuracy=accuracy,
                height_std_factor=height_std_factor,
                imaging_mode=imaging_mode,
                plot=plot,
                show_bad_energies=show_bad_energies,
                config=UCE_config,
                factor=factor,
                uniform_prefilter_size=uniform_prefilter_size,
            )
        )

    v = v1
    shape = (abseil(np.linalg.norm(v, axis=1))) // rfrac
    print(shape)
    u = np.zeros(shape)

    #gradient for objective dependend only on u
    objective_grad = jax.jit(jax.grad(objective))

    for i in range(300):
        info_dict = {}
        iterations = 0
        tau = 0
        #perform minimization of energy for motif u only
        u = GradientDescent(
            u,
            E=lambda t: objective(t, im, v),
            DE=lambda t: objective_grad(t, im, v),
            maxIter=50000,
            NonlinearCG=False,
            info_dict=info_dict,
        )

        iterations = max(iterations, info_dict["iterations"])
        tau = max(tau, info_dict["tau"])
        
        #perform minimization of energy for motif u and unit cell vector v on the same time
        #gradient is defined beforehand: objective_grad_full
        u, v = GradientDescent(
            np.array((u, v), dtype=object),
            E=lambda uv: objective_full(uv[0], uv[1], im),
            DE=lambda uv: np.array(objective_grad_full(uv[0], uv[1], im), dtype=object),
            maxIter=1000,
            NonlinearCG=False,
            info_dict=info_dict,
        )

        iterations = max(iterations, info_dict["iterations"])
        tau = max(tau, info_dict["tau"])

        if (iterations == 0) and (tau == 0):
            break
    
    check_path(path)

    print("refined v", v, "\n\n")
    if plot:
        plt.subplot(1, 3, 1)
        plt.imshow(u, cmap="gray")
        plt.title("u")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
    write_image(path / f"{name_prefix}motif.nc", u)
    reconstruction = image_from_motif(u, im, v)
    residue_img = im - reconstruction
    if plot:
        plt.subplot(1, 3, 2)
        plt.imshow(reconstruction, cmap="gray")
        plt.title("reconstructed image ")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1, 3, 3)
        plt.imshow(residue_img)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("difference")
        plt.axis("off")
        plt.show()
        plt.hist(np.ravel(reconstruction), bins=residue_img.size // 100)
        plt.title("reconstruction histogram")
        plt.show()
        plt.hist(np.ravel(residue_img), bins=residue_img.size // 100)
        plt.title("background histogram")
        plt.show()

    plt.imsave(str(path / f"{name_prefix}reconstructedimage_motif.png"), reconstruction)
    plt.imsave(str(path / f"{name_prefix}difference_motif.png"), residue_img, cmap="gray")
    write_image(path / f"{name_prefix}reconstructedimage_motif.nc", reconstruction)
    plt.close()
    print(v)
    np.savez(str(path / f"{name_prefix}motif_vectors"), v)
    return u, reconstruction, v


def project_positions_to_unit_cell(v, pos, np_mod=np):
    """
    This function projects atom positions to their corresponding unit cell positions.

    Parameters:

        v : *2x2 array* : Unit cell vectors.
        pos : *2D array* : Atom positions.
        np_mod : *module, optional, default=np* : The numpy module.


    Returns:

       Positions projected into the unit cell.
    """
    A = np_mod.linalg.inv(v)
    uc_coord = np_mod.einsum("...ij,...i", A, pos)
    frac = uc_coord - np_mod.floor(uc_coord)
    return np_mod.einsum("...ij,...i", v, frac)


def find_closest_lattice_point_index(v, pos):
    """
    This function finds the closest lattice point to a given position.

    Parameters:

        v : *2x2 array* : Unit cell vectors.
        pos : *1D array* : Position to project onto the lattice.


    Returns:

       Index of the closest lattice point.
    """
    A = np.linalg.inv(v)
    uc_coord = np.einsum("...ij,...i", A, pos)
    return rint(uc_coord)


def span_motif_to_image_domain(v, motif_atoms, image):
    """
    This function spans motif atoms across the image domain based on the unit cell vectors.

    Parameters:

        motif_atoms : *2D array* : positions of the atoms in the motif.
        image : *2D array* : the image in which the motif is to be spanned.


    Returns:

        **atom_positions** (*2D array*) contains the positions of motif atoms across the image domain, and **atom_types** (*list*) specifies the types of atoms, used for coloring purposes.

    """
    center = np.array(image.shape)[::-1] / 2
    index_shift = find_closest_lattice_point_index(v, center) #closest lattice point to image center
    num_motif_atoms = motif_atoms.shape[0]
    image_diam = np.linalg.norm(image.shape)
    num_v0 = int(np.ceil(image_diam / (2 * np.linalg.norm(v[0])))) #number of v0 to cover whole image
    num_v1 = int(np.ceil(image_diam / (2 * np.linalg.norm(v[1])))) #number of v1 to cover whole image
    atom_positions = []
    atom_types = []
    for i in range(-num_v0, num_v0 + 1) + index_shift[0]:
        for j in range(-num_v1, num_v1 + 1) + index_shift[1]:
            positions = [motif_atoms[:, 0] + i * v[0, 0] + j * v[1, 0], motif_atoms[:, 1] + i * v[0, 1] + j * v[1, 1]] #position of atoms in image, after shift along the unit cell vectors
            for k in range(num_motif_atoms):
                pos = [positions[0][k], positions[1][k]]
                # Only consider atoms that are in the image domain.
                if pos[0] >= 0 and pos[1] >= 0 and pos[0] <= image.shape[1] - 1 and pos[1] <= image.shape[0] - 1:
                    atom_positions.append(pos)
                    atom_types.append(k)
    return np.array(atom_positions), atom_types


def plot_motif_on_image(
    motif,
    v,
    image,
    show_outline=True,
    motif_atoms=None,
    extend_motif_atoms=False,
    show_plot=True,
    dot_size=40,
    show_motif_legend=False,
    show_v=False,
    legend_out=False,
):
    """
    This function plots a motif on an image, optionally extending the motif atoms and showing unit cell vectors.

    Parameters:

        motif : *2D array* : motif image to plot.
        v : *2x2 array* : unit cell vectors.
        image : Input of :py:meth:`plot_image_pixel_precise<msiplib.misc.plot_image_pixel_precise>`.
        show_outline : *bool, optional, default=True* : whether to outline the motif.
        motif_atoms : *2D array* : motif atom positions to plot.
        extend_motif_atoms : *bool, optional, default=False* : whether to extend the motif atoms across the entire image domain.
        show_plot : *bool, optional, default=True* : whether to display the plot.
        dot_size : *int, optional, default=40* : size of the atom dots.
        show_motif_legend : *bool, optional, default=False* : If True, a legend showing motif information will be included in the visual output.
        show_v : *bool, optional, default=False* : whether to show unit cell vectors .
        legend_out : *bool, optional, default=False* : whether to display the legend outside the plot.



    Returns:

        **fig** (*matplotlib figure*) is the generated plot, and **dpi** (*int*) refers to the resolution of the figure.
    """
    v_normed = np.zeros_like(v)
    for i in range(2):
        norm = np.linalg.norm(v[i])
        if np.isclose(norm, 0):
            raise ValueError(f"v[{i}] is approximately zero")
        v_normed[i] = v[i] / norm

    # First plot the image as background, which also prepares the figure.
    fig, ax, dpi = plot_image_pixel_precise(image)

    # Find an integer combination of the two unit cell vectors that is
    # close to the center of the image to use this as anchor for plotting
    # the motif.
    A = np.linalg.inv(v.T)
    center = np.array(image.shape)[::-1] / 2
    shift = np.matmul(v.T, np.round(np.matmul(A, center)))

    if motif is not None:
        transform = mtransforms.Affine2D().from_values(
            v_normed[0][0], v_normed[0][1], v_normed[1][0], v_normed[1][1], shift[0], shift[1]
        )

        # Plot the motif and transform its coordinate system to match the image
        im = ax.imshow(motif.T, interpolation="none", origin="upper", cmap="gray")
        trans_data = transform + ax.transData
        im.set_transform(trans_data)

        if show_outline:
            # Mark the outline of the transformed motif
            x1, x2, y1, y2 = im.get_extent()
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "r--", transform=trans_data)

    else:
        if show_outline:
            # Mark the outline of the transformed motif
            parallelogram = np.array([center, center + v[0], center + v[0] + v[1], center + v[1], center])
            ax.plot(parallelogram[:, 0], parallelogram[:, 1], "r--")

    if show_v:
        U, V = v.T
        ax.quiver(shift[0], shift[1], U[0], V[0], color="r", angles="xy", scale_units="xy", scale=1)
        ax.quiver(shift[0], shift[1], U[1], V[1], color="g", angles="xy", scale_units="xy", scale=1)

    if motif_atoms is not None:
        if extend_motif_atoms:
            num_motif_atoms = motif_atoms.shape[0]
            colors = create_segmentation_colormap_no_gray()
            if colors.shape[0] < num_motif_atoms:
                colors = cm.rainbow(np.linspace(0, 1, num_motif_atoms))
            else:
                colors = colors[:num_motif_atoms]

            atom_positions, atom_types = span_motif_to_image_domain(v, motif_atoms, image)
            plt.scatter(atom_positions[:, 0], atom_positions[:, 1], s=dot_size, marker=".", color=colors[atom_types])

            if show_motif_legend:
                import matplotlib.lines as mlines

                prox_handles = []
                for i in range(num_motif_atoms):
                    prox_handles.append(
                        mlines.Line2D([], [], color=colors[i], marker=".", linestyle="None", label=f"atom {i+1}")
                    )
                if legend_out:
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width, box.height])
                    ax.legend(handles=prox_handles, loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 24})
                else:
                    ax.legend(handles=prox_handles)
        else:
            plt.scatter(motif_atoms[:, 0] + shift[0], motif_atoms[:, 1] + shift[1], s=dot_size, marker=".")

    if show_plot:
        plt.show()
    return fig, dpi
