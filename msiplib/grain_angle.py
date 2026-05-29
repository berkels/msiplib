"""
Author: Amel Alhassan

This module is for the calculations related to two grains of the same type.

The main function is :py:meth:`get_angle_between_similar_grains <get_angle_between_similar_grains>` and can be called from outside the module.

    
Example of using the module
---------------------------

    >>> #run this example in msiplib/examples directory
    >>> from msiplib.grain_angle import get_angle_between_similar_grains
    >>> from msiplib.io import read_image
    >>>
    >>> img_name = 'experimental_data'
    >>> path = './'
    >>> left_segment = read_image('./images/left_grain.png')
    >>> right_segment = read_image('./images/right_grain.png')
    >>> angle = get_angle_between_similar_grains(left_segment,right_segment,path,img_name, accuracy=2, read_vectors= False)

References:
------------
    :cite:`AlBe23` A. S. A.Alhassan, S. Zhang, B. Berkels, Direct motif extraction from high resolution crystalline STEM images, Ultramicroscopy, https://doi.org/10.1016/j.ultramic.2023.113827

"""

from msiplib.unit_cell_from_real_space import get_primitive_unit_cell_vectors
import imageio, sys
from msiplib.registration import parametric_registration_mld as PR
from msiplib.registration import ParametricAffineDeformation, ParametricRigidBodyMotion
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from msiplib.io import read_image
from msiplib.motif import get_motif

"""
**Important note**

if the file name contain mathematical symbols such as "_", the user must change it in the output tex file [by hand].
"""


def angle_between_vectors(a, b):
    r"""
    Computes the directional angle between two 2D vectors ``a`` and ``b``.

    The angle is calculated using the inverse sine (arcsin) of the cross product
    of the vectors' components, normalized by the product of their magnitudes (Euclidean norms),
    as shown in the following equation (21) :cite:`AlBe23`:

    .. math::

        \angle(v_i^a, v_i^b) = \sin^{-1} \left( \frac{v_{1,2}^a v_{1,1}^b - v_{1,1}^a v_{1,2}^b}{\|v_i^a\|_2 \|v_i^b\|_2} \right), \quad i \in \{1, 2\}.


    parameters:

        a,b : array-like : 2D vectors in :math:`\mathbb{R}^2`.

    returns:

        The directional angle between vectors ``a`` and ``b`` in degree.
    """
    norms = np.linalg.norm((a, b), axis=1)
    return np.arcsin((a[1] * b[0] - a[0] * b[1]) / (norms[0] * norms[1]))  # equation 21 - AlBe23


def deform_img(v, im, dim, f_u, xy, path, name, index):
    """
    Applies a rigid body transformation based on the primitive unit cell vectors ``v`` to a given image ``im`` and saves the resulting deformed image.

    parameters:

        v : *2x2 ndarray* : A 2x2 matrix representing the two primitive unit cell vectors for the rigid body transformation.

        im: *2D ndarray* : The input lattice image to be deformed.

        dim: *int* : A parameter used for cropping the image to hide deformations at the borders. (Currently unused, but can be uncommented when needed).

        f_u : *object* : A continuous function of the interpolated input template image. This object must have an ``.evaluate`` method to apply the deformation.

        xy : *ndarray* : The interpolation grid used for evaluating the deformation.

        path : *str* : The directory where the output deformed image will be saved.

        name : *str, optional, default=``None``* : Identifier of the input image (e.g., the crystal type).

        index : *int* : Identifier of the grain or image index being considered.

    returns:

        This function saves the deformed image to the specified path with a name based on the grain index, angle, and input image identifier.
    """
    deformation = ParametricRigidBodyMotion(v, im.shape)
    im_shifted = f_u.evaluate(deformation.evaluate(xy))
    angle = str(int(np.degrees(v[2])))
    plt.imsave(
        path + "grain" + str(index) + "_" + angle + "_" + name + "_deformed.png",
        im_shifted,
        cmap="gray",
    )


def find_displacement(im1, im2, n, v, save_path, tex_file=None, make_plot=False, name=None):
    r"""
    Registers two images by finding the displacement (translation and rotation) from the first image to the second.

    This function calculates the transformation (translation and rotation) required to align ``im1`` with ``im2``.
    It performs a parametric rigid body motion registration, computes both forward and inverse transformations, and checks for consistency.
    If the registration is consistent, the translation vector and rotation angle are returned. Optionally, the function can save the deformed images,
    write the transformation details to a LaTeX file, and generate plots.

    parameters:

        im1 : *2D ndarray* : The input images to be registered. ``im1`` will be transformed to align with ``im2``.
                                Both  must be square and of the same size to ``im2``.
        im2 : *2D ndarray* : The input images to be registered. Both  must be square and of the same size.

        n :  *int* : The size of the grid used to compute the transformation. It can also be used for cropping the images to hide border deformations
                    (currently unused)

        v : *array-like of shape (2,)* : A 2D translation vector in :math:`\mathbb{R}^2` representing the initial guess for the translation
                        from ``im1`` to ``im2``.

        save_path: *str* : The directory where output files (such as images or LaTeX files) will be saved.

        tex_file : *file-like object or None, optional* : A writable file object (e.g., a LaTeX file) where the function will log the transformation
            details. If ``None``, no logging will occur.

        make_plot : *bool, optional, default=False* : whether to save the input and deformed images.

        name : *str, optional, default=None* : A string identifier for the input images (e.g., crystal type). This will be included in the
            output filenames.

    returns:
        A *2D ndarray* representing the translation vector in pixels that aligns ``im1`` with ``im2`` and a float representing
        the rotation angle in degrees between the two images.

    """
    if im1.shape != im2.shape or im1.shape[0] != im1.shape[1]:
        raise ValueError("images must be squares of the same shape")
    dim = int(n / 4)
    f_u1, xy1, v1 = PR(
        im1,
        im2,
        def_type=ParametricRigidBodyMotion,
        num_levels=4,
        minimizer="GaussNewton",
        v0=v,
    )
    v2 = ParametricRigidBodyMotion.get_inverse_deformation_params(v1)
    f_u2, xy2, v2 = PR(
        im2,
        im1,
        def_type=ParametricRigidBodyMotion,
        num_levels=4,
        minimizer="GaussNewton",
        v0=v2,
    )
    if np.allclose(
        np.degrees(v1[2]), -1 * np.degrees(v2[2]), atol=1
    ):  # if the angle from one side to the other is the same with reflected sign when the direction differs
        if make_plot:
            plt.imsave(save_path + "grain1" + name + ".png", im1, cmap="gray")  # [dim:-dim,dim:-dim]
            plt.imsave(save_path + "grain2" + name + ".png", im2, cmap="gray")  # [dim:-dim,dim:-dim]
            deform_img(v1, im1, dim, f_u1, xy1, save_path, name, 1)
            deform_img(v2, im2, dim, f_u2, xy2, save_path, name, 2)
        if tex_file is not None:
            tex_file.write("\n\nforward deformation parameters: ")
            tex_file.write(str(v1))
            tex_file.write("\n\n    shift: ")
            tex_file.write(str(v1[:2] * np.array([n, n])))
            tex_file.write(" (pxl)")
            tex_file.write(",\n    angle: ")
            tex_file.write(np.degrees(v1[2]).astype(str))
            tex_file.write(" (deg)\n%\n")
            tex_file.write("\\\\inverse deformation parameters: ")
            tex_file.write(str(v2))
            tex_file.write("\n\n    shift: ")
            tex_file.write(str(v2[:2] * np.array([n, n])))
            tex_file.write(" (pxl)")
            tex_file.write(",\n    angle: ")
            tex_file.write(np.degrees(v2[2]).astype(str))
            tex_file.write(" (deg)\n%\n")
            tex_file.write("\\\\comparison between the two deformations: ")
            tex_file.write(str(v1) + " \n")
            tex_file.write(str(ParametricRigidBodyMotion.get_inverse_deformation_params(v1)))
            tex_file.write("\n%------------------------------------------------------%\n")
        return v1[:2] * np.array([n, n]), np.degrees(v1[2])
    else:
        tex_file.write("registration is not consistent!")




def get_angle_between_similar_grains(
    left_segment,
    right_segment,
    path,
    img_name,
    read_vectors=True,
    accuracy=2,
    motif_vec=False,
):
    """
    Computes the rigid-body transformation parameters between two neighboring grains, including the rotational angle and translation vector.

    This function calculates the transformation between two neighboring grains of the same crystal structure, including the primitive unit cell vectors,
    rotational angle, and translation vector. Optionally, it can read the vectors from existing files or calculate them using more accurate motif-based
    methods. It also generates a LaTeX document summarizing the calculations and transformation parameters.

    parameters:

        left_segment :*2D ndarrays* : Left images of the neighboring grains of the same crystal.

        right_segment : *2D ndarrays* : Right images of the neighboring grains of the same crystal.

        path : *str* : Directory where output files (e.g., vectors, LaTeX files) will be saved.

        img-name : *str* : Identifier for the input images, typically the name of the crystal type. It is used to name the output files.

        read_vectors : *bool, optional, default=False* : If ``True``, the function reads the primitive unit cell vectors from existing ``.npz`` files in the output directory.
            The file names must follow the pattern ``{img_name}_left_vectors.npz`` and ``{img_name}_right_vectors.npz``.
            If ``False``, the vectors are calculated using :py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`.

        accuracy : *int, optional, default=2* : Specifies the accuracy at which the unit cell vector direction (angle) is calculated. Possible values are 1, 2, or 4.

        motif_vec : *bool, optional, default=False* : If ``True``, more accurate primitive unit cell vectors are calculated using `msiplib.motif.get_motif`.

    returns:
        The average rotational angle, ``mean_theta`` (*float*), (in degrees) between the two grains.

    note:
        - The function crops both input images to the largest possible square to ensure equal dimensions before processing.
        - If ``motif_vec`` is enabled, the function recalculates the primitive unit cell vectors with higher accuracy using motif-based methods.
        - The LaTeX document generated includes all calculated vectors, angles, and transformation details.
        
    Other Outputs

        A LaTeX file:
            The function generates a LaTeX file that includes a detailed description of the unit cell calculations and
            the rigid-body transformation parameters (angle and translation vector) between the two images.

        Files:
            - Vectors: If ``read_vectors`` is ``False``, the calculated vectors are saved as ``.npz`` files in the output directory.
            - Deformed images: If transformations are calculated, the deformed images are saved as output files.
    """

    left = left_segment
    right = right_segment
    dim = min(min(left.shape), min(right.shape))
    image = [left[:dim, :dim], right[:dim, :dim]]  # crop the images into equal squares
    names = [img_name + "_left", img_name + "_right"]

    # calculate the vectors when needed
    if read_vectors:
        vecs_left = np.load(path + img_name + "_left_vectors.npz")["arr_0"]
        vecs_right = np.load(path + img_name + "_right_vectors.npz")["arr_0"]
    else:
        vecs_left = get_primitive_unit_cell_vectors(
            image[0],
            path,
            names[0],
            write_to_file=False,
            accuracy=accuracy,
            save_vectors=True,
            plot=True,
        )
        vecs_right = get_primitive_unit_cell_vectors(
            image[1],
            path,
            names[1],
            write_to_file=False,
            accuracy=accuracy,
            save_vectors=True,
            plot=True,
        )
    # the more accurate vectors from the motif extraction code can be calculated
    if motif_vec:
        _, _, vecs_left = get_motif(left, names[0], path=path, read_vectors=read_vectors)
        _, _, vecs_right = get_motif(right, names[1], path=path, read_vectors=read_vectors)
        img_name = img_name + "_motif"

    tex_file = open(path + img_name + "_rotation_n_displacement.tex", "w")
    tex_file.write("\\documentclass[12pt]{article}\n")
    tex_file.write("\\usepackage{multicol}\n")
    tex_file.write("\\usepackage{amsmath}")
    tex_file.write("\n\\begin{document}\n\n")

    tex_file.write("\n right segment shape")
    tex_file.write(str(right.shape))
    tex_file.write("\n left segment shape")
    tex_file.write(str(left.shape) + "\n")

    tex_file.write("\n possible angles are: \n\n")
    theta = []  # find and list all possible rotation angles between the two grains

    for i in range(len(vecs_left)):
        ang = angle_between_vectors(vecs_left[i], vecs_right[i])
        tex_file.write(np.degrees(ang).astype(str))
        tex_file.write("\n")
        theta.append(ang)  # save angles between equivalent vectors

    tex_file.write("\n\n\n\n\\section{Interface}\n\n\n")
    theta = np.sort(theta)
    for angle in theta:
        tex_file.write("\n\n\n\\subsection{THE ANGLE:")
        tex_file.write(np.degrees(angle).astype(str))
        tex_file.write("}\n\n")
        v = np.array([0.0, 0.0, angle])
        find_displacement(
            image[0],
            image[1],
            dim,
            v,
            path,
            tex_file=tex_file,
            make_plot=True,
            name=img_name,
        )
    mean_theta = np.degrees(np.mean(theta))
    print("\nThe angle between the two grains is: ", mean_theta)
    tex_file.write("\n\n\n\\subsection{THE MEAN ANGLE:}")
    tex_file.write("\n\nThe mean calculated angle between the two grains is: ")
    tex_file.write(mean_theta.astype(str))
    tex_file.write("\n\n\\end{document}")

    tex_file.close()
    return mean_theta
