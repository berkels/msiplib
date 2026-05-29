"""
============
Registration
============

    Functions for the registration


    The main function for the registration is :func:`parametric_registration_mld`. All
    other functions are auxiliary functions and should usually not be called
    from outside this module.

    See ``msiplib/tests/test_registration.py`` for
    examples.


======================
 Interpolation Classes
======================

    * BilinearInterpolation(data)
    * ScipySplineInterpolationWrapper(data)
    * QuadraticSpline2DWrapper(data)

    convert a discrete image into a continuous function that can be evaluated on any set of positions.

    All interpolation classes have

    :Parameter:

    **data:** 2-D ndarray : image

    :Attributes:

    **xy:** array_like : 3-D array of discretized image coordinates

        xy.shape = ( len(y) , len(x) , 2 )

    *wrapper classes have the attribute **f_u** which calls the interpolation object from another library*

    :Methods:

    **evaluate(self, xy):** Evaluate the interpolation at given positions xy

        xy:  parameter: 3-D ndarray : grid :     xy.shape = ( len(x) , len(y) , 2 )


    **evaluate_gradient(self, xy):** Evaluate the interpolation at given positions xy

        xy:  parameter: 3-D ndarray : grid :     xy.shape = ( len(x) , len(y) , 2 )

====================
 Deformation Classes
====================

    * ParametricTranslation(b, shape)
    * ParametricRigidBodyMotion(params, shape)
    * ParametricAffineDeformation(params, shape)

    All Deformation classes have

    :Parameters:

    deformation characterizing parameter (called either '**b**' or '**params**')

    **shape:** tuple : shape of the image to be deformed

    :Methods:

    **evaluate(self, xy):** Evaluate the deformation at given positions xy

        xy: parameter: 3-D ndarray : grid :     xy.shape = ( len(x) , len(y) , 2 )

    **evaluate_gradient(self, xy*):** Evaluate the deformation gradient at given positions xy

        xy: parameter : 3-D ndarray : grid :     xy.shape = ( len(x) , len(y) , 2 )

        *- xy is omitted for ParametricTranslation gradient: evaluate_gradient(self, _)*

    **deform_image(self, image, cval=0.0):** apply scipy.ndimage.affine_transform using the given parameters on image

    **get_identity_deformation_params():** return identity parameters of the specific deformation class

    **get_translation_deformation_params(t):** return parameters of the specific deformation class given that the only deformation is translation by a vector t

        t: parameter: 1-D array: len(t) = 2

    **concatenate_deformation_params(params_one, params_two):** return the deformation parameters of a combined deformation, where params_two has occurred first, followed by params_one

        params_one: parameter : array_like : deformation parameters

        params_two: parameter : array_like : deformation parameters

"""
import os
import numpy as np
from scipy import interpolate
from skimage import img_as_float, img_as_float32
from scipy.optimize import minimize, least_squares
from scipy.ndimage import affine_transform, gaussian_filter
from skimage.transform import pyramid_gaussian
from skimage.registration import phase_cross_correlation
from numba import int32, float64, void
from numba.experimental import jitclass
from .optimization import GaussNewtonAlgorithm, GradientDescent
from .misc import rotation_matrix



spec = [
    ("grid_shape", int32[::1]),
    ("num_channels", void),
    ("data", float64[:, ::1]),
    ("xy", int32[:, :, ::1]),
]  # the last one is the fastest running


class InterpolationBase:
    def __init__(self, data):
        self.grid_shape = np.asarray(data.shape, dtype=int32)
        self.num_channels = None
        self.xy = np.zeros((data.shape[1], data.shape[0], 2), dtype=int32)
        # Numba doesn't support meshgrid.
        for y in np.arange(0, data.shape[1]):
            for x in np.arange(0, data.shape[0]):
                self.xy[y, x, 0] = x
                self.xy[y, x, 1] = y

    def get_coeffs(self, pos):
        """
        pos : an np.array of length 2, usually an element of the grid matrix (xy) of the output image

        """
        ind = np.floor(pos).astype(int32)
        c = pos - ind
        valid = True
        for k in range(2):
            if pos[k] < 0:
                valid = False
                ind[k] = 0
                c[k] = 0
            elif pos[k] > self.grid_shape[k] - 1:
                valid = False
                ind[k] = self.grid_shape[k] - 2
                c[k] = 1
            elif ind[k] == self.grid_shape[k] - 1:
                ind[k] -= 1
                c[k] = 1
        return valid, ind, c


@jitclass(spec)
class BilinearInterpolation(InterpolationBase):
    __init__base = InterpolationBase.__init__

    def __init__(self, data):
        self.__init__base(data)
        self.data = np.ascontiguousarray(data)

    def evaluate(self, pos):
        """
        pos: ndarray of the shape (x,y,2)

             grid for the output image (similar to xy )

        """
        result = np.zeros(pos.shape[:2])
        num_0 = pos.shape[0]
        num_1 = pos.shape[1]
        for i in range(num_0):
            for j in range(num_1):
                valid, ind, c = self.get_coeffs(pos[i, j, :])
                result[i, j] = (
                    self.data[ind[0], ind[1]] * (1 - c[0]) * (1 - c[1])
                    + self.data[ind[0] + 1, ind[1]] * c[0] * (1 - c[1])
                    + self.data[ind[0], ind[1] + 1] * (1 - c[0]) * c[1]
                    + self.data[ind[0] + 1, ind[1] + 1] * c[0] * c[1]
                )
        return result

    def evaluate_gradient(self, pos):
        """
        pos: ndarray of the shape (x,y,2)

             grid for the output image gradient (similar to xy )

        """
        result = np.zeros(pos.shape)
        num_0 = pos.shape[0]
        num_1 = pos.shape[1]
        for i in range(num_0):
            for j in range(num_1):
                valid, ind, c = self.get_coeffs(pos[i, j, :])
                if valid:
                    result[i, j, 0] = (self.data[ind[0] + 1, ind[1]] - self.data[ind[0], ind[1]]) * (1 - c[1]) + (
                        self.data[ind[0] + 1, ind[1] + 1] - self.data[ind[0], ind[1] + 1]
                    ) * c[1]
                    result[i, j, 1] = (self.data[ind[0], ind[1] + 1] - self.data[ind[0], ind[1]]) * (1 - c[0]) + (
                        self.data[ind[0] + 1, ind[1] + 1] - self.data[ind[0] + 1, ind[1]]
                    ) * c[0]
        return result




class ScipySplineInterpolationWrapper:
    def __init__(self, data):
        x = np.arange(0, data.shape[0])
        y = np.arange(0, data.shape[1])
        self.num_channels = data.shape[2] if data.ndim == 3 else None
        if self.num_channels:
            self.f_u = [
                interpolate.RectBivariateSpline(x, y, data[..., i], kx=3, ky=3) for i in range(self.num_channels)
            ]
        else:
            self.f_u = interpolate.RectBivariateSpline(x, y, data, kx=3, ky=3)
        self.xy = np.dstack(np.meshgrid(x, y))

    def evaluate(self, xy):
        # evaluate the continuous function f_u on a new mesh xy
        if self.num_channels:
            return np.stack([self.f_u[i].ev(xy[..., 0], xy[..., 1]) for i in range(self.num_channels)], axis=-1)
        else:
            return self.f_u.ev(xy[..., 0], xy[..., 1])

    def evaluate_gradient(self, xy):
        if self.num_channels:
            return np.stack(
                [
                    np.dstack(
                        [
                            self.f_u[i].ev(xy[..., 0], xy[..., 1], dx=1, dy=0),
                            self.f_u[i].ev(xy[..., 0], xy[..., 1], dx=0, dy=1),
                        ]
                    )
                    for i in range(self.num_channels)
                ],
                axis=-1,
            ).transpose((0, 1, 3, 2))
        else:
            return np.dstack(
                [self.f_u.ev(xy[..., 0], xy[..., 1], dx=1, dy=0), self.f_u.ev(xy[..., 0], xy[..., 1], dx=0, dy=1)]
            )




class ParametricTranslation:
    def __init__(self, b, shape):
        """
        **b:** array_like : 1-D ndarray len(b) = 2

             decimal translation deformation on x and y directions 0 < b[i] < 1

        """
        self.shape = shape
        self.b = b * (max(self.shape) - 1)

    def evaluate(self, xy):
        return np.add(xy, self.b)

    def evaluate_param_gradient(self, _):
        return (np.identity(2) * (max(self.shape) - 1))[np.newaxis, np.newaxis, ...]

    def deform_gray_image(self, image, cval, order):
        return affine_transform(image, matrix=np.identity(2), offset=self.b[::-1], cval=cval, order=order)

    def deform_image(self, image, cval=0.0, order=3):
        if image.ndim == 3:
            return np.stack(
                [self.deform_gray_image(image[..., i], cval, order) for i in range(image.shape[2])], axis=-1
            )
        else:
            return self.deform_gray_image(image, cval, order)

    @staticmethod
    def get_identity_deformation_params():
        return np.array([0.0, 0.0])

    @staticmethod
    def get_translation_deformation_params(t):
        return t

    @staticmethod
    def concatenate_deformation_params(params_one, params_two):
        return params_one + params_two


class ParametricRigidBodyMotion:
    def __init__(self, params, shape):
        """
        **params:**

            deformation parameters : 1-D ndarray of length 3

                 the first two entries are the translational deformation parameters as a percentage of the longest side of the image, and

                 the third element is the angle phi by which the image is rotated counter clock-wise with respect to the x axis about the image center.

        """
        self.shape = shape
        self.center = (np.array(shape) - 1) / 2
        self.alpha = params[2]
        self.rotation_matrix = rotation_matrix(self.alpha)
        self.rot_der_matrix = np.array(
            [[-np.sin(self.alpha), -np.cos(self.alpha)], [np.cos(self.alpha), -np.sin(self.alpha)]]
        )
        self.b = params[:2] * (max(self.shape) - 1) + self.center

    def evaluate(self, xy):
        return np.add(np.einsum("...ij,...j", self.rotation_matrix, xy - self.center), self.b)

    def evaluate_param_gradient(self, xy):
        mat = np.zeros(xy.shape[:-1] + (2, 2))
        mat[..., 0, 0] = max(self.shape) - 1
        mat[..., 1, 1] = max(self.shape) - 1
        der_rot = np.einsum("...ij,...j", self.rot_der_matrix, xy - self.center)
        return np.concatenate((mat, der_rot[:, :, np.newaxis, :]), axis=2)

    def deform_image(self, image, cval=0.0, order=3):
        return affine_transform(
            image,
            matrix=self.rotation_matrix.T,
            offset=(self.b - np.matmul(self.rotation_matrix, self.center))[::-1],
            cval=cval,
            order=order,
        )

    @staticmethod
    def get_identity_deformation_params():
        return np.array([0.0, 0.0, 0.0])

    @staticmethod
    def get_translation_deformation_params(t):
        """
        **t:** nparray with two elements representing the translation deformation in x and y in the continuous form (as a decimal)

        """
        return np.array([t[0], t[1], 0.0])

    @staticmethod
    def concatenate_deformation_params(params_one, params_two):
        """
        combines two transformations (params_one, params_two) where params_two has occurred first, followed by params_one

        **Output:**

        nparray of 3 elements:

            the first two are the continuous translation parameters (have values between 0 and 1)

            the third is the rotational transformation angle alpha about the image center

        """
        params = np.zeros((3))
        params[2] = params_one[2] + params_two[2]
        matrix_one = rotation_matrix(params_one[2])
        b_one = params_one[:2]
        b_two = params_two[:2]
        params[:2] = b_one + np.matmul(matrix_one, b_two)
        return params

    @staticmethod
    def get_inverse_deformation_params(params):
        rotation = np.linalg.inv(rotation_matrix(params[2]))
        t = np.einsum("...ij,...j", rotation, -params[:2])
        return np.array([t[0], t[1], -params[2]])


class ParametricAffineDeformation:
    def __init__(self, params, shape):
        """
        **params:**  deformation parameters : 1-D ndarray of length 6

                 The first two entries are the translational deformation parameters as a percentage of the longest side of the image, and

                    the last four represent the deformation matrix

        """
        self.shape = shape
        self.center = (np.array(shape) - 1) / 2
        self.matrix = params[2:].reshape(2, 2)
        self.b = params[:2] * (max(self.shape) - 1) + self.center

    def evaluate(self, xy):
        return np.add(np.einsum("...ij,...j", self.matrix, xy - self.center), self.b)

    def evaluate_param_gradient(self, xy):
        mat = np.zeros(xy.shape[:-1] + (6, 2))
        mat[..., 0, 0] = max(self.shape) - 1
        mat[..., 1, 1] = max(self.shape) - 1
        mat[..., 2, 0] = xy[..., 0] - self.center[0]
        mat[..., 3, 0] = xy[..., 1] - self.center[1]
        mat[..., 4, 1] = xy[..., 0] - self.center[0]
        mat[..., 5, 1] = xy[..., 1] - self.center[1]
        return mat

    def deform_image(self, image, cval=0.0, order=3):
        return affine_transform(
            image.T, matrix=self.matrix, offset=(self.b - np.matmul(self.matrix, self.center)), cval=cval, order=order
        ).T

    @staticmethod
    def get_identity_deformation_params():
        return np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    @staticmethod
    def get_translation_deformation_params(t):
        return np.array([t[0], t[1], 1.0, 0.0, 0.0, 1.0])

    @staticmethod
    def concatenate_deformation_params(params_one, params_two):
        """
        combines two transformations (params_one, params_two) where params_two has occurred first, followed by params_one

        **Output:**

        nparray of 6 elements:

                the first two are the continuous translation parameters (have values between 0 and 1)

                the last four represent the deformation matrix

        """
        params = np.zeros((6))
        matrix_one = params_one[2:].reshape(2, 2)
        matrix_two = params_two[2:].reshape(2, 2)
        params[2:] = np.matmul(matrix_one, matrix_two).reshape(-1)
        b_one = params_one[:2]
        b_two = params_two[:2]
        params[:2] = b_one + np.matmul(matrix_one, b_two)
        return params


# ======================
#  fitting
# ======================
def SSD_parametric_fit_residual(f_u, g, xy, phi):
    r"""
    return the registration residual S

    .. math::

        f ( \phi[p] (x) ) - g


    'Variational methods in image processing lecture notes SS 2020' ``"lectures-repo/VarBV/2020-SS``

    :parameters:

    **f_u:** object: continuous function of the interpolated input template image. It has  a method .evaluate.  e.g ParametricAffineDeformation


    **g:** array_like : 2-D ndarray : reference image

    **xy:** array_like : 3-D ndarray of the shape (y, x, 2) defining the positions upon which the template image will be evaluated upon the deformation.

    **phi:** object: deformation : it has a method "phi.evaluate"  e.g. scipy.interpolate.RectBivariateSpline.

    """
    deformed_xy = phi.evaluate(xy)
    return f_u.evaluate(deformed_xy) - g  # tmp


def SSD_parametric_fit_energy(f_u, g, xy, phi):
    r"""
    return the registration energy

    .. math::

        \int_{\Omega} | f ( \phi[p] (x) ) - g |^2  d x

    'Variational methods in image processing lecture notes SS 2020' ``"lectures-repo/VarBV/2020-SS``

    :parameters:

    **f_u:** object: continuous function of the interpolated input template image. It has  a method .evaluate.

    e.g ParametricAffineDeformation


    **g:** array_like : 2-D ndarray : reference image

    **xy:** array_like : 3-D ndarray of the shape (y, x, 2) defining the positions upon which the template image will be evaluated upon the deformation.

    **phi:** object: deformation : it has a method "phi.evaluate"

    e.g. scipy.interpolate.RectBivariateSpline.

    """
    return 1 / 2 * np.sum(SSD_parametric_fit_residual(f_u, g, xy, phi) ** 2) / np.prod(xy.shape[:-1])


def SSD_parametric_fit_residual_grad(f_u, g, xy, phi):
    """
    return the gradient of registration residual
    """
    deformed_xy = phi.evaluate(xy)
    phi_grad = phi.evaluate_param_gradient(xy)
    if f_u.num_channels:
        phi_grad = phi_grad[:, :, np.newaxis, ...]
    f_grad = f_u.evaluate_gradient(deformed_xy)
    return np.einsum("...ij,...j", phi_grad, f_grad)


def SSD_parametric_fit_energy_grad(f_u, g, xy, phi):
    """
    return the gradient of registration energy
    """
    deformed_xy = phi.evaluate(xy)
    tmp = f_u.evaluate(deformed_xy) - g
    phi_grad = phi.evaluate_param_gradient(xy)
    if f_u.num_channels:
        phi_grad = phi_grad[:, :, np.newaxis, ...]
        image_axes = (0, 1, 2)
    else:
        image_axes = (0, 1)
    f_grad = f_u.evaluate_gradient(deformed_xy)
    dxg = np.multiply(f_grad, tmp[..., np.newaxis])
    return np.sum(np.einsum("...ij,...j", phi_grad, dxg), axis=image_axes) / np.prod(xy.shape[:-1])


def parametric_registration_mld(
    template_image,
    reference_image,
    def_type,
    num_levels,
    stop_eps=0.0,
    early_stop_level=0,
    minimizer="trf",
    return_spline_and_grid=True,
    use_correlation_to_init_translation=False,
    interpolation="ScipySpline",
    v0=None,
    max_iter=100,
    pre_smooth_sigma=0,
    phase_cross_correlation_channel=None,
    phase_cross_correlation_overlap_ratio=None,
):
    """
    returns phi (deformation parameters) of the registration functional using multilevel descend

    :Parameters:

    **template_image**, **reference_image:** 2-D ndarrays, both must have the same shape and dtype

    **def_type:** class  Possible values ParametricTranslation, ParametricRigidBodyMotion, ParametricAffinedeformation.

              The deformation class has to be imported before calling parametric_registration_mld

    **num_levels:** int   (number of levels -1) is the "max_layer" value for of the skimage.transform.pyramid_gaussian function

     **stop_eps:** float : default value 0.,

    **early_stop_level** int : default value =0,

    **minimizer:** str     possible values: ``'GaussNewton'``, ``'L-BFGS-B'``, default value: ``'trf'``

    **return_spline_and_grid** boolean : default value =True,

    **use_correlation_to_init_translation** boolean : default value =False,

    **interpolation:** str     possible values: ``'ScipySpline'``, ``'bilinear'``, default value: ``'ScipySpline'``

    **v0:** translation deformation, default value None

    """
    if template_image.shape != reference_image.shape:
        raise ValueError(
            "Images to be registered must have the same size, but {} and {} differ.".format(
                template_image.shape, reference_image.shape
            )
        )

    if template_image.dtype != reference_image.dtype:
        raise ValueError(
            "Images to be registered are assumed to have the same dtype, but {} and {} differ.".format(
                template_image.dtype, reference_image.dtype
            )
        )

    if pre_smooth_sigma > 0:
        print("Pre-smoothing images ... ", end="", flush=True)
        template_image = gaussian_filter(template_image, sigma=pre_smooth_sigma)
        reference_image = gaussian_filter(reference_image, sigma=pre_smooth_sigma)
        print("done")

    print("Creating image hierarchy ... ", end="", flush=True)
    channel_axis = 2 if template_image.ndim == 3 else None
    pyramid_tem = tuple(
        pyramid_gaussian(
            img_as_float32(template_image) if template_image.dtype == np.uint8 else img_as_float(template_image),
            max_layer=num_levels - 1,
            downscale=2,
            channel_axis=channel_axis,
        )
    )
    pyramid_ref = tuple(
        pyramid_gaussian(
            img_as_float32(reference_image) if reference_image.dtype == np.uint8 else img_as_float(reference_image),
            max_layer=num_levels - 1,
            downscale=2,
            channel_axis=channel_axis,
        )
    )
    print("done")

    if interpolation == "ScipySpline":
        interpolation_type = ScipySplineInterpolationWrapper
    elif interpolation == "bilinear":
        interpolation_type = BilinearInterpolation
    else:
        raise NotImplementedError("Unknown interpolation")

    if v0 is not None:
        v = v0.copy()
    elif use_correlation_to_init_translation:
        # Estimate translation as pixel shift using the coarsest data.
        # phase_cross_correlation only supports 2D arrays.
        if channel_axis:
            if phase_cross_correlation_channel is not None:
                tem = pyramid_tem[num_levels - 1][..., phase_cross_correlation_channel]
                ref = pyramid_ref[num_levels - 1][..., phase_cross_correlation_channel]
            else:
                tem = pyramid_tem[num_levels - 1].sum(axis=channel_axis)
                ref = pyramid_ref[num_levels - 1].sum(axis=channel_axis)
        else:
            tem = pyramid_tem[num_levels - 1]
            ref = pyramid_ref[num_levels - 1]
        overlap_ratio_setting = (
            {"overlap_ratio": phase_cross_correlation_overlap_ratio, "reference_mask": np.ones_like(tem)}
            if phase_cross_correlation_overlap_ratio is not None
            else {}
        )
        shifts = phase_cross_correlation(tem, ref, **(overlap_ratio_setting))[0][::-1]
        # Scale the estimated shift by the pixel width and use this to get the initial v.
        v = def_type.get_translation_deformation_params(shifts / (np.amax(pyramid_tem[num_levels - 1].shape) - 1))
        print("Estimated initial translation:", v)
    else:
        v = def_type.get_identity_deformation_params()

    for i in reversed(range(early_stop_level, num_levels)):
        if max_iter == 0:
            continue

        image_tem = pyramid_tem[i]
        image_ref = pyramid_ref[i]

        grid_shape = image_tem.shape[:2]

        print("\n" + "-" * 56)
        print(f" Registration on resolution {grid_shape} started ".center(56))
        print("-" * 56 + "\n")

        f_u = interpolation_type(image_tem.swapaxes(0, 1))

        xy = f_u.xy
        scale = 1 / np.sqrt(image_tem.size)

        if minimizer == "GaussNewton":
            v = GaussNewtonAlgorithm(
                v,
                F=lambda v: scale
                * SSD_parametric_fit_residual(f_u, image_ref, xy, def_type(v, grid_shape)).reshape(-1),
                DF=lambda v: scale
                * SSD_parametric_fit_residual_grad(f_u, image_ref, xy, def_type(v, grid_shape)).reshape(-1, v.shape[0]),
                stopEpsilon=stop_eps,
                maxIter=max_iter,
            )
        elif minimizer == "trf":
            res = least_squares(
                lambda v: scale * SSD_parametric_fit_residual(f_u, image_ref, xy, def_type(v, grid_shape)).reshape(-1),
                v,
                jac=lambda v: scale
                * SSD_parametric_fit_residual_grad(f_u, image_ref, xy, def_type(v, grid_shape)).reshape(-1, v.shape[0]),
                method="trf",
                verbose=2,
                ftol=stop_eps,
                xtol=stop_eps,
                max_nfev=max_iter * len(v),
            )
            print(res)
            v = res.x
        elif minimizer == "L-BFGS-B":
            res = minimize(
                lambda v: SSD_parametric_fit_energy(f_u, image_ref, xy, def_type(v, grid_shape)),
                v,
                jac=lambda v: SSD_parametric_fit_energy_grad(f_u, image_ref, xy, def_type(v, grid_shape)),
                method="L-BFGS-B",
                options={"disp": True, "maxiter": max_iter},
            )
            print(res)
            v = res.x
        elif minimizer == "GradientDescent":
            v = GradientDescent(
                v,
                E=lambda v: SSD_parametric_fit_energy(f_u, image_ref, xy, def_type(v, grid_shape)),
                DE=lambda v: SSD_parametric_fit_energy_grad(f_u, image_ref, xy, def_type(v, grid_shape)),
                stopEpsilon=stop_eps,
                maxIter=max_iter,
                NonlinearCG=True,
            )
        else:
            raise NotImplementedError("Unknown minimizer")

    if not return_spline_and_grid:
        return v
    else:
        if early_stop_level > 0:
            f_u = interpolation_type(pyramid_tem[0].T)
            xy = f_u.xy

        return f_u, xy, v


def create_template_filename_list(params):
    if "templates" in params:
        directory = os.path.expandvars(params["templatesDirectory"])
        return [directory + os.path.expandvars(element) for element in params["templates"]]

    template_filenames = []
    template_num_offset = params["templateNumOffset"]
    template_num_step = params["templateNumStep"]
    num_templates = params["numTemplates"]
    template_name_pattern = os.path.expanduser(os.path.expandvars(params["templateNamePattern"]))

    for i in range(template_num_offset, template_num_offset + num_templates * template_num_step, template_num_step):
        template_filenames.append(template_name_pattern % i)

    return template_filenames
