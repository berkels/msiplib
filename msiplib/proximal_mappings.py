"""
    Implementations of proximal mappings
"""
import numpy as np


def project_canonical_simplex(u):
    """Python adaptation of http://ttic.uchicago.edu/~wwang5/papers/SimplexProj.m"""
    u_shape = u.shape
    N = np.prod(np.array(u.shape[:-1]))
    K = u_shape[-1]
    y = u.reshape((N.item(), K))
    x = -np.sort(-y, axis=1)
    xtmp = np.multiply(np.cumsum(x, axis=1) - 1, (1 / (np.arange(1, K + 1, like=x))))
    np.maximum(0, np.subtract(y, xtmp[np.arange(N), np.sum(x > xtmp, axis=1) - 1][:, np.newaxis]), out=x)
    return x.reshape(u_shape)


def project_unit_ball2D(p):
    p /= np.maximum(np.hypot(p[0, ...], p[1, ...]), 1)

    return p


def project_unit_ball3D(p):
    p /= np.maximum(np.sqrt(p[0, ...] ** 2 + p[1, ...] ** 2 + p[2, ...] ** 2), 1)

    return p


def proxL2Data(u, tau, f):
    r"""Implementation of the proximal map corresponding to
    :math:`G[u] = \int_\Omega (u-f)^2\mathrm{d}x`

    Args:
        u (array): position
        tau (float): step size
        f (array): input data, e.g. image to denoise
    """
    tmp = 1 / (1 + tau)
    return (u + tau * f) * tmp


class ProxMapBinaryUCSegmentation(object):
    r"""Implementation of the proximal map corresponding to
    :math:`G[u] = \int_\Omega u^2f_1+(1-u^2)f_2\mathrm{d}x`
    """

    def __init__(self, f):
        self.shift = 0.5
        self.indicator1 = f[..., 0] + self.shift
        self.indicator1Plus2 = self.indicator1 + f[..., 1] + self.shift

    def eval(self, u, t):
        return np.divide(u + 2 * t * self.indicator1, (1 + 2 * t * self.indicator1Plus2))

    def gamma(self):
        # The uniform convexity constant of the objective should be 2,
        # but smaller values seem to work better in practice.
        return 0.7


