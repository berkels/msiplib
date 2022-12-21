'''
    Implementations of proximal mappings
'''
import numpy as np


def project_canonical_simplex(u):
    '''Python adaptation of http://ttic.uchicago.edu/~wwang5/papers/SimplexProj.m'''
    u_shape = u.shape
    N = np.prod(np.array(u.shape[:-1]))
    K = u_shape[-1]
    y = u.reshape((N.item(), K))
    x = -np.sort(-y, axis=1)
    xtmp = np.multiply(np.cumsum(x, axis=1)-1, (1/(np.arange(1, K+1, like=x))))
    np.maximum(0, np.subtract(y, xtmp[np.arange(N), np.sum(x > xtmp, axis=1)-1][:, np.newaxis]), out=x)
    return x.reshape(u_shape)


def project_unit_ball2D(p):
    p /= np.maximum(np.hypot(p[0, ...], p[1, ...]), 1)

    return p


def project_unit_ball3D(p):
    p /= np.maximum(np.sqrt(p[0, ...]**2 + p[1, ...]**2 + p[2, ...]**2), 1)

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

