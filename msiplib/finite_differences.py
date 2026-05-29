"""
Operations on images using finite differences.
"""
import numpy as np
import scipy.sparse as sparse
import sys


def gradient_FD_1D(u, h, axis=0):
    """
    Approximates gradient of 2D function in only one dimension using finite differences.

    Args:
        u: the function (two or three dimensional)
        h: step size for finite differences scheme
        axis: specifies the direction in which the finite differences are applied
    """
    grad = np.zeros(u.shape)
    if u.ndim == 2:
        if axis == 0:
            grad[:-1, :] = (u[1:, :] - u[:-1, :]) / h
        else:
            grad[:, :-1] = (u[:, 1:] - u[:, :-1]) / h
    else:
        if axis == 0:
            grad[:-1, :, :] = (u[1:, :, :] - u[:-1, :, :]) / h
        else:
            grad[:, :-1, :] = (u[:, 1:, :] - u[:, :-1, :]) / h

    return grad


def gradient_FD2D(u):
    grad = np.zeros_like(u, shape=(2,) + u.shape)
    if u.ndim == 2:
        grad[0, 0:-1, :] = u[1:, :] - u[0:-1, :]
        grad[1, :, 0:-1] = u[:, 1:] - u[:, 0:-1]
    else:
        grad[0, 0:-1, :, :] = u[1:, :, :] - u[0:-1, :, :]
        grad[1, :, 0:-1, :] = u[:, 1:, :] - u[:, 0:-1, :]

    return grad


def gradient_FD3D(u, zDist=1):
    grad = np.zeros_like(u, shape=(3,) + u.shape)
    if u.ndim == 3:
        grad[0, 0:-1, :, :] = (u[1:, :, :] - u[0:-1, :, :]) / zDist
        grad[1, :, 0:-1, :] = u[:, 1:, :] - u[:, 0:-1, :]
        grad[2, :, :, 0:-1] = u[:, :, 1:] - u[:, :, 0:-1]
    else:
        grad[0, 0:-1, :, :, :] = (u[1:, :, :, :] - u[0:-1, :, :, :]) / zDist
        grad[1, :, 0:-1, :, :] = u[:, 1:, :, :] - u[:, 0:-1, :, :]
        grad[2, :, :, 0:-1, :] = u[:, :, 1:, :] - u[:, :, 0:-1, :]

    return grad


def divergence_FD2D(p):
    div = np.zeros_like(p, shape=p.shape[1:])
    if p.ndim == 3:
        div[0:-1, :] = p[0, 0:-1, :]
        div[1:, :] -= p[0, 0:-1, :]
        div[:, 0:-1] += p[1, :, 0:-1]
        div[:, 1:] -= p[1, :, 0:-1]
    else:
        div[0:-1, :, :] = p[0, 0:-1, :, :]
        div[1:, :, :] -= p[0, 0:-1, :, :]
        div[:, 0:-1, :] += p[1, :, 0:-1, :]
        div[:, 1:, :] -= p[1, :, 0:-1, :]

    return div


def divergence_FD3D(p, zDist=1):
    div = np.zeros_like(p, shape=p.shape[1:])
    if p.ndim == 4:
        div[0:-1, :, :] = p[0, 0:-1, :, :] / zDist
        div[1:, :, :] -= p[0, 0:-1, :, :] / zDist
        div[:, 0:-1, :] += p[1, :, 0:-1, :]
        div[:, 1:, :] -= p[1, :, 0:-1, :]
        div[:, :, 0:-1] += p[2, :, :, 0:-1]
        div[:, :, 1:] -= p[2, :, :, 0:-1]
    else:
        div[0:-1, :, :, :] = p[0, 0:-1, :, :, :] / zDist
        div[1:, :, :, :] -= p[0, 0:-1, :, :, :] / zDist
        div[:, 0:-1, :, :] += p[1, :, 0:-1, :, :]
        div[:, 1:, :, :] -= p[1, :, 0:-1, :, :]
        div[:, :, 0:-1, :] += p[2, :, :, 0:-1, :]
        div[:, :, 1:, :] -= p[2, :, :, 0:-1, :]

    return div


def xDerivative(num_rows, num_cols, h):
    """
    Computes matrix representation of the x-derivative.
    """
    d = sparse.csr_matrix(
        sparse.spdiags(np.array([[-1, 1]]).T * np.ones((1, num_cols)), np.array([0, 1]), num_cols, num_cols) / h
    )
    d[num_cols - 1, num_cols - 1] = 0
    dx = sparse.csr_matrix(sparse.kron(sparse.eye(num_rows), d))
    return dx


def yDerivative(num_rows, num_cols, h):
    """
    Computes matrix representation of the y-derivative.
    """
    d = sparse.csr_matrix(
        sparse.spdiags(np.array([[-1, 1]]).T * np.ones((1, num_rows)), np.array([0, 1]), num_rows, num_rows) / h
    )
    d[num_rows - 1, num_rows - 1] = 0
    dy = sparse.csr_matrix(sparse.kron(d, sparse.eye(num_cols)))
    return dy


if "jax" in sys.modules:
    import jax.numpy as jnp

    def jax_gradient_FD2D(u):
        grad = jnp.zeros((2,) + u.shape, dtype=u.dtype)
        grad = grad.at[0, 0:-1, :].set(u[1:, :] - u[0:-1, :])
        grad = grad.at[1, :, 0:-1].set(u[:, 1:] - u[:, 0:-1])
        return grad

    def jax_divergence_FD2D(p):
        div = jnp.zeros(p.shape[1:], dtype=p.dtype)
        div = div.at[0:-1, :].set(p[0, 0:-1, :])
        div = div.at[1:, :].add(-p[0, 0:-1, :])
        div = div.at[:, 0:-1].add(p[1, :, 0:-1])
        div = div.at[:, 1:].add(-p[1, :, 0:-1])
        return div
