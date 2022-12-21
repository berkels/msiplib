import numpy as np
import unittest
from msiplib.optimization import GradientDescent


def f(x, params):
    return params[2] * np.exp(-np.square(x - params[0]) / params[1])


def grad_f(x, params):
    grad = np.zeros((x.shape[0], 3))
    tmp = f(x, params)
    grad[..., 0] = tmp * 2 * (x - params[0]) / params[1]
    grad[..., 1] = tmp * np.square(x - params[0]) / (params[1]**2)
    grad[..., 2] = tmp / params[2]
    return grad


def hess_f(x, params):
    hess = np.zeros((x.shape[0], 3, 3))
    tmp = f(x, params)
    hess[:, 0, 0] = tmp * 2/params[1]**2 * \
        (2*x**2 - 4*x*params[0] + 2*params[0]**2 - params[1])
    hess[:, 0, 1] = tmp * 2/params[1]**3 * \
        (x-params[0]) * (x**2 - 2*x*params[0] + params[0]**2 - params[1])
    hess[:, 0, 2] = tmp * 2/params[1] * (x-params[0])/params[2]
    hess[:, 1, 0] = hess[:, 0, 1]
    hess[:, 1, 1] = tmp * (x-params[0])**2 * (x**2 - 2 *
                                              x*params[0] + params[0]**2 - 2*params[1])/params[1]**4
    hess[:, 1, 2] = tmp * (x-params[0])**2/params[1]**2 / params[2]
    hess[:, 2, 0] = hess[:, 0, 2]
    hess[:, 2, 1] = hess[:, 1, 2]
    # hess[:,2,2] = 0
    return hess


def F_fit(nodes, values, params):
    return f(nodes, params) - values


def E_fit(nodes, values, params):
    return np.sum(1/2*F_fit(nodes, values, params)**2)/np.prod(nodes.size)


def F_fit_grad(nodes, values, params):
    return grad_f(nodes, params)/np.prod(nodes.size)


def E_fit_grad(nodes, values, params):
    return np.einsum('i,ij', F_fit(nodes, values, params), F_fit_grad(nodes, values, params))


class TestOptimizationMethods(unittest.TestCase):

    def test_gradient_descent(self):
        params_true = np.array([1.028970500, 1.839322574, 5.046554164])
        sampling_nodes = np.linspace(-2, 3, 51)
        sampling_values = f(sampling_nodes, params_true)
        params_initial = np.array([1.0, 2.0, 5.0])
        params_GD = GradientDescent(params_initial, lambda params: E_fit(sampling_nodes, sampling_values, params),
                                    lambda params: E_fit_grad(sampling_nodes, sampling_values, params), maxIter=1000)
        self.assertTrue(np.linalg.norm(params_GD - params_true) <= 1e-10)


if __name__ == '__main__':
    unittest.main()
