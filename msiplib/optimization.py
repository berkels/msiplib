'''
    Implementation of various optimization algorithms
'''
import math
import numpy as np
import logging
from tqdm import tqdm
from scipy.linalg.interpolative import estimate_spectral_norm
from scipy.sparse.linalg import LinearOperator


def pd_hybrid_grad_alg(u0, proxG, proxHStar, A, MinusAStar, lambda_, maxNumIter,
                       stopEps, verbose=True, p=None, L=8, PDHGAlgType=1,
                       gamma=None):
    r"""Implementation of the primal-dual algorithm by Chambolle and Pock :cite:`ChPo11` to find
       :math:`(u',p')` such that

       .. math::
         A(u') \cdot p' + G(u') - H^*(p') = \inf_u \sup_p (A(u) \cdot p + G(u) - H^*(p))
    """

    if not np.issubdtype(u0.dtype, np.floating):
        raise ValueError(
            "Input u0 is expected to be floating point, but is {}.".format(u0.dtype))

    u = u0.copy()

    if L is None:
        L = estimate_spectral_norm_of_A(A, MinusAStar, u.shape)
        print(f"Estimated L={L}")

    # Default values that ensure convergence of the algorithm
    tau = 1 / L
    sigma = tau
    theta = 1.0

    # Use zero as the initial guess for p unless an initial guess was explicitly
    # specified
    if p is None:
        p = np.zeros_like(u0, shape=(2, ) + u0.shape)
    elif u0.dtype != p.dtype:
        raise ValueError(
            "If supplied, p is assumed to have the same dtype as u0, but {} and {} differ.".format(p.dtype, u0.dtype))

    u_new = u.copy()
    u_bar = u.copy()
    if verbose:
        pbar = tqdm(range(maxNumIter))
    else:
        pbar = range(maxNumIter)
    iterations = maxNumIter
    for i in pbar:
        # proximal operator of H^*
        p[...] = proxHStar(p + sigma * A(u_bar))

        # proximal operator of G
        u_new[...] = proxG(u + tau * MinusAStar(p), tau / lambda_)

        if PDHGAlgType == 2:
            theta = 1 / (math.sqrt(1 + 2 * gamma / lambda_ * tau))
            tau = theta * tau
            sigma = sigma / theta

        # acceleration step
        np.add(u_new, theta * (u_new - u), out=u_bar)

        # stopping criterion
        threshold = np.linalg.norm((u_new - u).ravel(), np.inf)

        # u = u_new
        np.copyto(u, u_new)

        # Display the norm of the updates to u every 10 iterations
        if verbose and np.mod(i+1, 10) == 0:
            pbar.set_postfix_str("update={0:1.3e}".format(threshold))

        # Check stopping criterion
        if (threshold < stopEps):
            iterations = i
            if verbose:
                pbar.close()
                print("stopping after {} iterations with update={}".format(
                    i, threshold))
            break

    return u, p, iterations


def estimate_spectral_norm_of_A(A, MinusAStar, grid_shape):
    num_rows = np.prod((len(grid_shape),)+grid_shape)
    num_cols = np.prod(grid_shape)

    def matvec(x):
        return A(x.reshape(grid_shape)).ravel()

    def matmat(X):
        raise NotImplementedError("not implemented")

    def rmatvec(x):
        return -MinusAStar(x.reshape((len(grid_shape),)+grid_shape)).ravel()

    A_opertor = LinearOperator(
        (num_rows, num_cols), matvec=matvec, matmat=matmat, rmatvec=rmatvec)

    return estimate_spectral_norm(A_opertor, its=100)


def validateDerivative(E, DE, position, N=100, h=0.0001, dir=None):
    """
    Calculates the Gateaux derivative (a generalization of the concept of the
    directional derivative) using the gradient of a given function.
    Then plots the Gateaux derivative and the original Energy.

    The gradient is valid if the Gateaux derivative is the tangent of the Energy function.


    :param E: The given Energy functional that calculates the energy depending on
              the position

    :param DE: The given derivative of the Energy functional that calculates the
               gradient depending on the position

    :param ndarray position: Position on which the tangent will be evaluated

    :param N: Number of points on which the Energy and derivative will be evaluated.

    :param h: Stepsize between the N coordinates of the Energy and Gateaux plots

    :param dir: The direction of the Energy plot. If None is given, then the classical
                -gradient will be taken.

    :returns: None, but plots the Gateaux derivative and Energy using matplotlib
    """

    initialEnergy = E(position)
    gradient = DE(position)
    if (dir is not None):
        direction = dir
    else:
        direction = -gradient
    coords = np.linspace(0, h*(N-1), num=N)
    Energy = np.zeros((N))
    Gateaux = np.zeros((N))
    gateauxDerivative = np.dot(gradient.ravel(), direction.ravel())
    for i in range(N):
        Energy[i] = E(position + coords[i]*direction)
        Gateaux[i] = initialEnergy + coords[i]*gateauxDerivative
    import matplotlib.pyplot as plt
    plt.plot(coords, Energy, label=r'$E[x+td]$')
    plt.plot(coords, Gateaux, label=r'$E[x]+tDE[x]\cdot d$')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.show()
    return


def validateDerivativeAllDirections(E, DE, position, N=100, h=0.0001):
    it = np.nditer(position, flags=['multi_index'])
    for x in it:
        dir = np.zeros_like(position)
        dir[it.multi_index] = 1
        validateDerivative(E, DE, position, N=N, h=h, dir=dir)


def validateSecondDerivative(DE, DDE, position, N=100, h=0.0001, dir_grad=None, dir_hess=None):
    """
    Extension of validateDerivative.
    Calculates the Gateaux derivative (a generalization of the concept of the
    directional derivative) using the gradient of a given function.
    Then plots the Gateaux derivative and the original Energy.

    The gradient is valid if the Gateaux derivative is the tangent of the Energy function.

    :param DE: The given first derivative functional that calculates the gradient depending on
              the position

    :param DDE: The given second derivative of the Energy functional that calculates the
               Hessian depending on the position

    :param ndarray position: Position on which the tangent will be evaluated

    :param N: Number of points on which the Energy and derivative will be evaluated.

    :param h: Stepsize between the N coordinates of the Energy and Gateaux plots

    :param dir_grad: The direction of the Gradient plot. If None is given, then the classical
                     -gradient will be taken.

    :param dir_hess: The direction of the Hessian plot. If None is given, then the classical
                     -gradient will be taken

    :returns: None, but plots the Gateaux derivative and Gradient using matplotlib
    """
    gradient = DE(position)
    if (dir_grad is not None):
        direction_grad = dir_grad
    else:
        direction_grad = -gradient

    if (dir_hess is not None):
        direction_hess = dir_hess
    else:
        direction_hess = -gradient

    coords = np.linspace(0, h*(N-1), num=N)
    Gradient = np.zeros((N))
    Gateaux = np.zeros((N))
    gateauxDerivative = np.dot(np.dot(DDE(position), direction_grad), direction_hess)
    initial = np.dot(gradient.ravel(), direction_hess.ravel())
    for i in range(N):
        Gradient[i] = np.dot(
            DE(position + coords[i]*direction_grad), direction_hess)
        Gateaux[i] = initial + coords[i]*gateauxDerivative
    import matplotlib.pyplot as plt
    plt.plot(coords, Gradient, label=r'$DE[x+td]\cdot h$')
    plt.plot(coords, Gateaux, label=r'$DE[x]+tDDE[x]d\cdot h$')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.show()
    return


def GradientDescent(x0, E, DE, beta=0.5, sigma=0.5, maxIter=100, stopEpsilon=0., startTau=1., inverseSP=None, NonlinearCG=False, info_dict=None):

    def ArmijoRule(x, initialTau, beta, sigma, energy, descentDir, gradientAtX, E):

        logging.debug(
            'Calculating tau with Armijo Line Search (starting with tau^0 = {})'.format(initialTau))

        if descentDir.dtype == object:
            tangentSlope = 0
            for i in range(len(descentDir)):
                tangentSlope += np.dot(descentDir[i].ravel(), gradientAtX[i].ravel())
        else:
            tangentSlope = np.dot(descentDir.ravel(), gradientAtX.ravel())

        if tangentSlope >= 0:
            return 0

        if not np.isfinite(energy):
            logging.debug(
                'Error: Initial energy is not finite. Rejecting step.)')

        def ArmijoCondition(tau):
            secantSlope = (E(x + tau * descentDir) - energy)/tau
            cond = (secantSlope / tangentSlope) >= sigma
            return cond

        tauMin = 1e-10
        tauMax = 1e10
        tau = max(min(initialTau, tauMax), tauMin)

        condition = ArmijoCondition(tau)
        if condition:
            logging.debug('initial stepsize too small. Will increase tau ...')
            while (condition and (tau < tauMax)):
                tau = tau / beta
                condition = ArmijoCondition(tau)

            tau = beta * tau
        else:
            logging.debug('initial stepsize too large. Will decrease tau ...')
            while ((not condition) and (tau > tauMin)):
                tau = beta * tau
                condition = ArmijoCondition(tau)

        if (tau > tauMin):
            logging.debug('accepted tau = {}'.format(tau))
            return tau
        else:
            logging.debug('tau <= tauMin, returning 0')
            return 0

    x = x0
    energyNew = E(x)
    tau = startTau

    if NonlinearCG:
        oldGradient = np.zeros_like(x)
        oldDirection = np.zeros_like(x)

    print('Initial energy {:.6f}'.format(energyNew))
    pbar = tqdm(range(maxIter))
    for i in pbar:
        energy = energyNew
        gradient = DE(x)
        descentDir = - gradient
        if (inverseSP is not None):
            descentDir = inverseSP(descentDir)

        # Fletcher-Reeves nonlinear conjugate gradient method.
        if NonlinearCG:
            nonlinCGBeta = (np.linalg.norm(descentDir.ravel()) /
                            np.linalg.norm(oldGradient.ravel()))**2 if (i % 10) else 0
            oldGradient = descentDir.copy()
            descentDir += nonlinCGBeta*oldDirection
            oldDirection = descentDir.copy()

        tau = ArmijoRule(x, tau, beta, sigma, energy, descentDir, gradient, E)

        if (NonlinearCG and (tau == 0)):
            descentDir = oldGradient.copy()
            # To be Jax compatible, do not use "oldDirection.fill(0)""
            oldDirection = np.zeros_like(x)
            tau = ArmijoRule(x, tau, beta, sigma, energy,
                             descentDir, gradient, E)

        x = x + tau * descentDir
        energyNew = E(x)

        pbar.set_postfix_str(f'tau={tau:.6f} E={energyNew:.6f}')

        if ((energy - energyNew) <= stopEpsilon):
            pbar.close()
            print(f"stopping after {i+1} iterations with energy {energyNew:.6f}")
            break

    if info_dict is not None:
        info_dict['iterations'] = i
        info_dict['tau'] = tau
        info_dict['energy'] = energyNew

    return x


def nesterov_II(x0, f_grad, alpha0, mu, L, restart, max_num_iter, stop_eps, proj_c=None):
    """
    Implementation of the Nesterov optimizer for strong convex functions with convexity constant mu >= 0
    that are continuously differentiable and have a Lipschitz continuous gradient with Lipschitz constant L.
    The "Constant Step Scheme, II" from "Introductory Lectures on Convex Programming" :cite:`Ne04` was implemented.

    Args:
        x0: Initial guess
        f_grad: Gradient of function that is to be minimized
        alpha0: scalar in (0,1)
        mu: strong convexity constant of f
        L: Lipschitz constant of the gradient of f
        restart: new initialization of alpha every 'restart' iterations
        max_num_iter: maximal number of iterations
        stop_eps: stopping tolerance
        proj_c: if provided, algorithm performs a constrained optimization by projecting the result of
                the gradient step on a feasible convex set C
    """
    x_old = x0.copy()
    y_old = x0.copy()
    q = mu / L
    alpha_old = alpha0

    # use tqdm to show a progress bar
    kbar = tqdm(range(max_num_iter))

    # main iteration
    for k in kbar:
        # perform (projected) gradient step on f
        if proj_c is not None:
            x_new = proj_c(y_old - 1 / L * f_grad(y_old))
        else:
            x_new = y_old - 1 / L * f_grad(y_old)

        # compute new alpha by solving alpha_new^2 = (1 - alpha_new) * alpha^2 + q * alpha_new
        alpha_new = - (alpha_old**2 - q) / 2 + \
            math.sqrt(((alpha_old**2 - q)**2) / 4 + alpha_old**2)
        assert 0 < alpha_new < 1, 'alpha_new not in intervall (0, 1).'

        # compute step size beta for extrapolation step
        beta = (alpha_old * (1 - alpha_old)) / (alpha_old**2 + alpha_new)

        # perform extrapolation step
        y_new = x_new + beta * (x_new - x_old)

        # stopping criterion: compare difference between last and current iterate with a pre-defined threshold
        res = np.linalg.norm((x_new - x_old).ravel())
        if res < stop_eps:
            kbar.close()
            print("Stopping after {} iterations with residuum {}".format(k, res))
            break

        # remember new values for alpha, x and y
        alpha_old = alpha_new
        np.copyto(x_old, x_new)
        np.copyto(y_old, y_new)

        # restart nesterov every 'restart' iterations
        if np.mod(k + 1, restart) == 0:
            alpha_old = alpha0
            np.copyto(y_old, x_new)

        if np.mod(k+1, 10) == 0:
            kbar.set_postfix(res=res)

    return x_new


def SplitBregman(v0, psi0, H, DH, zeta, DzetaExt, lambda_, minAlg, maxIter=10,
                 stopEps=0.0, stopEpsSlice=None, b0=None, verbose=True,
                 saveFunc=None):
    r"""
    Performs the split Bregman iteration to minimize
    :math:`E[v] := ||\zeta(v)||_1 + H[v]`, where both
    :math:`v\mapsto||\zeta||_1` and :math:`v\mapsto H[v]` are convex
    functionals.

    A single step of the split Bregman iteration is given by

    .. math::
        v^{k+1} &= \mathrm{argmin}_v\; H[v]+\lambda||\psi^k-\zeta[v]-b^k||_2^2\\

        \psi^{k+1} &= \mathrm{shrink}(\zeta[v^{k+1}]+b^k, 2/\lambda)\\

        b^{k+1} &= b^k - \psi^{k+1} + \zeta[v^{k+1}],

    where :math:`\mathrm{shrink}(z, w) := z/|z|\max(|z|-w, 0)` :cite:`GoOs09`.
    The update of v is performed with the minimization algorithm given by the
    ``minAlg`` parameter.

    :param ndarray v0: Initial guess for v (i.e. the value of :math:`v^0`)

    :param ndarray psi0: Value of :math:`\psi^0`

    :param H: A convex functional that takes an ndarray of the same shape as
              ``v0`` and returns a float

    :param DH: The gradient of H, taking an ndarray of the same shape as ``v0``
               and returning an ndarray of the same shape

    :param zeta: A functional that takes a numpy array of the same shape as
                 ``v0`` and returns a numpy array of the same shape as ``psi0``.

    :param DzetaExt: The gradient of :math:`||\psi-\zeta[v]-b||_2^2` with
                     respect to v. This must be a function with the signature
                     ``(v, psi, b)``, where ``v`` is an ndarray of the same
                     shape as ``v0`` and ``psi`` and ``b`` are ndarrays
                     of the same shape as ``psi0``. The returned gradient
                     must be of the same shape as ``v0``.

    :param float lambda_: A positive scalar value (see the update of v above)

    :param minAlg: The minimization algorithm that is used for the update of v.
                   This should be a function with the signature ``(x0, E, DE)``
                   that returns a numpy array of the same shape as ``x0``, where
                   ``x0`` is the initial guess, ``E`` is the energy functional
                   that is to be minimized and ``DE`` is its gradient.

    :param int maxIter: Maximum number of split Bregman iterations that are
                        performed, i.e. maximum number of times that ``k`` is
                        incremented.

    :param float stopEps: Stopping criterion (the optimization is stopped as
                          soon as the updates of the 2-norm of v are smaller
                          than ``stopEps``).

    :param stopEpsSlice: A numpy slice object that describes the subsection of
                         v that should be used for the calculation of the
                         updates. This can be used to crop a part of the array
                         near the boundary for example. By default, the entire
                         array v is used for the computation of the update of v
                         from the previous iteration.

    :param ndarray b0: Value of :math:`b^0`, which must be of the same shape as
                       ``psi0``. By default, ``b0`` is set to a numpy array of
                       zeros.

    :param bool verbose: Determines if the progress is printed to ``sys.stdout``

    :param saveFunc: A function with the signature ``(k, v, psi, b)`` that is
                     called with the current iteration number ``k`` and the
                     values of :math:`v^k, \psi^k, b^k` after every iteration.
                     This function can be used to save intermediate results for
                     example. Note that it must not modify any of its arguments.

    :returns: The resulting value of v, which is of the same shape as ``v0``
    """
    # Initialization
    v = v0.copy()
    v_old = v0.copy()
    psi = psi0.copy()
    if b0 is not None:
        b = b0.copy()
    else:
        b = np.zeros_like(psi0)

    def shrink(z, w):
        z_abs = np.abs(z)
        z_nonzero = (z_abs != 0)
        res = np.zeros_like(z)
        res[z_nonzero] = z[z_nonzero] / z_abs[z_nonzero] * \
            np.maximum(z_abs[z_nonzero] - w, 0)
        return res

    # Split Bregman iterations
    for i in range(maxIter):
        if verbose:
            print("\nSplit-Bregman iteration {} / {} ...".format(i+1, maxIter))

        # The energy functional and its gradient for the update of v
        def E(x): return H(x) + lambda_ * np.linalg.norm(psi - zeta(x) - b)**2
        def DE(x): return DH(x) + lambda_ * DzetaExt(x, psi, b)

        # Update v, psi and b
        v_old = v
        v = minAlg(v, E, DE)
        zeta_v = zeta(v)
        psi = shrink(zeta_v + b, 2/lambda_)
        b += zeta_v - psi

        # Save intermediate results
        if saveFunc is not None:
            saveFunc(i+1, v, psi, b)

        # Check the stopping criterion
        if stopEpsSlice is None:
            update = np.linalg.norm(v-v_old)
        else:
            update = np.linalg.norm(v[stopEpsSlice] - v_old[stopEpsSlice])

        if verbose:
            print("\nUpdate = {}".format(update))

        if update < stopEps:
            print("\nStopping after {} iterations with update="
                  "{}.".format(i+1, update))
            break

    return v


def getTimestepWidthWithSimpleLineSearch(E, direction, x, start_tau=1, tau_min=2**-30):
    # extreme simple timestep width control, just ensures, that fnew < fnew

    tau = start_tau

    e = E(x)
    eNew = E(x+tau*direction)

    while ((eNew >= e) and (tau >= tau_min)):
        tau = tau * 0.5
        eNew = E(x+tau*direction)

    # No energy descent for tau >= tauMin, so we don't want the step to be done.
    # The stopping criterion also handles this case.
    if (tau < tau_min):
        tau = 0

    return tau


def GaussNewtonAlgorithm(x0, F, DF, maxIter=50, stopEpsilon=0):
    x = x0.copy()
    f = F(x)
    fNormSqrOld = np.linalg.norm(f)**2
    print("Initial fNormSqr {:#.6g}".format(fNormSqrOld))
    tau = 1

    # use tqdm to show a progress bar
    i_bar = tqdm(range(maxIter))
    for i in i_bar:
        matDF = DF(x)
        # from scipy.linalg import qr, solve, solve_triangular
        # Q, R = np.linalg.qr(matDF, mode='reduced')
        # b = np.matmul(Q.T, f)
        # direction = solve_triangular(R, b, lower=False)
        direction = np.linalg.lstsq(matDF, f, rcond=None)[0]

        if not np.all(np.isfinite(direction)):
            print("Error: lstsq failed.")

        x -= direction

        f = F(x)
        fNormSqr = np.linalg.norm(f)**2

        # If the target functional did not decrease with the update, try to find a smaller step
        # so that it does. This step size control is extremely simple and not very efficient, but
        # it's certainly better than letting the algorithm diverge.
        if fNormSqr >= fNormSqrOld:
            x += direction
            direction *= -1
            # getTimestepWidthWithSimpleLineSearch doesn't support "widening", so let it start with 2*tau.
            tau = getTimestepWidthWithSimpleLineSearch(
                lambda v: np.linalg.norm(F(v))**2, direction, x, start_tau=min(2*tau, 1))
            x += tau*direction
            f = F(x)
            fNormSqr = np.linalg.norm(f)**2
        else:
            tau = 1

        i_bar.set_description("\u03C4={:#.2g}, E={:#.5g}, \u0394={:.1e}".format(
            tau, fNormSqr, fNormSqrOld - fNormSqr))

        if ((fNormSqrOld - fNormSqr)) <= stopEpsilon * fNormSqr or np.isclose(fNormSqr, 0):
            break

        fNormSqrOld = fNormSqr

    return x
