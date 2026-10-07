"""SCRiBLe algorithm specialized to the probability simplex.

Implements Algorithm 4 from Abernethy, Hazan & Rakhlin (IEEE TIT 2012)
in reduced coordinates z in R^m where m = n-1 and x_n = 1 - sum(z).

Functional API: explore, estimate_gradient, ftrl_update.
Stateful wrapper: SCRiBLeSolver.
Randomness controlled externally via np.random.seed().
"""

import numpy as np
from scipy.optimize import brentq

class SCRiBLeSolver:
    """Stateful SCRiBLe solver for bandit linear optimization on Delta_n.

    Usage
    -----
    >>> solver = SCRiBLeSolver(n=3, eta=0.05)
    >>> for t in range(T):
    ...     action = solver.act()           # point on Delta_n
    ...     reward = my_env(action)         # external; caller's responsibility
    ...     solver.update(reward)

    The solver hides z (current iterate) and g (cumulative gradient) and
    enforces an alternating act -> update call pattern.

    Parameters
    ----------
    n : int
        Number of actions (vertices of the simplex).
    eta : float
        Learning rate.
    kappa : float
        Discount factor for past gradients in the exponential moving average.
        Theoretical results hold for kappa close to 1, but smaller values may
        lead to better performance in practice.
    """

    def __init__(self, n: int, eta: float, kappa: float = 0.9999):
        self.n = n
        self.eta = eta
        self.m = n - 1
        self.z = np.full(self.m, 1.0 / n)
        self.g = np.zeros(self.m)
        self.kappa = kappa
        self._pending: dict | None = None

    def suggest_action(self) -> np.ndarray:
        """Return the next action to play, a point on Delta_n.

        Raises
        ------
        RuntimeError
            If called twice without an intervening update().
        """
        if self._pending is not None:
            raise RuntimeError("suggest_action() called twice without update() in between")
        action, perturbation_info = explore(self.z)
        self._pending = perturbation_info
        return action

    def update(self, cost: float, round_num: int) -> None:
        """Feed the scalar cost for the last action played.

        Internally uses loss = cost (SCRiBLe minimizes loss),
        builds the gradient estimator, accumulates, and runs the FTRL update.

        Raises
        ------
        RuntimeError
            If called before suggest_action().
        """
        if round_num == 0:
            return
        if self._pending is None:
            raise RuntimeError("update() called before suggest_action()")
        loss = cost
        f_tilde = estimate_gradient(loss, self._pending)
        # discounting past gradients with exponential moving average
        self.g = self.kappa * self.g + f_tilde
        self.z = ftrl_update(self.g, self.eta)
        self._pending = None

def explore(z: np.ndarray) -> tuple[np.ndarray, dict]:
    """Explore: perturb z along a random eigenvector of the Hessian.

    Parameters
    ----------
    z : ndarray(m,)
        Current reduced iterate in int(K).

    Returns
    -------
    action : ndarray(n,)
        Point on Delta_n to play, where n = m + 1.
    perturbation_info : dict
        Contains "epsilon", "eigval_i", "eigvec_i" needed by
        estimate_gradient().
    """
    m = len(z)
    H = log_barrier_hessian(z)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    i_t = np.random.randint(0, m)
    eps_t = np.random.choice([-1, 1])

    lam_i = eigenvalues[i_t]
    e_i = eigenvectors[:, i_t]

    # Exploration point in reduced coordinates
    y = z + eps_t * lam_i ** (-0.5) * e_i

    # Reconstruct full action on Delta_n
    action = np.empty(m + 1)
    action[:m] = y
    action[m] = 1.0 - np.sum(y)

    perturbation_info = {
        "epsilon": eps_t,
        "eigval_i": lam_i,
        "eigvec_i": e_i,
    }
    return action, perturbation_info


def estimate_gradient(loss: float, perturbation_info: dict) -> np.ndarray:
    """One-point gradient estimator.

    Parameters
    ----------
    loss : float
        Observed scalar loss f_t^T a_t.
    perturbation_info : dict
        Returned by explore().

    Returns
    -------
    f_tilde : ndarray(m,)
        Gradient estimator: m * loss * epsilon * lambda^{1/2} * e_i.
    """
    eps = perturbation_info["epsilon"]
    lam_i = perturbation_info["eigval_i"]
    e_i = perturbation_info["eigvec_i"]
    m = len(e_i)
    return m * loss * eps * lam_i**0.5 * e_i


def ftrl_update(g: np.ndarray, eta: float) -> np.ndarray:
    """FTRL update: solve argmin_{z in K} [ eta * g^T z + R(z) ].

    KKT gives z_i = 1 / (eta * g_i + nu), where nu = 1/s > 0
    satisfies sum_i 1/(eta*g_i + nu) + 1/nu = 1.

    Parameters
    ----------
    g : ndarray(m,)
        Cumulative gradient estimator.
    eta : float
        Learning rate.

    Returns
    -------
    z_new : ndarray(m,)
        Next iterate in reduced coordinates.
    """
    eg = eta * g

    # Lower bound for nu: must have eta*g_i + nu > 0 for all i
    nu_min = max(0.0, -np.min(eg)) + 1e-12

    def phi(nu):
        return np.sum(1.0 / (eg + nu)) + 1.0 / nu - 1.0

    # Find upper bracket
    nu_max = nu_min + 1.0
    while phi(nu_max) > 0:
        nu_max *= 2.0

    nu_star = brentq(phi, nu_min, nu_max, xtol=1e-14)
    return 1.0 / (eg + nu_star)

"""
Log-barrier for the m-dimensional simplex K = {z in R^m : z >= 0, sum(z) <= 1}.

R(z) = -sum(log(z_i)) - log(1 - sum(z_i))

This is an n-self-concordant barrier where n = m + 1 (one log per constraint).
"""


def log_barrier(z: np.ndarray) -> float:
    """R(z) = -sum(log(z_i)) - log(1 - sum(z_i))."""
    s = 1.0 - np.sum(z)
    return -np.sum(np.log(z)) - np.log(s)


def log_barrier_grad(z: np.ndarray) -> np.ndarray:
    """Gradient: [nabla R(z)]_i = -1/z_i + 1/s."""
    s = 1.0 - np.sum(z)
    return -1.0 / z + 1.0 / s


def log_barrier_hessian(z: np.ndarray) -> np.ndarray:
    """Hessian: H = diag(1/z_i^2) + (1/s^2) * 1*1^T.

    Returns the full m x m matrix.
    """
    s = 1.0 - np.sum(z)
    m = len(z)
    H = np.diag(1.0 / z**2) + (1.0 / s**2) * np.ones((m, m))
    return H
