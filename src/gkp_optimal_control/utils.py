from functools import lru_cache

import jax.numpy as jnp
import jaxquantum as jqt
import numpy as np

# Bridge factor between qutip's quadrature convention (alpha = (x+ip)/sqrt(2))
# and jaxquantum's alpha-as-coord convention. The grid going into jqt.wigner
# is divided by sqrt(2); the output is multiplied by 1/2 (Jacobian of the
# coordinate change so that integrate(W) dx dp = 1 is preserved).
_SQRT2 = np.sqrt(2.0)
_JACOBIAN = 0.5


@lru_cache(maxsize=16)
def _wigner_grid(x_bound: float, y_bound: float, grid_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cached ``(xvec, yvec)`` quadrature grids for a given grid spec."""
    xvec = np.linspace(-x_bound, x_bound, grid_points)
    yvec = np.linspace(-y_bound, y_bound, grid_points)
    return xvec, yvec


def _alpha_grid(xvec: np.ndarray, yvec: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert quadrature grids ``(xvec, yvec)`` to alpha grids for jqt.wigner."""
    return jnp.asarray(xvec / _SQRT2), jnp.asarray(yvec / _SQRT2)


def _to_qarray(state: jnp.ndarray) -> jqt.Qarray:
    """Wrap a raw JAX array into a jaxquantum Qarray for the wigner call."""
    arr = jnp.asarray(state)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return jqt.Qarray.create(arr)


def compute_wigner(
    state: jnp.ndarray,
    x_bound: float,
    y_bound: float,
    grid_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute the Wigner distribution of a single bosonic state.

    Uses the quadrature convention: the returned ``xvec`` and ``yvec`` are
    the position and momentum coordinates :math:`(x, p)` with
    :math:`[x, p] = i`. A coherent state :math:`D(\alpha)|0\rangle` with real
    :math:`\alpha` is centered at :math:`x = \sqrt{2}\,\alpha`, :math:`p = 0`.

    Parameters
    ----------
    state : jnp.ndarray
        Either a 1D ket of shape ``(n_fock,)`` or ``(n_fock, 1)``, or a 2D
        density matrix of shape ``(n_fock, n_fock)``.
    x_bound, y_bound : float
        Half-widths of the :math:`x`- and :math:`p`-axes (quadratures).
    grid_points : int, default 200
        Number of samples along each phase-space axis.

    Returns
    -------
    xvec, yvec : numpy.ndarray
        1D grid samples along the :math:`x` and :math:`p` axes.
    wigner : numpy.ndarray
        2D Wigner distribution of shape ``(len(yvec), len(xvec))``,
        normalized so that :math:`\int W\, dx\, dp = 1`.
    """
    xvec, yvec = _wigner_grid(x_bound, y_bound, grid_points)
    av, bv = _alpha_grid(xvec, yvec)
    qa = _to_qarray(state)
    wigner = _JACOBIAN * jqt.wigner(qa, av, bv)
    return xvec, yvec, np.asarray(wigner)


def wigner_trajectory(
    states,
    x_bound: float,
    y_bound: float,
    grid_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute Wigner distributions along a trajectory of states.

    Same convention as :func:`compute_wigner`: outputs are in
    quadrature coordinates with the Jacobian baked in.

    Parameters
    ----------
    states : jnp.ndarray or sequence of arrays
        Trajectory of states. Accepted shapes:

        * ``(T, n_fock)``        -- batch of kets
        * ``(T, n_fock, n_fock)`` -- batch of density matrices
        * a Python sequence of 1D or 2D arrays
    x_bound, y_bound : float
        Half-widths of the :math:`x`- and :math:`p`-axes (quadratures).
    grid_points : int, default 100
        Number of samples along each phase-space axis.

    Returns
    -------
    xvec, yvec : numpy.ndarray
        1D grid samples along the :math:`x` and :math:`p` axes.
    wigner_trajectory : numpy.ndarray
        Wigner distributions stacked along the time axis, shape
        ``(T, len(yvec), len(xvec))``.
    """
    if isinstance(states, (list, tuple)):
        states_arr = jnp.stack([jnp.asarray(s) for s in states])
    else:
        states_arr = jnp.asarray(states)

    xvec, yvec = _wigner_grid(x_bound, y_bound, grid_points)
    av, bv = _alpha_grid(xvec, yvec)

    frames = []
    for k in range(states_arr.shape[0]):
        state_k = states_arr[k]
        if state_k.ndim == 2 and state_k.shape[0] == state_k.shape[1]:
            qa = jqt.Qarray.create(state_k)
        else:
            qa = jqt.Qarray.create(state_k.reshape(-1, 1))
        frames.append(np.asarray(_JACOBIAN * jqt.wigner(qa, av, bv)))

    return xvec, yvec, np.stack(frames)
