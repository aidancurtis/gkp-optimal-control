from functools import lru_cache

import jax.numpy as jnp
import jaxquantum as jqt
import numpy as np

# Anything the coercion helpers below accept as "a state": a jaxquantum
# Qarray, or a raw jax/numpy ket or density matrix.
StateLike = jqt.Qarray | jnp.ndarray | np.ndarray

# Bridge factor between qutip's quadrature convention (alpha = (x+ip)/sqrt(2))
# and jaxquantum's alpha-as-coord convention. The grid going into jqt.wigner
# is divided by sqrt(2); the output is multiplied by 1/2 (Jacobian of the
# coordinate change so that integrate(W) dx dp = 1 is preserved).
_SQRT2 = np.sqrt(2.0)
_JACOBIAN = 0.5


def _to_array(s):
    if isinstance(s, jqt.Qarray):
        if getattr(s, "is_sparse_bcoo", False):
            s = s.to_dense()
        return s.data
    return jnp.asarray(s)


@lru_cache(maxsize=16)
def _wigner_grid(x_bound: float, y_bound: float, grid_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cached ``(xvec, yvec)`` quadrature grids for a given grid spec."""
    xvec = np.linspace(-x_bound, x_bound, grid_points)
    yvec = np.linspace(-y_bound, y_bound, grid_points)
    return xvec, yvec


def _alpha_grid(xvec: np.ndarray, yvec: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert quadrature grids ``(xvec, yvec)`` to alpha grids for jqt.wigner."""
    return jnp.asarray(xvec / _SQRT2), jnp.asarray(yvec / _SQRT2)


def to_qarray(state: StateLike) -> jqt.Qarray:
    r"""Coerce a state into a jaxquantum ``Qarray``.

    This is the single sanctioned crossing from the raw-array layer
    (``grape``, ``hamiltonians``, ``gate_optimization``) into the ``Qarray``
    layer (``jqt.wigner``, ``jqt.sesolve``, plotting). It is idempotent, so
    every call site can pass whichever representation it happens to hold.

    Parameters
    ----------
    state : jaxquantum.Qarray or array_like
        A ``Qarray`` (returned unchanged), a 1D ket ``(n,)``, a column ket
        ``(n, 1)``, or a density matrix ``(n, n)``.

    Returns
    -------
    jaxquantum.Qarray
        The state as a ``Qarray``.

    Raises
    ------
    ValueError
        If ``state`` is a 2D array that is neither square nor a column
        vector, which is ambiguous. The common case is an ECD qubit block of
        shape ``(2, n_fock)``: trace the qubit out first, e.g.
        ``rho = arr.T @ arr.conj()``.

    Notes
    -----
    The ``isinstance`` check must come first. ``Qarray.__getattr__`` raises
    ``NotImplementedError`` rather than ``AttributeError`` for unknown
    attributes, so any duck-typing probe -- including ``jnp.asarray``, which
    reaches for ``.aval`` internally, and ``hasattr`` -- crashes instead of
    falling through when handed a ``Qarray``.
    """
    if isinstance(state, jqt.Qarray):
        return state

    arr = jnp.asarray(state)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[0] != arr.shape[1] and arr.shape[1] != 1:
        raise ValueError(
            f"Ambiguous state shape {arr.shape}: expected a ket (n,) or (n, 1), "
            "or a density matrix (n, n). An ECD qubit block (2, n_fock) must be "
            "reduced to a cavity density matrix first, e.g. arr.T @ arr.conj()."
        )
    elif arr.ndim > 2:
        raise ValueError(
            f"Expected a single state, got shape {arr.shape}. "
            "Use wigner_trajectory for a batch of states."
        )
    return jqt.Qarray.create(arr)


def to_ket(state: StateLike, normalize: bool = True) -> jnp.ndarray:
    r"""Coerce a state into a flat, normalized 1D JAX ket.

    The reverse crossing to :func:`to_qarray`: use it when handing a state
    from the ``Qarray`` layer into ``grape``, ``gate_optimization``, or any
    other raw-array code, or when inspecting Fock amplitudes directly.

    Parameters
    ----------
    state : jaxquantum.Qarray or array_like
        A ``Qarray`` ket, a 1D ket ``(n,)``, or a column ket ``(n, 1)``.
    normalize : bool, default True
        Rescale to unit norm. Pass ``False`` where the norm carries meaning,
        e.g. for a state about to be propagated linearly.

    Returns
    -------
    jnp.ndarray
        Normalized 1D array of Fock amplitudes, shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``state`` is a density matrix, which has no ket representation.
    """
    data = state.data if isinstance(state, jqt.Qarray) else state
    arr = jnp.asarray(data)
    if arr.ndim == 2 and arr.shape[1] != 1:
        raise ValueError(
            f"Cannot convert shape {arr.shape} to a ket; a density matrix has "
            "no ket representation."
        )
    amps = arr.reshape(-1)
    return amps / jnp.linalg.norm(amps) if normalize else amps


def to_ket_batch(state, n_fock: int | None = None, name: str = "state"):
    """Coerce one or more kets into a normalized ``(K, n_fock)`` complex array.

    The batched counterpart of :func:`to_ket`, and the entry point for states
    going into :mod:`gate_optimization`. ``n_fock`` checks the dimension and
    ``name`` labels the argument in error messages.

    Accepts a ``jaxquantum.Qarray`` (single or batched), a raw ``(d,)``,
    ``(d, 1)``, ``(K, d)`` or ``(K, d, 1)`` array, or a list of any of those.
    """
    if isinstance(state, (list, tuple)):
        rows = [to_ket_batch(s, n_fock, name) for s in state]
        data = jnp.concatenate(rows, axis=0)
    else:
        if isinstance(state, jqt.Qarray):
            data = state.data
        else:
            data = state
        data = jnp.asarray(data)
        if data.ndim >= 2 and data.shape[-1] == 1:
            data = data[..., 0]
        if data.ndim == 1:
            data = data[None, :]
        elif data.ndim > 2:
            data = data.reshape(-1, data.shape[-1])

    if n_fock is not None and data.shape[-1] != n_fock:
        raise ValueError(f"{name} has Fock dimension {data.shape[-1]}, expected n_fock={n_fock}.")

    data = data.astype(jnp.result_type(complex))
    norms = jnp.linalg.norm(data, axis=-1, keepdims=True)
    if bool(jnp.any(norms == 0)):
        raise ValueError(f"{name} contains a zero vector.")
    return data / norms


def compute_wigner(
    state: StateLike,
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
    state : jaxquantum.Qarray or array_like
        Either a 1D ket of shape ``(n_fock,)`` or ``(n_fock, 1)``, or a 2D
        density matrix of shape ``(n_fock, n_fock)``. Coerced by
        :func:`to_qarray`, so either representation works.
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
    qa = to_qarray(state)
    wigner = _JACOBIAN * jqt.wigner(qa, av, bv)
    return xvec, yvec, np.asarray(wigner)


def wigner_trajectory(
    states, x_bound: float, y_bound: float, grid_points: int = 100, slices: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute Wigner distributions along a trajectory of states.

    Same convention as :func:`compute_wigner`: outputs are in
    quadrature coordinates with the Jacobian baked in.

    Parameters
    ----------
    states : jnp.ndarray or sequence of arrays
        Trajectory of states. Accepted shapes:

        * ``(T, n_fock)``        -- batch of kets
        * ``(T, n_fock, 1)``      -- batch of column kets
        * ``(T, n_fock, n_fock)`` -- batch of density matrices
        * a Python sequence of 1D or 2D arrays

        Each frame is coerced by :func:`to_qarray`, so a trajectory of ECD
        qubit blocks ``(T, 2, n_fock)`` raises rather than being flattened
        into meaningless ``2 n_fock``-dimensional kets. Trace the qubit out
        first.
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
        states_arr = _to_array(states)

    if slices is None:
        slices = np.arange(states_arr.shape[0])

    xvec, yvec = _wigner_grid(x_bound, y_bound, grid_points)
    av, bv = _alpha_grid(xvec, yvec)

    frames = []
    for k in slices:
        qa = to_qarray(states_arr[k])
        frames.append(np.asarray(_JACOBIAN * jqt.wigner(qa, av, bv)))

    return xvec, yvec, np.stack(frames)
