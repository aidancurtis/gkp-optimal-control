import warnings
from _collections_abc import Callable
from typing import cast

import diffrax
import jax.numpy as jnp

warnings.filterwarnings(
    "ignore",
    message="Complex dtype support in Diffrax",
    category=UserWarning,
)


def _build_vector_field(
    h_drift: jnp.ndarray,
    h_controls: jnp.ndarray,
    coeffs: Callable[..., jnp.ndarray],
):
    """Construct the right-hand side -i H(t) psi as a diffrax-compatible f(t, y, args).

    The returned function has a deliberately loose signature so it satisfies
    diffrax's ``vector_field`` protocol (which expects ``RealScalarLike`` for
    time, not strictly ``float``).
    """

    def f(t, psi, _):
        # coeffs(t): (n_controls,), h_controls: (n_controls, d, d)
        # tensordot along the controls axis -> (d, d).
        h_sys = h_drift + jnp.tensordot(coeffs(t), h_controls, axes=1)
        return -1j * (h_sys @ psi)

    return f


def evolve(
    psi0: jnp.ndarray,
    h_drift: jnp.ndarray,
    h_controls: jnp.ndarray,
    coeffs: Callable[..., jnp.ndarray],
    t0: float,
    t1: float,
    *,
    saveat: jnp.ndarray | None = None,
    solver: diffrax.AbstractSolver | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 100_000,
    dt0: float | None = None,
) -> jnp.ndarray:
    r"""Integrate the Schrödinger equation from ``t0`` to ``t1``.

    Parameters
    ----------
    psi0 : jnp.ndarray
        Initial ket of shape ``(dim,)``.
    h_drift : jnp.ndarray
        Time-independent drift Hamiltonian, shape ``(dim, dim)``.
    h_controls : jnp.ndarray
        Stack of control Hamiltonians, shape ``(n_controls, dim, dim)``.
        Use ``jnp.zeros((0, dim, dim))`` if there are no controls.
    coeffs : Callable
        A function ``t -> jnp.ndarray`` returning the ``(n_controls,)``
        coefficient vector at time ``t``. Must be a pure JAX function (no
        Python control flow on ``t``); see :func:`piecewise_constant` and
        :func:`time_independent` for common patterns.
    t0, t1 : float
        Start and end times.
    saveat : jnp.ndarray, optional
        Times at which to save the state. If ``None``, only the final state
        at ``t1`` is returned. If provided, must be a sorted 1D array with
        all values in ``[t0, t1]``.
    solver : diffrax.AbstractSolver, optional
        Override the default solver (``Tsit5``). For stiff or strongly
        oscillatory problems, try ``Dopri8`` or ``Kvaerno5``.
    rtol, atol : float
        Relative and absolute tolerances for adaptive step control.
    max_steps : int
        Cap on the number of internal ODE steps. Increase if you see
        truncation errors at high tolerance.
    dt0 : float, optional
        Initial step size hint. If ``None``, diffrax picks one. Usually
        fine; specify if you want deterministic step patterns under jit.

    Returns
    -------
    jnp.ndarray
        If ``saveat`` is None: final state, shape ``(dim,)``.
        Otherwise: trajectory, shape ``(len(saveat), dim)``.
    """
    if solver is None:
        solver = diffrax.Tsit5()

    if saveat is None:
        saveat_obj = diffrax.SaveAt(t1=True)
    else:
        saveat_obj = diffrax.SaveAt(ts=saveat)

    term = diffrax.ODETerm(_build_vector_field(h_drift, h_controls, coeffs))
    controller = diffrax.PIDController(rtol=rtol, atol=atol)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=psi0,
        saveat=saveat_obj,
        stepsize_controller=controller,
        max_steps=max_steps,
        # The default RecursiveCheckpointAdjoint works well with jax.grad
        # and is enabled implicitly; we don't need to set it.
    )

    # diffrax types sol.ys as PyTree | None (None means the solve failed
    # before saving anything). We've configured SaveAt to always save, so
    # at runtime sol.ys is always a jnp.ndarray; cast lets pyright see that.
    ys = cast(jnp.ndarray, sol.ys)
    if saveat is None:
        return ys[0]  # shape (dim,)
    return ys


# ---------------------------------------------------------------------------
# Convenience builders for `coeffs`
# ---------------------------------------------------------------------------


def time_independent(n_controls: int = 0) -> Callable[..., jnp.ndarray]:
    """Return a ``coeffs`` callable that always returns the zero vector.

    Useful when ``h_drift`` alone defines the dynamics. ``n_controls=0``
    pairs with ``h_controls = jnp.zeros((0, dim, dim))``.
    """
    zero = jnp.zeros(n_controls)

    def coeffs(_):
        return zero

    return coeffs


def piecewise_constant(pulses: jnp.ndarray, t0: float, t1: float) -> Callable[..., jnp.ndarray]:
    r"""Return a piecewise-constant ``coeffs`` callable from a pulse array.

    Maps the time axis ``[t0, t1]`` onto ``pulses.shape[0]`` equally sized
    bins. At time ``t``, returns the pulses row for the bin containing ``t``
    (clamped to the last bin at ``t == t1``).

    Parameters
    ----------
    pulses : jnp.ndarray
        Pulse array of shape ``(n_steps, n_controls)``.
    t0, t1 : float
        Time bounds defining the bin spacing.

    Returns
    -------
    Callable
        A pure-JAX function ``t -> (n_controls,)`` suitable for use as
        ``coeffs`` in :func:`evolve`.

    Notes
    -----
    Piecewise-constant means the time-dependence has step discontinuities
    at bin boundaries; for an adaptive solver, this is fine but the solver
    may slow down near the steps. If GRAPE-style speed matters, prefer a
    propagator-product method (slice-wise ``expm``) over this; if you want
    smooth controls, interpolate first (e.g. with a spline).
    """
    n_steps = pulses.shape[0]
    duration = t1 - t0

    def coeffs(t):
        # Fractional position in [0, n_steps]; clamp last point inside the
        # final bin so t == t1 doesn't index out of bounds.
        idx = jnp.clip(
            jnp.floor((t - t0) / duration * n_steps).astype(jnp.int32),
            0,
            n_steps - 1,
        )
        return pulses[idx]

    return coeffs
