from functools import partial

import jax
import jax.numpy as jnp


def _annihilation(n: int) -> jnp.ndarray:
    """Annihilation operator on an n-level truncated harmonic oscillator."""
    return jnp.diag(jnp.sqrt(jnp.arange(1, n, dtype=jnp.complex128)), k=1)


@partial(jax.jit, static_argnames=("n_fock",))
def cavity_operators(
    n_fock: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Return :math:`(a,\, a^\dagger,\, n)` on a truncated cavity."""
    a = _annihilation(n_fock)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n


@partial(jax.jit, static_argnames=("n_fock",))
def kerr_cavity_drift(n_fock: int, kerr) -> jnp.ndarray:
    r"""Single-cavity Kerr drift :math:`H = (K/2)\, (a^\dagger)^2 a^2`.

    Parameters
    ----------
    n_fock : int
        Fock truncation (static).
    kerr : scalar
        Kerr coefficient :math:`K` in rad / time-unit. Traceable.
    """
    a, adag, _ = cavity_operators(n_fock)
    return (kerr / 2.0) * (adag @ adag @ a @ a)


@partial(jax.jit, static_argnames=("n_fock",))
def kerr_cavity_squeezing_controls(n_fock: int) -> jnp.ndarray:
    r"""Two-photon (squeezing) drives :math:`a^2 + a^{\dagger 2}` and
    :math:`i(a^2 - a^{\dagger 2})`.

    Returns
    -------
    jnp.ndarray
        Stack of shape ``(2, n_fock, n_fock)``.
    """
    a, adag, _ = cavity_operators(n_fock)
    h_i = a @ a + adag @ adag
    h_q = 1j * (a @ a - adag @ adag)
    return jnp.stack([h_i, h_q])


@partial(jax.jit, static_argnames=("n_fock",))
def cavity_displacement_controls(n_fock: int) -> jnp.ndarray:
    r"""Linear (displacement) drives :math:`a + a^\dagger` and
    :math:`i(a - a^\dagger)`.

    Returns
    -------
    jnp.ndarray
        Stack of shape ``(2, n_fock, n_fock)``.
    """
    a, adag, _ = cavity_operators(n_fock)
    h_i = a + adag
    h_q = 1j * (a - adag)
    return jnp.stack([h_i, h_q])


def _cavity_transmon_modes(
    n_cav: int, n_tr: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(a, a_dag, b, b_dag)`` on the joint cavity ⊗ transmon space."""
    a_c = _annihilation(n_cav)
    b_t = _annihilation(n_tr)
    i_c = jnp.eye(n_cav, dtype=jnp.complex128)
    i_t = jnp.eye(n_tr, dtype=jnp.complex128)
    a = jnp.kron(a_c, i_t)
    b = jnp.kron(i_c, b_t)
    return a, a.conj().T, b, b.conj().T


@partial(jax.jit, static_argnames=("n_cav", "n_tr"))
def cavity_transmon_drift(
    n_cav: int,
    n_tr: int,
    chi,
    k=0.0,
    alpha=0.0,
) -> jnp.ndarray:
    r"""Dispersive cavity-transmon drift in the rotating frame, on resonance.

    :math:`H = \chi\, n_c n_t + (K/2)\, (a^\dagger)^2 a^2 + (\alpha/2)\,
    (b^\dagger)^2 b^2`.

    Parameters
    ----------
    n_cav, n_tr : int
        Hilbert-space dimensions (static).
    chi, k, alpha : scalar
        Dispersive shift, cavity self-Kerr, and transmon self-Kerr.
        Traceable. ``k`` and ``alpha`` default to zero, in which case
        the corresponding terms drop out by virtue of multiplication.
    """
    a, adag, b, bdag = _cavity_transmon_modes(n_cav, n_tr)
    n_c = adag @ a
    n_t = bdag @ b
    return (
        chi * (n_c @ n_t)
        + (k / 2.0) * (adag @ adag @ a @ a)
        + (alpha / 2.0) * (bdag @ bdag @ b @ b)
    )


@partial(jax.jit, static_argnames=("n_cav", "n_tr"))
def cavity_transmon_iq_controls(n_cav: int, n_tr: int) -> jnp.ndarray:
    r"""I/Q drives on both cavity and transmon.

    Returns
    -------
    jnp.ndarray
        Stack of shape ``(4, n_cav * n_tr, n_cav * n_tr)``: cavity-I,
        cavity-Q, transmon-I, transmon-Q.
    """
    a, adag, b, bdag = _cavity_transmon_modes(n_cav, n_tr)
    return jnp.stack(
        [
            a + adag,
            1j * (a - adag),
            b + bdag,
            1j * (b - bdag),
        ]
    )
