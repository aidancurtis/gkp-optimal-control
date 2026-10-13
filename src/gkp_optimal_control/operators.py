r"""Truncated-space operators shared by the rest of the package.

Everything here builds a matrix (or a matrix-free action) from a truncation
size and a few parameters, with no physical model attached: ladder and
quadrature operators, the joint cavity-transmon embedding, Pauli matrices and
qubit rotations, displacements and squeezers. Hamiltonians built from these
live in :mod:`hamiltonians`.

All functions return JAX arrays; NumPy callers wrap with ``np.asarray``.

Conventions
-----------
Joint operators are ordered ``kron(cavity, transmon)``, i.e. joint index
``n_cav * n_transmon + s``. Qubit matrices are in the ``(|g>, |e>)`` basis.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import jax
import jax.numpy as jnp

__all__ = [
    "SIGMA_X",
    "SIGMA_Y",
    "SIGMA_Z",
    "annihilation",
    "cavity_operators",
    "displacement",
    "joint_operators",
    "make_displacement",
    "make_displacement_apply",
    "make_squeeze",
    "make_squeeze_apply",
    "quadrature_operators",
    "qubit_rotation",
]

SIGMA_X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
SIGMA_Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
SIGMA_Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)


# ---------------------------------------------------------------------------
# Ladder and quadrature operators
# ---------------------------------------------------------------------------


def annihilation(n_fock: int) -> jnp.ndarray:
    """Annihilation operator on an ``n_fock``-level truncated oscillator."""
    return jnp.diag(jnp.sqrt(jnp.arange(1, n_fock, dtype=jnp.complex128)), k=1)


def cavity_operators(n_fock: int):
    """Annihilation, creation, number ops on a truncated cavity."""
    a = annihilation(n_fock)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n


def quadrature_operators(n_fock: int):
    """q = (a + a†)/√2,  p = -i(a − a†)/√2   (so [q, p] = i, matching utils.compute_wigner)."""
    a, adag, _ = cavity_operators(n_fock)
    q = (a + adag) / jnp.sqrt(2.0)
    p = -1j * (a - adag) / jnp.sqrt(2.0)
    return q, p


def joint_operators(n_fock: int, n_transmon: int = 2) -> dict:
    """Cavity and transmon operators on ``kron(cavity, transmon)``."""
    a_c = annihilation(n_fock)
    b_t = annihilation(n_transmon)
    i_c = jnp.eye(n_fock, dtype=jnp.complex128)
    i_t = jnp.eye(n_transmon, dtype=jnp.complex128)
    a = jnp.kron(a_c, i_t)
    b = jnp.kron(i_c, b_t)
    return {"a": a, "adag": a.conj().T, "b": b, "bdag": b.conj().T, "a_c": a_c}


# ---------------------------------------------------------------------------
# Displacements and squeezers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def make_displacement(n_fock: int) -> Callable:
    r"""Return a jitted ``alpha -> D(alpha)`` closure, cached per ``n_fock``.

    :math:`D(\alpha) = \exp(\alpha a^\dagger - \alpha^* a)` of the truncated
    generator, exactly unitary in the truncated space. Built from
    :math:`D(r e^{i\theta}) = e^{i\theta \hat n} D(r) e^{-i\theta \hat n}`:
    conjugation by the diagonal :math:`e^{i\theta \hat n}` commutes with Fock
    truncation, so one eigendecomposition of the fixed matrix
    :math:`i(a^\dagger - a)` serves every :math:`\alpha`. This agrees with a
    dense matrix exponential to round-off and is an order of magnitude faster.

    The polar form is not differentiable at :math:`\alpha = 0`; optimize
    through :func:`make_displacement_apply` instead, which is smooth in
    ``Re alpha`` and ``Im alpha``.
    """
    a = annihilation(n_fock)
    lam, vec = jnp.linalg.eigh(1j * (a.conj().T - a))
    vec_dag = vec.conj().T
    n = jnp.arange(n_fock)

    @jax.jit
    def displace(alpha):
        alpha = jnp.asarray(alpha, dtype=a.dtype)
        d_r = (vec * jnp.exp(-1j * jnp.abs(alpha) * lam)) @ vec_dag
        ph = jnp.exp(1j * jnp.angle(alpha) * n)
        return ph[:, None] * d_r * ph.conj()[None, :]

    return displace


def displacement(n_fock: int, alpha) -> jnp.ndarray:
    r""":math:`D(\alpha)` as an ``(n_fock, n_fock)`` matrix; see :func:`make_displacement`."""
    return make_displacement(n_fock)(alpha)


def make_displacement_apply(n_fock: int) -> Callable:
    r"""Return a jittable ``(psi, alpha, adjoint=False) -> D(alpha) psi``.

    Matrix-free displacement from the exact BCH factorization
    :math:`D(\alpha) = e^{-i x_0 p_0} e^{i\sqrt{2} p_0 \hat{x}} e^{-i\sqrt{2} x_0 \hat{p}}`
    (:math:`\alpha = x_0 + i p_0`), using eigendecompositions of
    :math:`\hat x` and :math:`\hat p` computed once. With
    :math:`W = V_x^\dagger V_p`,

    .. math::
        D(\alpha)\psi = e^{-i x_0 p_0}\, V_x\, e^{i\sqrt2 p_0 w_x} \odot
            \Big(W \big(e^{-i\sqrt2 x_0 w_p} \odot V_p^\dagger \psi\big)\Big),

    three matrix-vector products per state. Exactly unitary in the truncated
    space, smooth in ``Re alpha`` and ``Im alpha``, and gradients flow through
    diagonal phases only. It equals :func:`make_displacement` away from the
    Fock cutoff; the two factorizations differ only at the truncation edge.
    ``psi`` holds row kets on its last axis, shape ``(..., n_fock)``.
    ``adjoint=True`` applies ``D(alpha)^dag``, which in the truncated space is
    *not* ``D(-alpha)`` for this factorization (the two exponentials do not
    commute once truncated), so ECD uses it for the conjugate branch.
    """
    a, adag, _ = cavity_operators(n_fock)
    root2 = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.real(a).dtype))
    w_x, v_x = jnp.linalg.eigh((a + adag) / root2)
    w_p, v_p = jnp.linalg.eigh(1j * (adag - a) / root2)
    w_xp = v_x.conj().T @ v_p  # p-eigenbasis -> x-eigenbasis
    # Row-vector forms: x @ M.T == (M x).T
    vp_dag_t, w_xp_t, vx_t = v_p.conj(), w_xp.T, v_x.T
    vx_dag_t, w_xp_dag_t, vp_t = v_x.conj(), w_xp.conj(), v_p.T

    def apply(psi, alpha, adjoint: bool = False):
        alpha = jnp.asarray(alpha, dtype=a.dtype)
        x_0, p_0 = jnp.real(alpha), jnp.imag(alpha)
        ph_x = jnp.exp(1j * root2 * p_0 * w_x)
        ph_p = jnp.exp(-1j * root2 * x_0 * w_p)
        glob = jnp.exp(-1j * x_0 * p_0)
        if not adjoint:  # D = glob * V_x ph_x W ph_p V_p^dag
            y = (psi @ vp_dag_t) * ph_p
            y = (y @ w_xp_t) * ph_x
            return glob * (y @ vx_t)
        # D^dag = conj(glob) * V_p conj(ph_p) W^dag conj(ph_x) V_x^dag
        y = (psi @ vx_dag_t) * jnp.conj(ph_x)
        y = (y @ w_xp_dag_t) * jnp.conj(ph_p)
        return jnp.conj(glob) * (y @ vp_t)

    return apply


def make_squeeze(n_fock: int) -> Callable:
    """Return a jittable ``(r, phi) -> S(r, phi)`` closure.

    ``S(r, phi) = exp[(r/2)(a^2 e^{-i phi} - a^dag^2 e^{i phi})]`` (Schiaffino,
    Lombardo & Paz, Eq. 5), from the eigendecomposition of its Hermitian
    generator. Use :func:`make_squeeze_apply` when only ``S psi`` is needed.
    """
    a2 = annihilation(n_fock) @ annihilation(n_fock)
    a2d = a2.conj().T

    def squeeze(r, phi):
        gen = 0.5 * r * (a2 * jnp.exp(-1j * phi) - a2d * jnp.exp(1j * phi))  # anti-Hermitian
        lam, v = jnp.linalg.eigh(1j * gen)
        return (v * jnp.exp(-1j * lam)) @ v.conj().T  # exp(G) = exp(-i (iG))

    return squeeze


def make_squeeze_apply(n_fock: int) -> Callable:
    r"""Return a jittable ``(psi, r, phi) -> S(r, phi) psi`` without forming ``S``.

    ``psi`` holds row kets on its last axis, shape ``(..., n_fock)``; ``phi``
    broadcasts against ``psi.shape[:-1]``, so ``phi`` of shape ``(2,)`` applies
    a different phase to each qubit block of a ``(K, 2, n_fock)`` state in one
    call. Uses

    .. math::
        S(r, \varphi) = e^{i\varphi\hat n/2}\, V e^{-i r\lambda} V^\dagger\,
                         e^{-i\varphi\hat n/2},

    where ``(lam, V) = eigh(i/2 (a^2 - a^dag^2))`` is computed once. Exact in
    the truncated space: conjugation by the diagonal ``e^{i theta n}`` maps
    ``a^2 -> e^{-2 i theta} a^2`` with truncated operators too.
    """
    rdtype = jnp.asarray(0.0).dtype
    a = annihilation(n_fock)
    a2 = a @ a
    lam, v = jnp.linalg.eigh(0.5j * (a2 - a2.conj().T))  # Hermitian
    v_conj = v.conj()
    v_t = v.T
    n = jnp.arange(n_fock, dtype=rdtype)

    def apply(psi, r, phi):
        phi = jnp.asarray(phi)[..., None]
        rot = jnp.exp(0.5j * phi * n)                   # e^{i phi n / 2}
        y = (psi * jnp.conj(rot)) @ v_conj              # V^dag e^{-i phi n/2} psi
        y = y * jnp.exp(-1j * r * lam)
        return (y @ v_t) * rot

    return apply


# ---------------------------------------------------------------------------
# Qubit rotations
# ---------------------------------------------------------------------------


def qubit_rotation(theta, phi):
    r"""Equatorial qubit rotation :math:`R(\theta,\phi)` as a 2x2 matrix.

    ``R = exp[-i theta/2 (sigma_x cos phi + sigma_y sin phi)]``, the unitary
    generated by a resonant drive of pulse area ``theta`` and phase ``phi``.
    """
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)
    cos = jnp.cos(theta / 2) + 0j
    sin = jnp.sin(theta / 2) + 0j
    off_lo = -1j * jnp.exp(1j * phi) * sin
    off_hi = -1j * jnp.exp(-1j * phi) * sin
    return jnp.stack([jnp.stack([cos, off_hi]), jnp.stack([off_lo, cos])])
