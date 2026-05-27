from functools import partial

import jax
import jax.numpy as jnp


def _coherent_state_vectors(n_fock: int, alphas: jnp.ndarray) -> jnp.ndarray:
    """Compute coherent-state Fock-basis vectors for a batch of alphas.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation.
    alphas : jnp.ndarray
        Complex array of arbitrary shape ``(...,)`` of displacement amplitudes.
        Scalars are accepted.

    Returns
    -------
    jnp.ndarray
        Array of shape ``(..., n_fock)`` where the trailing axis indexes the
        Fock basis and the leading axes broadcast over the input shape.
    """
    alphas = jnp.asarray(alphas)
    n = jnp.arange(n_fock)
    log_factorial = jax.scipy.special.gammaln(n + 1)  # shape (n_fock,)

    a = alphas[..., None]

    log_abs = jnp.where(jnp.abs(a) > 0, jnp.log(jnp.abs(a)), -jnp.inf)
    log_mag = n * log_abs - 0.5 * log_factorial - 0.5 * jnp.abs(a) ** 2
    phase = jnp.exp(1j * n * jnp.angle(a))

    coeffs = jnp.exp(log_mag) * phase  # shape (..., n_fock)

    # Handle alpha = 0 exactly: |0> coherent state is the Fock vacuum.
    is_zero = (jnp.abs(alphas) == 0)[..., None]  # shape (..., 1)
    vacuum = jnp.zeros(n_fock, dtype=coeffs.dtype).at[0].set(1.0)
    coeffs = jnp.where(is_zero, vacuum, coeffs)

    return coeffs


def cat_states(n_fock: int, alphas: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Compute normalized even and odd Schrödinger-cat states.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation dimension.
    alphas : jnp.ndarray
        Complex array of arbitrary shape ``(...,)`` of cat amplitudes.
        Scalars are accepted.

    Returns
    -------
    even, odd : jnp.ndarray
        Arrays of shape ``(..., n_fock)`` containing the normalized even and
        odd cat states
        :math:`(|\alpha\rangle \pm |-\alpha\rangle) / \mathcal{N}_\pm`.
    """
    alphas = jnp.asarray(alphas)
    # Stack +alpha and -alpha along a new leading axis, then evaluate both
    # in one shot. Shape: (2, ..., n_fock).
    stacked = jnp.stack([alphas, -alphas], axis=0)
    coherents = _coherent_state_vectors(n_fock, stacked)
    plus, minus = coherents[0], coherents[1]  # each (..., n_fock)

    even = plus + minus
    odd = plus - minus

    # Norms along the Fock axis only; broadcast back over leading dims.
    even = even / jnp.linalg.norm(even, axis=-1, keepdims=True)
    odd = odd / jnp.linalg.norm(odd, axis=-1, keepdims=True)
    return even, odd


@partial(jax.jit, static_argnames=("n_fock", "cutoff"))
def gkp_states(
    n_fock: int,
    alpha: complex,
    beta: complex,
    delta: float,
    cutoff: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Return the two finite-energy GKP logical states as Fock-basis vectors.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation dimension.
    alpha : complex
        Primitive lattice displacement along the logical-:math:`Z` axis.
    beta : complex
        Primitive lattice displacement along the logical-:math:`X` axis.
    delta : float
        Envelope width parameter setting the finite-energy cutoff.
    cutoff : int
        Lattice-sum truncation; each state sums over a
        :math:`(2\,\text{cutoff}+1)^2` grid of displaced peaks.

    Returns
    -------
    gkp_0, gkp_1 : jnp.ndarray
        Normalized finite-energy logical :math:`|0_L\rangle` and
        :math:`|1_L\rangle` as 1D Fock-basis vectors of length ``n_fock``.
    """
    ks = jnp.arange(-cutoff, cutoff + 1)
    js = jnp.arange(-cutoff, cutoff + 1)
    k_grid, j_grid = jnp.meshgrid(ks, js, indexing="ij")
    k_flat = k_grid.ravel()
    j_flat = j_grid.ravel()

    envelope = 0.5 * (1.0 - jnp.exp(-2.0 * delta**2))

    def build_logical(i: int) -> jnp.ndarray:
        displacements = (2 * k_flat + i) * alpha + j_flat * beta
        peaks = _coherent_state_vectors(n_fock, displacements)
        phases = jnp.exp(-1j * jnp.pi * (k_flat + i / 2) * j_flat)
        weights = jnp.exp(-envelope * jnp.abs(displacements) ** 2)
        combined = (phases * weights)[:, None] * peaks
        return jnp.sum(combined, axis=0)

    gkp_0 = build_logical(0)
    gkp_1 = build_logical(1)
    gkp_0 = gkp_0 / jnp.linalg.norm(gkp_0)
    gkp_1 = gkp_1 / jnp.linalg.norm(gkp_1)
    return gkp_0, gkp_1


def fock_state(n_fock: int, n: int) -> jnp.ndarray:
    r"""Return the number state :math:`|n\rangle` as a Fock-basis vector.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation dimension.
    n : int
        Photon number, must satisfy ``0 <= n < n_fock``.

    Returns
    -------
    jnp.ndarray
        Unit vector of length ``n_fock`` with a 1 at index ``n``.
    """
    return jnp.zeros(n_fock, dtype=jnp.complex128).at[n].set(1.0)


@partial(jax.jit, static_argnames=("n_fock",))
def squeezed_vacuum(n_fock: int, r) -> jnp.ndarray:
    r"""Return the squeezed-vacuum state :math:`S(r)|0\rangle`.

    The single-mode squeezing operator is
    :math:`S(r) = \exp\!\bigl(\tfrac{1}{2}(r^* a^2 - r\, a^{\dagger 2})\bigr)`.
    For real ``r > 0`` the squeezing is along the position quadrature, with
    variance reduced by a factor :math:`e^{-2r}`.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation dimension (static).
    r : scalar
        Squeezing parameter (real or complex). Traceable.

    Returns
    -------
    jnp.ndarray
        Normalized squeezed-vacuum ket of length ``n_fock``.

    Notes
    -----
    Computed as a direct matrix exponential rather than via the
    closed-form Fock-basis expansion, so the implementation is short and
    differentiable in ``r``. The matrix exponential of a small antihermitian
    operator is well-conditioned at typical squeezing levels.
    """
    n = jnp.arange(n_fock)
    a = jnp.diag(jnp.sqrt(n[1:]), k=1).astype(jnp.complex128)
    a2 = a @ a
    adag2 = a2.conj().T
    r_c = jnp.asarray(r, dtype=jnp.complex128)
    generator = 0.5 * (jnp.conj(r_c) * a2 - r_c * adag2)
    squeeze = jax.scipy.linalg.expm(generator)
    psi = squeeze[:, 0]  # S @ |0> picks out the first column
    return psi / jnp.linalg.norm(psi)
