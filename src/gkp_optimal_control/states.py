r"""Cavity state constructors.

Every function returns a normalized ket as a plain 1-D complex JAX array of
shape ``(n_fock,)``. That is the one state representation used across the
package: it works directly with ``jax.numpy``, and every library function
accepts it (they also accept a ``jaxquantum.Qarray`` or a column vector and
convert on entry). To hand a state to jaxquantum itself, wrap it with
:func:`utils.to_qarray`.
"""

from functools import partial

import jax
import jax.numpy as jnp


def _unit(ket: jnp.ndarray) -> jnp.ndarray:
    return ket / jnp.linalg.norm(ket)


def cat_states(n_fock: int, alpha: complex) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Return normalized even and odd Schrödinger-cat states.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation dimension.
    alpha : complex
        Coherent-state amplitude.

    Returns
    -------
    even : jnp.ndarray
        Normalized even cat, :math:`(|\alpha\rangle + |-\alpha\rangle) / \mathcal{N}_+`,
        shape ``(n_fock,)``.
    odd : jnp.ndarray
        Normalized odd cat, :math:`(|\alpha\rangle - |-\alpha\rangle) / \mathcal{N}_-`.
    """
    plus, minus = coherent_state_vectors(n_fock, jnp.asarray([alpha, -alpha], dtype=complex))
    return _unit(plus + minus), _unit(plus - minus)


def coherent_state_vectors(n_fock: int, alphas: jnp.ndarray) -> jnp.ndarray:
    """Compute coherent-state Fock-basis column vectors for a batch of alphas.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation.
    alphas : jnp.ndarray
        Complex array of shape (K,) of displacement amplitudes.

    Returns
    -------
    jnp.ndarray
        Array of shape (K, n_fock) where row k is |alpha_k> in the Fock basis.
    """
    # log|n!| via lgamma(n+1) for numerical stability when n_fock is large.
    n = jnp.arange(n_fock)
    log_factorial = jax.scipy.special.gammaln(n + 1)  # shape (n_fock,)

    # alphas: (K,) -> (K, 1); n: (n_fock,) -> (1, n_fock)
    a = alphas[:, None]
    n_row = n[None, :]

    # Work in log space for the magnitude, keep phase separately.
    # |alpha|^n / sqrt(n!) = exp(n*log|alpha| - 0.5*log(n!))
    # Combined with the e^{-|alpha|^2/2} prefactor and alpha^n phase.
    log_abs = jnp.where(jnp.abs(a) > 0, jnp.log(jnp.abs(a)), -jnp.inf)
    log_mag = n_row * log_abs - 0.5 * log_factorial - 0.5 * jnp.abs(a) ** 2
    phase = jnp.exp(1j * n_row * jnp.angle(a))

    coeffs = jnp.exp(log_mag) * phase  # shape (K, n_fock)

    # Handle alpha = 0 exactly: |0> coherent state is the Fock vacuum.
    is_zero = jnp.abs(alphas) == 0
    vacuum = jnp.zeros(n_fock, dtype=coeffs.dtype).at[0].set(1.0)
    coeffs = jnp.where(is_zero[:, None], vacuum[None, :], coeffs)

    return coeffs


@partial(jax.jit, static_argnames=("n_fock", "cutoff"))
def gkp_states(
    n_fock: int,
    alpha: complex,
    beta: complex,
    delta: float,
    cutoff: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Return the two finite-energy GKP logical states.

    The states are :math:`E_\Delta|\mu\rangle` with
    :math:`E_\Delta = e^{-\Delta^2 \hat n}`, the convention of Eickbusch et al.
    and Sivak et al. They are built from the coherent-state lattice expansion
    of the ideal state, using the exact identity

    .. math::
        e^{-\Delta^2 \hat n}\,|d\rangle
            = e^{-\frac{1}{2}(1 - e^{-2\Delta^2})|d|^2}\,|e^{-\Delta^2} d\rangle,

    so that

    .. math::
        |\mu_\Delta\rangle \propto \sum_{k,j} e^{i\phi_{kj}}\,
            e^{-\frac{1}{2}(1 - e^{-2\Delta^2})|d_{kj}|^2}\,
            |e^{-\Delta^2} d_{kj}\rangle,
        \qquad d_{kj} = (2k + \mu)\,\alpha + j\,\beta.

    The Gaussian weight is evaluated at the lattice point :math:`d_{kj}`, but
    each coherent state is centred at the contracted point
    :math:`e^{-\Delta^2} d_{kj}`. Dropping the contraction gives a different
    state: narrower peaks, a broader envelope and roughly 0.8 extra photons,
    with overlap 0.991--0.998 for :math:`\Delta = 0.35`--:math:`0.25`. With
    it, the result equals :math:`E_\Delta|\mu\rangle` up to the lattice and
    Fock truncations; at ``cutoff=10`` it matches an independent Hermite-function
    construction to machine precision for :math:`\Delta \ge 0.25`.

    In position space, each peak has variance :math:`\tanh(\Delta^2)/2`,
    sits at :math:`x_s\,\mathrm{sech}(\Delta^2)`, and the peak weights fall
    off as :math:`e^{-\tanh(\Delta^2)\,x_s^2}`.

    The two returned states are each normalized but not mutually orthogonal:
    finite-energy :math:`|0_L\rangle` and :math:`|1_L\rangle` overlap slightly
    (about :math:`10^{-5}` at :math:`\Delta = 0.25`, :math:`3\times10^{-3}` at
    :math:`\Delta = 0.35`). Orthonormalize them before using them as a logical
    basis.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation dimension.
    alpha : complex
        Primitive lattice displacement along the logical-:math:`Z` axis.
    beta : complex
        Primitive lattice displacement along the logical-:math:`X` axis.
    delta : float
        Finite-energy parameter :math:`\Delta` in :math:`E_\Delta = e^{-\Delta^2 \hat n}`.
    cutoff : int
        Lattice-sum truncation; each state sums over a
        :math:`(2\,\text{cutoff}+1)^2` grid of displaced peaks.

    Returns
    -------
    gkp_0 : jnp.ndarray
        Normalized finite-energy logical :math:`|0_L\rangle`, shape ``(n_fock,)``.
    gkp_1 : jnp.ndarray
        Normalized finite-energy logical :math:`|1_L\rangle`.
    """
    # Build the 2D grid of lattice indices (k, j) once.
    ks = jnp.arange(-cutoff, cutoff + 1)
    js = jnp.arange(-cutoff, cutoff + 1)
    k_grid, j_grid = jnp.meshgrid(ks, js, indexing="ij")
    k_flat = k_grid.ravel()  # shape (M,)
    j_flat = j_grid.ravel()  # shape (M,)

    envelope = 0.5 * (1.0 - jnp.exp(-2.0 * delta**2))
    contraction = jnp.exp(-(delta**2))

    def build_logical(i: int) -> jnp.ndarray:
        """Build the i-th logical state (i = 0 or 1) as a Fock-basis vector."""
        # Displacement amplitudes for every (k, j) on the grid.
        displacements = (2 * k_flat + i) * alpha + j_flat * beta  # (M,)

        # Peaks: each row is the coherent state |e^{-Delta^2} d> in the Fock
        # basis, i.e. E_Delta |d> up to the scalar weight applied below.
        peaks = coherent_state_vectors(n_fock, contraction * displacements)  # (M, n_fock)

        # Phase factor that alternates the sublattice signs.
        phases = jnp.exp(-1j * jnp.pi * (k_flat + i / 2) * j_flat)  # (M,)

        # Scalar factor from E_Delta |d>, evaluated at the uncontracted d.
        weights = jnp.exp(-envelope * jnp.abs(displacements) ** 2)  # (M,)

        # Weighted sum over the lattice.
        combined = (phases * weights)[:, None] * peaks  # (M, n_fock)
        return jnp.sum(combined, axis=0)  # (n_fock,)

    return _unit(build_logical(0)), _unit(build_logical(1))


@partial(jax.jit, static_argnames=("n_fock",))
def squeezed_vacuum(n_fock: int, r: float, phi: float = 0.0) -> jnp.ndarray:
    r"""Return the squeezed vacuum :math:`S(\xi)|0\rangle`, :math:`\xi = r e^{i\phi}`.

    Uses :math:`S(\xi) = \exp[\tfrac{1}{2}(\xi^* a^2 - \xi a^{\dagger 2})]`, the same
    convention as the squeeze gate in ``gate_optimization``. For ``phi = 0`` and
    ``r > 0`` the state is squeezed in :math:`x`.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation dimension.
    r : float
        Squeezing parameter (squeezing in dB is ``20 * r * log10(e)``).
    phi : float, optional
        Squeezing angle.

    Returns
    -------
    jnp.ndarray
        Normalized squeezed vacuum (renormalized after truncation), shape ``(n_fock,)``.
    """
    m = jnp.arange((n_fock + 1) // 2)
    log_mag = (
        0.5 * jax.scipy.special.gammaln(2 * m + 1)
        - jax.scipy.special.gammaln(m + 1)
        - m * jnp.log(2.0)
        - 0.5 * jnp.log(jnp.cosh(r))
    )
    coeffs = jnp.exp(log_mag) * (-jnp.exp(1j * phi) * jnp.tanh(r)) ** m

    return _unit(jnp.zeros(n_fock, dtype=coeffs.dtype).at[2 * m].set(coeffs))


def fock_state(n_fock: int, n: int) -> jnp.ndarray:
    r"""Return the Fock state :math:`|n\rangle`.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation dimension.
    n : int
        Photon number, with ``0 <= n < n_fock``.

    Returns
    -------
    jnp.ndarray
        The basis ket :math:`|n\rangle`, shape ``(n_fock,)``.
    """
    if not 0 <= n < n_fock:
        raise ValueError(f"n={n} is outside the truncated space [0, {n_fock}).")
    return jnp.zeros(n_fock, dtype=jnp.complex128).at[n].set(1.0)