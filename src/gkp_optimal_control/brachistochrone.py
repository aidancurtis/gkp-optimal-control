import jax
import jax.numpy as jnp

from .utils import StateLike, to_ket


def quantum_brachistochrone_hamiltonian(
    initial_state: StateLike,
    final_state: StateLike,
    energy_bound: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Return the time-optimal Hamiltonian driving one state into another.

    Constructs the constant Hamiltonian that evolves ``initial_state`` to
    ``final_state`` (up to a global phase) in the minimum time allowed by the
    Mandelstam-Tamm quantum speed limit, given a spectral-norm bound
    ``energy_bound`` on :math:`H`. The Hamiltonian acts nontrivially only on
    the 2D subspace spanned by the initial state and the component of the
    final state orthogonal to it.

    Parameters
    ----------
    initial_state : array_like or jaxquantum.Qarray
        Initial ket :math:`|\psi_i\rangle`; any ket shape, normalized on entry.
    final_state : array_like or jaxquantum.Qarray
        Target ket :math:`|\psi_f\rangle`; any ket shape, normalized on entry.
        Must not be orthogonal to ``initial_state``.
    energy_bound : float, default 1.0
        Spectral-norm bound on the Hamiltonian, :math:`\lVert H \rVert`.

    Returns
    -------
    h_optimal : jnp.ndarray
        Time-optimal Hamiltonian driving :math:`|\psi_i\rangle` to
        :math:`|\psi_f\rangle`, shape ``(dim, dim)``.
    min_time : jnp.ndarray
        Minimum evolution time :math:`T = \theta_B / \lVert H \rVert`, where
        :math:`\theta_B = \arccos|\langle \psi_i | \psi_f \rangle|` is the
        Bures angle between the two states.

    References
    ----------
    Yang and Del Campo
    """
    return _brachistochrone(to_ket(initial_state), to_ket(final_state), energy_bound)


@jax.jit
def _brachistochrone(initial_state, final_state, energy_bound):
    # Overlap <psi_i | psi_f> as a complex scalar.
    # jnp.vdot conjugates the first argument, matching the bra-ket convention.
    overlap = jnp.vdot(initial_state, final_state)

    # Orthogonal component of |psi_f> relative to |psi_i>, normalized.
    perp = final_state - overlap * initial_state
    psi_f_perp = perp / jnp.linalg.norm(perp)

    # Phase of the overlap and the Bures angle. arccos's argument is clipped
    # to [0, 1] to stay safe near (anti-)parallel states.
    phi = jnp.angle(overlap)
    overlap_abs = jnp.clip(jnp.abs(overlap), 0.0, 1.0)
    bures_angle = jnp.arccos(overlap_abs)

    # Effective Pauli operators inside the 2D subspace {|psi_i>, |psi_f_perp>}.
    # outer(a, b.conj()) = |a><b|.
    proj_if = jnp.outer(initial_state, psi_f_perp.conj())
    proj_fi = jnp.outer(psi_f_perp, initial_state.conj())

    sigma_x_eff = proj_if + proj_fi
    sigma_y_eff = -1j * (proj_if - proj_fi)

    h_optimal = energy_bound * (jnp.sin(phi) * sigma_x_eff + jnp.cos(phi) * sigma_y_eff)
    min_time = bures_angle / jnp.abs(energy_bound)

    return h_optimal, min_time
