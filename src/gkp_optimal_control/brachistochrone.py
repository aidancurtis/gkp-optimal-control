import cmath

import numpy as np
import qutip as qt


def quantum_brachistochrone_hamiltonian(
    initial_state: qt.Qobj,
    final_state: qt.Qobj,
    energy_bound: float = 1.0,
) -> tuple[qt.Qobj, float]:
    r"""Return the time-optimal Hamiltonian driving one state into another.

    Constructs the constant Hamiltonian that evolves ``initial_state`` to
    ``final_state`` (up to a global phase) in the minimum time allowed by the
    Mandelstam-Tamm quantum speed limit, given a spectral-norm bound
    ``energy_bound`` on :math:`H`. The Hamiltonian acts nontrivially only on
    the 2D subspace spanned by the initial state and the component of the
    final state orthogonal to it.

    Parameters
    ----------
    initial_state : qutip.Qobj
        Normalized initial ket :math:`|\psi_i\rangle`.
    final_state : qutip.Qobj
        Normalized target ket :math:`|\psi_f\rangle`. Must not be orthogonal
        to ``initial_state``.
    energy_bound : float, default 1.0
        Spectral-norm bound on the Hamiltonian, :math:`\lVert H \rVert`.

    Returns
    -------
    h_optimal : qutip.Qobj
        Time-optimal Hamiltonian driving :math:`|\psi_i\rangle` to
        :math:`|\psi_f\rangle`.
    min_time : float
        Minimum evolution time :math:`T = \theta_B / \lVert H \rVert`, where
        :math:`\theta_B = \arccos|\langle \psi_i | \psi_f \rangle|` is the
        Bures angle between the two states.

    References
    ----------
    Carlini, A., Hosoya, A., Koike, T., & Okudaira, Y. (2006).
    Time-optimal quantum evolution. *Physical Review Letters*, 96(6), 060503.
    """
    psi_f_perp = (final_state - initial_state.overlap(final_state) * initial_state).unit()
    phi = cmath.phase(initial_state.overlap(final_state))
    bures_angle = np.arccos(np.abs(initial_state.overlap(final_state)))

    sigma_x_eff = initial_state * psi_f_perp.dag() + psi_f_perp * initial_state.dag()
    sigma_y_eff = -1j * (initial_state * psi_f_perp.dag() - psi_f_perp * initial_state.dag())
    h_optimal = energy_bound * (np.sin(phi) * sigma_x_eff + np.cos(phi) * sigma_y_eff)

    min_time = bures_angle / np.abs(energy_bound)
    return h_optimal, min_time
