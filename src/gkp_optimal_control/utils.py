from typing import Any

import numpy as np
import qutip as qt


def compute_wigner(
    state: qt.Qobj,
    x_bound: float,
    y_bound: float,
    grid_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute the Wigner distribution of a bosonic state on a square grid.

    Single source of truth for grid construction: all Wigner computations in
    this package route through here so that static plots, trajectories, and
    animations use identical axes.

    Parameters
    ----------
    state : qutip.Qobj
        Ket or density matrix of the bosonic state.
    x_bound : float
        Half-width of the :math:`q`-axis; the grid spans
        :math:`[-x_{\text{bound}}, x_{\text{bound}}]`.
    y_bound : float
        Half-width of the :math:`p`-axis; the grid spans
        :math:`[-y_{\text{bound}}, y_{\text{bound}}]`.
    grid_points : int, default 200
        Number of samples along each phase-space axis.

    Returns
    -------
    xvec : numpy.ndarray
        1D :math:`q`-axis samples.
    yvec : numpy.ndarray
        1D :math:`p`-axis samples.
    wigner : numpy.ndarray
        2D Wigner distribution of shape ``(grid_points, grid_points)``.

    Raises
    ------
    ValueError
        If the Wigner function could not be computed for ``state``.
    """
    xvec = np.linspace(-x_bound, x_bound, grid_points)
    yvec = np.linspace(-y_bound, y_bound, grid_points)
    wigner = qt.wigner(state, xvec, yvec)

    if not isinstance(wigner, np.ndarray):
        raise ValueError("Could not compute Wigner function for the given state.")

    return xvec, yvec, wigner


def wigner_trajectory(
    result,
    x_bound: float,
    y_bound: float,
    grid_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    r"""Compute one Wigner distribution per state in a QuTiP solver result.

    Parameters
    ----------
    result : qutip.solver.Result
        Solver result exposing a ``states`` attribute containing a sequence
        of kets or density matrices.
    x_bound : float
        Half-width of the :math:`q`-axis; the grid spans
        :math:`[-x_{\text{bound}}, x_{\text{bound}}]`.
    y_bound : float
        Half-width of the :math:`p`-axis; the grid spans
        :math:`[-y_{\text{bound}}, y_{\text{bound}}]`.
    grid_points : int, default 100
        Number of samples along each phase-space axis.

    Returns
    -------
    xvec : numpy.ndarray
        1D :math:`q`-axis samples shared by every frame.
    yvec : numpy.ndarray
        1D :math:`p`-axis samples shared by every frame.
    wigner_states : list of numpy.ndarray
        One 2D Wigner distribution per state in ``result.states``, each of
        shape ``(grid_points, grid_points)``.
    """
    xvec = np.linspace(-x_bound, x_bound, grid_points)
    yvec = np.linspace(-y_bound, y_bound, grid_points)
    wigner_states = [qt.wigner(state, xvec, yvec) for state in result.states]
    return xvec, yvec, wigner_states


def assemble_qt_hamiltonian(
    drift: qt.Qobj,
    controls: list[qt.Qobj],
    pulses: np.ndarray,
    pulse_time: float,
) -> tuple[list[qt.Qobj | list], np.ndarray]:
    r"""Assemble a QuTiP list-format time-dependent Hamiltonian and matching tlist.

    Parameters
    ----------
    drift : qutip.Qobj
        Time-independent drift Hamiltonian.
    controls : sequence of qutip.Qobj
        Control Hamiltonians, one per control channel.
    pulses : numpy.ndarray
        Control amplitudes with shape ``(n_timesteps, n_controls)``.
    pulse_time : float
        Total evolution time spanned by the pulses.

    Returns
    -------
    h_tot : list
        QuTiP list-format Hamiltonian ``[drift, [H_c0, pulse_0], ...]``
        suitable for passing to ``qt.mesolve`` or ``qt.sesolve``.
    tlist : numpy.ndarray
        1D array of ``n_timesteps`` evenly spaced time points from
        ``0`` to ``pulse_time``.

    Raises
    ------
    ValueError
        If ``pulses`` is not 2D, or if its second dimension does not match
        the number of control Hamiltonians.
    """
    controls = list(controls)
    if pulses.ndim != 2:
        raise ValueError(
            f"pulses must be 2D with shape (n_timesteps, n_controls); got {pulses.shape}."
        )
    if pulses.shape[1] != len(controls):
        raise ValueError(
            f"pulses has {pulses.shape[1]} control channels but {len(controls)} "
            f"control Hamiltonians were provided."
        )

    h_tot: list[qt.Qobj | list] = [drift]
    for i, hc in enumerate(controls):
        h_tot.append([hc, pulses[:, i]])

    tlist = np.linspace(0.0, pulse_time, pulses.shape[0])
    return h_tot, tlist
