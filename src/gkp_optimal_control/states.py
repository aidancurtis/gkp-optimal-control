import numpy as np
import qutip as qt


def cat_states(n_fock: int, alpha: complex) -> tuple[qt.Qobj, qt.Qobj]:
    r"""Return normalized even and odd Schrödinger-cat states.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation dimension.
    alpha : complex
        Coherent-state amplitude.

    Returns
    -------
    even : qutip.Qobj
        Normalized even cat, :math:`(|\alpha\rangle + |-\alpha\rangle) / \mathcal{N}_+`.
    odd : qutip.Qobj
        Normalized odd cat, :math:`(|\alpha\rangle - |-\alpha\rangle) / \mathcal{N}_-`.
    """
    even_cat = qt.coherent(n_fock, alpha) + qt.coherent(n_fock, -alpha)
    odd_cat = qt.coherent(n_fock, alpha) - qt.coherent(n_fock, -alpha)
    return even_cat.unit(), odd_cat.unit()


def gkp_states(
    n_fock: int,
    alpha: complex,
    beta: complex,
    delta: float,
    cutoff: int,
) -> tuple[qt.Qobj, qt.Qobj]:
    r"""Return the two finite-energy GKP logical states.

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
    gkp_0 : qutip.Qobj
        Normalized finite-energy logical :math:`|0_L\rangle`.
    gkp_1 : qutip.Qobj
        Normalized finite-energy logical :math:`|1_L\rangle`.
    """
    gkps = [qt.zero_ket(n_fock), qt.zero_ket(n_fock)]
    vac = qt.basis(n_fock, 0)
    envelope = 0.5 * (1.0 - np.exp(-2.0 * delta**2))

    for k in range(-cutoff, cutoff + 1):
        for j in range(-cutoff, cutoff + 1):
            for i in range(2):
                displacement = (2 * k + i) * alpha + j * beta
                d_op = qt.displace(n_fock, displacement)
                peak = np.exp(-1j * np.pi * (k + i / 2) * j) * d_op * vac
                weight = np.exp(-envelope * np.abs(displacement) ** 2)
                gkps[i] += weight * peak

    gkp_0, gkp_1 = gkps
    return gkp_0.unit(), gkp_1.unit()
