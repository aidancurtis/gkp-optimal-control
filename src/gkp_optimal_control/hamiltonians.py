import jax.numpy as jnp

from .operators import cavity_operators, joint_operators, quadrature_operators


def kerr_cavity_drift(n_fock: int, kerr: float):
    """H = (K/2) (a†)² a²  on a single Kerr cavity. K in rad/(time-unit)."""
    a, adag, _ = cavity_operators(n_fock)
    return (kerr / 2) * (adag @ adag @ a @ a)


def kerr_cavity_squeezing_controls(n_fock: int):
    """Two-photon (squeezing) drives:  a²+a†²  and  i(a²−a†²)."""
    a, adag, _ = cavity_operators(n_fock)
    h_i = a @ a + adag @ adag
    h_q = 1j * (a @ a - adag @ adag)
    return jnp.stack([h_i, h_q])


def cavity_displacement_controls(n_fock: int):
    """Linear (displacement) drives:  a+a†  and  i(a−a†)."""
    a, adag, _ = cavity_operators(n_fock)
    h_i = a + adag
    h_q = 1j * (a - adag)
    return jnp.stack([h_i, h_q])


def cavity_transmon_drift(n_cav: int, n_tr: int, chi: float, k: float = 0.0, alpha: float = 0.0):
    """Dispersive cavity-transmon system in rotating frame, on resonance."""
    o = joint_operators(n_cav, n_tr)
    a, adag, b, bdag = o["a"], o["adag"], o["b"], o["bdag"]
    n_c = adag @ a
    n_t = bdag @ b

    sys_h = chi * (n_c @ n_t)
    if k != 0.0:
        sys_h = sys_h + (k / 2) * (adag @ adag @ a @ a)
    if alpha != 0.0:
        sys_h = sys_h + (alpha / 2) * (bdag @ bdag @ b @ b)
    return sys_h


def cavity_transmon_iq_controls(n_cav: int, n_tr: int):
    """I/Q drives on both cavity and transmon."""
    o = joint_operators(n_cav, n_tr)
    a, adag, b, bdag = o["a"], o["adag"], o["b"], o["bdag"]
    return jnp.stack(
        [
            a + adag,
            1j * (a - adag),
            b + bdag,
            1j * (b - bdag),
        ]
    )


def gkp_cosine_hamiltonian(n_fock: int, n: int = 2, symmetric: bool = True, pad: int = 3):
    r"""H = 2 − [cos p + cos(2π n q)]  on a truncated cavity.

    cos p shifts q by ±1 and cos(2πnq) shifts p by ±2πn; for integer n these
    commute, so the zero-energy subspace is an n-dimensional GKP code with
    codewords |j⟩ ∝ Σ_k |q = k + j/n⟩.

    Parameters
    ----------
    n_fock : int
        Fock truncation. Since the two cosines commute, H has a continuous
        spectrum and no normalisable ground state: the truncation is what sets
        the finite-energy envelope of the low-lying eigenstates.
    n : int
        Code dimension (n = 2 is a GKP qubit).
    symmetric : bool, default True
        If True, (q, p) are the canonically rescaled quadratures
        q = x/√(2πn), p = √(2πn) p_x  with x = (a+a†)/√2, p_x = -i(a−a†)/√2,
        so H = 2 − cos(√(2πn) p_x) − cos(√(2πn) x): a square lattice in the
        cavity's own phase space (for n = 2 the √π lattice of ``gkp_states``).
        If False, q and p are taken literally as x and p_x, which gives a
        strongly anisotropic lattice (spacing 1/n in x, 2π in p_x) needing far
        more photons.
    pad : int, default 3
        Cosines are exponentiated in a ``pad × n_fock`` space and cropped so
        their matrix elements are accurate inside the kept space.
    """
    from jax.scipy.linalg import expm

    n_big = pad * n_fock
    x, p_x = quadrature_operators(n_big)
    s = jnp.sqrt(2 * jnp.pi * n) if symmetric else 1.0
    q, p = x / s, p_x * s

    e_p = expm(1j * p)
    e_q = expm(1j * 2 * jnp.pi * n * q)
    cos_p = 0.5 * (e_p + e_p.conj().T)
    cos_q = 0.5 * (e_q + e_q.conj().T)
    h = (2 * jnp.eye(n_big) - cos_p - cos_q)[:n_fock, :n_fock]
    return 0.5 * (h + h.conj().T)
