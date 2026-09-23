r"""Time-dependent simulation of compiled pulse sequences, and Wigner readout.

Takes a :class:`pulses.PulseSequence` and propagates the joint cavity-transmon
state under the time-dependent Hamiltonian of Eq. (S2) of Eickbusch et al.,
then hands the trajectory to :func:`utils.wigner_trajectory`.

Why the displaced frame
-----------------------
ECD control drives the oscillator to :math:`|\alpha|^2 \sim 10^3` photons, so a
lab-frame simulation would need ``n_fock`` of order 1500 and a joint dimension
of 3000. Instead the classical response :math:`\alpha(t)` is solved exactly
(Eq. S3) and factored out, :math:`|\psi\rangle = D(\alpha(t))
|\tilde\psi(t)\rangle`. What is left in the frame is the *residual* state,
which stays within a few photons of the origin, so ``n_fock ~ 30-60`` suffices.

Two consequences worth internalizing:

* The frame does **not** close. :math:`\alpha(T) \neq 0` in general, because the
  compiled amplitude ratios null the mean of the two *conditional*
  trajectories, while :math:`\alpha(t)` follows the ground branch alone. The
  physical final state is :math:`D(\alpha(T))|\tilde\psi(T)\rangle`, and
  forgetting this costs several parts in :math:`10^4` of fidelity even though
  :math:`|\alpha(T)|` is only a few percent.
* Lab-frame Wigner functions are a *coordinate shift* of the co-moving ones:
  :math:`W_\mathrm{lab}(x, p) = \tilde W(x - \sqrt2\,\mathrm{Re}\,\alpha,
  p - \sqrt2\,\mathrm{Im}\,\alpha)`. There is no need to build a huge Hilbert
  space to see the lab-frame picture; :func:`wigner_frames` returns the offsets.

Conventions
-----------
Operator ordering is ``kron(cavity, transmon)``, i.e. joint index
``n_cav * n_transmon + s``, matching :func:`hamiltonians.cavity_transmon_drift`
and :func:`gate_optimization.to_joint_ket`.

The Hamiltonian carries the paper's signs (:math:`-\chi a^\dagger a q^\dagger q`,
:math:`-K_c a^{\dagger 2} a^2`); see :meth:`pulses.SystemParams.drift_kwargs`
for the translation to :mod:`hamiltonians`.

Every time-dependent coefficient is evaluated at the *midpoint* of its step and
every operator in the decomposition is Hermitian with a real coefficient, so
the propagator is unitary by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import expm

try:  # package-relative imports, with a flat-layout fallback
    from . import pulses as _pulses
    from .utils import wigner_trajectory
except ImportError:  # pragma: no cover
    import pulses as _pulses
    from utils import wigner_trajectory

SystemParams = _pulses.SystemParams
PulseSequence = _pulses.PulseSequence

__all__ = [
    "SimResult",
    "joint_operators",
    "hamiltonian_terms",
    "simulate_sequence",
    "gate_boundary_states",
    "wigner_frames",
    "lab_extent",
]

_SQRT2 = np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def joint_operators(n_fock: int, n_transmon: int = 2) -> dict:
    """Cavity and transmon operators on ``kron(cavity, transmon)``."""
    a_c = jnp.diag(jnp.sqrt(jnp.arange(1, n_fock, dtype=jnp.complex128)), k=1)
    b_t = jnp.diag(jnp.sqrt(jnp.arange(1, n_transmon, dtype=jnp.complex128)), k=1)
    i_c = jnp.eye(n_fock, dtype=jnp.complex128)
    i_t = jnp.eye(n_transmon, dtype=jnp.complex128)
    a = jnp.kron(a_c, i_t)
    b = jnp.kron(i_c, b_t)
    return {"a": a, "adag": a.conj().T, "b": b, "bdag": b.conj().T, "a_c": a_c}


def hamiltonian_terms(
    params: SystemParams,
    n_fock: int,
    n_transmon: int = 2,
):
    r"""Hermitian operator basis for the displaced-frame Hamiltonian.

    Returns ``(h_static, ops)`` where ``ops`` is a ``(n_ops, dim, dim)`` stack
    of Hermitian operators whose real coefficients are produced by
    :func:`_coefficients`. Splitting the Hamiltonian this way means the
    per-step assembly is a single ``einsum`` and no complex bookkeeping happens
    inside the propagation loop.

    Term order (Eq. S2):

    ==== ====================================  ==========================================
    idx  operator                              coefficient
    ==== ====================================  ==========================================
    0    :math:`n_c`                           :math:`\Delta - 4 K_c|\alpha|^2`
    1    :math:`n_c n_t`                       :math:`-4\chi'|\alpha|^2`
    2    :math:`n_t`                           :math:`-(\chi|\alpha|^2 + \chi'|\alpha|^4)`
    3    :math:`(a + a^\dagger) n_t`           :math:`-(\chi + 2\chi'|\alpha|^2)\mathrm{Re}\,\alpha`
    4    :math:`i(a^\dagger - a) n_t`          :math:`-(\chi + 2\chi'|\alpha|^2)\mathrm{Im}\,\alpha`
    5    :math:`a^{\dagger 2}a + \mathrm{h.c.}`         :math:`-2 K_c \mathrm{Re}\,\alpha`
    6    :math:`i(a^\dagger a^2 - a^{\dagger 2}a)`      :math:`+2 K_c \mathrm{Im}\,\alpha`
    7    :math:`a^{\dagger 2} + a^2`                    :math:`-K_c \mathrm{Re}\,\alpha^2`
    8    :math:`i(a^{\dagger 2} - a^2)`                 :math:`-K_c \mathrm{Im}\,\alpha^2`
    9-12 same four, times :math:`n_t`                   same with :math:`K_c \to \chi'`
    13   :math:`a + a^\dagger`                 :math:`\mathrm{Re}\,A`  (residual drive)
    14   :math:`i(a^\dagger - a)`              :math:`\mathrm{Im}\,A`
    15   :math:`b + b^\dagger`                 :math:`\mathrm{Re}\,\Omega`
    16   :math:`i(b^\dagger - b)`              :math:`\mathrm{Im}\,\Omega`
    ==== ====================================  ==========================================

    The static part holds :math:`-\chi n_c n_t`, :math:`-K_c a^{\dagger2}a^2`,
    :math:`-\chi' a^{\dagger2}a^2 n_t` and the transmon anharmonicity.
    """
    o = joint_operators(n_fock, n_transmon)
    a, adag, b, bdag = o["a"], o["adag"], o["b"], o["bdag"]
    n_c = adag @ a
    n_t = bdag @ b
    a2, ad2 = a @ a, adag @ adag
    quart = ad2 @ a2
    cube = ad2 @ a

    x_lin = a + adag
    y_lin = 1j * (adag - a)
    cx = cube + cube.conj().T
    cy = 1j * (cube.conj().T - cube)
    sx = ad2 + a2
    sy = 1j * (ad2 - a2)

    p = params
    h_static = -p.chi * (n_c @ n_t) - p.kerr * quart - p.chi_prime * (quart @ n_t)
    if n_transmon > 2:
        h_static = h_static - 0.5 * p.anharm * (bdag @ bdag @ b @ b)

    ops = jnp.stack(
        [
            n_c,
            n_c @ n_t,
            n_t,
            x_lin @ n_t,
            y_lin @ n_t,
            cx,
            cy,
            sx,
            sy,
            cx @ n_t,
            cy @ n_t,
            sx @ n_t,
            sy @ n_t,
            x_lin,
            y_lin,
            b + bdag,
            1j * (bdag - b),
        ]
    )
    return h_static, ops


def _coefficients(
    alpha: np.ndarray,
    dalpha: np.ndarray,
    eps: np.ndarray,
    omega: np.ndarray,
    params: SystemParams,
    delta: float,
    kappa_h: float,
) -> np.ndarray:
    """Real coefficients of :func:`hamiltonian_terms`, shape ``(n_steps, 17)``.

    Built in NumPy: this is a cheap elementwise pass over the waveform and is
    not differentiated.
    """
    p = params
    n = np.abs(alpha) ** 2
    xa, ya = np.real(alpha), np.imag(alpha)
    a2 = alpha**2
    cond = -(p.chi + 2.0 * p.chi_prime * n)

    # Residual linear drive. Identically zero when alpha solves the frame ODE
    # with the same kappa the Hamiltonian carries; kept so that any mismatch
    # (e.g. a loss-free frame with a lossy compilation) stays exact rather than
    # silently wrong.
    amp = (
        delta * alpha
        - 2.0 * p.kerr * n * alpha
        - 1j * dalpha
        - 1j * 0.5 * kappa_h * alpha
        + eps
    )

    return np.stack(
        [
            delta - 4.0 * p.kerr * n,
            -4.0 * p.chi_prime * n,
            -(p.chi * n + p.chi_prime * n * n),
            cond * xa,
            cond * ya,
            -2.0 * p.kerr * xa,
            +2.0 * p.kerr * ya,
            -p.kerr * np.real(a2),
            -p.kerr * np.imag(a2),
            -2.0 * p.chi_prime * xa,
            +2.0 * p.chi_prime * ya,
            -p.chi_prime * np.real(a2),
            -p.chi_prime * np.imag(a2),
            np.real(amp),
            np.imag(amp),
            np.real(omega),
            np.imag(omega),
        ],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    """Output of :func:`simulate_sequence`.

    Attributes
    ----------
    states : ndarray, complex, shape (n_save, dim)
        Displaced-frame joint states at the saved times. Multiply by
        ``D(alpha)`` to get the lab-frame state.
    save_indices : ndarray
        Waveform sample index of each saved state; index 0 is the input state.
    times : ndarray
        Saved times in us.
    alpha : ndarray, complex, shape (n_save,)
        Frame trajectory at the saved times.
    alpha_full : ndarray, complex, shape (n_samples + 1,)
        Frame trajectory on the full grid.
    n_fock, n_transmon : int
    frame : str
        ``"displaced"`` or ``"lab"``.
    seq : PulseSequence
    """

    states: np.ndarray
    save_indices: np.ndarray
    times: np.ndarray
    alpha: np.ndarray
    alpha_full: np.ndarray
    n_fock: int
    n_transmon: int
    frame: str
    seq: PulseSequence = field(repr=False, default=None)

    # -- state views -----------------------------------------------------
    def blocks(self) -> np.ndarray:
        """States as ``(n_save, n_transmon, n_fock)``."""
        arr = self.states.reshape(-1, self.n_fock, self.n_transmon)
        return np.swapaxes(arr, 1, 2)

    def cavity_states(self, project: str = "ground", normalize: bool = True) -> np.ndarray:
        """Cavity states along the trajectory.

        Parameters
        ----------
        project : {"ground", "traced"}
            ``"ground"`` returns the ``|g>`` component as kets, which is what
            the experiment postselects on and what the reported fidelities use.
            ``"traced"`` returns cavity density matrices with the transmon
            traced out, appropriate whenever the transmon is still entangled
            (i.e. anywhere mid-sequence).
        normalize : bool
            Renormalize the projected states. The unnormalized norm is
            :math:`P(g)`, available from :meth:`p_ground`.
        """
        blk = self.blocks()
        if project == "ground":
            psi = blk[:, 0, :]
            if normalize:
                nrm = np.linalg.norm(psi, axis=-1, keepdims=True)
                psi = psi / np.where(nrm == 0, 1.0, nrm)
            return psi
        if project == "traced":
            return np.einsum("tsi,tsj->tij", blk, blk.conj())
        raise ValueError(f"project must be 'ground' or 'traced'; got {project!r}")

    def p_ground(self) -> np.ndarray:
        """Transmon ground-state population along the trajectory."""
        return np.linalg.norm(self.blocks()[:, 0, :], axis=-1) ** 2

    def photon_number(self, lab: bool = True) -> np.ndarray:
        r"""Mean photon number.

        With ``lab=True`` this is the physical
        :math:`\langle a^\dagger a\rangle = \langle \tilde a^\dagger \tilde a
        \rangle + 2\mathrm{Re}[\alpha^*\langle \tilde a\rangle] + |\alpha|^2`,
        which is the quantity plotted in Fig. 2c and reaches
        :math:`\sim 900` at :math:`\alpha_0 = 30`. With ``lab=False`` it is the
        residual occupation in the frame, a useful truncation diagnostic.
        """
        blk = self.blocks()
        n_vec = np.arange(self.n_fock)
        pop = np.abs(blk) ** 2
        n_res = np.einsum("tsi,i->t", pop, n_vec)
        if not lab:
            return n_res
        a_c = np.diag(np.sqrt(np.arange(1, self.n_fock)), k=1)
        a_exp = np.einsum("tsi,ij,tsj->t", blk.conj(), a_c, blk)
        return n_res + 2.0 * np.real(np.conj(self.alpha) * a_exp) + np.abs(self.alpha) ** 2

    def truncation_error(self, n_tail: int = 4) -> float:
        """Peak population in the top ``n_tail`` Fock levels of the frame.

        The single number to check before trusting a run: if this is not tiny,
        raise ``n_fock``.
        """
        pop = np.abs(self.blocks()) ** 2
        return float(pop[:, :, -n_tail:].sum(axis=(1, 2)).max())

    # -- endpoint helpers -------------------------------------------------
    def final_cavity_state(self, project: str = "ground", apply_frame: bool = True):
        r"""Final cavity state, with the frame displacement :math:`D(\alpha(T))`.

        ``apply_frame=True`` is almost always what you want; see the module
        docstring. The displacement is applied in the truncated space, which is
        accurate only for small :math:`|\alpha(T)|` -- a warning is raised
        otherwise.
        """
        psi = self.cavity_states(project=project)[-1]
        if not apply_frame:
            return psi
        alpha_t = complex(self.alpha[-1])
        if abs(alpha_t) > 1.0:
            import warnings

            warnings.warn(
                f"|alpha(T)| = {abs(alpha_t):.2f}; applying D(alpha(T)) inside a "
                f"{self.n_fock}-level truncation is inaccurate. Compare in the "
                "co-moving frame instead, or displace the target state.",
                stacklevel=2,
            )
        d = np.asarray(_displace_np(self.n_fock, alpha_t))
        if project == "traced":
            return d @ psi @ d.conj().T
        return d @ psi

    def fidelity(self, psi_target, project: str = "ground", apply_frame: bool = True) -> float:
        r"""State-preparation fidelity :math:`\langle\psi_t|\rho|\psi_t\rangle`.

        With ``project="ground"`` this is the paper's definition: the cavity
        state after projecting the transmon onto :math:`|g\rangle`.
        """
        target = np.asarray(psi_target, dtype=complex).reshape(-1)
        target = target / np.linalg.norm(target)
        state = self.final_cavity_state(project=project, apply_frame=apply_frame)
        if project == "traced":
            return float(np.real(np.conj(target) @ state @ target))
        return float(abs(np.vdot(target, state)) ** 2)


def _displace_np(n_fock: int, alpha: complex) -> np.ndarray:
    from scipy.linalg import expm as _expm

    a = np.diag(np.sqrt(np.arange(1, n_fock)), k=1).astype(complex)
    return _expm(alpha * a.conj().T - np.conj(alpha) * a)


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def _make_stepper(h_static, ops, dt, method: str):
    if method == "expm":

        def prop(coeff):
            h = h_static + jnp.einsum("c,cij->ij", coeff.astype(ops.dtype), ops)
            return expm(-1j * dt * h)

    elif method == "eigh":

        def prop(coeff):
            h = h_static + jnp.einsum("c,cij->ij", coeff.astype(ops.dtype), ops)
            w, v = jnp.linalg.eigh(h)
            return v @ (jnp.exp(-1j * dt * w)[:, None] * v.conj().T)

    else:
        raise ValueError(f"method must be 'expm' or 'eigh'; got {method!r}")
    return prop


def simulate_sequence(
    seq: PulseSequence,
    psi0=None,
    n_fock: int = 40,
    n_transmon: int = 2,
    save_every: int | None = None,
    include_kappa_force: bool = False,
    method: str = "eigh",
    frame: str = "displaced",
) -> SimResult:
    r"""Propagate a compiled sequence and return the trajectory.

    Parameters
    ----------
    seq : PulseSequence
        From :func:`pulses.compile_ecd_sequence` or
        :func:`pulses.compile_snap_sequence`.
    psi0 : array_like, optional
        Initial state. Accepts a joint ket of length ``n_fock * n_transmon``, a
        block array ``(n_transmon, n_fock)``, or a bare cavity ket
        ``(n_fock,)``, in which case the transmon starts in :math:`|g\rangle`.
        Defaults to :math:`|0\rangle|g\rangle`.
    n_fock : int
        Cavity truncation *in the frame*. 40 is ample for ECD sequences from
        vacuum; check :meth:`SimResult.truncation_error` afterwards.
    n_transmon : int
        Transmon levels. Use 3 or 4 to look for leakage driven by the
        broadband rotation pulses (the :math:`K \gg \chi` requirement);
        ``params.anharm`` then matters.
    save_every : int, optional
        Save a state every this many samples. Defaults to a stride that keeps
        roughly 400 frames, and gate boundaries are always included.
    include_kappa_force : bool
        Add the deterministic re-centring force at rate :math:`\kappa/2` to the
        Hamiltonian and use the lossy frame trajectory, consistent with the
        compilation. The propagation stays unitary either way -- this only
        affects which classical trajectory is factored out. Leave it ``False``
        for a loss-free benchmark.
    method : {"eigh", "expm"}
        Per-step propagator. ``"eigh"`` is faster and exactly unitary for a
        Hermitian generator.
    frame : {"displaced", "lab"}
        ``"lab"`` sets :math:`\alpha \equiv 0` and keeps the cavity drive
        explicitly, which requires ``n_fock`` large enough to hold
        :math:`\max|\alpha|^2` photons. Useful only as a cross-check at small
        :math:`\alpha_0`.
    """
    p = seq.params
    dim = n_fock * n_transmon

    # ----- initial state -------------------------------------------------
    if psi0 is None:
        blocks = np.zeros((n_transmon, n_fock), dtype=complex)
        blocks[0, 0] = 1.0
        psi_init = np.swapaxes(blocks, 0, 1).reshape(-1)
    else:
        arr = np.asarray(psi0, dtype=complex)
        if arr.ndim == 2 and arr.shape == (n_transmon, n_fock):
            psi_init = np.swapaxes(arr, 0, 1).reshape(-1)
        elif arr.size == dim:
            psi_init = arr.reshape(-1)
        elif arr.size == n_fock:
            blocks = np.zeros((n_transmon, n_fock), dtype=complex)
            blocks[0] = arr.reshape(-1)
            psi_init = np.swapaxes(blocks, 0, 1).reshape(-1)
        else:
            raise ValueError(
                f"cannot interpret psi0 of shape {arr.shape} for n_fock={n_fock}, "
                f"n_transmon={n_transmon}"
            )
    psi_init = psi_init / np.linalg.norm(psi_init)

    # ----- frame trajectory and coefficients ------------------------------
    kappa_h = p.kappa if include_kappa_force else 0.0
    frame_params = p.replace(kappa=kappa_h)
    if frame == "lab":
        alpha_full = np.zeros(seq.n_samples + 1, dtype=complex)
    elif frame == "displaced":
        alpha_full = _pulses.frame_trajectory(seq.eps, frame_params, seq.delta)
    else:
        raise ValueError(f"frame must be 'displaced' or 'lab'; got {frame!r}")

    # Midpoint sampling of the frame trajectory: the |alpha|^2 terms are large
    # and fast, so left-endpoint sampling loses accuracy.
    alpha_mid = 0.5 * (alpha_full[:-1] + alpha_full[1:])

    # The residual-drive coefficient needs the *actual* time derivative of the
    # trajectory being factored out. In the displaced frame alpha solves the
    # frame ODE, so the ODE right-hand side is that derivative and the residual
    # cancels to zero. In the lab frame alpha is identically zero, so its
    # derivative is zero too -- evaluating the ODE right-hand side there would
    # return -i eps and silently cancel the entire cavity drive.
    if frame == "lab":
        dalpha = np.zeros(seq.n_samples, dtype=complex)
    else:
        dalpha = np.array(
            [
                _pulses._alpha_rhs(alpha_mid[k], seq.eps[k], 0.0, frame_params, seq.delta)
                for k in range(seq.n_samples)
            ]
        )
    coeff = _coefficients(
        alpha_mid, dalpha, seq.eps, seq.omega, p, seq.delta, kappa_h
    )

    # ----- save schedule --------------------------------------------------
    if save_every is None:
        save_every = max(1, seq.n_samples // 400)
    keep = np.zeros(seq.n_samples, dtype=bool)
    keep[save_every - 1 :: save_every] = True
    keep[-1] = True
    for i in np.asarray(seq.gate_indices, dtype=int):
        if 0 < i <= seq.n_samples:
            keep[i - 1] = True

    h_static, ops = hamiltonian_terms(p, n_fock, n_transmon)
    prop = _make_stepper(h_static, ops, p.dt, method)

    coeff_j = jnp.asarray(coeff)
    keep_j = jnp.asarray(keep)
    zero = jnp.zeros((dim,), dtype=jnp.complex128)

    def step(psi, layer):
        c, flag = layer
        psi = prop(c) @ psi
        return psi, jnp.where(flag, psi, zero)

    @jax.jit
    def run(psi):
        return lax.scan(step, psi, (coeff_j, keep_j))

    psi_final, stacked = run(jnp.asarray(psi_init))
    stacked = np.asarray(stacked)[np.asarray(keep)]

    save_indices = np.concatenate([[0], np.nonzero(keep)[0] + 1])
    states = np.concatenate([psi_init[None, :], stacked], axis=0)

    return SimResult(
        states=states,
        save_indices=save_indices,
        times=save_indices * p.dt,
        alpha=alpha_full[save_indices],
        alpha_full=alpha_full,
        n_fock=n_fock,
        n_transmon=n_transmon,
        frame=frame,
        seq=seq,
    )


def gate_boundary_states(result: SimResult, project: str = "ground") -> tuple:
    """Saved states at logical gate boundaries only.

    Returns ``(labels, states, alpha)``, letting you line the simulated
    trajectory up with :func:`gate_optimization.sequence_history`.
    """
    seq = result.seq
    wanted = np.concatenate([[0], np.asarray(seq.gate_indices, dtype=int)])
    sel = np.searchsorted(result.save_indices, wanted)
    sel = np.clip(sel, 0, result.save_indices.size - 1)
    labels = ["input"] + [s.label for s in seq.segments if s.stop in set(wanted.tolist())]
    states = result.cavity_states(project=project)[sel]
    return labels, states, result.alpha[sel]


# ---------------------------------------------------------------------------
# Wigner readout
# ---------------------------------------------------------------------------


def wigner_frames(
    result: SimResult,
    x_bound: float = 5.0,
    y_bound: float = 5.0,
    grid_points: int = 100,
    project: str = "traced",
    frame: str = "comoving",
):
    r"""Wigner distributions along the trajectory.

    Parameters
    ----------
    project : {"traced", "ground"}
        ``"traced"`` (default) uses cavity density matrices with the transmon
        traced out, which is the honest object mid-sequence: during an ECD gate
        the cavity and transmon are deliberately entangled, and projecting on
        :math:`|g\rangle` there would show an artificially pure state.
        ``"ground"`` matches the experiment's postselection and is the right
        choice for the final frame.
    frame : {"comoving", "lab"}
        ``"comoving"`` returns one fixed grid and the Wigner function of the
        residual state. ``"lab"`` returns the same arrays plus per-frame
        offsets: the physical Wigner function is the co-moving array plotted on
        a grid translated by :math:`(\sqrt2\,\mathrm{Re}\,\alpha,
        \sqrt2\,\mathrm{Im}\,\alpha)`. Use :func:`lab_extent` to get the
        ``imshow`` extent for a frame.

    Returns
    -------
    xvec, yvec : ndarray
        Quadrature grids, in the convention of :func:`utils.compute_wigner`.
    wigner : ndarray, shape (n_save, len(yvec), len(xvec))
    offsets : ndarray, shape (n_save, 2)
        :math:`(\sqrt2\,\mathrm{Re}\,\alpha, \sqrt2\,\mathrm{Im}\,\alpha)` at
        each frame. All zeros when ``frame="comoving"``.
    """
    states = result.cavity_states(project=project)
    xvec, yvec, w = wigner_trajectory(states, x_bound, y_bound, grid_points)
    if frame == "comoving":
        offsets = np.zeros((w.shape[0], 2))
    elif frame == "lab":
        offsets = np.stack(
            [_SQRT2 * np.real(result.alpha), _SQRT2 * np.imag(result.alpha)], axis=-1
        )
    else:
        raise ValueError(f"frame must be 'comoving' or 'lab'; got {frame!r}")
    return xvec, yvec, w, offsets


def lab_extent(xvec: np.ndarray, yvec: np.ndarray, offset) -> tuple:
    """``imshow`` extent for one lab-frame Wigner frame."""
    dx, dp = float(offset[0]), float(offset[1])
    return (xvec[0] + dx, xvec[-1] + dx, yvec[0] + dp, yvec[-1] + dp)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    # --- a bare displacement, in both frames.
    # This catches a whole class of bug in the residual-drive coefficient: if
    # the derivative of the factored-out trajectory is inconsistent with the
    # trajectory itself, the cavity drive cancels silently and the state simply
    # never moves. In the lab frame <n> must equal |alpha|^2; in the displaced
    # frame the residual state must stay at the origin.
    p0 = SystemParams()
    g_env, area = _pulses.gaussian_envelope(p0.sigma_disp, p0.n_sigma_disp, p0.dt)
    alpha_test = 1.3 - 0.4j
    eps_test = (1j * alpha_test / area) * g_env
    seq_d = PulseSequence(
        eps=eps_test.astype(complex),
        omega=np.zeros(eps_test.size, dtype=complex),
        params=p0,
        delta=0.0,
        segments=[_pulses.PulseSegment("displacement", 0, eps_test.size, "D")],
        gate_indices=np.array([eps_test.size]),
        meta={"gate_set": "test"},
    )
    for frame_name, expected in (("lab", abs(alpha_test) ** 2), ("displaced", 0.0)):
        r = simulate_sequence(seq_d, n_fock=30, frame=frame_name, save_every=10)
        n_end = r.photon_number(lab=False)[-1]
        ok = abs(n_end - expected) < 1e-3
        print(
            f"{frame_name:>9} frame: <n> in frame = {n_end:.6f}, expected "
            f"{expected:.6f}  {'ok' if ok else 'FAIL'}"
        )
        assert ok, f"{frame_name} frame displacement is wrong"
    print()

    p = SystemParams(chi=2 * np.pi * 0.2)  # 200 kHz, so the pulses stay short
    seq = _pulses.compile_ecd_sequence(
        betas=np.array([0.7 + 0.3j, -1.1 + 0.2j]),
        thetas=np.array([np.pi / 2, 0.7 * np.pi, np.pi / 3]),
        phis=np.array([0.0, 0.4 * np.pi, -0.3 * np.pi]),
        alpha0=8.0,
        params=p,
    )
    print(seq.summary())
    res = simulate_sequence(seq, n_fock=40)
    print(f"\nsaved frames    : {res.states.shape[0]}")
    print(f"truncation error: {res.truncation_error():.2e}")
    print(f"P(g) at the end : {res.p_ground()[-1]:.5f}")
    print(f"max <n> (lab)   : {res.photon_number().max():.1f}")
    print(f"|alpha(T)|      : {abs(res.alpha[-1]):.4f}")
