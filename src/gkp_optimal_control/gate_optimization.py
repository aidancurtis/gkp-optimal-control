r"""Gate-level optimization of continuous-variable circuits.

This module optimizes *gate parameters* of a fixed-depth circuit ansatz, in
contrast to :mod:`grape`, which optimizes time-domain control pulses. The
gates are treated as ideal unitaries; no Kerr, no finite-duration effects, no
dispersive-shift phase accumulation.

Three gate sets are supported, each built by its own function
(:func:`ecd_sequence`, :func:`snap_sequence`, :func:`csq_sequence`) and passed
to :func:`optimize_gate_sequence`.

**ECD + qubit rotations** (Eickbusch et al., *Nat. Phys.* **18**, 1464 (2022))

.. math::
    U = R(\theta_{N+1}, \phi_{N+1}) \prod_{i=N}^{1}
        \mathrm{ECD}(\beta_i)\, R(\theta_i, \phi_i),

with :math:`N` echoed conditional displacements and :math:`N+1` equatorial
qubit rotations, i.e. :math:`4N+2` real parameters. Conventions:

.. math::
    \mathrm{ECD}(\beta) = D(\beta/2)\,|e\rangle\langle g|
                        + D(-\beta/2)\,|g\rangle\langle e|, \qquad
    R(\theta,\phi) = \exp\!\left[-\tfrac{i\theta}{2}
        \left(\sigma_x\cos\phi + \sigma_y\sin\phi\right)\right].

The qubit is *not* represented explicitly. Because ECD and the rotations are
2x2 block operators on the qubit index, the state is carried as a pair of
cavity kets ``(psi_g, psi_e)``, halving memory and the cost of every
displacement relative to a ``2 * n_fock`` joint space.

**SNAP + displacements** (Heeres et al., *PRL* **115**, 137002 (2015);
Fösel, Krastanov et al., arXiv:2004.14256)

.. math::
    U = D(\alpha_{N+1}) \prod_{i=N}^{1} S(\vec\theta_i)\, D(\alpha_i),
    \qquad S(\vec\theta) = \sum_n e^{i\theta_n} |n\rangle\langle n|,

with :math:`N` SNAP gates and :math:`N+1` displacements. This gate set acts on
the cavity alone.

**Conditional squeezing + qubit rotations** (Schiaffino, Lombardo & Paz,
Eq. 5)

.. math::
    U = R(\theta_{N+1}, \phi_{N+1}) \prod_{i=N}^{1}
        \mathrm{CSq}(r_i, \varphi_{0,i}, \varphi_{1,i})\, R(\theta_i, \phi_i),

with :math:`N` conditional squeezers and :math:`N+1` equatorial qubit
rotations, i.e. :math:`5N+2` real parameters. Each squeezer applies a
squeeze of common strength but independent phase in each qubit branch:

.. math::
    \mathrm{CSq}(r, \varphi_0, \varphi_1) =
        |g\rangle\langle g| \otimes S(r, \varphi_0)
      + |e\rangle\langle e| \otimes S(r, \varphi_1), \qquad
    S(r, \varphi) = \exp\!\left[\tfrac{r}{2}\left(a^2 e^{-i\varphi}
        - a^{\dagger 2} e^{i\varphi}\right)\right].

The squeezing axis in branch :math:`j` is :math:`\varphi_j/2`, and the
qubit ground state plays the role of the paper's :math:`|0\rangle`; the
paper's encoding gate is :math:`\varphi_0 = 0,\ \varphi_1 = \pi`. The gate is
diagonal in the qubit basis (no echo / qubit flip), and the state is carried
in the same ``(psi_g, psi_e)`` block representation as the ECD set.

Every element of this gate set commutes with cavity photon-number parity
(:math:`\Delta n = \pm 2` only, and the rotations act on the qubit alone), so
the circuit can never move weight between the even and odd Fock sectors. From
vacuum it reaches only even-parity cavity states. ``optimize_gate_sequence``
warns when the resulting fidelity ceiling is below 0.999.

Optimization is batched multi-start Adam (vmapped over random seeds) followed
by an L-BFGS-B polish of the best seed, which is the standard recipe for these
ansaetze: the landscape is far more non-convex than GRAPE's, and single-start
gradient descent routinely stalls in poor local minima. Optional trajectory
penalties (:class:`TrajectoryPenalty`) regularize the fidelity *path* and the
layer-to-layer control variation, not only the endpoint.

Notes
-----
Run with 64-bit precision enabled::

    import jax
    jax.config.update("jax_enable_x64", True)

Displacements (ECD and SNAP sets) are applied by one of two methods, selected
with the builders' ``disp_method`` argument. Both are exactly unitary in the
truncated space and agree away from the Fock cutoff; they differ only in how
the truncation edge is treated.

``"fixed"`` (default)
    The BCH factorization of :func:`operators.make_displacement_apply`, applied
    to the state without forming :math:`D`: three matrix-vector products per
    state, smooth in ``Re alpha`` and ``Im alpha``, with gradients flowing
    through diagonal phases only.
``"exact"``
    :math:`D(\alpha) = \exp(\alpha a^\dagger - \alpha^* a)` of the truncated
    generator (:func:`operators.make_displacement`). Slower, and not
    differentiable at :math:`\alpha = 0`; use it to reproduce results that were
    optimized with it.

Squeezers (CSQ set) are applied by :func:`operators.make_squeeze_apply`, also
without forming the operator.

Building :math:`\langle m|D(\alpha)|n\rangle` from its closed form is
deliberately *not* offered. Those are the exact infinite-dimensional matrix
elements restricted to the truncated block, so the resulting matrix is not
unitary (numerically ``||D^dag D - I|| ~ 0.5`` at ``n_fock = 40``); norm is not
conserved and an optimizer will happily exploit that to report fidelities it
has not achieved.
"""

from __future__ import annotations

import inspect
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jaxquantum as jqt
import numpy as np
from jax import lax, value_and_grad, vmap
from scipy.optimize import minimize

from .operators import (
    make_displacement,
    make_displacement_apply,
    make_squeeze_apply,
    qubit_rotation,
)

__all__ = [
    "GateBounds",
    "OptimizerConfig",
    "TrajectoryPenalty",
    "GateOptResult",
    "GateSequence",
    "ecd_sequence",
    "snap_sequence",
    "csq_sequence",
    "build_sequence",
    "optimize_gate_sequence",
    "sequence_history",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateBounds:
    """Soft constraints on the gate parameters and on Fock-space leakage.

    All terms are differentiable penalties added to the loss, not hard bounds:
    hard box bounds on ``Re beta`` / ``Im beta`` would constrain a square rather
    than a disk, and would not be respected by Adam anyway.

    Parameters
    ----------
    max_disp : float or None
        Soft cap on displacement magnitude -- ``|beta_i|`` for the ECD set,
        ``|alpha_i|`` for the SNAP set. Penalizes ``max(|d| - max_disp, 0)**2``.
    max_disp_weight : float
        Weight of the displacement-cap penalty.
    max_squeeze : float or None
        Soft cap on the squeeze strength ``|r_i|`` of every conditional
        squeezer in the CSQ set. Used instead of ``max_disp`` for that gate set.
    max_squeeze_weight : float
        Weight of the squeezing-cap penalty.
    leakage_weight : float
        Weight of the Fock-leakage penalty. Population in the top ``n_leak``
        Fock levels (set on the sequence builder) is penalized after every
        displacement or squeeze in the circuit, not only at the end, since
        intermediate gates are what push a state into the truncation boundary.
        Set to 0 to disable.
    """

    max_disp: float | None = None
    max_disp_weight: float = 1.0
    max_squeeze: float | None = None
    max_squeeze_weight: float = 1.0
    leakage_weight: float = 1.0


@dataclass(frozen=True)
class OptimizerConfig:
    """Multi-start Adam followed by an L-BFGS-B polish.

    Parameters
    ----------
    n_seeds : int
        Number of random initializations, optimized in parallel via ``vmap``.
    n_adam_iters : int
        Adam iterations per seed.
    peak_lr : float
        Peak learning rate, reached after ``warmup_frac`` of the iterations and
        then cosine-decayed to ``final_lr_frac * peak_lr``.
    polish : bool
        Whether to refine the best Adam seed with L-BFGS-B.
    polish_maxiter : int
        Maximum L-BFGS-B iterations.
    seed : int
        PRNG seed for the initializations.
    """

    n_seeds: int = 8
    n_adam_iters: int = 1500
    peak_lr: float = 0.03
    warmup_frac: float = 0.05
    final_lr_frac: float = 0.05
    polish: bool = True
    polish_maxiter: int = 500
    seed: int = 0


@dataclass(frozen=True)
class TrajectoryPenalty:
    r"""Opt-in regularizers on the fidelity path and the control sequence.

    With :math:`F_k` the fidelity after layer :math:`k` (:math:`F_0` is the
    input state), the added loss is

    .. math::
        \lambda_\mathrm{mono} \sum_k \max(0, F_{k-1} - F_k)^2
        + \lambda_\mathrm{curve} \sum_k (F_{k+1} - 2F_k + F_{k-1})^2
        + \lambda_\mathrm{control} \sum_k \lVert c_{k+1} - c_k \rVert^2
        + \lambda_\mathrm{rot} \sum_k \theta_k^2
        + \lambda_\mathrm{geo} \sum_k D_k .

    :math:`F_k` is always the *reduced-oscillator* fidelity
    :math:`\langle t|\rho_\mathrm{cav}|t\rangle`, averaged over state pairs,
    even when the endpoint objective uses ``qubit_target="ground"``.
    Mid-circuit the qubit is generically entangled, so the ground-projected
    overlap is not a meaningful progress measure.

    In the control term, angles (qubit phases, squeezing phases, SNAP phases)
    are differenced on the circle. All other controls are differenced
    linearly. Differences are not divided by the layer spacing, so the weights
    absorb that scale.

    **Geodesic (brachistochrone) term.** For each state pair the time-optimal
    path of :func:`brachistochrone.quantum_brachistochrone_hamiltonian` is the
    Fubini-Study geodesic

    .. math::
        |\psi(a)\rangle = \cos a\,|\psi_i\rangle + \sin a\,|\psi_\perp\rangle,
        \qquad a \in [0, \theta_B],\quad
        \theta_B = \arccos|\langle\psi_i|\psi_f\rangle|,

    with :math:`|\psi_\perp\rangle` the normalized component of the
    (rephased) target orthogonal to :math:`|\psi_i\rangle`. With
    :math:`f_k(a) = \langle\psi(a)|\rho^{(k)}_\mathrm{cav}|\psi(a)\rangle`,
    the per-layer deviation is

    ``geodesic_mode="tube"``
        :math:`D_k = 1 - \max_{a\in[0,\theta_B]} f_k(a)`: distance to the
        nearest point of the arc. Schedule-free, so layers may advance by
        unequal amounts. The maximum is analytic: :math:`f(a)` is a sinusoid
        in :math:`2a`.
    ``geodesic_mode="schedule"``
        :math:`D_k = 1 - f_k(\theta_B\, k/N)`: the state must sit at the
        brachistochrone point for its layer, i.e. constant Bures speed, which
        is the brachistochrone schedule itself. Strictest.

    Both are evaluated on the reduced cavity state, so any mid-circuit
    qubit-cavity entanglement (a mixed :math:`\rho_\mathrm{cav}`) counts as
    leaving the geodesic. For ECD this is a strong constraint, because ECD
    normally works by entangling and then disentangling.

    All weights default to zero (off). :meth:`notes_defaults` returns the
    values quoted in the notes. :meth:`brachistochrone` is a starting point
    for geodesic following.
    """

    mono_weight: float = 0.0
    curve_weight: float = 0.0
    control_weight: float = 0.0
    rot_weight: float = 0.0
    geodesic_weight: float = 0.0
    geodesic_mode: str = "tube"

    def __post_init__(self):
        if self.geodesic_mode not in ("tube", "schedule"):
            raise ValueError(
                f"unknown geodesic_mode {self.geodesic_mode!r}; expected 'tube' or 'schedule'"
            )

    @classmethod
    def notes_defaults(cls) -> TrajectoryPenalty:
        return cls(mono_weight=2.0, curve_weight=1.0, control_weight=0.02, rot_weight=0.002)

    @classmethod
    def brachistochrone(cls, weight: float = 1.0, mode: str = "tube") -> TrajectoryPenalty:
        """Geodesic following plus a mild monotonicity term."""
        return cls(mono_weight=1.0, geodesic_weight=weight, geodesic_mode=mode)

    @property
    def active(self) -> bool:
        return any(
            w != 0.0
            for w in (
                self.mono_weight,
                self.curve_weight,
                self.control_weight,
                self.rot_weight,
                self.geodesic_weight,
            )
        )

    @property
    def needs_path(self) -> bool:
        return self.mono_weight != 0.0 or self.curve_weight != 0.0 or self.geodesic_weight != 0.0


@dataclass
class GateOptResult:
    """Outcome of a gate-sequence optimization."""

    gate_set: str
    n_gates: int
    n_fock: int
    fidelity: float
    loss: float
    leakage: float
    params: dict[str, np.ndarray]
    flat_params: np.ndarray
    final_states: np.ndarray
    per_seed_fidelity: np.ndarray
    best_seed: int
    adam_history: np.ndarray
    sequence: GateSequence = field(repr=False)
    polish_info: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    trajectory: np.ndarray | None = None
    penalties: dict = field(default_factory=dict)
    qubit_purity: float | None = None
    geodesic: dict = field(default_factory=dict)
    """Per-layer brachistochrone diagnostics (pair-averaged): ``deviation``
    (tube distance to the arc), ``progress`` (fraction of the Bures angle
    covered, ``a*/theta_B``) and ``deviation_schedule`` (distance from the
    uniform-speed brachistochrone point). Always computed, penalized only when
    ``TrajectoryPenalty.geodesic_weight > 0``."""
    initial_state: np.ndarray | None = None
    """The cavity ket(s) the circuit was optimized from, ``(K, n_fock)``."""
    ansatz: dict = field(default_factory=dict)
    """Builder options a loaded result was rebuilt with (see :mod:`gate_results`)."""
    metadata: dict = field(default_factory=dict)
    """Free-form fields stored alongside a saved result."""

    def summary(self) -> str:
        lines = [
            f"gate set        : {self.gate_set}",
            f"n_gates         : {self.n_gates}",
            f"n_params        : {self.flat_params.size}",
            f"fidelity        : {self.fidelity:.6f}",
            f"infidelity      : {1 - self.fidelity:.3e}",
            f"loss            : {self.loss:+.6e}",
            f"leakage         : {self.leakage:.3e}",
            f"best seed       : {self.best_seed} of {self.per_seed_fidelity.size}",
            f"seed F spread   : {self.per_seed_fidelity.min():.4f} .. "
            f"{self.per_seed_fidelity.max():.4f}",
        ]
        if self.qubit_purity is not None:
            lines.append(f"qubit purity    : {self.qubit_purity:.6f}")
        if self.trajectory is not None and self.trajectory.size > 1:
            drops = np.maximum(self.trajectory[:-1] - self.trajectory[1:], 0.0)
            lines.append(f"max F drop      : {drops.max():.3e}  (reduced-oscillator path)")
        if self.geodesic:
            lines.append(f"geodesic dev max: {self.geodesic['deviation'].max():.3e}")
        for key, val in self.penalties.items():
            lines.append(f"{key:<16}: {val:.3e}")
        return "\n".join(lines)

    def repropagate(self, psi_init=None) -> np.ndarray:
        """State after every gate from the stored parameters; see :func:`sequence_history`."""
        psi = self.initial_state if psi_init is None else psi_init
        if psi is None:
            raise ValueError("this result has no initial_state; pass psi_init explicitly")
        return sequence_history(self, psi)

    def final_qarray(self):
        """Return the final cavity state(s) as a ``jaxquantum.Qarray``.

        For qubit gate sets (ECD, CSQ) this is the ``|g>`` component only,
        renormalized; inspect ``final_states`` directly if you need the ``|e>``
        component or the joint state.
        """
        states = np.atleast_2d(self.final_states)
        if states.ndim == 3:  # (K, 2, d) -> take |g> block
            states = states[:, 0, :]
        if states.shape[0] == 1:
            return jqt.Qarray.create(states[0].reshape(-1, 1)).unit()
        return [jqt.Qarray.create(s.reshape(-1, 1)).unit() for s in states]


# ---------------------------------------------------------------------------
# State coercion
# ---------------------------------------------------------------------------


def _as_ket_batch(state, n_fock: int | None = None, name: str = "state"):
    """Coerce a state into a normalized ``(K, n_fock)`` complex array.

    Accepts a ``jaxquantum.Qarray`` (single or batched), a raw ``(d,)``,
    ``(d, 1)``, ``(K, d)`` or ``(K, d, 1)`` array, or a list of any of those.
    """
    if isinstance(state, (list, tuple)):
        rows = [_as_ket_batch(s, n_fock, name) for s in state]
        data = jnp.concatenate(rows, axis=0)
    else:
        if isinstance(state, jqt.Qarray):
            data = state.data
        else:
            data = state
        data = jnp.asarray(data)
        if data.ndim >= 2 and data.shape[-1] == 1:
            data = data[..., 0]
        if data.ndim == 1:
            data = data[None, :]
        elif data.ndim > 2:
            data = data.reshape(-1, data.shape[-1])

    if n_fock is not None and data.shape[-1] != n_fock:
        raise ValueError(f"{name} has Fock dimension {data.shape[-1]}, expected n_fock={n_fock}.")

    data = data.astype(jnp.result_type(complex))
    norms = jnp.linalg.norm(data, axis=-1, keepdims=True)
    if bool(jnp.any(norms == 0)):
        raise ValueError(f"{name} contains a zero vector.")
    return data / norms


# ---------------------------------------------------------------------------
# Fidelity / loss helpers
# ---------------------------------------------------------------------------


def _reduce_fidelity(overlaps, reduction: str):
    """Combine per-pair overlaps into a scalar fidelity."""
    if reduction == "mean":
        return jnp.mean(jnp.abs(overlaps) ** 2)
    if reduction == "coherent":
        # Phase-coherent average: correct for optimizing a *gate* on a
        # subspace spanned by orthonormal inputs, up to one global phase.
        return jnp.abs(jnp.mean(overlaps)) ** 2
    raise ValueError(f"unknown batch_reduction {reduction!r}")


def _apply_loss_style(fid, loss_type: str, eps: float = 1e-12):
    if loss_type == "infidelity":
        return 1.0 - fid
    if loss_type == "neg_fidelity":
        return -fid
    if loss_type == "log_infidelity":
        # Sharpens gradients once F > 0.99, where 1 - F is nearly flat.
        return jnp.log(jnp.clip(1.0 - fid, eps, None))
    raise ValueError(
        f"unknown loss_type {loss_type!r}; expected 'infidelity', "
        "'log_infidelity' or 'neg_fidelity'"
    )


def _leakage(psi, n_leak: int):
    """Mean population in the top ``n_leak`` Fock levels, averaged over batch."""
    if n_leak <= 0:
        return jnp.zeros((), dtype=jnp.real(psi).dtype)
    tail = jnp.abs(psi[..., -n_leak:]) ** 2
    return jnp.sum(tail) / psi.shape[0]


def _disp_penalty(mags, max_disp, weight):
    if max_disp is None or weight == 0.0:
        return jnp.zeros((), dtype=mags.dtype)
    excess = jnp.maximum(mags - max_disp, 0.0)
    return weight * jnp.sum(excess**2)


def _traced_fidelity(psi, targets):
    """Mean reduced-cavity fidelity ``<t|rho_cav|t>`` over the batch.

    ``psi`` is ``(K, d)`` (cavity only) or ``(K, 2, d)`` (qubit blocks).
    """
    if psi.ndim == 3:
        ov = jnp.sum(jnp.conj(targets)[:, None, :] * psi, axis=-1)
        return jnp.mean(jnp.sum(jnp.abs(ov) ** 2, axis=-1))
    ov = jnp.sum(jnp.conj(targets) * psi, axis=-1)
    return jnp.mean(jnp.abs(ov) ** 2)


def _probe_amplitudes(psi, probes):
    """Overlaps ``<P_m,k | psi_k,q>`` -> ``(M, K, Q)``; ``Q = 1`` without a qubit."""
    if psi.ndim == 3:
        return jnp.einsum("mkd,kqd->mkq", jnp.conj(probes), psi)
    return jnp.einsum("mkd,kd->mk", jnp.conj(probes), psi)[..., None]


def _path_fidelity(amps):
    """Reduced-cavity target fidelity per layer from ``(L, M, K, Q)`` amplitudes."""
    return jnp.mean(jnp.sum(jnp.abs(amps[:, 0]) ** 2, axis=-1), axis=-1)


def _geodesic_frame(psi_i, psi_t):
    """``(psi_perp, theta_B)`` per pair for the geodesic from ``psi_i`` to ``psi_t``."""
    psi_i = np.asarray(psi_i)
    psi_t = np.asarray(psi_t)
    ov = np.sum(np.conj(psi_i) * psi_t, axis=-1)
    theta = np.maximum(np.arccos(np.clip(np.abs(ov), 0.0, 1.0)), 1e-9)
    phase = np.where(np.abs(ov) > 0, ov / np.maximum(np.abs(ov), 1e-300), 1.0)
    t_rephased = psi_t * np.conj(phase)[:, None]
    resid = t_rephased - np.abs(ov)[:, None] * psi_i
    norms = np.linalg.norm(resid, axis=-1, keepdims=True)
    perp = np.where(norms > 1e-12, resid / np.maximum(norms, 1e-300), 0.0)
    return perp, theta


def _geodesic_overlap(amps, theta, mode: str):
    r"""Per-layer, per-pair ``f_k(a)`` on the geodesic; returns ``(f, a)``.

    ``amps`` is ``(L, M, K, Q)`` with probes ``[target, psi_i, psi_perp]``.
    ``f(a) = A + B cos 2a + C sin 2a`` with ``A = (p_i + p_perp)/2``,
    ``B = (p_i - p_perp)/2`` and ``C = Re <psi_i|rho|psi_perp>``.
    """
    amp_i, amp_p = amps[:, 1], amps[:, 2]
    p_i = jnp.sum(jnp.abs(amp_i) ** 2, axis=-1)
    p_p = jnp.sum(jnp.abs(amp_p) ** 2, axis=-1)
    c = jnp.real(jnp.sum(amp_i * jnp.conj(amp_p), axis=-1))
    big_a, big_b = 0.5 * (p_i + p_p), 0.5 * (p_i - p_p)
    n_layers = amps.shape[0]

    def f_at(a):
        return big_a + big_b * jnp.cos(2 * a) + c * jnp.sin(2 * a)

    if mode == "schedule":
        s = jnp.linspace(0.0, 1.0, n_layers)[:, None]
        a = s * theta[None, :]
        return f_at(a), a
    # Unconstrained maximizer of the sinusoid, clipped into the arc. If it lies
    # beyond the arc the true constrained maximum is an endpoint, possibly the
    # far one, so compare all three candidates.
    #
    # The best candidate is chosen with explicit `where` comparisons, not
    # jnp.max: the max JVP divides by the number of entries equal to the
    # output, and under jit XLA may recompute fused values with different
    # rounding, so no entry matches and the gradient becomes 0/0 = NaN.
    th = jnp.broadcast_to(theta[None, :], p_i.shape)
    # arctan2 has a 0/0 derivative at (0, 0); substitute a harmless point there.
    r2 = c**2 + big_b**2
    ok = r2 > 1e-300
    a_star = 0.5 * jnp.arctan2(jnp.where(ok, c, 0.0), jnp.where(ok, big_b, 1.0))
    cands = (jnp.clip(a_star, 0.0, th), jnp.zeros_like(th), th)
    f, a = f_at(cands[0]), cands[0]
    for cand in cands[1:]:
        val = f_at(cand)
        take = val > f
        f = jnp.where(take, val, f)
        a = jnp.where(take, cand, a)
    return f, a


def _wrap_angle(d):
    return jnp.arctan2(jnp.sin(d), jnp.cos(d))


def _control_roughness(controls):
    """Sum of squared layer-to-layer differences, angles wrapped on the circle."""
    total = jnp.zeros(())
    for arr, is_angle in controls:
        if arr.shape[0] < 2:
            continue
        d = arr[1:] - arr[:-1]
        if is_angle:
            d = _wrap_angle(d)
        total = total + jnp.sum(d**2)
    return total


def _trajectory_terms(amps, controls, rot_angles, theta=None, geo_mode="tube"):
    """Unweighted trajectory penalty terms (dict of scalars).

    ``amps`` is the ``(L, M, K, Q)`` probe-amplitude path or ``None``; the
    geodesic term is computed only when ``theta`` is given.
    """
    zero = jnp.zeros(())
    fids = None if amps is None else _path_fidelity(amps)
    terms = {"mono": zero, "curve": zero, "geodesic": zero}
    if theta is not None and amps is not None:
        f_geo, _ = _geodesic_overlap(amps, theta, geo_mode)
        terms["geodesic"] = jnp.sum(jnp.mean(1.0 - f_geo, axis=-1))
    if fids is not None and fids.shape[0] >= 2:
        terms["mono"] = jnp.sum(jnp.maximum(fids[:-1] - fids[1:], 0.0) ** 2)
    if fids is not None and fids.shape[0] >= 3:
        terms["curve"] = jnp.sum((fids[2:] - 2 * fids[1:-1] + fids[:-2]) ** 2)
    terms["control"] = _control_roughness(controls)
    terms["rot"] = jnp.sum(rot_angles**2) if rot_angles.size else zero
    return terms


def _weighted_trajectory_penalty(terms, cfg: TrajectoryPenalty):
    return (
        cfg.mono_weight * terms["mono"]
        + cfg.curve_weight * terms["curve"]
        + cfg.control_weight * terms["control"]
        + cfg.rot_weight * terms["rot"]
        + cfg.geodesic_weight * terms["geodesic"]
    )


def _qubit_purity(psi_blocks):
    """Mean purity Tr(rho_q^2) of the reduced qubit over a ``(K, 2, d)`` batch."""
    rho = jnp.einsum("kid,kjd->kij", psi_blocks, jnp.conj(psi_blocks))
    return jnp.mean(jnp.sum(jnp.abs(rho) ** 2, axis=(-2, -1)))


def _parity_fidelity_ceiling(psi_i, psi_t):
    """Upper bound on F for any parity-preserving circuit, per state pair.

    With even/odd weights ``p_e, p_o`` of input and target, a circuit that
    commutes with cavity parity (and a qubit starting in |g>) cannot exceed
    ``(sqrt(pe_i pe_t) + sqrt(po_i po_t))**2``, for either ``qubit_target``.
    """
    pi_ = np.abs(np.asarray(psi_i)) ** 2
    pt_ = np.abs(np.asarray(psi_t)) ** 2
    pe_i, pe_t = pi_[:, ::2].sum(-1), pt_[:, ::2].sum(-1)
    po_i, po_t = pi_[:, 1::2].sum(-1), pt_[:, 1::2].sum(-1)
    return (np.sqrt(pe_i * pe_t) + np.sqrt(po_i * po_t)) ** 2


# ---------------------------------------------------------------------------
# Gate-sequence definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateSequence:
    """A concrete circuit ansatz: parameter layout plus propagation rules.

    Attributes
    ----------
    name, n_gates, n_fock, n_params
        Problem sizes. ``n_gates`` counts ECDs, SNAPs, or conditional-squeezing
        layers, not total gates.
    has_qubit : bool
        Whether the state carries a qubit block (ECD, CSQ).
    strength_kind : {"disp", "squeeze"}
        Which :class:`GateBounds` cap applies to the reported gate strengths.
    qubit_target : {"ground", "traced"} or None
        How the qubit is scored at the end (see :func:`ecd_sequence`); ``None``
        for SNAP, which has no qubit.
    n_leak : int
        Number of top Fock levels counted as leakage.
    unpack : callable
        ``flat -> pytree`` of gate parameters.
    lift : callable
        ``(K, n_fock) -> initial internal state`` (adds the qubit block).
    propagate : callable
        ``(flat, psi0, probes=None) -> (psi_final, leakage, strengths, amps)``.
        With ``probes`` of shape ``(M, K, d)``, ``amps`` is ``(L, M, K, Q)``:
        overlaps of every probe ket with every qubit block after every layer
        (``L`` includes the input state; ``Q = 2`` for qubit sets, 1 for SNAP).
        ``amps`` is ``None`` without probes.
    history : callable
        ``(flat, psi0) -> stacked states after every gate``, for plotting.
    overlaps : callable
        ``(psi_final, targets) -> complex overlaps``, one per state pair.
    fidelity : callable
        ``(psi_final, targets, reduction) -> scalar F``.
    init : callable
        ``key -> flat params`` random initialization.
    describe : callable
        ``flat -> dict`` of human-readable numpy parameter arrays.
    controls : callable
        ``flat -> list[(array (L, c), is_angle)]`` control sequences along the
        layer axis, for the smoothness penalty.
    rotation_angles : callable
        ``flat -> array`` of qubit rotation angles (empty for SNAP).
    options : dict
        Builder options that change propagation (``echoed``, ``qubit_target``,
        ``n_snap``, ``n_leak``), recorded so :mod:`gate_results` can rebuild
        the sequence from a saved file.
    """

    name: str
    n_gates: int
    n_fock: int
    n_params: int
    has_qubit: bool
    strength_kind: str
    qubit_target: str | None
    n_leak: int
    unpack: Callable
    lift: Callable
    propagate: Callable
    history: Callable
    overlaps: Callable
    fidelity: Callable
    init: Callable
    describe: Callable
    controls: Callable
    rotation_angles: Callable
    options: dict = field(default_factory=dict)


def _layered_sequence(
    *,
    name: str,
    n_gates: int,
    n_fock: int,
    n_params: int,
    has_qubit: bool,
    strength_kind: str,
    qubit_target: str | None,
    n_leak: int,
    unpack: Callable,
    to_layers: Callable,
    first: Callable,
    layer: Callable,
    leak_after_first: bool,
    strengths: Callable,
    init: Callable,
    describe: Callable,
    controls: Callable,
    rotation_angles: Callable,
    options: dict,
) -> GateSequence:
    r"""Shared scaffold: one opening gate, then ``n_gates`` identical layers.

    Parameters
    ----------
    unpack : callable
        ``flat -> pytree`` of gate parameters, in the set's natural grouping.
    to_layers : callable
        ``flat -> (head, layers)``: the opening gate's parameters and a pytree
        of per-layer parameters with leading axis ``n_gates``.
    first : callable
        ``(psi, head) -> psi``, the opening gate.
    layer : callable
        ``(psi, layer_params) -> list[(psi, counts_leak)]``, the state after
        each gate of one layer and whether leakage is accumulated there.
    leak_after_first : bool
        Whether the opening gate contributes to leakage. When it does, the
        state after it is also probed (it moves the cavity).
    strengths : callable
        ``flat -> 1-D array`` of gate strengths to cap.
    """

    def lift(psi_cav):
        """(K, d) cavity kets -> internal state; adds the qubit block in |g>."""
        if not has_qubit:
            return psi_cav
        return jnp.stack([psi_cav, jnp.zeros_like(psi_cav)], axis=1)

    def propagate(flat, psi0, probes=None):
        head, layers = to_layers(flat)
        track = probes is not None
        psi = first(psi0, head)
        leak = _leakage(psi, n_leak) if leak_after_first else jnp.zeros((), jnp.real(psi).dtype)
        psi_first = psi

        def step(carry, layer_params):
            psi, leak = carry
            for sub, counts_leak in layer(psi, layer_params):
                if counts_leak:
                    leak = leak + _leakage(sub, n_leak)
                psi = sub
            return (psi, leak), (_probe_amplitudes(psi, probes) if track else None)

        (psi, leak), amps = lax.scan(step, (psi, leak), layers)
        if track:
            opening = [psi0, psi_first] if leak_after_first else [psi0]
            amps = jnp.concatenate(
                [jnp.stack([_probe_amplitudes(x, probes) for x in opening]), amps]
            )
        return psi, leak, strengths(flat), amps

    def history(flat, psi0):
        """The input state followed by the state after every gate."""
        head, layers = to_layers(flat)
        psi = first(psi0, head)
        out = [psi0, psi]
        for i in range(n_gates):
            for psi, _ in layer(psi, jax.tree.map(lambda x: x[i], layers)):
                out.append(psi)
        return jnp.stack(out)

    def overlaps(psi_final, targets):
        if qubit_target == "traced":
            raise ValueError(
                "overlaps are only defined for qubit_target='ground'; "
                "batch_reduction='coherent' is incompatible with 'traced'."
            )
        block = psi_final[:, 0, :] if has_qubit else psi_final
        return jnp.sum(jnp.conj(targets) * block, axis=-1)

    def fidelity(psi_final, targets, reduction):
        if qubit_target == "traced":
            # F_k = <t| rho_cav |t> = |<t|psi_g>|^2 + |<t|psi_e>|^2
            return _traced_fidelity(psi_final, targets)
        return _reduce_fidelity(overlaps(psi_final, targets), reduction)

    return GateSequence(
        name=name,
        n_gates=n_gates,
        n_fock=n_fock,
        n_params=n_params,
        has_qubit=has_qubit,
        strength_kind=strength_kind,
        qubit_target=qubit_target,
        n_leak=n_leak,
        unpack=unpack,
        lift=lift,
        propagate=propagate,
        history=history,
        overlaps=overlaps,
        fidelity=fidelity,
        init=init,
        describe=describe,
        controls=controls,
        rotation_angles=rotation_angles,
        options={"n_leak": n_leak, **options},
    )


def _qubit_block_sequence(
    *,
    name: str,
    n_gates: int,
    n_fock: int,
    n_gate_params: int,
    conditional: Callable,
    strengths: Callable,
    strength_kind: str,
    gate_controls: Callable,
    init_gate: Callable,
    describe_gate: Callable,
    qubit_target: str,
    n_leak: int,
    options: dict | None = None,
) -> GateSequence:
    r"""``R_{N+1} prod_i C_i R_i`` on qubit-block states ``(K, 2, d)``.

    Flat layout: ``[(theta, phi) x (N+1), gate params x N]``. ``conditional``
    is ``(psi, gate_params) -> psi``, the qubit-conditional cavity gate.
    """
    if qubit_target not in ("ground", "traced"):
        raise ValueError(f"unknown qubit_target {qubit_target!r}")

    n_rot = n_gates + 1

    def unpack(flat):
        flat = jnp.asarray(flat)
        rots = flat[: 2 * n_rot].reshape(n_rot, 2)
        gps = flat[2 * n_rot :].reshape(n_gates, n_gate_params)
        return rots, gps

    def to_layers(flat):
        rots, gps = unpack(flat)
        return rots[0], (gps, rots[1:])

    def rotate(psi, rot):
        return jnp.einsum("ij,kjd->kid", qubit_rotation(rot[0], rot[1]), psi)

    def layer(psi, layer_params):
        gp, rot = layer_params
        psi = conditional(psi, gp)
        return [(psi, True), (rotate(psi, rot), False)]

    def init(key):
        k_rot, k_phi, k_gate = jax.random.split(key, 3)
        thetas = jax.random.uniform(k_rot, (n_rot,), minval=0.0, maxval=jnp.pi)
        phis = jax.random.uniform(k_phi, (n_rot,), minval=-jnp.pi, maxval=jnp.pi)
        gps = init_gate(k_gate)
        return jnp.concatenate([jnp.stack([thetas, phis], axis=-1).ravel(), gps.ravel()])

    def describe(flat):
        rots, gps = unpack(flat)
        rots = np.asarray(rots)
        return {"thetas": rots[:, 0], "phis": rots[:, 1], **describe_gate(np.asarray(gps))}

    def controls(flat):
        rots, gps = unpack(flat)
        return [(rots[:, 0:1], False), (rots[:, 1:2], True), *gate_controls(gps)]

    return _layered_sequence(
        name=name,
        n_gates=n_gates,
        n_fock=n_fock,
        n_params=2 * n_rot + n_gate_params * n_gates,
        has_qubit=True,
        strength_kind=strength_kind,
        qubit_target=qubit_target,
        n_leak=n_leak,
        unpack=unpack,
        to_layers=to_layers,
        first=rotate,
        layer=layer,
        leak_after_first=False,
        strengths=lambda flat: strengths(unpack(flat)[1]),
        init=init,
        describe=describe,
        controls=controls,
        rotation_angles=lambda flat: unpack(flat)[0][:, 0],
        options={"qubit_target": qubit_target, **(options or {})},
    )


def _displacer(n_fock: int, method: str) -> Callable:
    """``(psi, alpha, adjoint=False) -> D(alpha) psi`` for a ``disp_method``."""
    if method == "fixed":
        return make_displacement_apply(n_fock)
    if method == "exact":
        displace = make_displacement(n_fock)

        def apply(psi, alpha, adjoint: bool = False):
            d_op = displace(alpha)
            return psi @ (d_op.conj() if adjoint else d_op.T)  # row kets

        return apply
    raise ValueError(f"unknown disp_method {method!r}; expected 'fixed' or 'exact'")


def ecd_sequence(
    n_gates: int,
    n_fock: int,
    *,
    disp_method: str = "fixed",
    echoed: bool = True,
    qubit_target: str = "ground",
    n_leak: int = 5,
    init_disp_scale: float = 1.0,
) -> GateSequence:
    r"""ECD + equatorial rotations, propagated in the 2x2 qubit block basis.

    Parameters
    ----------
    n_gates : int
        Number of conditional displacements; the circuit has ``n_gates + 1``
        rotations.
    n_fock : int
        Fock-space truncation.
    disp_method : {"fixed", "exact"}
        How ``D(alpha)`` is applied; see the module docstring.
    echoed : bool
        ``True`` for the echoed conditional displacement (includes the qubit
        flip), ``False`` for the bare conditional displacement
        ``D(beta/2)|g><g| + D(-beta/2)|e><e|``.
    qubit_target : {"ground", "traced"}
        ``"ground"`` scores ``|<psi_t| psi_g>|**2``, which rewards returning the
        qubit to ``|g>`` and disentangling it -- the standard choice.
        ``"traced"`` scores ``<psi_t| rho_cav |psi_t>``, allowing the qubit to
        end anywhere, which is only meaningful if you intend to discard it.
    n_leak : int
        Number of top Fock levels counted as leakage (weighted by
        ``GateBounds.leakage_weight``).
    init_disp_scale : float
        Standard deviation of the random initial ``Re beta``, ``Im beta``.
    """
    displace = _displacer(n_fock, disp_method)

    def conditional(psi, gp):
        # D(beta/2) on one branch and D(beta/2)^dag on the other; echoed swaps them.
        half = (gp[0] + 1j * gp[1]) / 2
        forward = displace(psi[:, 0, :], half)
        backward = displace(psi[:, 1, :], half, adjoint=True)
        return jnp.stack([backward, forward] if echoed else [forward, backward], axis=1)

    return _qubit_block_sequence(
        name="ecd",
        n_gates=n_gates,
        n_fock=n_fock,
        n_gate_params=2,
        conditional=conditional,
        strengths=lambda gps: jnp.linalg.norm(gps, axis=-1),
        strength_kind="disp",
        gate_controls=lambda gps: [(gps, False)],
        init_gate=lambda key: init_disp_scale * jax.random.normal(key, (n_gates, 2)),
        describe_gate=lambda gps: {"betas": gps[:, 0] + 1j * gps[:, 1]},
        qubit_target=qubit_target,
        n_leak=n_leak,
        options={"disp_method": disp_method, "echoed": echoed},
    )


def csq_sequence(
    n_gates: int,
    n_fock: int,
    *,
    qubit_target: str = "ground",
    n_leak: int = 5,
    init_squeeze_scale: float = 0.2,
) -> GateSequence:
    r"""Conditional squeezing gate of Schiaffino, Lombardo & Paz (Eq. 5).

    Each layer is ``CSq(r, phi0, phi1) = |g><g| (x) S(r, phi0) + |e><e| (x) S(r, phi1)``
    with ``S(r, phi) = exp[(r/2)(a^2 e^{-i phi} - a^dag^2 e^{i phi})]``, parameters
    ``(r, phi0, phi1)``. The squeezing axis in branch j is phi_j / 2. The qubit
    ground state plays the role of the paper's |0>. The encoding gate of the paper
    is ``phi0 = 0, phi1 = pi``.

    Parameters
    ----------
    n_gates : int
        Number of conditional squeezers; the circuit has ``n_gates + 1``
        rotations.
    n_fock : int
        Fock-space truncation.
    qubit_target : {"ground", "traced"}
        As in :func:`ecd_sequence`.
    n_leak : int
        Number of top Fock levels counted as leakage (weighted by
        ``GateBounds.leakage_weight``).
    init_squeeze_scale : float
        Standard deviation of the random initial ``r``. Phases are drawn
        uniformly on ``[-pi, pi)``.
    """
    squeeze = make_squeeze_apply(n_fock)

    def init_gate(key):
        k_r, k_p = jax.random.split(key)
        r = init_squeeze_scale * jax.random.normal(k_r, (n_gates, 1))
        phases = jax.random.uniform(k_p, (n_gates, 2), minval=-jnp.pi, maxval=jnp.pi)
        return jnp.concatenate([r, phases], axis=-1)

    def describe_gate(gps):
        return {
            "r": gps[:, 0],
            "phi0": gps[:, 1],
            "phi1": gps[:, 2],
            "axis0": gps[:, 1] / 2,
            "axis1": gps[:, 2] / 2,
        }

    return _qubit_block_sequence(
        name="csq",
        n_gates=n_gates,
        n_fock=n_fock,
        n_gate_params=3,
        # Both qubit branches in one pass: phase phi0 on |g>, phi1 on |e>.
        conditional=lambda psi, gp: squeeze(psi, gp[0], gp[1:3]),
        strengths=lambda gps: jnp.abs(gps[:, 0]),
        strength_kind="squeeze",
        gate_controls=lambda gps: [(gps[:, 0:1], False), (gps[:, 1:3], True)],
        init_gate=init_gate,
        describe_gate=describe_gate,
        qubit_target=qubit_target,
        n_leak=n_leak,
    )


def snap_sequence(
    n_gates: int,
    n_fock: int,
    *,
    disp_method: str = "fixed",
    n_snap: int | None = None,
    n_leak: int = 5,
    init_disp_scale: float = 1.0,
) -> GateSequence:
    r"""SNAP + displacements, sandwiched as ``D S D S ... D``.

    Flat layout: ``[phases x N, (Re alpha, Im alpha) x (N+1)]``.

    Parameters
    ----------
    n_gates : int
        Number of SNAP gates; the circuit has ``n_gates + 1`` displacements.
    n_fock : int
        Fock-space truncation.
    disp_method : {"fixed", "exact"}
        How ``D(alpha)`` is applied; see the module docstring.
    n_snap : int or None
        Number of Fock phases optimized per SNAP gate. ``None`` (default)
        optimizes all ``n_fock`` phases. A smaller value pins the phases of
        levels ``n >= n_snap`` to zero, which is the physically honest choice
        when the selective qubit pulses only resolve low photon numbers.
    n_leak : int
        Number of top Fock levels counted as leakage (weighted by
        ``GateBounds.leakage_weight``).
    init_disp_scale : float
        Standard deviation of the random initial ``Re alpha``, ``Im alpha``.
    """
    n_snap_eff = int(n_fock if n_snap is None else n_snap)
    if not 1 <= n_snap_eff <= n_fock:
        raise ValueError(f"n_snap must lie in [1, n_fock]; got {n_snap}")

    displace = _displacer(n_fock, disp_method)
    n_disp = n_gates + 1
    n_theta = n_gates * n_snap_eff
    pad = n_fock - n_snap_eff

    def unpack(flat):
        flat = jnp.asarray(flat)
        return flat[:n_theta].reshape(n_gates, n_snap_eff), flat[n_theta:].reshape(n_disp, 2)

    def to_layers(flat):
        thetas, alphas = unpack(flat)
        return alphas[0], (thetas, alphas[1:])

    def apply_disp(psi, alpha_ri):
        return displace(psi, alpha_ri[0] + 1j * alpha_ri[1])

    def layer(psi, layer_params):
        theta, alpha_ri = layer_params
        phases = jnp.concatenate([theta, jnp.zeros((pad,), dtype=theta.dtype)])
        psi = psi * jnp.exp(1j * phases)[None, :]
        return [(psi, False), (apply_disp(psi, alpha_ri), True)]

    def init(key):
        k_theta, k_alpha = jax.random.split(key)
        thetas = jax.random.uniform(k_theta, (n_gates, n_snap_eff), minval=-jnp.pi, maxval=jnp.pi)
        alphas = init_disp_scale * jax.random.normal(k_alpha, (n_disp, 2))
        return jnp.concatenate([thetas.ravel(), alphas.ravel()])

    def describe(flat):
        thetas, alphas = unpack(flat)
        alphas = np.asarray(alphas)
        return {"snap_phases": np.asarray(thetas), "alphas": alphas[:, 0] + 1j * alphas[:, 1]}

    def controls(flat):
        thetas, alphas = unpack(flat)
        return [(thetas, True), (alphas, False)]

    return _layered_sequence(
        name="snap",
        n_gates=n_gates,
        n_fock=n_fock,
        n_params=n_theta + 2 * n_disp,
        has_qubit=False,
        strength_kind="disp",
        qubit_target=None,
        n_leak=n_leak,
        unpack=unpack,
        to_layers=to_layers,
        first=apply_disp,
        layer=layer,
        leak_after_first=True,
        strengths=lambda flat: jnp.linalg.norm(unpack(flat)[1], axis=-1),
        init=init,
        describe=describe,
        controls=controls,
        rotation_angles=lambda flat: jnp.zeros((0,)),
        options={"disp_method": disp_method, "n_snap": n_snap_eff},
    )


def build_sequence(gate_set: str, n_gates: int, n_fock: int, **ansatz) -> GateSequence:
    """Build a :class:`GateSequence` from its gate-set name and builder options.

    Options the gate set's builder does not take are ignored, so one ``ansatz``
    dict can be passed for any gate set and files saved with options that have
    since been removed still load.
    """
    try:
        builder = {"ecd": ecd_sequence, "snap": snap_sequence, "csq": csq_sequence}[gate_set]
    except KeyError:
        raise ValueError(
            f"unknown gate set {gate_set!r}; expected 'ecd', 'snap' or 'csq'"
        ) from None
    accepted = inspect.signature(builder).parameters
    kwargs = {k: v for k, v in ansatz.items() if k in accepted and v is not None}
    return builder(n_gates, n_fock, **kwargs)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


def _cosine_schedule(n_iters: int, peak_lr: float, warmup_frac: float, final_frac: float):
    n_warm = max(1, int(round(warmup_frac * n_iters)))
    n_decay = max(1, n_iters - n_warm)
    final_lr = peak_lr * final_frac

    def lr_at(i):
        i = jnp.asarray(i).astype(jnp.result_type(float))
        lr_warm = peak_lr * (i + 1) / n_warm
        prog = jnp.clip((i - n_warm) / n_decay, 0.0, 1.0)
        lr_cos = final_lr + (peak_lr - final_lr) * 0.5 * (1 + jnp.cos(jnp.pi * prog))
        return jnp.where(i < n_warm, lr_warm, lr_cos)

    return lr_at


def _adam_run(loss_and_grad, params0, n_iters, lr_at, b1=0.9, b2=0.999, eps=1e-8):
    """Plain Adam on a single parameter vector; returns (params, loss_history)."""

    def step(carry, i):
        params, mom, vel = carry
        val, grad = loss_and_grad(params)
        mom = b1 * mom + (1 - b1) * grad
        vel = b2 * vel + (1 - b2) * grad**2
        m_hat = mom / (1 - b1 ** (i + 1))
        v_hat = vel / (1 - b2 ** (i + 1))
        params = params - lr_at(i) * m_hat / (jnp.sqrt(v_hat) + eps)
        return (params, mom, vel), val

    init = (params0, jnp.zeros_like(params0), jnp.zeros_like(params0))
    (params, _, _), history = lax.scan(step, init, jnp.arange(n_iters))
    return params, history


def optimize_gate_sequence(
    seq: GateSequence,
    psi_init,
    psi_targ,
    *,
    loss_type: str = "infidelity",
    batch_reduction: str = "mean",
    bounds: GateBounds | None = None,
    trajectory: TrajectoryPenalty | None = None,
    optimizer: OptimizerConfig | None = None,
    params0: np.ndarray | None = None,
    verbose: bool = True,
) -> GateOptResult:
    r"""Optimize the gate parameters of a CV circuit for state preparation.

    Parameters
    ----------
    seq : GateSequence
        The circuit ansatz, built with :func:`ecd_sequence`,
        :func:`snap_sequence` or :func:`csq_sequence`. Depth, truncation and
        every ansatz-specific option (echo, SNAP phase count, qubit target,
        displacement / squeeze construction, leakage levels, initialization
        scale) are fixed there.
    psi_init, psi_targ : Qarray or array_like
        Cavity kets, shape ``(n_fock,)``, ``(n_fock, 1)``, ``(K, n_fock)`` or a
        list thereof, with ``n_fock == seq.n_fock``. With ``K > 1`` one
        parameter set is optimized for all pairs simultaneously, which is how
        you target a *gate* on a logical subspace rather than a single state.
        For the qubit gate sets these are cavity states only: the qubit is
        assumed to start in ``|g>`` and (with ``qubit_target="ground"``) is
        required to return there. Targets are renormalized.
    loss_type : {"infidelity", "log_infidelity", "neg_fidelity"}
        ``"log_infidelity"`` is usually the better choice once you are pushing
        past ``F = 0.99``, where ``1 - F`` is nearly flat.
    batch_reduction : {"mean", "coherent"}
        How multiple state pairs are combined. ``"mean"`` averages the
        fidelities. ``"coherent"`` averages the *overlaps* before squaring,
        which is the right objective for a gate on a subspace (it fixes the
        relative phases between logical basis states, up to one global phase).
        Requires ``qubit_target="ground"`` for the qubit gate sets.
    bounds : GateBounds, optional
        Soft displacement / squeezing caps and the leakage-penalty weight.
    trajectory : TrajectoryPenalty, optional
        Opt-in penalties on the fidelity path and control roughness. These are
        added to the loss in both the Adam and L-BFGS-B stages. Seeds are
        still ranked, and the polish accepted, by endpoint fidelity alone.
    optimizer : OptimizerConfig, optional
        Multi-start Adam plus L-BFGS-B polish settings.
    params0 : ndarray, optional
        Explicit initial parameters. Shape ``(n_params,)`` skips the multi-start
        and optimizes that point alone; ``(n_seeds, n_params)`` replaces the
        random initialization.
    verbose : bool
        Print progress.

    Returns
    -------
    GateOptResult
        Optimized parameters (both flat and in a labelled dict), achieved
        fidelity, final states, per-seed fidelities, the Adam loss history,
        the reduced-cavity fidelity trajectory, penalty breakdown, (for qubit
        gate sets) the final qubit purity, and the ``GateSequence`` itself.

    Examples
    --------
    >>> import jax, jax.numpy as jnp, jaxquantum as jqt
    >>> jax.config.update("jax_enable_x64", True)
    >>> from gkp_optimal_control.states import gkp_states
    >>> ell = jnp.sqrt(jnp.pi / 2)
    >>> gkp_0, _ = gkp_states(80, ell, 1j * ell, 0.4, 5)
    >>> res = optimize_gate_sequence(
    ...     ecd_sequence(12, 80, n_leak=8),
    ...     jqt.basis(80, 0), gkp_0,
    ...     loss_type="log_infidelity",
    ...     bounds=GateBounds(max_disp=4.0),
    ...     optimizer=OptimizerConfig(n_seeds=16, n_adam_iters=2000),
    ... )
    >>> res_csq = optimize_gate_sequence(
    ...     csq_sequence(32, 80, qubit_target="traced", n_leak=8),
    ...     jqt.basis(80, 0), gkp_0,
    ...     loss_type="log_infidelity",
    ...     bounds=GateBounds(max_squeeze=0.6),
    ...     trajectory=TrajectoryPenalty.notes_defaults(),
    ... )
    >>> print(res_csq.summary())
    """
    if jnp.zeros(1).dtype != jnp.float64:
        warnings.warn(
            "jax_enable_x64 is disabled; gate-level optimization is run in "
            "complex64 and fidelities above ~1 - 1e-6 will not be trustworthy. "
            "Enable it with jax.config.update('jax_enable_x64', True).",
            RuntimeWarning,
            stacklevel=2,
        )

    bounds = bounds or GateBounds()
    optimizer = optimizer or OptimizerConfig()
    trajectory = trajectory or TrajectoryPenalty()

    psi_i = _as_ket_batch(psi_init, seq.n_fock, "psi_init")
    psi_t = _as_ket_batch(psi_targ, seq.n_fock, "psi_targ")
    if psi_i.shape[0] != psi_t.shape[0]:
        raise ValueError(
            f"psi_init has {psi_i.shape[0]} state(s) but psi_targ has "
            f"{psi_t.shape[0]}; they must pair up."
        )
    n_pairs = int(psi_i.shape[0])

    overlaps_defined = (not seq.has_qubit) or seq.qubit_target == "ground"
    if batch_reduction == "coherent" and not overlaps_defined:
        raise ValueError(
            "batch_reduction='coherent' requires qubit_target='ground' "
            "(a traced-out qubit has no well-defined overlap phase)."
        )

    if seq.name == "csq":
        ceiling = _parity_fidelity_ceiling(psi_i, psi_t)
        if ceiling.min() < 0.999:
            warnings.warn(
                "the CSQ gate set conserves cavity photon-number parity; for the "
                f"given states the per-pair fidelity ceiling is {ceiling.min():.4f}. "
                "Odd-parity weight (or a displaced target) cannot be reached from "
                "an even-parity input.",
                RuntimeWarning,
                stacklevel=2,
            )

    if seq.strength_kind == "squeeze":
        cap, cap_weight, cap_label = bounds.max_squeeze, bounds.max_squeeze_weight, "|r|"
    else:
        cap, cap_weight, cap_label = bounds.max_disp, bounds.max_disp_weight, "|disp|"

    psi0 = seq.lift(psi_i)
    # Probe kets tracked after every layer: [target, psi_i, psi_perp].
    geo_perp, geo_theta = _geodesic_frame(psi_i, psi_t)
    probes = jnp.stack([psi_t, psi_i, jnp.asarray(geo_perp)])
    if trajectory.geodesic_weight != 0.0 and np.any(geo_theta < 1e-6):
        raise ValueError("geodesic penalty: an initial state already equals its target.")
    geo_theta = jnp.asarray(geo_theta)
    path_probes = probes if trajectory.needs_path else None
    geo_arg = geo_theta if trajectory.geodesic_weight != 0.0 else None

    # ----- objective -----------------------------------------------------
    def loss_fn(flat):
        psi_f, leak, mags, amps = seq.propagate(flat, psi0, path_probes)
        fid = seq.fidelity(psi_f, psi_t, batch_reduction)
        loss = _apply_loss_style(fid, loss_type)
        loss = loss + bounds.leakage_weight * leak
        loss = loss + _disp_penalty(mags, cap, cap_weight)
        if trajectory.active:
            terms = _trajectory_terms(
                amps,
                seq.controls(flat),
                seq.rotation_angles(flat),
                geo_arg,
                trajectory.geodesic_mode,
            )
            loss = loss + _weighted_trajectory_penalty(terms, trajectory)
        return loss

    def diagnose(flat):
        flat = jnp.asarray(flat)
        psi_f, leak, mags, amps = seq.propagate(flat, psi0, probes)
        fid = seq.fidelity(psi_f, psi_t, batch_reduction)
        per_pair = None
        if overlaps_defined:
            per_pair = np.asarray(jnp.abs(seq.overlaps(psi_f, psi_t)) ** 2)
        terms = _trajectory_terms(
            amps,
            seq.controls(flat),
            seq.rotation_angles(flat),
            geo_theta,
            trajectory.geodesic_mode,
        )
        purity = float(_qubit_purity(psi_f)) if seq.has_qubit else None
        f_tube, a_tube = _geodesic_overlap(amps, geo_theta, "tube")
        geo = {
            "deviation": np.asarray(jnp.mean(1.0 - f_tube, axis=-1)),
            "progress": np.asarray(jnp.mean(a_tube / geo_theta[None, :], axis=-1)),
            "deviation_schedule": np.asarray(
                jnp.mean(1.0 - _geodesic_overlap(amps, geo_theta, "schedule")[0], axis=-1)
            ),
        }
        return (
            psi_f,
            float(fid),
            float(leak),
            np.asarray(mags),
            per_pair,
            np.asarray(_path_fidelity(amps)),
            {k: float(v) for k, v in terms.items()},
            purity,
            geo,
        )

    loss_and_grad = jax.jit(value_and_grad(loss_fn))
    fid_only = jax.jit(
        lambda flat: seq.fidelity(seq.propagate(flat, psi0)[0], psi_t, batch_reduction)
    )

    # ----- initial parameters -------------------------------------------
    if params0 is None:
        # Built with an explicit loop rather than vmap: n_seeds is small, and
        # vmapping over jax.random.split is not portable across JAX versions.
        keys = jax.random.split(jax.random.PRNGKey(optimizer.seed), optimizer.n_seeds)
        init_params = jnp.stack([seq.init(keys[i]) for i in range(optimizer.n_seeds)])
    else:
        init_params = jnp.asarray(params0, dtype=jnp.result_type(float))
        if init_params.ndim == 1:
            init_params = init_params[None, :]
        if init_params.shape[-1] != seq.n_params:
            raise ValueError(
                f"params0 has {init_params.shape[-1]} parameters, expected {seq.n_params}"
            )
    n_seeds = int(init_params.shape[0])

    if verbose:
        print(f"gate set : {seq.name}   depth : {seq.n_gates}   n_fock : {seq.n_fock}")
        print(
            f"params   : {seq.n_params}   state pairs : {n_pairs}   reduction : {batch_reduction}"
        )
        print(f"loss     : {loss_type}   seeds : {n_seeds}")
        if trajectory.active:
            print(
                f"traj     : mono {trajectory.mono_weight}  curve {trajectory.curve_weight}"
                f"  control {trajectory.control_weight}  rot {trajectory.rot_weight}"
                f"  geodesic {trajectory.geodesic_weight} ({trajectory.geodesic_mode})"
            )
        print("-" * 62)

    # ----- stage 1: batched Adam ----------------------------------------
    t0 = time.time()
    lr_at = _cosine_schedule(
        optimizer.n_adam_iters,
        optimizer.peak_lr,
        optimizer.warmup_frac,
        optimizer.final_lr_frac,
    )

    @jax.jit
    def adam_all(p0_batch):
        return vmap(lambda p0: _adam_run(loss_and_grad, p0, optimizer.n_adam_iters, lr_at))(
            p0_batch
        )

    adam_params, adam_hist = adam_all(init_params)
    adam_params.block_until_ready()
    t_adam = time.time() - t0

    seed_fids = np.asarray(vmap(fid_only)(adam_params))
    best = int(np.argmax(seed_fids))
    best_params = adam_params[best]

    if verbose:
        print(f"Adam ({optimizer.n_adam_iters} iters x {n_seeds} seeds) in {t_adam:.1f}s")
        print(
            f"  seed fidelities: best {seed_fids.max():.6f}, "
            f"median {np.median(seed_fids):.6f}, worst {seed_fids.min():.6f}"
        )
        print(f"  best seed: {best}")

    # ----- stage 2: L-BFGS-B polish -------------------------------------
    polish_info: dict = {}
    if optimizer.polish:
        t1 = time.time()

        def scipy_obj(flat):
            val, grad = loss_and_grad(jnp.asarray(flat))
            return float(val), np.asarray(grad, dtype=np.float64)

        res = minimize(
            scipy_obj,
            np.asarray(best_params, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": optimizer.polish_maxiter,
                "ftol": 1e-14,
                "gtol": 1e-12,
            },
        )
        polished = jnp.asarray(res.x)
        f_polished = float(fid_only(polished))
        polish_info = {
            "nit": int(res.nit),
            "success": bool(res.success),
            "message": str(res.message),
            "fidelity_before": float(seed_fids[best]),
            "fidelity_after": f_polished,
            "time": time.time() - t1,
        }
        if f_polished >= seed_fids[best]:
            best_params = polished
        elif verbose:
            print("  polish did not improve fidelity; keeping the Adam result")
        if verbose:
            print(
                f"L-BFGS-B polish: {polish_info['nit']} iters in "
                f"{polish_info['time']:.1f}s  ->  F = {f_polished:.6f}"
            )

    # ----- report --------------------------------------------------------
    psi_f, fid, leak, mags, per_pair, fid_path, terms, purity, geo = diagnose(best_params)
    loss_val = float(loss_fn(best_params))

    params = seq.describe(best_params)
    params["strength_magnitudes"] = mags
    if seq.strength_kind == "disp":
        params["disp_magnitudes"] = mags  # backwards-compatible key
    if per_pair is not None:
        params["per_pair_fidelity"] = per_pair

    penalties = {}
    if trajectory.active:
        penalties = {f"pen_{k}": v for k, v in terms.items()}

    if verbose:
        print("-" * 62)
        print(f"Final fidelity : {fid:.6f}   (infidelity {1 - fid:.3e})")
        print(f"Leakage        : {leak:.3e}  (top {seq.n_leak} Fock levels)")
        print(f"Peak {cap_label:<10}: {mags.max():.3f}")
        if purity is not None:
            print(f"Qubit purity   : {purity:.6f}")
        drops = np.maximum(fid_path[:-1] - fid_path[1:], 0.0)
        print(f"Max F drop     : {drops.max():.3e}  (reduced-cavity path)")
        print(
            f"Geodesic dev.  : max {geo['deviation'].max():.3e}  (distance to brachistochrone arc)"
        )
        print(f"Total time     : {time.time() - t0:.1f}s")

    return GateOptResult(
        gate_set=seq.name,
        n_gates=seq.n_gates,
        n_fock=seq.n_fock,
        fidelity=fid,
        loss=loss_val,
        leakage=leak,
        params=params,
        flat_params=np.asarray(best_params),
        final_states=np.asarray(psi_f),
        per_seed_fidelity=seed_fids,
        best_seed=best,
        adam_history=np.asarray(adam_hist),  # (n_seeds, n_iters)
        polish_info=polish_info,
        config={
            "loss_type": loss_type,
            "batch_reduction": batch_reduction,
            "bounds": bounds,
            "trajectory": trajectory,
            "optimizer": optimizer,
            "n_pairs": n_pairs,
        },
        sequence=seq,
        trajectory=fid_path,
        penalties=penalties,
        qubit_purity=purity,
        geodesic=geo,
        initial_state=np.asarray(psi_i),
    )


def sequence_history(result: GateOptResult, psi_init) -> np.ndarray:
    """Re-run an optimized sequence and return the state after every gate.

    Useful for Wigner-function movies of the preparation: pair the output with
    :func:`utils.wigner_trajectory` or :mod:`animation`. Uses
    ``result.sequence``, so the history is computed with exactly the ansatz
    that was optimized.

    Returns
    -------
    ndarray
        The input state followed by the state after every gate:
        ``(2 * n_gates + 2, K, n_fock)`` for SNAP and
        ``(2 * n_gates + 2, K, 2, n_fock)`` for ECD and CSQ. For qubit gate
        sets the second-to-last axis indexes the qubit block ``(|g>, |e>)``.
    """
    seq = result.sequence
    psi_i = _as_ket_batch(psi_init, seq.n_fock, "psi_init")
    return np.asarray(seq.history(jnp.asarray(result.flat_params), seq.lift(psi_i)))
