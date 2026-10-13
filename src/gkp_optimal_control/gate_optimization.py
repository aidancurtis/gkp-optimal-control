r"""Gate-level optimization of continuous-variable circuits.

This module optimizes the *gate parameters* of a fixed-depth circuit ansatz
from :mod:`sequences`, in contrast to :mod:`grape`, which optimizes time-domain
control pulses.

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
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, value_and_grad, vmap
from scipy.optimize import minimize

from .sequences import GateSequence
from .utils import to_ket_batch

__all__ = [
    "GateBounds",
    "OptimizerConfig",
    "TrajectoryPenalty",
    "GateOptResult",
    "adam_run",
    "cosine_schedule",
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
    target_fidelity : float, optional
        Stop the Adam stage as soon as any seed reaches this fidelity. Checked
        every ``prune_every`` iterations, or every 100 if that is unset.
    prune_every : int, optional
        Every this many Adam iterations, keep only the best ``keep_fraction``
        of the seeds still running (successive halving). Seeds that stall
        early stop costing time; a late bloomer can be lost.
    keep_fraction : float
        Fraction of seeds kept at each pruning step; at least one survives.
    """

    n_seeds: int = 8
    n_adam_iters: int = 1500
    peak_lr: float = 0.03
    warmup_frac: float = 0.05
    final_lr_frac: float = 0.05
    polish: bool = True
    polish_maxiter: int = 500
    seed: int = 0
    target_fidelity: float | None = None
    prune_every: int | None = None
    keep_fraction: float = 0.5


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
    target_state: np.ndarray | None = None
    """The cavity ket(s) it was optimized towards, ``(K, n_fock)``."""
    init_label: str | None = None
    target_label: str | None = None
    """Short names for the two states (e.g. ``"vac"``, ``"gkp:0"``), used by
    :mod:`gate_results` to organize and index saved runs."""
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

    def final_ket(self):
        """The final cavity ket, ``(n_fock,)``, or a ``(K, n_fock)`` array for ``K`` pairs.

        For qubit gate sets (ECD, CSQ) this is the ``|g>`` component only,
        renormalized; inspect ``final_states`` directly if you need the ``|e>``
        component or the joint state.
        """
        states = np.atleast_2d(self.final_states)
        if states.ndim == 3:  # (K, 2, d) -> take |g> block
            states = states[:, 0, :]
        states = states / np.linalg.norm(states, axis=-1, keepdims=True)
        return states[0] if states.shape[0] == 1 else states


# ---------------------------------------------------------------------------
# Fidelity / loss helpers
# ---------------------------------------------------------------------------


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


def _disp_penalty(mags, max_disp, weight):
    if max_disp is None or weight == 0.0:
        return jnp.zeros((), dtype=mags.dtype)
    excess = jnp.maximum(mags - max_disp, 0.0)
    return weight * jnp.sum(excess**2)


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
# Optimizers
# ---------------------------------------------------------------------------


def cosine_schedule(n_iters: int, peak_lr: float, warmup_frac: float, final_frac: float):
    """Linear warm-up to ``peak_lr`` then cosine decay to ``final_frac * peak_lr``."""
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


def _adam_steps(loss_and_grad, state, steps, lr_at, b1=0.9, b2=0.999, eps=1e-8):
    """Advance an Adam ``(params, mom, vel)`` state over the iteration indices ``steps``."""

    def step(carry, i):
        params, mom, vel = carry
        val, grad = loss_and_grad(params)
        mom = b1 * mom + (1 - b1) * grad
        vel = b2 * vel + (1 - b2) * grad**2
        m_hat = mom / (1 - b1 ** (i + 1))
        v_hat = vel / (1 - b2 ** (i + 1))
        params = params - lr_at(i) * m_hat / (jnp.sqrt(v_hat) + eps)
        return (params, mom, vel), val

    return lax.scan(step, state, steps)


def adam_run(loss_and_grad, params0, n_iters, lr_at, b1=0.9, b2=0.999, eps=1e-8):
    """Plain Adam on a single parameter vector; returns (params, loss_history)."""
    init = (params0, jnp.zeros_like(params0), jnp.zeros_like(params0))
    (params, _, _), history = _adam_steps(
        loss_and_grad, init, jnp.arange(n_iters), lr_at, b1, b2, eps
    )
    return params, history


#: Compiled objective functions, keyed by sequence and loss configuration.
_COMPILED: dict = {}
_COMPILED_MAX = 16


def _compiled_objective(
    seq: GateSequence,
    loss_type: str,
    batch_reduction: str,
    bounds: GateBounds,
    trajectory: TrajectoryPenalty,
) -> dict:
    """Jitted loss, fidelity, diagnostics and Adam stepper for one problem type.

    The states are arguments rather than closure constants, so calls that
    differ only in their initial or target states, seeds or starting point
    reuse the compiled code. Sequences built by the :mod:`sequences` builders
    with equal options share an entry; hand-built ones are keyed by identity.
    """
    # Builder-made sequences with the same options run the same code objects.
    code = tuple(
        id(getattr(fn, "__code__", fn))
        for fn in (seq.propagate, seq.fidelity, seq.controls, seq.rotation_angles)
    )
    ident = tuple(sorted(seq.options.items())) if seq.options else id(seq)
    key = (seq.name, seq.n_gates, seq.n_fock, ident, code)
    key += (loss_type, batch_reduction, bounds, trajectory)
    hit = _COMPILED.get(key)
    if hit is not None and (seq.options or hit["seq"] is seq):
        return hit

    if seq.strength_kind == "squeeze":
        cap, cap_weight = bounds.max_squeeze, bounds.max_squeeze_weight
    else:
        cap, cap_weight = bounds.max_disp, bounds.max_disp_weight
    overlaps_defined = (not seq.has_qubit) or seq.qubit_target == "ground"

    def loss_fn(flat, psi0, psi_t, probes, geo_theta):
        path_probes = probes if trajectory.needs_path else None
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
                geo_theta if trajectory.geodesic_weight != 0.0 else None,
                trajectory.geodesic_mode,
            )
            loss = loss + _weighted_trajectory_penalty(terms, trajectory)
        return loss

    def fidelity(flat, psi0, psi_t):
        return seq.fidelity(seq.propagate(flat, psi0)[0], psi_t, batch_reduction)

    def diagnose(flat, psi0, psi_t, probes, geo_theta):
        psi_f, leak, mags, amps = seq.propagate(flat, psi0, probes)
        f_tube, a_tube = _geodesic_overlap(amps, geo_theta, "tube")
        f_schedule = _geodesic_overlap(amps, geo_theta, "schedule")[0]
        out = {
            "psi_f": psi_f,
            "fidelity": seq.fidelity(psi_f, psi_t, batch_reduction),
            "loss": loss_fn(flat, psi0, psi_t, probes, geo_theta),
            "leakage": leak,
            "mags": mags,
            "path": _path_fidelity(amps),
            "terms": _trajectory_terms(
                amps,
                seq.controls(flat),
                seq.rotation_angles(flat),
                geo_theta,
                trajectory.geodesic_mode,
            ),
            "geodesic": {
                "deviation": jnp.mean(1.0 - f_tube, axis=-1),
                "progress": jnp.mean(a_tube / geo_theta[None, :], axis=-1),
                "deviation_schedule": jnp.mean(1.0 - f_schedule, axis=-1),
            },
        }
        if overlaps_defined:
            out["per_pair"] = jnp.abs(seq.overlaps(psi_f, psi_t)) ** 2
        if seq.has_qubit:
            out["purity"] = _qubit_purity(psi_f)
        return out

    loss_and_grad = jax.jit(value_and_grad(loss_fn))

    @jax.jit
    def adam_chunk(state, steps, schedule, *data):
        """Advance every seed's Adam state over ``steps``; ``schedule`` is traced."""
        n_iters, peak_lr, warmup_frac, final_frac = schedule
        n_warm = jnp.maximum(1.0, jnp.round(warmup_frac * n_iters))
        n_decay = jnp.maximum(1.0, n_iters - n_warm)
        final_lr = peak_lr * final_frac

        def lr_at(i):
            i = i.astype(jnp.result_type(float))
            prog = jnp.clip((i - n_warm) / n_decay, 0.0, 1.0)
            lr_cos = final_lr + (peak_lr - final_lr) * 0.5 * (1 + jnp.cos(jnp.pi * prog))
            return jnp.where(i < n_warm, peak_lr * (i + 1) / n_warm, lr_cos)

        return vmap(lambda one: _adam_steps(lambda p: loss_and_grad(p, *data), one, steps, lr_at))(
            state
        )

    compiled = {
        "seq": seq,
        "loss_and_grad": loss_and_grad,
        "fidelities": jax.jit(vmap(fidelity, in_axes=(0, None, None))),
        "diagnose": jax.jit(diagnose),
        "adam_chunk": adam_chunk,
    }
    while len(_COMPILED) >= _COMPILED_MAX:
        _COMPILED.pop(next(iter(_COMPILED)))
    _COMPILED[key] = compiled
    return compiled


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
    init_label: str | None = None,
    target_label: str | None = None,
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
    init_label, target_label : str, optional
        Short names for the states (e.g. ``"vac"``, ``"gkp:0"``), recorded on
        the result so saved runs can be organized and searched by problem.
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
    >>> import jax, jax.numpy as jnp
    >>> jax.config.update("jax_enable_x64", True)
    >>> from gkp_optimal_control.states import fock_state, gkp_states
    >>> ell = jnp.sqrt(jnp.pi / 2)
    >>> gkp_0, _ = gkp_states(80, ell, 1j * ell, 0.4, 5)
    >>> res = optimize_gate_sequence(
    ...     ecd_sequence(12, 80, n_leak=8),
    ...     fock_state(80, 0), gkp_0,
    ...     loss_type="log_infidelity",
    ...     bounds=GateBounds(max_disp=4.0),
    ...     optimizer=OptimizerConfig(n_seeds=16, n_adam_iters=2000),
    ... )
    >>> res_csq = optimize_gate_sequence(
    ...     csq_sequence(32, 80, qubit_target="traced", n_leak=8),
    ...     fock_state(80, 0), gkp_0,
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

    psi_i = to_ket_batch(psi_init, seq.n_fock, "psi_init")
    psi_t = to_ket_batch(psi_targ, seq.n_fock, "psi_targ")
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

    # Probe kets tracked after every layer: [target, psi_i, psi_perp].
    geo_perp, geo_theta = _geodesic_frame(psi_i, psi_t)
    if trajectory.geodesic_weight != 0.0 and np.any(geo_theta < 1e-6):
        raise ValueError("geodesic penalty: an initial state already equals its target.")
    probes = jnp.stack([psi_t, psi_i, jnp.asarray(geo_perp)])
    psi0 = seq.lift(psi_i)
    data = (psi0, psi_t, probes, jnp.asarray(geo_theta))

    fn = _compiled_objective(seq, loss_type, batch_reduction, bounds, trajectory)

    # ----- initial parameters -------------------------------------------
    if params0 is None:
        keys = jax.random.split(jax.random.PRNGKey(optimizer.seed), optimizer.n_seeds)
        init_params = vmap(seq.init)(keys)
    else:
        init_params = jnp.asarray(params0, dtype=jnp.result_type(float))
        if init_params.ndim == 1:
            init_params = init_params[None, :]
        if init_params.shape[-1] != seq.n_params:
            raise ValueError(
                f"params0 has {init_params.shape[-1]} parameters, expected {seq.n_params}"
            )
    n_seeds = int(init_params.shape[0])
    n_iters = optimizer.n_adam_iters

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

    # ----- stage 1: batched Adam, in chunks ------------------------------
    # One chunk unless a fidelity target or seed pruning asks for checkpoints.
    t0 = time.time()
    checkpoints = optimizer.target_fidelity is not None or optimizer.prune_every is not None
    chunk = min(n_iters, optimizer.prune_every or 100) if checkpoints else n_iters
    schedule = jnp.asarray(
        [n_iters, optimizer.peak_lr, optimizer.warmup_frac, optimizer.final_lr_frac],
        dtype=jnp.result_type(float),
    )

    alive = np.arange(n_seeds)  # original indices of the seeds still running
    state = (init_params, jnp.zeros_like(init_params), jnp.zeros_like(init_params))
    final_params = np.array(init_params)
    seed_fids = np.zeros(n_seeds)
    adam_hist = np.full((n_seeds, n_iters), np.nan)
    done = 0
    while done < n_iters:
        steps = jnp.arange(done, min(done + chunk, n_iters))
        state, hist = fn["adam_chunk"](state, steps, schedule, *data)
        adam_hist[alive, done : done + steps.size] = np.asarray(hist)
        done += int(steps.size)
        fids = np.asarray(fn["fidelities"](state[0], psi0, psi_t))
        final_params[alive], seed_fids[alive] = np.asarray(state[0]), fids
        if optimizer.target_fidelity is not None and fids.max() >= optimizer.target_fidelity:
            break
        if optimizer.prune_every is not None and done < n_iters and alive.size > 1:
            n_keep = max(1, int(np.ceil(optimizer.keep_fraction * alive.size)))
            keep = np.sort(np.argsort(fids)[-n_keep:])
            if keep.size < alive.size:
                alive = alive[keep]
                state = tuple(x[keep] for x in state)
    t_adam = time.time() - t0

    best = int(np.argmax(seed_fids))
    best_params = jnp.asarray(final_params[best])

    if verbose:
        print(f"Adam ({done} iters x {n_seeds} seeds, {alive.size} finished) in {t_adam:.1f}s")
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
            val, grad = fn["loss_and_grad"](jnp.asarray(flat), *data)
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
        f_polished = float(fn["fidelities"](polished[None], psi0, psi_t)[0])
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
    diag = fn["diagnose"](best_params, *data)
    fid, leak, mags = float(diag["fidelity"]), float(diag["leakage"]), np.asarray(diag["mags"])
    fid_path = np.asarray(diag["path"])
    geo = {k: np.asarray(v) for k, v in diag["geodesic"].items()}
    purity = float(diag["purity"]) if "purity" in diag else None

    params = seq.describe(best_params)
    params["strength_magnitudes"] = mags
    if seq.strength_kind == "disp":
        params["disp_magnitudes"] = mags  # backwards-compatible key
    if "per_pair" in diag:
        params["per_pair_fidelity"] = np.asarray(diag["per_pair"])

    penalties = {}
    if trajectory.active:
        penalties = {f"pen_{k}": float(v) for k, v in diag["terms"].items()}

    if verbose:
        cap_label = "|r|" if seq.strength_kind == "squeeze" else "|disp|"
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
        loss=float(diag["loss"]),
        leakage=leak,
        params=params,
        flat_params=np.asarray(best_params),
        final_states=np.asarray(diag["psi_f"]),
        per_seed_fidelity=seed_fids,
        best_seed=best,
        adam_history=adam_hist,  # (n_seeds, n_iters); NaN after a seed stops
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
        target_state=np.asarray(psi_t),
        init_label=init_label,
        target_label=target_label,
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
    psi_i = to_ket_batch(psi_init, seq.n_fock, "psi_init")
    return np.asarray(seq.history(jnp.asarray(result.flat_params), seq.lift(psi_i)))
