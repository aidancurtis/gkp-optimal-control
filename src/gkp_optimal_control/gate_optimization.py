r"""Gate-level optimization of continuous-variable circuits.

This module optimizes *gate parameters* of a fixed-depth circuit ansatz, in
contrast to :mod:`grape`, which optimizes time-domain control pulses. The
gates are treated as ideal unitaries; no Kerr, no finite-duration effects, no
dispersive-shift phase accumulation.

Two gate sets are supported.

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
displacement relative to a ``2 * n_fock`` joint space. When a joint vector is
needed (e.g. to hand a state to ``jaxquantum``), the cavity-first convention
``index = n_cav * 2 + s_qubit`` is used, matching
:func:`hamiltonians.cavity_transmon_drift`.

**SNAP + displacements** (Heeres et al., *PRL* **115**, 137002 (2015);
Fösel, Krastanov et al., arXiv:2004.14256)

.. math::
    U = D(\alpha_{N+1}) \prod_{i=N}^{1} S(\vec\theta_i)\, D(\alpha_i),
    \qquad S(\vec\theta) = \sum_n e^{i\theta_n} |n\rangle\langle n|,

with :math:`N` SNAP gates and :math:`N+1` displacements. This gate set acts on
the cavity alone.

Optimization is batched multi-start Adam (vmapped over random seeds) followed
by an L-BFGS-B polish of the best seed, which is the standard recipe for these
ansaetze: the landscape is far more non-convex than GRAPE's, and single-start
gradient descent routinely stalls in poor local minima.

Notes
-----
Run with 64-bit precision enabled::

    import jax
    jax.config.update("jax_enable_x64", True)

Displacements are built by one of two methods, selected with ``disp_method``:

``"expm"`` (default)
    :math:`D(\alpha) = \exp(\alpha a^\dagger - \alpha^* a)` of the truncated
    generator. Exactly unitary in the truncated space.
``"quadrature"``
    Exact BCH factorization :math:`D(\alpha) = e^{-i x_0 p_0}
    e^{i\sqrt{2} p_0 \hat{x}} e^{-i\sqrt{2} x_0 \hat{p}}` using
    eigendecompositions of :math:`\hat x` and :math:`\hat p` precomputed once.
    Also exactly unitary, benchmarked to the same truncation accuracy as
    ``"expm"``, roughly 4x faster to evaluate, and its gradients flow through a
    diagonal exponential rather than an ``expm`` Frechet derivative. Recommended
    for large seed batches or deep circuits.

A third possibility -- building :math:`\langle m|D(\alpha)|n\rangle` from its
closed form -- is deliberately *not* offered. Those are the exact
infinite-dimensional matrix elements restricted to the truncated block, so the
resulting matrix is not unitary (numerically ``||D^dag D - I|| ~ 0.5`` at
``n_fock = 40``); norm is not conserved and an optimizer will happily exploit
that to report fidelities it has not achieved.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, value_and_grad, vmap
from jax.scipy.linalg import expm
from scipy.optimize import minimize

try:  # package-relative import, with a flat-layout fallback
    from .hamiltonians import cavity_operators
except ImportError:  # pragma: no cover
    from hamiltonians import cavity_operators

try:
    import jaxquantum as jqt
except ImportError:  # pragma: no cover
    jqt = None


__all__ = [
    "GateBounds",
    "OptimizerConfig",
    "GateOptResult",
    "GateSequence",
    "optimize_gate_sequence",
    "build_sequence",
    "make_displacement",
    "sequence_history",
    "to_joint_ket",
    "GATE_SETS",
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
    n_leak : int
        Number of top Fock levels treated as leakage. Population there is
        penalized after every displacement in the circuit, not only at the end,
        since intermediate displacements are what push a state into the
        truncation boundary.
    leakage_weight : float
        Weight of the leakage penalty. Set to 0 to disable.
    """

    max_disp: float | None = None
    max_disp_weight: float = 1.0
    n_leak: int = 5
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
    seed : int
        PRNG seed for the initializations.
    init_disp_scale : float
        Scale of the random initial displacements (``|beta|`` or ``|alpha|``).
    """

    n_seeds: int = 8
    n_adam_iters: int = 1500
    peak_lr: float = 0.03
    warmup_frac: float = 0.05
    final_lr_frac: float = 0.05
    polish: bool = True
    polish_maxiter: int = 500
    seed: int = 0
    init_disp_scale: float = 1.0


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
    polish_info: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

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
        return "\n".join(lines)

    def final_qarray(self):
        """Return the final cavity state(s) as a ``jaxquantum.Qarray``.

        For the ECD gate set this is the ``|g>`` component only, renormalized;
        inspect ``final_states`` directly if you need the ``|e>`` component or
        the joint state.
        """
        if jqt is None:  # pragma: no cover
            raise ImportError("jaxquantum is required for final_qarray().")
        states = np.atleast_2d(self.final_states)
        if states.ndim == 3:  # (K, 2, d) -> take |g> block
            states = states[:, 0, :]
        if states.shape[0] == 1:
            return jqt.Qarray.create(states[0].reshape(-1, 1)).unit()
        return [jqt.Qarray.create(s.reshape(-1, 1)).unit() for s in states]


# ---------------------------------------------------------------------------
# Displacement operators
# ---------------------------------------------------------------------------


def make_displacement(n_fock: int, method: str = "expm") -> Callable:
    r"""Return a jittable ``alpha -> D(alpha)`` closure.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation.
    method : {"expm", "quadrature"}
        See the module docstring. Both are exactly unitary on the truncated
        space and benchmark to the same accuracy; ``"quadrature"`` is faster.
    """
    a, adag, _ = cavity_operators(n_fock)

    if method == "expm":

        def displace(alpha):
            alpha = jnp.asarray(alpha, dtype=a.dtype)
            return expm(alpha * adag - jnp.conj(alpha) * a)

    elif method == "quadrature":
        root2 = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.real(a).dtype))
        x_op = (a + adag) / root2
        p_op = 1j * (adag - a) / root2
        w_x, v_x = jnp.linalg.eigh(x_op)
        w_p, v_p = jnp.linalg.eigh(p_op)
        v_x_dag = v_x.conj().T
        v_p_dag = v_p.conj().T

        def displace(alpha):
            alpha = jnp.asarray(alpha, dtype=a.dtype)
            x_0 = jnp.real(alpha)
            p_0 = jnp.imag(alpha)
            e_x = v_x @ (jnp.exp(1j * root2 * p_0 * w_x)[:, None] * v_x_dag)
            e_p = v_p @ (jnp.exp(-1j * root2 * x_0 * w_p)[:, None] * v_p_dag)
            return jnp.exp(-1j * x_0 * p_0) * (e_x @ e_p)

    else:
        raise ValueError(
            f"unknown disp_method {method!r}; expected 'expm' or 'quadrature'. "
            "The closed-form Fock matrix elements are not offered because they "
            "are not unitary under truncation (see module docstring)."
        )

    return displace


def qubit_rotation(theta, phi):
    r"""Equatorial qubit rotation :math:`R(\theta,\phi)` as a 2x2 matrix.

    ``R = exp[-i theta/2 (sigma_x cos phi + sigma_y sin phi)]``, the unitary
    generated by a resonant drive of pulse area ``theta`` and phase ``phi``.
    """
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)
    cos = jnp.cos(theta / 2) + 0j
    sin = jnp.sin(theta / 2) + 0j
    off_lo = -1j * jnp.exp(1j * phi) * sin
    off_hi = -1j * jnp.exp(-1j * phi) * sin
    return jnp.stack([jnp.stack([cos, off_hi]), jnp.stack([off_lo, cos])])


def to_joint_ket(psi_blocks: np.ndarray) -> np.ndarray:
    """Flatten a block state ``(..., 2, n_fock)`` into a cavity-first joint ket.

    The output index convention is ``index = n_cav * 2 + s_qubit``, i.e.
    ``kron(cavity, qubit)``, matching :mod:`hamiltonians`.
    """
    arr = np.asarray(psi_blocks)
    return np.swapaxes(arr, -1, -2).reshape(*arr.shape[:-2], -1)


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
        if jqt is not None and isinstance(state, jqt.Qarray):
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


# ---------------------------------------------------------------------------
# Gate-sequence definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateSequence:
    """A concrete circuit ansatz: parameter layout plus propagation rules.

    Attributes
    ----------
    name, n_gates, n_fock, n_params
        Problem sizes. ``n_gates`` counts ECDs or SNAPs, not total gates.
    unpack : callable
        ``flat -> pytree`` of gate parameters.
    lift : callable
        ``(K, n_fock) -> initial internal state`` (adds the qubit block for ECD).
    propagate : callable
        ``(flat, psi0) -> (psi_final, leakage, disp_mags)``.
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
    """

    name: str
    n_gates: int
    n_fock: int
    n_params: int
    unpack: Callable
    lift: Callable
    propagate: Callable
    history: Callable
    overlaps: Callable
    fidelity: Callable
    init: Callable
    describe: Callable


def _build_ecd_sequence(
    n_gates: int,
    n_fock: int,
    *,
    disp_method: str = "expm",
    echoed: bool = True,
    qubit_target: str = "ground",
    n_leak: int = 5,
    init_disp_scale: float = 1.0,
    **_ignored,
) -> GateSequence:
    r"""ECD + equatorial rotations, propagated in the 2x2 qubit block basis.

    Parameters
    ----------
    echoed : bool
        ``True`` for the echoed conditional displacement (includes the qubit
        flip), ``False`` for the bare conditional displacement
        ``D(beta/2)|g><g| + D(-beta/2)|e><e|``.
    qubit_target : {"ground", "traced"}
        ``"ground"`` scores ``|<psi_t| psi_g>|**2``, which rewards returning the
        qubit to ``|g>`` and disentangling it -- the standard choice.
        ``"traced"`` scores ``<psi_t| rho_cav |psi_t>``, allowing the qubit to
        end anywhere, which is only meaningful if you intend to discard it.
    """
    if qubit_target not in ("ground", "traced"):
        raise ValueError(f"unknown qubit_target {qubit_target!r}")

    displace = make_displacement(n_fock, disp_method)
    n_rot = n_gates + 1
    n_params = 2 * n_rot + 2 * n_gates  # (theta, phi) per rotation, (Re, Im) per beta

    def unpack(flat):
        flat = jnp.asarray(flat)
        rots = flat[: 2 * n_rot].reshape(n_rot, 2)
        betas = flat[2 * n_rot :].reshape(n_gates, 2)
        return rots, betas

    def lift(psi_cav):
        """(K, d) cavity kets -> (K, 2, d) with the qubit in |g>."""
        zeros = jnp.zeros_like(psi_cav)
        return jnp.stack([psi_cav, zeros], axis=1)

    def apply_rot(psi, theta, phi):
        rot = qubit_rotation(theta, phi)
        return jnp.einsum("ij,kjd->kid", rot, psi)

    def apply_ecd(psi, beta):
        d_op = displace(beta / 2)
        psi_g, psi_e = psi[:, 0, :], psi[:, 1, :]
        if echoed:
            out_g = psi_e @ d_op.conj()  # D(-beta/2) psi_e  ==  D(beta/2)^dag psi_e
            out_e = psi_g @ d_op.T  # D(beta/2) psi_g
        else:
            out_g = psi_g @ d_op.T
            out_e = psi_e @ d_op.conj()
        return jnp.stack([out_g, out_e], axis=1)

    def propagate(flat, psi0):
        rots, betas = unpack(flat)
        psi = apply_rot(psi0, rots[0, 0], rots[0, 1])
        leak0 = jnp.zeros((), dtype=jnp.real(psi).dtype)

        def step(carry, layer):
            psi, leak = carry
            beta_ri, rot = layer
            beta = beta_ri[0] + 1j * beta_ri[1]
            psi = apply_ecd(psi, beta)
            leak = leak + _leakage(psi, n_leak)
            psi = apply_rot(psi, rot[0], rot[1])
            return (psi, leak), None

        (psi, leak), _ = lax.scan(step, (psi, leak0), (betas, rots[1:]))
        disp_mags = jnp.linalg.norm(betas, axis=-1)
        return psi, leak, disp_mags

    def history(flat, psi0):
        """States after every gate: ``(2*n_gates+2, K, 2, n_fock)``."""
        rots, betas = unpack(flat)
        psi = psi0
        out = [psi]
        psi = apply_rot(psi, rots[0, 0], rots[0, 1])
        out.append(psi)
        for i in range(n_gates):
            psi = apply_ecd(psi, betas[i, 0] + 1j * betas[i, 1])
            out.append(psi)
            psi = apply_rot(psi, rots[i + 1, 0], rots[i + 1, 1])
            out.append(psi)
        return jnp.stack(out)

    def overlaps(psi_final, targets):
        if qubit_target != "ground":
            raise ValueError(
                "overlaps are only defined for qubit_target='ground'; "
                "batch_reduction='coherent' is incompatible with 'traced'."
            )
        return jnp.sum(jnp.conj(targets) * psi_final[:, 0, :], axis=-1)

    def fidelity(psi_final, targets, reduction):
        if qubit_target == "ground":
            return _reduce_fidelity(overlaps(psi_final, targets), reduction)
        # traced: F_k = <t| rho_cav |t> = |<t|psi_g>|^2 + |<t|psi_e>|^2
        ov = jnp.sum(jnp.conj(targets)[:, None, :] * psi_final, axis=-1)
        return jnp.mean(jnp.sum(jnp.abs(ov) ** 2, axis=-1))

    def init(key):
        k_rot, k_phi, k_beta = jax.random.split(key, 3)
        thetas = jax.random.uniform(k_rot, (n_rot,), minval=0.0, maxval=jnp.pi)
        phis = jax.random.uniform(k_phi, (n_rot,), minval=-jnp.pi, maxval=jnp.pi)
        betas = init_disp_scale * jax.random.normal(k_beta, (n_gates, 2))
        return jnp.concatenate([jnp.stack([thetas, phis], axis=-1).ravel(), betas.ravel()])

    def describe(flat):
        rots, betas = unpack(flat)
        rots = np.asarray(rots)
        betas = np.asarray(betas)
        return {
            "thetas": rots[:, 0],
            "phis": rots[:, 1],
            "betas": betas[:, 0] + 1j * betas[:, 1],
        }

    return GateSequence(
        name="ecd",
        n_gates=n_gates,
        n_fock=n_fock,
        n_params=n_params,
        unpack=unpack,
        lift=lift,
        propagate=propagate,
        history=history,
        overlaps=overlaps,
        fidelity=fidelity,
        init=init,
        describe=describe,
    )


def _build_snap_sequence(
    n_gates: int,
    n_fock: int,
    *,
    disp_method: str = "expm",
    n_snap: int | None = None,
    n_leak: int = 5,
    init_disp_scale: float = 1.0,
    **_ignored,
) -> GateSequence:
    r"""SNAP + displacements, sandwiched as ``D S D S ... D``.

    Parameters
    ----------
    n_snap : int or None
        Number of Fock phases optimized per SNAP gate. ``None`` (default)
        optimizes all ``n_fock`` phases. A smaller value pins the phases of
        levels ``n >= n_snap`` to zero, which is the physically honest choice
        when the selective qubit pulses only resolve low photon numbers.
    """
    n_snap_eff = int(n_fock if n_snap is None else n_snap)
    if not 1 <= n_snap_eff <= n_fock:
        raise ValueError(f"n_snap must lie in [1, n_fock]; got {n_snap}")

    displace = make_displacement(n_fock, disp_method)
    n_disp = n_gates + 1
    n_params = n_gates * n_snap_eff + 2 * n_disp
    pad = n_fock - n_snap_eff

    def unpack(flat):
        flat = jnp.asarray(flat)
        n_theta = n_gates * n_snap_eff
        thetas = flat[:n_theta].reshape(n_gates, n_snap_eff)
        alphas = flat[n_theta:].reshape(n_disp, 2)
        return thetas, alphas

    def lift(psi_cav):
        return psi_cav

    def apply_snap(psi, theta):
        phases = jnp.concatenate([theta, jnp.zeros((pad,), dtype=theta.dtype)])
        return psi * jnp.exp(1j * phases)[None, :]

    def apply_disp(psi, alpha):
        return psi @ displace(alpha).T

    def propagate(flat, psi0):
        thetas, alphas = unpack(flat)
        psi = apply_disp(psi0, alphas[0, 0] + 1j * alphas[0, 1])
        leak = _leakage(psi, n_leak)

        def step(carry, layer):
            psi, leak = carry
            theta, alpha_ri = layer
            psi = apply_snap(psi, theta)
            psi = apply_disp(psi, alpha_ri[0] + 1j * alpha_ri[1])
            leak = leak + _leakage(psi, n_leak)
            return (psi, leak), None

        (psi, leak), _ = lax.scan(step, (psi, leak), (thetas, alphas[1:]))
        disp_mags = jnp.linalg.norm(alphas, axis=-1)
        return psi, leak, disp_mags

    def history(flat, psi0):
        """States after every gate: ``(2*n_gates+2, K, n_fock)``."""
        thetas, alphas = unpack(flat)
        psi = psi0
        out = [psi]
        psi = apply_disp(psi, alphas[0, 0] + 1j * alphas[0, 1])
        out.append(psi)
        for i in range(n_gates):
            psi = apply_snap(psi, thetas[i])
            out.append(psi)
            psi = apply_disp(psi, alphas[i + 1, 0] + 1j * alphas[i + 1, 1])
            out.append(psi)
        return jnp.stack(out)

    def overlaps(psi_final, targets):
        return jnp.sum(jnp.conj(targets) * psi_final, axis=-1)

    def fidelity(psi_final, targets, reduction):
        return _reduce_fidelity(overlaps(psi_final, targets), reduction)

    def init(key):
        k_theta, k_alpha = jax.random.split(key)
        thetas = jax.random.uniform(k_theta, (n_gates, n_snap_eff), minval=-jnp.pi, maxval=jnp.pi)
        alphas = init_disp_scale * jax.random.normal(k_alpha, (n_disp, 2))
        return jnp.concatenate([thetas.ravel(), alphas.ravel()])

    def describe(flat):
        thetas, alphas = unpack(flat)
        alphas = np.asarray(alphas)
        return {
            "snap_phases": np.asarray(thetas),
            "alphas": alphas[:, 0] + 1j * alphas[:, 1],
        }

    return GateSequence(
        name="snap",
        n_gates=n_gates,
        n_fock=n_fock,
        n_params=n_params,
        unpack=unpack,
        lift=lift,
        propagate=propagate,
        history=history,
        overlaps=overlaps,
        fidelity=fidelity,
        init=init,
        describe=describe,
    )


GATE_SETS: dict[str, Callable[..., GateSequence]] = {
    "ecd": _build_ecd_sequence,
    "snap": _build_snap_sequence,
}

_GATE_SET_ALIASES = {
    "ecd": "ecd",
    "ecd+rotations": "ecd",
    "ecd_rotations": "ecd",
    "ecd+qubit_rotations": "ecd",
    "snap": "snap",
    "snap+displacements": "snap",
    "snap_displacements": "snap",
    "snap+disp": "snap",
}


def build_sequence(gate_set: str, n_gates: int, n_fock: int, **kwargs) -> GateSequence:
    """Construct a :class:`GateSequence` for a named gate set."""
    key = _GATE_SET_ALIASES.get(str(gate_set).strip().lower())
    if key is None:
        raise ValueError(
            f"unknown gate_set {gate_set!r}; expected one of {sorted(set(_GATE_SET_ALIASES))}"
        )
    if n_gates < 1:
        raise ValueError(f"n_gates must be >= 1; got {n_gates}")
    return GATE_SETS[key](n_gates, n_fock, **kwargs)


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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def optimize_gate_sequence(
    gate_set: str,
    n_gates: int,
    psi_init,
    psi_targ,
    n_fock: int | None = None,
    *,
    # --- ansatz options -------------------------------------------------
    n_snap: int | None = None,
    echoed: bool = True,
    qubit_target: str = "ground",
    disp_method: str = "expm",
    # --- objective ------------------------------------------------------
    loss_type: str = "infidelity",
    batch_reduction: str = "mean",
    bounds: GateBounds | None = None,
    # --- optimizer ------------------------------------------------------
    optimizer: OptimizerConfig | None = None,
    params0: np.ndarray | None = None,
    verbose: bool = True,
) -> GateOptResult:
    r"""Optimize the gate parameters of a CV circuit for state preparation.

    Parameters
    ----------
    gate_set : {"ecd", "snap"}
        ``"ecd"`` for echoed conditional displacements interleaved with
        equatorial qubit rotations; ``"snap"`` for SNAP gates sandwiched between
        displacements. Aliases such as ``"ecd+rotations"`` and
        ``"snap+displacements"`` are accepted.
    n_gates : int
        Circuit depth, counted in ECDs or SNAPs. The ECD ansatz then carries
        ``n_gates + 1`` rotations, the SNAP ansatz ``n_gates + 1`` displacements.
    psi_init, psi_targ : Qarray or array_like
        Cavity kets, shape ``(n_fock,)``, ``(n_fock, 1)``, ``(K, n_fock)`` or a
        list thereof. With ``K > 1`` one parameter set is optimized for all
        pairs simultaneously, which is how you target a *gate* on a logical
        subspace rather than a single state. For the ECD set these are cavity
        states only: the qubit is assumed to start in ``|g>`` and (with
        ``qubit_target="ground"``) is required to return there.
    n_fock : int, optional
        Truncation. Inferred from the states if omitted.
    n_snap : int, optional
        Fock phases optimized per SNAP gate; default all ``n_fock``.
    echoed : bool
        Use the echoed conditional displacement (default) or the bare one.
    qubit_target : {"ground", "traced"}
        Whether the qubit must return to ``|g>`` or may be discarded.
    disp_method : {"expm", "quadrature"}
        How to build ``D(alpha)``. ``"quadrature"`` is ~4x faster at equal
        accuracy and is recommended for large seed batches.
    loss_type : {"infidelity", "log_infidelity", "neg_fidelity"}
        ``"log_infidelity"`` is usually the better choice once you are pushing
        past ``F = 0.99``, where ``1 - F`` is nearly flat.
    batch_reduction : {"mean", "coherent"}
        How multiple state pairs are combined. ``"mean"`` averages the
        fidelities. ``"coherent"`` averages the *overlaps* before squaring,
        which is the right objective for a gate on a subspace (it fixes the
        relative phases between logical basis states, up to one global phase).
    bounds : GateBounds, optional
        Soft displacement cap and Fock-leakage penalty.
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
        fidelity, final states, per-seed fidelities and the Adam loss history.

    Examples
    --------
    >>> import jax, jax.numpy as jnp, jaxquantum as jqt
    >>> jax.config.update("jax_enable_x64", True)
    >>> from states import gkp_states
    >>> ell = jnp.sqrt(jnp.pi / 2)
    >>> gkp_0, _ = gkp_states(80, ell, 1j * ell, 0.4, 5)
    >>> res = optimize_gate_sequence(
    ...     "ecd", 12, jqt.basis(80, 0), gkp_0,
    ...     loss_type="log_infidelity",
    ...     bounds=GateBounds(max_disp=4.0, n_leak=8),
    ...     optimizer=OptimizerConfig(n_seeds=16, n_adam_iters=2000),
    ... )
    >>> print(res.summary())
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

    psi_i = _as_ket_batch(psi_init, n_fock, "psi_init")
    n_fock = int(psi_i.shape[-1]) if n_fock is None else int(n_fock)
    psi_t = _as_ket_batch(psi_targ, n_fock, "psi_targ")
    if psi_i.shape[0] != psi_t.shape[0]:
        raise ValueError(
            f"psi_init has {psi_i.shape[0]} state(s) but psi_targ has "
            f"{psi_t.shape[0]}; they must pair up."
        )
    n_pairs = int(psi_i.shape[0])

    seq = build_sequence(
        gate_set,
        n_gates,
        n_fock,
        disp_method=disp_method,
        echoed=echoed,
        qubit_target=qubit_target,
        n_snap=n_snap,
        n_leak=bounds.n_leak,
        init_disp_scale=optimizer.init_disp_scale,
    )
    if batch_reduction == "coherent" and seq.name == "ecd" and qubit_target != "ground":
        raise ValueError(
            "batch_reduction='coherent' requires qubit_target='ground' "
            "(a traced-out qubit has no well-defined overlap phase)."
        )

    psi0 = seq.lift(psi_i)

    # ----- objective -----------------------------------------------------
    def loss_fn(flat):
        psi_f, leak, disp_mags = seq.propagate(flat, psi0)
        fid = seq.fidelity(psi_f, psi_t, batch_reduction)
        loss = _apply_loss_style(fid, loss_type)
        loss = loss + bounds.leakage_weight * leak
        loss = loss + _disp_penalty(disp_mags, bounds.max_disp, bounds.max_disp_weight)
        return loss

    overlaps_defined = seq.name == "snap" or qubit_target == "ground"

    def diagnose(flat):
        psi_f, leak, disp_mags = seq.propagate(jnp.asarray(flat), psi0)
        fid = seq.fidelity(psi_f, psi_t, batch_reduction)
        per_pair = None
        if overlaps_defined:
            per_pair = np.asarray(jnp.abs(seq.overlaps(psi_f, psi_t)) ** 2)
        return psi_f, float(fid), float(leak), np.asarray(disp_mags), per_pair

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
        print(f"gate set : {seq.name}   depth : {n_gates}   n_fock : {n_fock}")
        print(
            f"params   : {seq.n_params}   state pairs : {n_pairs}   reduction : {batch_reduction}"
        )
        print(
            f"loss     : {loss_type}   disp : {disp_method}   seeds : {n_seeds}"
            + (f"   n_snap : {n_snap}" if seq.name == "snap" and n_snap else "")
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
    psi_f, fid, leak, disp_mags, per_pair = diagnose(best_params)
    loss_val = float(loss_fn(best_params))

    params = seq.describe(best_params)
    params["disp_magnitudes"] = disp_mags
    if per_pair is not None:
        params["per_pair_fidelity"] = per_pair

    if verbose:
        print("-" * 62)
        print(f"Final fidelity : {fid:.6f}   (infidelity {1 - fid:.3e})")
        print(f"Leakage        : {leak:.3e}  (top {bounds.n_leak} Fock levels)")
        print(f"Peak |disp|    : {disp_mags.max():.3f}")
        print(f"Total time     : {time.time() - t0:.1f}s")

    return GateOptResult(
        gate_set=seq.name,
        n_gates=n_gates,
        n_fock=n_fock,
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
            "disp_method": disp_method,
            "echoed": echoed,
            "qubit_target": qubit_target,
            "n_snap": n_snap,
            "bounds": bounds,
            "optimizer": optimizer,
            "n_pairs": n_pairs,
        },
    )


def sequence_history(result: GateOptResult, psi_init, **kwargs) -> np.ndarray:
    """Re-run an optimized sequence and return the state after every gate.

    Useful for Wigner-function movies of the preparation: pair the output with
    :func:`utils.wigner_trajectory` or :mod:`animation`.

    Returns
    -------
    ndarray
        ``(2 * n_gates + 2, K, n_fock)`` for the SNAP set, or
        ``(2 * n_gates + 2, K, 2, n_fock)`` for the ECD set, where the
        second-to-last axis indexes the qubit block ``(|g>, |e>)``.
    """
    cfg = result.config
    psi_i = _as_ket_batch(psi_init, result.n_fock, "psi_init")
    seq = build_sequence(
        result.gate_set,
        result.n_gates,
        result.n_fock,
        disp_method=kwargs.pop("disp_method", cfg.get("disp_method", "expm")),
        echoed=cfg.get("echoed", True),
        qubit_target=cfg.get("qubit_target", "ground"),
        n_snap=cfg.get("n_snap"),
        n_leak=cfg.get("bounds", GateBounds()).n_leak,
        **kwargs,
    )
    return np.asarray(seq.history(jnp.asarray(result.flat_params), seq.lift(psi_i)))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    n_fock = 30
    vac = jnp.zeros(n_fock, dtype=jnp.complex128).at[0].set(1.0)

    # Displacement methods must agree and be unitary.
    for method in ("expm", "quadrature"):
        d_op = make_displacement(n_fock, method)(1.0 + 0.5j)
        err = jnp.abs(d_op.conj().T @ d_op - jnp.eye(n_fock)).max()
        print(f"{method:>11}: ||D^dag D - I|| = {err:.2e}")
    d_a = make_displacement(n_fock, "expm")(1.0 + 0.5j)
    d_b = make_displacement(n_fock, "quadrature")(1.0 + 0.5j)
    print(f"expm vs quadrature: {jnp.abs(d_a - d_b).max():.2e}\n")

    # Fock |2> with a shallow SNAP circuit.
    fock2 = jnp.zeros(n_fock, dtype=jnp.complex128).at[2].set(1.0)
    res_snap = optimize_gate_sequence(
        "snap+displacements",
        3,
        vac,
        fock2,
        n_snap=8,
        loss_type="log_infidelity",
        optimizer=OptimizerConfig(n_seeds=6, n_adam_iters=600, seed=1),
    )
    print()

    # Even cat with a shallow ECD circuit.
    alpha = 1.5
    n_vec = jnp.arange(n_fock, dtype=jnp.float64)
    log_coh = n_vec * jnp.log(alpha) - 0.5 * jax.scipy.special.gammaln(n_vec + 1.0)
    cat = jnp.exp(log_coh) * (1 + (-1.0) ** n_vec)  # even cat, unnormalized
    cat = (cat / jnp.linalg.norm(cat)).astype(jnp.complex128)
    res_ecd = optimize_gate_sequence(
        "ecd",
        4,
        vac,
        cat,
        disp_method="quadrature",
        loss_type="log_infidelity",
        bounds=GateBounds(max_disp=4.0, n_leak=4),
        optimizer=OptimizerConfig(n_seeds=6, n_adam_iters=600, seed=2),
    )
    print()
    print(res_ecd.summary())
