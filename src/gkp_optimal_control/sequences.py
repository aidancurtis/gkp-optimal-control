r"""Gate-sequence ansaetze: ECD, SNAP and conditional-squeezing circuits.

Each builder (:func:`ecd_sequence`, :func:`snap_sequence`, :func:`csq_sequence`)
returns a :class:`GateSequence`: a parameter layout plus jittable propagation
rules, which :func:`gate_optimization.optimize_gate_sequence` optimizes. The
gates are ideal unitaries; no Kerr, no finite-duration effects, no
dispersive-shift phase accumulation.

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

Notes
-----
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
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .operators import (
    make_displacement,
    make_displacement_apply,
    make_squeeze_apply,
    qubit_rotation,
)

__all__ = [
    "GateSequence",
    "ecd_sequence",
    "snap_sequence",
    "csq_sequence",
    "build_sequence",
    "leakage",
    "reduce_fidelity",
]


# ---------------------------------------------------------------------------
# Fidelity and leakage of a propagated state
# ---------------------------------------------------------------------------


def reduce_fidelity(overlaps, reduction: str):
    """Combine per-pair overlaps into a scalar fidelity."""
    if reduction == "mean":
        return jnp.mean(jnp.abs(overlaps) ** 2)
    if reduction == "coherent":
        # Phase-coherent average: correct for optimizing a *gate* on a
        # subspace spanned by orthonormal inputs, up to one global phase.
        return jnp.abs(jnp.mean(overlaps)) ** 2
    raise ValueError(f"unknown batch_reduction {reduction!r}")


def leakage(psi, n_leak: int):
    """Mean population in the top ``n_leak`` Fock levels, averaged over batch."""
    if n_leak <= 0:
        return jnp.zeros((), dtype=jnp.real(psi).dtype)
    tail = jnp.abs(psi[..., -n_leak:]) ** 2
    return jnp.sum(tail) / psi.shape[0]


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
        leak = leakage(psi, n_leak) if leak_after_first else jnp.zeros((), jnp.real(psi).dtype)
        psi_first = psi

        def step(carry, layer_params):
            psi, leak = carry
            for sub, counts_leak in layer(psi, layer_params):
                if counts_leak:
                    leak = leak + leakage(sub, n_leak)
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
        return reduce_fidelity(overlaps(psi_final, targets), reduction)

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
